import argparse
import time
import os
import sys
from pathlib import Path
import threading
import queue
import cv2
import numpy as np
import insightface

"""
Multi-threaded face identification benchmark using InsightFace FaceAnalysis.

Architecture:
 - Reader thread decodes video -> (frame_index, frame) into input_queue.
 - N worker threads each create their own FaceAnalysis instance (to avoid
   potential thread-safety issues) and pull frames, run detection + similarity,
   annotate frame, push (frame_index, annotated_frame, stats_dict) to output_queue.
 - Writer thread preserves original ordering: buffers out-of-order results in a dict
   and writes sequentially to VideoWriter once the next expected index is ready.
 - Main thread gathers stats & prints summary.

Ordering Strategy:
 Results may finish out-of-order due to differing processing times. Writer keeps
 a "next_to_write" counter and a dictionary buffer. When a result arrives:
   buffer[idx] = result
   while next_to_write in buffer: flush/write & next_to_write += 1
 This preserves ordering with minimal locking.

Graceful Shutdown:
 Reader sends a sentinel (None) for each worker after finishing.
 Workers forward a sentinel to output_queue when they exit so writer knows when
 all workers are done. Writer stops when it has written all frames AND received
 all worker sentinels.

Why threads (vs processes)?
 ONNXRuntime / CoreML provider releases the GIL during heavy compute, so threads
 can still offer concurrency without duplicating large model memory per process.

Benchmark Metrics:
 - Total frames, processed frames, frames with identification.
 - End-to-end elapsed time and effective FPS.
 - Per-frame processing latency distribution (optional CSV export).

Example:
 python detect_face_multithreaded.py \
   --model-pack buffalo_s \
   --ref-images assets/raw_face.webp assets/person1.webp \
   --video assets/raw_video.mp4 \
   --workers 4 --providers CoreMLExecutionProvider CPUExecutionProvider

"""

MODEL_PACKS = ["antelopev2", "buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc"]
SENTINEL = None


def parse_args():
    p = argparse.ArgumentParser(description="Multi-threaded InsightFace video face identifier")
    p.add_argument('--model-pack', default='buffalo_l', choices=MODEL_PACKS,
                   help='InsightFace model pack name')
    # UPDATED: allow multiple reference images
    p.add_argument('--ref-images', required=True, nargs='+', help='Paths to one or more reference (target) face images')
    p.add_argument('--video', required=True, help='Path to input video file')
    p.add_argument('--output', default='output_mt.mp4', help='Output annotated video path')
    p.add_argument('--providers', nargs='*', default=None,
                   help='Execution providers list. Example: CoreMLExecutionProvider CPUExecutionProvider')
    p.add_argument('--det-size', type=int, nargs=2, default=[640, 640],
                   help='Detection input size (width height)')
    p.add_argument('--similarity-threshold', type=float, default=0.3,
                   help='Cosine similarity threshold (0-1). Higher = stricter.')
    p.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    p.add_argument('--queue-size', type=int, default=32, help='Max frames queued to workers')
    p.add_argument('--no-display', action='store_true', help='Disable preview window')
    p.add_argument('--show-all-sim', action='store_true', help='Print similarity of all faces per frame')
    p.add_argument('--save-csv', default=None, help='Optional CSV of per-frame stats')
    p.add_argument('--max-frames', type=int, default=None, help='Limit frames (debug)')
    return p.parse_args()


def select_providers(user_providers):
    if user_providers:
        return user_providers
    providers = []
    try:
        providers.append('CoreMLExecutionProvider')
    except Exception:
        pass
    providers.append('CPUExecutionProvider')
    return providers


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# NEW: load multiple reference images
# Returns (names:list[str], embeddings: np.ndarray [N, D])
def load_reference_embeddings(pack, providers, det_size, ref_image_paths):
    app = insightface.app.FaceAnalysis(name=pack, providers=providers)
    app.prepare(ctx_id=0, det_size=tuple(det_size))
    names = []
    embeddings = []
    for path in ref_image_paths:
        if not os.path.isfile(path):
            print(f"[WARN] Reference image not found, skipping: {path}")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Failed to read reference image, skipping: {path}")
            continue
        faces = app.get(img)
        if len(faces) == 0:
            print(f"[WARN] No face detected in reference image, skipping: {path}")
            continue
        names.append(Path(path).stem)
        embeddings.append(faces[0].normed_embedding)
        print(f"Loaded reference '{names[-1]}' from {path}")
    if not embeddings:
        raise ValueError("No valid faces extracted from provided reference images.")
    emb_arr = np.stack(embeddings, axis=0)
    print(f"Total reference identities loaded: {len(names)}")
    return names, emb_arr


def open_video(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def create_writer(cap, output_path: str):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))
    return writer, fps_in


# UPDATED: accept multiple reference embeddings
# ref_embeddings: np.ndarray [N, D], ref_names: list[str]
def worker_thread(idx, args, providers, ref_embeddings, ref_names, det_size, in_q: queue.Queue, out_q: queue.Queue,
                   similarity_threshold: float, show_all_sim: bool):
    model = insightface.app.FaceAnalysis(name=args.model_pack, providers=providers)
    model.prepare(ctx_id=0, det_size=tuple(det_size))
    while True:
        item = in_q.get()
        if item is SENTINEL:
            out_q.put(SENTINEL)
            in_q.task_done()
            break
        frame_index, frame = item
        t_start = time.time()
        faces = model.get(frame)
        frame_best_sim = None
        identified = False
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            emb = face.normed_embedding
            # Vectorized similarity (embeddings are already L2-normalized)
            sims = ref_embeddings @ emb
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_name = ref_names[best_idx]
            if show_all_sim:
                sim_str = ", ".join(f"{n}:{s:.4f}" for n, s in zip(ref_names, sims))
                print(f"[W{idx}] Frame {frame_index} face sims -> {sim_str}")
            if frame_best_sim is None or best_sim > frame_best_sim:
                frame_best_sim = best_sim
            if best_sim >= similarity_threshold:
                color = (0, 255, 0)
                label = f"{best_name} ({best_sim:.2f})"
                identified = True
            else:
                color = (255, 0, 0)
                label = f"Unknown ({best_sim:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        t_end = time.time()
        out_q.put((frame_index, frame, {
            'faces_detected': len(faces),
            'identified': int(identified),
            'best_similarity': frame_best_sim if frame_best_sim is not None else -1.0,
            'latency_ms': (t_end - t_start) * 1000.0,
            'worker': idx
        }))
        in_q.task_done()


def reader_thread(cap, in_q: queue.Queue, max_frames: int | None, workers: int):
    frame_index = 0
    while True:
        if max_frames is not None and frame_index >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        in_q.put((frame_index, frame))
        frame_index += 1
    # Send one sentinel per worker
    for _ in range(workers):
        in_q.put(SENTINEL)


def writer_thread(out_q: queue.Queue, writer: cv2.VideoWriter, total_workers: int, preview: bool,
                  stats_collector, max_frames: int | None):
    next_to_write = 0
    buffer = {}
    finished_workers = 0
    while True:
        item = out_q.get()
        if item is SENTINEL:
            finished_workers += 1
            out_q.task_done()
            if finished_workers == total_workers and not buffer:
                break
            continue
        frame_index, frame, stats = item
        buffer[frame_index] = (frame, stats)
        # Flush in-order frames
        while next_to_write in buffer:
            f, s = buffer.pop(next_to_write)
            writer.write(f)
            stats_collector.record(next_to_write, s)
            if preview:
                # cv2.imshow('Preview (MT)', f)
                if cv2.waitKey(1) & 0xFF == 27:
                    stats_collector.stop_early = True
                    buffer.clear()
                    break
            next_to_write += 1
            if max_frames is not None and next_to_write >= max_frames:
                buffer.clear()
                break
        out_q.task_done()
        if stats_collector.stop_early:
            # Drain remaining sentinels
            continue
    if preview:
        cv2.destroyAllWindows()


class StatsCollector:
    def __init__(self):
        self.rows = []  # (frame_index, faces_detected, identified, best_similarity, latency_ms, worker)
        self.identified_frames = 0
        self.stop_early = False

    def record(self, frame_index, stats):
        self.rows.append((frame_index, stats['faces_detected'], stats['identified'],
                          stats['best_similarity'], stats['latency_ms'], stats['worker']))
        if stats['identified']:
            self.identified_frames += 1


def main():
    args = parse_args()
    providers = select_providers(args.providers)
    print(f"Providers: {providers}")

    ref_names, ref_embeddings = load_reference_embeddings(args.model_pack, providers, args.det_size, args.ref_images)

    cap = open_video(args.video)
    writer, fps_in = create_writer(cap, args.output)

    in_q = queue.Queue(maxsize=args.queue_size)
    out_q = queue.Queue(maxsize=args.queue_size * 2)

    stats = StatsCollector()

    # Start reader
    rt = threading.Thread(target=reader_thread, name='reader', args=(cap, in_q, args.max_frames, args.workers), daemon=True)
    rt.start()

    # Start workers
    workers = []
    for i in range(args.workers):
        t = threading.Thread(target=worker_thread, name=f'worker-{i}',
                             args=(i, args, providers, ref_embeddings, ref_names, args.det_size, in_q, out_q,
                                   args.similarity_threshold, args.show_all_sim), daemon=True)
        t.start()
        workers.append(t)

    # Start writer
    wt = threading.Thread(target=writer_thread, name='writer',
                          args=(out_q, writer, args.workers, not args.no_display, stats, args.max_frames), daemon=True)
    wt.start()

    t0 = time.time()

    # Wait for reader to finish feeding and workers to finish processing
    rt.join()
    in_q.join()  # all items processed by workers
    out_q.join()  # all results consumed by writer
    wt.join()

    elapsed = time.time() - t0
    total_frames = len(stats.rows)
    identified_frames = stats.identified_frames
    effective_fps = total_frames / elapsed if elapsed > 0 else 0

    # Aggregate latency stats
    latencies = [r[4] for r in stats.rows]
    if latencies:
        lat_arr = np.array(latencies)
        latency_summary = {
            'p50_ms': float(np.percentile(lat_arr, 50)),
            'p90_ms': float(np.percentile(lat_arr, 90)),
            'p95_ms': float(np.percentile(lat_arr, 95)),
            'p99_ms': float(np.percentile(lat_arr, 99)),
            'mean_ms': float(lat_arr.mean()),
            'min_ms': float(lat_arr.min()),
            'max_ms': float(lat_arr.max()),
        }
    else:
        latency_summary = {}

    print("\n==== Multi-Thread Summary ====")
    print(f"Model pack          : {args.model_pack}")
    print(f"Providers           : {providers}")
    print(f"Workers             : {args.workers}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print(f"Frames processed    : {total_frames}")
    print(f"Frames identified   : {identified_frames}")
    print(f"Elapsed (s)         : {elapsed:.2f}")
    print(f"Effective FPS       : {effective_fps:.2f}")
    print(f"Elapsed (s)         : {elapsed:.2f}")
    if latency_summary:
        print("Latency (ms)        : " + ", ".join(f"{k}={v:.1f}" for k,v in latency_summary.items()))
    print(f"Output saved        : {args.output}")

    if args.save_csv:
        try:
            import csv
            with open(args.save_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['frame_index', 'faces_detected', 'identified', 'best_similarity', 'latency_ms', 'worker'])
                for row in stats.rows:
                    w.writerow(row)
            print(f"Stats CSV saved: {args.save_csv}")
        except Exception as e:
            print(f"Failed to save CSV: {e}")

    cap.release()
    writer.release()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(1)
