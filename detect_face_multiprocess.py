import argparse
import time
import os
import sys
import multiprocessing as mp
from multiprocessing import queues
import cv2
import numpy as np
import insightface
import threading  # Added for concurrent reader

"""
Multi-process face identification benchmark using InsightFace FaceAnalysis.

Pipeline (Reader -> Workers -> Writer in main process):
 1. Main process decodes frames from video and pushes (frame_index, frame) into an input Queue.
 2. N worker processes each hold their own FaceAnalysis model instance and:
       - run detection & similarity scoring
       - annotate the frame
       - optionally JPEG-compress the annotated frame to reduce IPC payload
       - push (frame_index, encoded_or_array, stats_dict) to output Queue
 3. Main process collects results (may arrive out of order), reorders, writes to VideoWriter.

Ordering Strategy:
  Uses a buffer dict and a next_to_write counter (same as multithread version).

Sentinels:
  Reader sends one SENTINEL per worker via input queue when finished.
  Each worker sends one SENTINEL into output queue when it exits.
  Main stops after all workers finished and buffer flushed.

Why multiprocessing version?
  - True parallelism even for any Python-bound work (independent GILs).
  - Isolation (a crash in a worker need not bring down orchestrator if guarded).
Trade-offs:
  - Higher memory use (model per process).
  - (De)serialization overhead of frames across processes.
  - Start-up cost.

Optimizations Provided:
  --compress : encode frames to JPEG before sending to main writer (reduces IPC size, adds CPU cost).
  Future improvements (not implemented to keep code concise):
    - Use shared_memory for zero-copy frame transfer.
    - Split detection & embedding stages into separate pools.

Example:
  python detect_face_multiprocess.py \
     --model-pack buffalo_s \
     --ref-image assets/raw_face.webp \
     --video assets/raw_video.mp4 \
     --workers 4 --providers CoreMLExecutionProvider CPUExecutionProvider --compress

"""

MODEL_PACKS = ["antelopev2", "buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc"]
SENTINEL = None


def parse_args():
    p = argparse.ArgumentParser(description="Multi-process InsightFace video face identifier")
    p.add_argument('--model-pack', default='buffalo_l', choices=MODEL_PACKS)
    p.add_argument('--ref-image', required=True)
    p.add_argument('--video', required=True)
    p.add_argument('--output', default='output_mp.mp4')
    p.add_argument('--providers', nargs='*', default=None)
    p.add_argument('--det-size', type=int, nargs=2, default=[640, 640])
    p.add_argument('--similarity-threshold', type=float, default=0.3)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--queue-size', type=int, default=16, help='Max pending frames in input queue')
    p.add_argument('--output-queue-size', type=int, default=32)
    p.add_argument('--compress', action='store_true', help='JPEG compress frames between processes')
    p.add_argument('--jpeg-quality', type=int, default=85, help='JPEG quality when --compress')
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--show-all-sim', action='store_true')
    p.add_argument('--save-csv', default=None)
    p.add_argument('--max-frames', type=int, default=None)
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


def extract_reference_embedding(pack, providers, det_size, ref_image_path):
    if not os.path.isfile(ref_image_path):
        raise FileNotFoundError(ref_image_path)
    img = cv2.imread(ref_image_path)
    if img is None:
        raise ValueError("Failed to read reference image")
    app = insightface.app.FaceAnalysis(name=pack, providers=providers)
    app.prepare(ctx_id=0, det_size=tuple(det_size))
    faces = app.get(img)
    if not faces:
        raise ValueError("No face in reference image")
    return faces[0].normed_embedding


def open_video(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    return cap


def create_writer(cap, output_path: str):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))
    return writer, fps_in


def worker_process(idx, args, providers, det_size, ref_embedding, in_q, out_q):
    # Each process loads model
    model = insightface.app.FaceAnalysis(name=args.model_pack, providers=providers)
    model.prepare(ctx_id=0, det_size=tuple(det_size))
    similarity_threshold = args.similarity_threshold
    show_all_sim = args.show_all_sim
    compress = args.compress
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]

    while True:
        item = in_q.get()
        if item is SENTINEL:
            out_q.put(SENTINEL)
            break
        frame_index, frame = item
        t0 = time.time()
        faces = model.get(frame)
        best_sim = None
        identified = False
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            emb = face.normed_embedding
            sim = cosine_similarity(ref_embedding, emb)
            if show_all_sim:
                print(f"[P{idx}] Frame {frame_index} sim {sim:.4f}")
            if best_sim is None or sim > best_sim:
                best_sim = sim
            color = (255, 0, 0)
            label = f"Unknown" if best_sim is None else f"Unknown ({sim:.2f})"
            if sim >= similarity_threshold:
                color = (0, 255, 0)
                label = f"Target ({sim:.2f})"
                identified = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        latency_ms = (time.time() - t0) * 1000.0
        if compress:
            ok, buf = cv2.imencode('.jpg', frame, encode_params)
            if not ok:
                # Fallback to raw if encoding fails
                payload = frame
                compressed = False
            else:
                payload = buf.tobytes()
                compressed = True
        else:
            payload = frame
            compressed = False
        out_q.put((frame_index, payload, compressed, {
            'faces_detected': len(faces),
            'identified': int(identified),
            'best_similarity': best_sim if best_sim is not None else -1.0,
            'latency_ms': latency_ms,
            'worker': idx
        }))


def reader(cap, in_q, max_frames, workers):
    frame_index = 0
    while True:
        if max_frames is not None and frame_index >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        in_q.put((frame_index, frame))
        frame_index += 1
    # Termination signals
    for _ in range(workers):
        in_q.put(SENTINEL)
    return frame_index  # total frames enqueued


class Stats:
    def __init__(self):
        self.rows = []  # (frame_index, faces_detected, identified, best_similarity, latency_ms, worker)
        self.identified = 0

    def add(self, idx, st):
        self.rows.append((idx, st['faces_detected'], st['identified'], st['best_similarity'], st['latency_ms'], st['worker']))
        if st['identified']:
            self.identified += 1


def writer_loop(out_q, writer, total_workers, preview, max_frames, compress):
    next_to_write = 0
    buffer = {}
    finished = 0
    stats = Stats()
    while True:
        item = out_q.get()
        if item is SENTINEL:
            finished += 1
            if finished == total_workers and not buffer:
                break
            continue
        frame_index, payload, is_compressed, st = item
        if is_compressed:
            arr = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            frame = payload
        buffer[frame_index] = (frame, st)
        while next_to_write in buffer:
            f, st2 = buffer.pop(next_to_write)
            writer.write(f)
            stats.add(next_to_write, st2)
            if preview:
                # cv2.imshow('Preview (MP)', f)
                if cv2.waitKey(1) & 0xFF == 27:
                    buffer.clear()
                    finished = total_workers  # force exit after flush
                    break
            next_to_write += 1
            if max_frames is not None and next_to_write >= max_frames:
                buffer.clear()
                finished = total_workers  # force exit
                break
    if preview:
        cv2.destroyAllWindows()
    return stats


def main():
    args = parse_args()
    providers = select_providers(args.providers)
    print(f"Providers: {providers}")

    # Extract reference embedding once (small vector) to broadcast
    ref_embedding = extract_reference_embedding(args.model_pack, providers, args.det_size, args.ref_image)

    cap = open_video(args.video)
    writer, fps_in = create_writer(cap, args.output)

    ctx = mp.get_context('spawn')  # explicit for cross-platform consistency

    in_q = ctx.Queue(maxsize=args.queue_size)
    out_q = ctx.Queue(maxsize=args.output_queue_size)

    # Start worker processes
    workers = []
    for i in range(args.workers):
        p = ctx.Process(target=worker_process, args=(i, args, providers, args.det_size, ref_embedding, in_q, out_q), daemon=True)
        p.start()
        workers.append(p)

    # Run reader concurrently so writer can drain output queue to avoid deadlock when both queues fill.
    count_holder = {'n': 0}
    def _reader_wrapper():
        count_holder['n'] = reader(cap, in_q, args.max_frames, args.workers)
    t_reader = threading.Thread(target=_reader_wrapper, name='ReaderThread', daemon=True)

    t0 = time.time()
    t_reader.start()

    stats = writer_loop(out_q, writer, args.workers, not args.no_display, args.max_frames, args.compress)

    # Ensure reader finished (if not already). Timeout to avoid infinite wait if early termination.
    t_reader.join(timeout=2.0)
    total_enqueued = count_holder['n']

    # Ensure workers exit
    for p in workers:
        p.join()

    elapsed = time.time() - t0
    processed = len(stats.rows)
    effective_fps = processed / elapsed if elapsed > 0 else 0

    latencies = [r[4] for r in stats.rows]
    if latencies:
        arr = np.array(latencies)
        latency_summary = {
            'p50_ms': float(np.percentile(arr, 50)),
            'p90_ms': float(np.percentile(arr, 90)),
            'p95_ms': float(np.percentile(arr, 95)),
            'p99_ms': float(np.percentile(arr, 99)),
            'mean_ms': float(arr.mean()),
            'min_ms': float(arr.min()),
            'max_ms': float(arr.max()),
        }
    else:
        latency_summary = {}

    print("\n==== Multi-Process Summary ====")
    print(f"Model pack          : {args.model_pack}")
    print(f"Providers           : {providers}")
    print(f"Workers             : {args.workers}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print(f"Frames enqueued     : {total_enqueued}")
    print(f"Frames processed    : {processed}")
    print(f"Frames identified   : {stats.identified}")
    print(f"Elapsed (s)         : {elapsed:.2f}")
    print(f"Effective FPS       : {effective_fps:.2f}")
    if latency_summary:
        print("Latency (ms)        : " + ", ".join(f"{k}={v:.1f}" for k,v in latency_summary.items()))
    print(f"Compression         : {args.compress}")
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
        mp.freeze_support()  # safe on all platforms
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(1)
