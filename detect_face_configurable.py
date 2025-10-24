import argparse
import time
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import insightface

"""
Configurable face detection / identification script using InsightFace model packs.
Supports selectable model packs, provider selection (CoreML / CPU), frame rate control,
frame skipping, similarity threshold configuration, and basic performance benchmarking.

Tested on Apple Silicon (M-series). For M4 you typically want CoreMLExecutionProvider
for acceleration; CPUExecutionProvider is the fallback.

Example usages:

1) Basic run (auto providers, buffalo_l pack):
   python detect_face_configurable.py \
       --model-pack buffalo_l \
       --ref-image raw_face.webp \
       --video raw_video.mp4

2) Force CPU only:
   python detect_face_configurable.py --providers CPUExecutionProvider \
       --model-pack buffalo_s --ref-image raw_face.webp --video raw_video.mp4

3) Faster by skipping frames (process every 3rd frame) & output at 10 fps:
   python detect_face_configurable.py --frame-skip 2 --target-fps 10 \
       --model-pack buffalo_s --ref-image raw_face.webp --video raw_video.mp4

4) Increase similarity strictness:
   python detect_face_configurable.py --similarity-threshold 0.8 ...

Model Packs (size / speed tradeoff):
 - antelopev2 : Largest (~407MB) highest accuracy
 - buffalo_l  : Large (~326MB)
 - buffalo_m  : Medium (~313MB)
 - buffalo_s  : Small (~159MB) faster
 - buffalo_sc : Smallest / compact

Similarity threshold meaning: we compute cosine similarity of normalized embeddings.
A value close to 1 means highly similar. Default 0.7 (can tune). If you see too many
false positives, raise it (e.g., 0.8). If misses target, lower it (e.g., 0.6).
"""

MODEL_PACKS = ["antelopev2", "buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc"]


def parse_args():
    p = argparse.ArgumentParser(description="Configurable InsightFace video face identifier")
    p.add_argument('--model-pack', default='buffalo_l', choices=MODEL_PACKS,
                   help='InsightFace model pack name')
    p.add_argument('--ref-image', required=True, help='Path to reference (target) face image')
    p.add_argument('--video', required=True, help='Path to input video file')
    p.add_argument('--output', default='output.mp4', help='Path to output annotated video')
    p.add_argument('--providers', nargs='*', default=None,
                   help='Execution providers list. Example: CoreMLExecutionProvider CPUExecutionProvider')
    p.add_argument('--det-size', type=int, nargs=2, default=[640, 640],
                   help='Detection input size (width height)')
    p.add_argument('--similarity-threshold', type=float, default=0.3,
                   help='Cosine similarity threshold (0-1). Higher = stricter.')
    p.add_argument('--frame-skip', type=int, default=0,
                   help='Number of frames to skip after each processed frame (0 = process all)')
    p.add_argument('--target-fps', type=float, default=None,
                   help='Re-encode output at this FPS (default: original). Does not affect processing unless combined with frame skipping.')
    p.add_argument('--max-frames', type=int, default=None,
                   help='Limit number of frames processed (debug)')
    p.add_argument('--no-display', action='store_true', help='Do not open preview window')
    p.add_argument('--show-all-sim', action='store_true', help='Print similarity for all faces per frame')
    p.add_argument('--save-csv', default=None, help='Optional path to save per-frame stats CSV')
    return p.parse_args()


def select_providers(user_providers):
    if user_providers:
        return user_providers
    # Auto-detect: try CoreML then CPU
    providers = []
    try:
        # CoreML provider name
        providers.append('CoreMLExecutionProvider')
    except Exception:
        pass
    providers.append('CPUExecutionProvider')
    return providers


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_model(pack: str, providers, det_size):
    print(f"Loading model pack '{pack}' with providers {providers} and det_size={det_size}...")
    app = insightface.app.FaceAnalysis(name=pack, providers=providers)
    # ctx_id ignored when providers list given; keep 0
    app.prepare(ctx_id=0, det_size=tuple(det_size))
    return app


def load_reference_embedding(model, ref_image_path: str):
    if not os.path.isfile(ref_image_path):
        raise FileNotFoundError(f"Reference image not found: {ref_image_path}")
    ref_img = cv2.imread(ref_image_path)
    if ref_img is None:
        raise ValueError(f"Failed to read reference image: {ref_image_path}")
    faces = model.get(ref_img)
    if len(faces) == 0:
        raise ValueError("No face detected in reference image.")
    print(f"Reference image: detected {len(faces)} face(s); using first for embedding.")
    return faces[0].normed_embedding


def open_video(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def create_writer(cap, output_path: str, target_fps: float | None):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps_out = target_fps if target_fps else fps_in
    print(f"Input FPS: {fps_in:.2f} -> Output FPS: {fps_out:.2f}")
    return cv2.VideoWriter(output_path, fourcc, fps_out, (width, height)), fps_in, fps_out


def main():
    args = parse_args()
    providers = select_providers(args.providers)

    model = load_model(args.model_pack, providers, args.det_size)
    ref_embedding = load_reference_embedding(model, args.ref_image)

    cap = open_video(args.video)
    writer, fps_in, fps_out = create_writer(cap, args.output, args.target_fps)

    similarity_threshold = args.similarity_threshold
    frame_skip = max(0, args.frame_skip)

    frame_index = 0
    processed_frames = 0
    identified_frames = 0
    t0 = time.time()

    stats_rows = []  # (frame_index, processed_flag, faces_detected, identified, best_similarity)

    print("Starting processing...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            process_this = True
            if frame_skip > 0 and processed_frames > 0:
                # If we've processed the previous frame, skip next 'frame_skip' frames
                if (processed_frames % (frame_skip + 1)) != 0:
                    process_this = False
                    processed_frames += 1  # count skipped frames too

            best_sim = None
            identified = False

            if process_this:
                faces = model.get(frame)
                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    emb = face.normed_embedding
                    sim = cosine_similarity(ref_embedding, emb)
                    if args.show_all_sim:
                        print(f"Frame {frame_index} face sim: {sim:.4f}")
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
                processed_frames += 1
                if identified:
                    identified_frames += 1
            else:
                # Optionally annotate skipped frames
                cv2.putText(frame, "SKIPPED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            writer.write(frame)

            stats_rows.append((frame_index, int(process_this), len(faces) if process_this else 0,
                               int(identified), best_sim if best_sim is not None else -1))

            if not args.no_display:
                # cv2.imshow('Preview', frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    print("ESC pressed; exiting.")
                    break

            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                print("Reached max frame limit.")
                break

    finally:
        cap.release()
        writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = time.time() - t0
    total_frames = frame_index
    effective_fps = processed_frames / elapsed if elapsed > 0 else 0
    realtime_ratio = (processed_frames / fps_in) / elapsed if fps_in > 0 and elapsed > 0 else 0

    print("\n==== Summary ====")
    print(f"Model pack          : {args.model_pack}")
    print(f"Providers           : {providers}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Frames total        : {total_frames}")
    print(f"Frames processed    : {processed_frames}")
    print(f"Time elapsed       : {elapsed:.2f} sec")
    print(f"Frames identified   : {identified_frames}")
    print(f"Processing FPS      : {effective_fps:.2f} frames/sec")
    print(f"Real-time factor    : {realtime_ratio:.2f} (>=1 means faster than real-time)")
    print(f"Output saved        : {args.output}")

    if args.save_csv:
        try:
            import csv
            with open(args.save_csv, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['frame_index', 'processed', 'faces_detected', 'identified', 'best_similarity'])
                for row in stats_rows:
                    w.writerow(row)
            print(f"Stats CSV saved: {args.save_csv}")
        except Exception as e:
            print(f"Failed to save CSV: {e}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(1)
