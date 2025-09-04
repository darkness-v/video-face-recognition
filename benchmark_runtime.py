import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import onnxruntime as ort

print("Available providers:", ort.get_available_providers())

def run_inference(provider, img_path="person1.webp"):
    print(f"\n--- Running with {provider} ---")

    app = FaceAnalysis(
        name="buffalo_l",
        providers=[provider, "CPUExecutionProvider"]  # GPU first, fallback CPU
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load {img_path}")

    # Warm-up (avoid first-call overhead)
    for _ in range(2):
        _ = app.get(img)

    # Benchmark
    start = time.time()
    for _ in range(5):  # run multiple times for stable timing
        faces = app.get(img)
    elapsed = (time.time() - start) / 5

    print(f"Avg inference time: {elapsed:.3f} sec")
    if faces:
        print("Faces detected:", len(faces))
        print("Embedding shape:", faces[0].embedding.shape)

# Run both CPU and CoreML if available
run_inference("CPUExecutionProvider")

if "CoreMLExecutionProvider" in ort.get_available_providers():
    run_inference("CoreMLExecutionProvider")
