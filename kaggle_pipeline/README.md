# Kaggle T4 Remote Face Recognition Pipeline

This folder contains assets to run the existing `detect_face_configurable.py` pipeline on a Kaggle Notebook (2x T4) and expose a simple HTTP inference endpoint so you can call it from your local machine while keeping your private video & image data local (only embeddings or cropped face encodings can be optionally sent).

## Overview

There are two modes:

1. Remote Embedding Service (Recommended):
   - Kaggle notebook loads InsightFace models on T4 GPU.
   - You send face crops (or a single reference face + query face) as base64 images.
   - Service returns embeddings or similarity score.
   - Local script handles video decoding & face detection using CPU/CoreML locally, and delegates ONLY embedding computation & similarity to GPU service.

2. Remote Full Frame Inference (Optional):
   - You send frames (heavier bandwidth). Not recommended unless necessary.

## Components

- `kaggle_inference_service.ipynb` : Kaggle notebook that launches a small FastAPI server inside the notebook, serving embedding & similarity endpoints.
- `client/remote_embed_client.py` : Local helper to call remote service.
- `client/remote_similarity_cli.py` : Example CLI usage from local terminal.
- `service/requirements.txt` : Minimal pinned deps for Kaggle (fastapi, uvicorn, insightface, opencv-python-headless, numpy, pillow).

## Flow (Recommended Mode)

Local (Mac M4):
1. Extract reference embedding locally or (optionally) send ref face to remote once and cache remote embedding id.
2. For each detected face in video frames, send crop to remote `/embed` endpoint -> receive embedding vector.
3. Compute cosine similarity locally (or ask remote `/similarity`).
4. Annotate frames & write video locally (privacy preserved).

Remote (Kaggle T4):
- Loads model pack (configurable) once.
- Serves endpoints:
  - `POST /embed` -> returns embedding list for provided face images.
  - `POST /similarity` -> returns similarity between provided (already embedded or raw images) faces.
  - `POST /ping` -> health + model info.

## Security / Access

Kaggle notebooks are ephemeral & not intended for long-running public services. For ad‑hoc benchmarking:
- Start notebook.
- Expose tunnel via `cloudflared` (or `ngrok` if allowed) to obtain a stable URL for a session.
- Use a temporary bearer token set as env var `SERVICE_TOKEN` in both sides.

If tunnels are disallowed, fall back to pulling embeddings by downloading JSON after batch processing (batch mode section below).

## Batch Mode (No Live Tunnel)

1. Upload a zip of face crops & a manifest JSON to Kaggle `/kaggle/working/input/face_batch.zip`.
2. Run notebook cell that processes all images and produces `embeddings.json`.
3. Download artifacts manually (or using Kaggle API) to local.
4. Perform similarity offline.

## Next Steps

1. Open `kaggle_inference_service.ipynb` on Kaggle.
2. Run all cells until the server prints the public tunnel URL.
3. From local machine, export `SERVICE_TOKEN=...` and run: `python client/remote_similarity_cli.py --ref raw_face.webp --query some_face_crop.jpg --url https://YOUR_TUNNEL`.

---
