import time

import pandas as pd

from tqdm.auto import tqdm


from insightface.app import FaceAnalysis
import cv2
from insightface.data import get_image as ins_get_image
import os
import numpy as np
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor

# import torch


import torch
import cv2
from torch.utils.data import IterableDataset, DataLoader


class VidLoader(IterableDataset):

    def __init__(self, vid_path):
        self.vid_path = vid_path

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None and worker_info.num_workers > 1:
            raise ValueError(
                f"does not support num_workers > 1 for single video files. "
                f"Detected {worker_info.num_workers} workers."
            )

        cap = cv2.VideoCapture(self.vid_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {self.vid_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            yield frame

        cap.release()


def get_dl(vid_path, batch_size=32, use_thread=False):
    ds = VidLoader(vid_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1 if use_thread else 0,
        collate_fn=lambda x: x,
    )


def initializer_worker(local, name="buffalo_l", ctx_id=0, det_size=(640, 640)):
    # one model for each worker
    app = FaceAnalysis(name=name)
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    local.app = app


def task(local, img):
    app = local.app

    # onnx seems allows sending kernels to gpu simultaneously
    # without explicit seperate cuda stream like in pytorch

    # with torch.cuda.stream():
    faces = app.get(img)

    return faces


class FaceAnalysis_multithread:

    def __init__(self, name="buffalo_l", ctx_id=0, det_size=(640, 640), n_workers=2):
        self.name = name
        self.det_size = det_size

        self.n_workers = n_workers

        if n_workers <= 1:
            app = FaceAnalysis(name=name, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=ctx_id, det_size=det_size)
            self.app = app
        else:
            self.local = threading.local()
            self.executor = ThreadPoolExecutor(
                max_workers=n_workers,
                initializer=initializer_worker,
                initargs=(self.local, name, ctx_id, det_size),
            )

    def get(self, imgs):

        if self.n_workers <= 1:
            return [self.app.get(i) for i in imgs]

        else:
            futures = [self.executor.submit(task, self.local, i) for i in imgs]
            return [f.result() for f in futures]


bench_option = {
    "models": [
        # "antelopev2",
        "buffalo_l",
        "buffalo_m",
        "buffalo_s",
        "buffalo_sc",
    ],
    "n_workers": [1, 2, 4],
}


def bench(name, vid_path, frame_batch_size, n_workers, n_tries=10):

    app = FaceAnalysis_multithread(name=name, n_workers=n_workers)

    dl = get_dl(vid_path)

    # cold run
    temp_batch = next(iter(dl))
    app.get(temp_batch)

    # then the benchmark

    start_time = time.time()
    for i in range(n_tries):
        for b in tqdm(dl):
            temp = app.get(b)
    end_time = time.time()

    duration = (end_time - start_time) / n_tries

    return duration


def main():

    vid = "assets/vid_1.mp4"

    results = []
    start_time = time.time()
    for n_worker in bench_option["n_workers"]:
        for model_name in bench_option["models"]:

            duration = bench(model_name, vid, 32, n_worker, 10)

            log = {
                "model": model_name,
                "n_workers": n_worker,
                "processing_time": duration,
            }
            print(log)
            results.append(log)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total elapsed time: {elapsed:.2f} sec")
    df = pd.DataFrame(results)

    df.to_csv("result.csv")


if __name__ == "__main__":
    main()
 