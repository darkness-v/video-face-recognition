import base64
import json
import os
import time
from typing import List, Sequence
import requests
import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image

SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "changeme")

class RemoteEmbeddingClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {SERVICE_TOKEN}"})
        self.timeout = timeout

    @staticmethod
    def encode_image(img_path: str | Path) -> str:
        with Image.open(img_path) as im:
            im = im.convert('RGB')
            buf = BytesIO()
            im.save(buf, format='JPEG', quality=95)
            return base64.b64encode(buf.getvalue()).decode('utf-8')

    def health(self):
        r = self.session.post(self.base_url + '/ping', timeout=self.timeout, json={})
        r.raise_for_status()
        return r.json()

    def embed_images(self, img_paths: Sequence[str | Path]) -> List[np.ndarray]:
        payload = {"images": [self.encode_image(p) for p in img_paths]}
        t0 = time.time()
        r = self.session.post(self.base_url + '/embed', json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        emb_list = [np.array(v, dtype='float32') for v in data['embeddings']]
        dt = time.time() - t0
        return emb_list

    def similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        payload = {"emb_a": emb_a.tolist(), "emb_b": emb_b.tolist()}
        r = self.session.post(self.base_url + '/similarity', json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()['similarity']

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--query', required=True)
    args = ap.parse_args()

    client = RemoteEmbeddingClient(args.url)
    info = client.health()
    print('Service info:', json.dumps(info, indent=2))
    ref_emb, = client.embed_images([args.ref])
    qry_emb, = client.embed_images([args.query])
    # Cosine similarity local
    sim = float(np.dot(ref_emb, qry_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(qry_emb)))
    print(f'Local cosine similarity: {sim:.4f}')
    remote_sim = client.similarity(ref_emb, qry_emb)
    print(f'Remote cosine similarity: {remote_sim:.4f}')
