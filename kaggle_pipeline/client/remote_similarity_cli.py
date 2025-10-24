import argparse
import numpy as np
from remote_embed_client import RemoteEmbeddingClient
import os


def main():
    ap = argparse.ArgumentParser(description='Quick remote similarity test (ref vs query)')
    ap.add_argument('--url', required=True, help='Base URL of remote Kaggle service (e.g. https://abc.trycloudflare.com)')
    ap.add_argument('--ref', required=True, help='Reference face image path')
    ap.add_argument('--query', required=True, help='Query face image path')
    args = ap.parse_args()

    client = RemoteEmbeddingClient(args.url)
    ref_emb, = client.embed_images([args.ref])
    qry_emb, = client.embed_images([args.query])
    sim = float(np.dot(ref_emb, qry_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(qry_emb)))
    print(f'Cosine similarity: {sim:.4f}')

if __name__ == '__main__':
    main()
