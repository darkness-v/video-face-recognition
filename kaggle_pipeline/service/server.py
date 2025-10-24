import os
import base64
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import insightface
import uvicorn

SERVICE_TOKEN = os.getenv('SERVICE_TOKEN', 'changeme')
MODEL_PACK = os.getenv('MODEL_PACK', 'buffalo_l')
DET_SIZE = tuple(int(x) for x in os.getenv('DET_SIZE', '640,640').split(','))

app = FastAPI(title='InsightFace Remote Embedding Service', version='0.1.0')

model = None  # lazy load

class EmbedRequest(BaseModel):
    images: List[str]  # base64 JPEG/PNG

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    count: int

class SimilarityRequest(BaseModel):
    emb_a: List[float]
    emb_b: List[float]

class SimilarityResponse(BaseModel):
    similarity: float

class PingResponse(BaseModel):
    model_pack: str
    det_size: List[int]
    provider_list: List[str]


def auth_check(authorization: Optional[str] = Header(None)):
    if not SERVICE_TOKEN:
        return
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing bearer token')
    token = authorization.split(' ', 1)[1]
    if token != SERVICE_TOKEN:
        raise HTTPException(status_code=403, detail='Invalid token')


def load_model():
    global model
    if model is None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        model = insightface.app.FaceAnalysis(name=MODEL_PACK, providers=providers)
        model.prepare(ctx_id=0, det_size=DET_SIZE)
    return model


def decode_image(b64_str: str):
    raw = base64.b64decode(b64_str)
    im = Image.open(BytesIO(raw)).convert('RGB')
    return np.array(im)[:, :, ::-1]  # to BGR for insightface

@app.post('/ping', response_model=PingResponse)
async def ping(_: None = Depends(auth_check)):
    m = load_model()
    return PingResponse(model_pack=MODEL_PACK, det_size=list(DET_SIZE), provider_list=m.providers)

@app.post('/embed', response_model=EmbedResponse)
async def embed(req: EmbedRequest, _: None = Depends(auth_check)):
    m = load_model()
    out_emb = []
    for img_b64 in req.images:
        img = decode_image(img_b64)
        faces = m.get(img)
        if not faces:
            out_emb.append([])
        else:
            out_emb.append(faces[0].normed_embedding.tolist())
    return EmbedResponse(embeddings=out_emb, count=len(out_emb))

@app.post('/similarity', response_model=SimilarityResponse)
async def similarity(req: SimilarityRequest, _: None = Depends(auth_check)):
    a = np.array(req.emb_a, dtype='float32')
    b = np.array(req.emb_b, dtype='float32')
    sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return SimilarityResponse(similarity=sim)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
