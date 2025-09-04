from insightface.app import FaceAnalysis
import onnxruntime as ort
import cv2

# Init with CoreML if available
app = FaceAnalysis(name="buffalo_l", providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640,640))

img = cv2.imread("/Users/tani/TechJDI/person1.webp")
faces = app.get(img)

print("Faces detected:", len(faces))
if faces:
    print("Embedding shape:", faces[0].embedding.shape)
