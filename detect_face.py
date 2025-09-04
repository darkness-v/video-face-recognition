import cv2
import insightface
import numpy as np

# Load InsightFace model (face analysis)
model = insightface.app.FaceAnalysis(providers=['CoreMLExecutionProvider','CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(640, 640))

# Load target image and extract embedding
target_img_path = "raw_face.webp"  # path to the reference face image
target_img = cv2.imread(target_img_path)
target_faces = model.get(target_img)

if len(target_faces) == 0:
    raise ValueError("No face detected in the target image.")

# Take the first detected face in the reference image
target_embedding = target_faces[0].normed_embedding

# Open video
video_path = "raw_video.mp4"
cap = cv2.VideoCapture(video_path)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Threshold for cosine similarity
SIMILARITY_THRESHOLD = 0.7  # adjust if too strict/loose

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

while True:
    ret, frame = cap.read()
    print(1)
    if not ret:
        break

    faces = model.get(frame)
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        emb = face.normed_embedding

        # Draw all detected faces in blue
        color = (255, 0, 0)
        label = "Unknown"

        # Compare with target embedding
        sim = cosine_similarity(target_embedding, emb)
        print(sim)
        if sim > (1 - SIMILARITY_THRESHOLD):
            color = (0, 255, 0)  # Green for identified target
            label = f"Target ({sim:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved as output.mp4")
