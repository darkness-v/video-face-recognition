import cv2, os
import onnxruntime as ort
import onnxruntime
print(onnxruntime.__file__)
from onnxruntime import InferenceSession

path = "assets/person1.webp"
print("Current working directory:", os.getcwd())
print("File exists:", os.path.exists(path))

image = cv2.imread(path)
print("Image loaded:", image is not None)
