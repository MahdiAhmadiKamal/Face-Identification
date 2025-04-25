import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis


parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default="input\image.jpg")
parser.add_argument('--threshold', type=str, default=25)

opt = parser.parse_args()

app = FaceAnalysis(name="buffalo_s", providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

input_image = cv2.imread(opt.image)

results = app.get(input_image)

face_bank = np.load("face_bank.npy", allow_pickle=True)

for result in results:
    cv2.rectangle(input_image, (int(result.bbox[0]), int(result.bbox[1])), (int(result.bbox[2]), int(result.bbox[3])), 
                  (0, 255, 0), 2)
    
    for person in face_bank:
        face_bank_person_embedding = person["embedding"]
        new_person_embedding = result["embedding"]
        distance = np.sqrt(np.sum((face_bank_person_embedding - new_person_embedding)**2))
        if distance <= opt.threshold:
            cv2.putText(input_image, person["name"], (int(result.bbox[0])-50, int(result.bbox[1])-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=4, 
                        lineType=cv2.LINE_AA)
            cv2.putText(input_image, person["name"], (int(result.bbox[0])-50, int(result.bbox[1])-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2, 
                        lineType=cv2.LINE_AA)
            break
    else:
        cv2.putText(input_image, "Unknown", (int(result.bbox[0])-50, int(result.bbox[1])-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=4, 
                        lineType=cv2.LINE_AA)
        cv2.putText(input_image, "Unknown", (int(result.bbox[0])-50, int(result.bbox[1])-10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, 
                        lineType=cv2.LINE_AA)
        
cv2.imwrite("output/result_image.jpg", input_image)