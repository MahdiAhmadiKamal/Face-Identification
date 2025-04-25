import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis


app = FaceAnalysis(name="buffalo_s", providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

face_bank_path = "./face_bank/"

face_bank = []
for person_name in os.listdir(face_bank_path):
    folder_path = os.path.join(face_bank_path, person_name)
    if os.path.isdir(folder_path):
        # print(folder_path)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # print(file_path)
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = app.get(image)

            if len(result) > 1:
                print(f"Warning: more than one face detected in image: {file_path}")
                continue

            embedding = result[0]["embedding"]
            dict = {"name": person_name, "embedding": embedding}
            face_bank.append(dict)

