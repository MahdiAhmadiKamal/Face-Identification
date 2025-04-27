import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class CreateFaceBank:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_s", providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.face_bank_path = "./face_bank/"
        