import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceIdentification:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_s", providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.threshold = 25

    def load_image(self, opt):
        self.input_image = cv2.imread(opt.image)
        self.results = self.app.get(self.input_image)

    def load_face_bank(self):
        self.face_bank = np.load("face_bank.npy", allow_pickle=True)
        
    def identification(self):
        ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default="input\image.jpg")
    parser.add_argument('--threshold', type=str, default=25)
    parser.add_argument('--update')

    opt = parser.parse_args()