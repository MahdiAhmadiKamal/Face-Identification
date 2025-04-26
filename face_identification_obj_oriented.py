import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceIdentification:
    def __init__(self, opt):
        self.opt = opt
        self.app = FaceAnalysis(name="buffalo_s", providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_image(self, opt):
        self.input_image = cv2.imread(opt.image)
        
    def identification(self, app):
        self.results = app.get(self.input_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default="input\image.jpg")
    parser.add_argument('--threshold', type=str, default=25)
    parser.add_argument('--update')

    opt = parser.parse_args()