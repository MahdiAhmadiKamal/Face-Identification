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
        print("load")
        self.input_image = cv2.imread(opt.image)
        self.results = self.app.get(self.input_image)

    def load_face_bank(self):
        self.face_bank = np.load("face_bank.npy", allow_pickle=True)
        
    def identification(self):
        for result in self.results:
            cv2.rectangle(self.input_image, (int(result.bbox[0]), int(result.bbox[1])), 
                        (int(result.bbox[2]), int(result.bbox[3])), (0, 255, 0), 2)
        
            for person in self.face_bank:
                face_bank_person_embedding = person["embedding"]
                new_person_embedding = result["embedding"]
                distance = np.sqrt(np.sum((face_bank_person_embedding - new_person_embedding)**2))
                if distance <= opt.threshold:
                    cv2.putText(self.input_image, person["name"], (int(result.bbox[0])-50, int(result.bbox[1])-10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=4, 
                                lineType=cv2.LINE_AA)
                    cv2.putText(self.input_image, person["name"], (int(result.bbox[0])-50, int(result.bbox[1])-10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2, 
                                lineType=cv2.LINE_AA)
                    break
            else:
                cv2.putText(self.input_image, "Unknown", (int(result.bbox[0])-50, int(result.bbox[1])-10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=4, 
                                lineType=cv2.LINE_AA)
                cv2.putText(self.input_image, "Unknown", (int(result.bbox[0])-50, int(result.bbox[1])-10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, 
                                lineType=cv2.LINE_AA)
                
        cv2.imwrite("output/result_image.jpg", self.input_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default="input/image.jpg")
    parser.add_argument('--threshold', type=str, default=25)
    parser.add_argument('--update')

    opt = parser.parse_args()

    cls = FaceIdentification()
    cls.load_image(opt)
    cls.load_face_bank()
    cls.identification()
