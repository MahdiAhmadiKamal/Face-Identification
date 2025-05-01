import argparse
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from create_face_bank_obj_orntd import CreateFaceBank


class FaceIdentification:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_s", providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.threshold = 25
        self.face_bank_path = "./face_bank/"

    def load_image(self, opt):
        self.input_image = cv2.imread(opt.image)
        self.results = self.app.get(self.input_image)

    def load_face_bank(self):
        self.face_bank = np.load("face_bank.npy", allow_pickle=True)

    def update_face_bank(self, opt):
        CreateFaceBank.update(opt.update, self.face_bank_path, self.app)
        
    def identification(self):
        for result in self.results:
            cv2.rectangle(self.input_image, (int(result.bbox[0]), int(result.bbox[1])), 
                        (int(result.bbox[2]), int(result.bbox[3])), (0, 255, 0), 2)
        
            for person in self.face_bank:
                face_bank_person_embedding = person["embedding"]
                new_person_embedding = result["embedding"]
                distance = np.sqrt(np.sum((face_bank_person_embedding - new_person_embedding)**2))
                if distance <= opt.threshold:
                    cv2.putText(self.input_image, person["name"], 
                                (int(result.bbox[0])-50, int(result.bbox[1])-10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                                color=(0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
                    cv2.putText(self.input_image, person["name"], 
                                (int(result.bbox[0])-50, int(result.bbox[1])-10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                                color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                    break
            else:
                cv2.putText(self.input_image, "Unknown", 
                            (int(result.bbox[0])-50, int(result.bbox[1])-10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                                color=(0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
                cv2.putText(self.input_image, "Unknown", 
                            (int(result.bbox[0])-50, int(result.bbox[1])-10),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                                color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                
        cv2.imwrite("output/result_image.jpg", self.input_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default="input/image.jpg")
    parser.add_argument('--threshold', type=str, default=25)
    parser.add_argument('--update', action='store_true')

    opt = parser.parse_args()

    obj = FaceIdentification()
    obj.load_image(opt)

    if opt.update:
        obj.update_face_bank(opt)

    obj.load_face_bank()
    obj.identification()