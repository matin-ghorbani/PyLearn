import os
from time import time
from argparse import ArgumentParser, BooleanOptionalAction

from insightface.app import FaceAnalysis
import cv2 as cv
import numpy as np
from tqdm import tqdm


class FaceIdentification:
    def __init__(self, model: str, thresh: float, face_bank: str):
        self.threshold = thresh
        self.app = FaceAnalysis(name=model, providers=[
                                'CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.face_bank = np.load(face_bank, allow_pickle=True)

    def update_face_bank(self, face_bank_path: str, face_bank_name: str):
        files = filter(lambda x: os.path.isdir(
            os.path.join(face_bank_path, x)
        ), os.listdir(face_bank_path))

        face_bank = []
        img_name: str

        for person in tqdm(files):
            person_folder_path = os.path.join(face_bank_path, person)

            for img_name in os.listdir(person_folder_path):
                if img_name.endswith('.jpg') or img_name.endswith('.png'):
                    img_path = os.path.join(person_folder_path, img_name)

                    img = cv.cvtColor(cv.imread(img_path),  cv.COLOR_BGR2RGB)
                    result = self.app.get(img)

                    if len(result) > 1:
                        print(
                            f'WARNING: more than one face detected in {img_path}')
                        continue

                    embedding = result[0]['embedding']
                    person_dict = {
                        'name': person,
                        'embedding': embedding
                    }
                    face_bank.append(person_dict)

        np.save(face_bank_name, face_bank)
        print('\nFace Bank Updated!\n')

    def get_embeddings(self, face_image):
        faces = self.app.get(face_image)
        embeddings = []
        bboxes = []
        kpses = []

        for face in faces:
            embeddings.append(face['embedding'])
            bboxes.append(map(lambda x: int(x), face['bbox']))
            kpses.append(face['kps'])

        return embeddings, bboxes, kpses

    @staticmethod
    def calc_distance(embedding1, embedding2):
        return float(np.sqrt(np.sum(
            (embedding1 - embedding2) ** 2
        )))

    @staticmethod
    def draw_info(img, bbox, kps, name='unknown'):
        for lm in kps:
            cv.circle(img, (int(lm[0]), int(lm[1])), 5, (255, 0, 255), -1)

        x1, y1, x2, y2 = bbox
        if name == 'unknown':
            cv.putText(img, 'unknown',
                       (x1, y1 - 20), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            cv.putText(img, name,
                       (x1, y1 - 20), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return img

    def recognize(self, img):
        recognized = []
        embeddings, bboxes, kpses = self.get_embeddings(
            cv.cvtColor(img, cv.COLOR_BGR2RGB))
        
        for embedding, bbox, kps in zip(embeddings, bboxes, kpses):
            for person in self.face_bank:
                dis = FaceIdentification.calc_distance(
                    embedding, person['embedding'])
                if dis <= self.threshold:
                    FaceIdentification.draw_info(
                        img, bbox, kps, person['name'])
                    recognized.append(person['name'])
                    return img, recognized
            else:
                FaceIdentification.draw_info(img, bbox, kps, 'unknown')

        return img, []

    @staticmethod
    def make_online(model, thresh, face_bank_path, camera_id, is_show: bool = True) -> list[bool, str]:
        iden = FaceIdentification(model, thresh, face_bank_path)
        cap = cv.VideoCapture(camera_id)

        is_done = False
        c_time, p_time = 0, 0

        once = False

        while True:
            success, frame = cap.read()
            if not success:
                return False, 'Failed to open camera'

            c_time = time()
            frame, recognized = iden.recognize(frame)

            if c_time - p_time >= 10:
                is_done = True
                p_time = time()
            else:
                is_done = False
                if len(recognized):
                    print(recognized)
                    return True, 'Welcome'
            
            if is_done:
                if once:
                    return False, 'Access Denied'
                else:
                    once = True
            
            if is_show:
                cv.imshow(str(camera_id), frame)
            
            if cv.waitKey(1) & 0xFF == 27:
                return False, 'Access Denied'


def main():
    arg = ArgumentParser()
    arg.add_argument('--model', type=str, default='buffalo_s',
                     help='Enter the model')
    arg.add_argument('--img', type=str, required=True,
                     help='Enter the image path')
    arg.add_argument('--thresh', type=float, default=25.,
                     help='Threshold of compare')
    arg.add_argument('--face-bank', type=str, default='face_bank.npy',
                     help='Enter the face bank file path')
    arg.add_argument('--face-bank-path', type=str, default='face_bank',
                     help='Enter the face bank directory path')
    arg.add_argument('--update', type=bool, default=False, action=BooleanOptionalAction,
                     help='Update the face bank')
    opt = arg.parse_args()

    iden = FaceIdentification(opt.model, opt.thresh, opt.face_bank)
    if opt.update:
        iden.update_face_bank(opt.face_bank_path, opt.face_bank)
    result_img, recognized = iden.recognize(cv.imread(opt.img))
    cv.imshow('Result Image', result_img)
    cv.waitKey()


if __name__ == '__main__':
    main()
