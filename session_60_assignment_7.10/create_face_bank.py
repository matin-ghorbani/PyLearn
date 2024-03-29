import os
from argparse import ArgumentParser

import cv2 as cv
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis

arg = ArgumentParser()
arg.add_argument('--model', type=str, default='buffalo_s',
                 help='Enter the model')
arg.add_argument('--face-bank', type=str, required=True,
                 help='Enter the face bank directory path')
arg.add_argument('--face-bank-name', type=str, default='face_bank',
                 help='Enter the face bank file name to save')

opt = arg.parse_args()


def main():
    app = FaceAnalysis(name=opt.model, providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    face_bank_path = opt.face_bank

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
                result = app.get(img)

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

    np.save(f'{opt.face_bank_name}.npy', face_bank)
    print('Face Bank Created')


if __name__ == '__main__':
    main()
