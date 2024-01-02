import pandas as pd
import cv2 as cv
import os
from deepface import DeepFace


def generate_dataset(dataset_path: str, save_df: bool = True, df_name_to_save: str = 'extracted_features.csv') -> pd.DataFrame | None:
    subjects_list = os.listdir(dataset_path)
    subject_feature = {}
    all_features = []

    for subject in subjects_list:
        print(f'{subject}...')
        for image in os.listdir(f'{dataset_path}/{subject}'):
            # TODO: Extract features with deepface
            try:
              encoded_features = DeepFace.represent(img_path=os.path.join(
                  dataset_path, subject, image), model_name='ArcFace', enforce_detection=False)
              subject_feature['subject'] = subject
              for i in range(len(encoded_features[0]['embedding'])):
                  subject_feature[f'feature{i + 1}'] = encoded_features[0]['embedding'][i]
              all_features.append(subject_feature.copy())
              subject_feature = {}

            except ValueError:
              print(f'the {image} has non-english characters!')

    df = pd.DataFrame(all_features)

    if save_df:
        df.to_csv(df_name_to_save, index=False)
    else:
        return df



def main() -> None:
    df = generate_dataset(
        './persian_face', df_name_to_save='Test_df.csv', save_df=True)
    if df:
        print(df.head())


if __name__ == '__main__':
    main()
