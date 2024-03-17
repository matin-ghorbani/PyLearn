import argparse
import sqlite3 as sl
from difflib import SequenceMatcher
import cv2 as cv
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes

from deep_text_recognition_benchmark.dtrb import DTRB


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=0)
parser.add_argument('--batch_size', type=int,
                    default=192, help='input batch size')
""" Data processing """
parser.add_argument('--batch_max_length', type=int,
                    default=25, help='maximum-label-length')
parser.add_argument('--imgH', type=int, default=32,
                    help='the height of the input image')
parser.add_argument('--imgW', type=int, default=100,
                    help='the width of the input image')
parser.add_argument('--rgb', action='store_true', help='use rgb input')
parser.add_argument('--character', type=str,
                    default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
parser.add_argument('--sensitive', action='store_true',
                    help='for sensitive character mode')
parser.add_argument('--PAD', action='store_true',
                    help='whether to keep ratio then pad for image resize')
""" Model Architecture """
parser.add_argument('--Transformation', type=str,
                    default="TPS", help='Transformation stage. None|TPS')
parser.add_argument('--FeatureExtraction', type=str, default="ResNet",
                    help='FeatureExtraction stage. VGG|RCNN|ResNet')
parser.add_argument('--SequenceModeling', type=str,
                    default="BiLSTM", help='SequenceModeling stage. None|BiLSTM')
parser.add_argument('--Prediction', type=str, default="Attn",
                    help='Prediction stage. CTC|Attn')
parser.add_argument('--num_fiducial', type=int, default=20,
                    help='number of fiducial points of TPS-STN')
parser.add_argument('--input_channel', type=int, default=1,
                    help='the number of input channel of Feature extractor')
parser.add_argument('--output_channel', type=int, default=512,
                    help='the number of output channel of Feature extractor')
parser.add_argument('--hidden_size', type=int, default=256,
                    help='the size of the LSTM hidden state')

parser.add_argument('--detector-weight', type=str,
                    default='weights/yolov8-detector/yolov8-s-license-plate-detector.pt', help='Path of detector weight')
parser.add_argument('--recognizer-weight', type=str,
                    default='weights/dtrb-recognizer/dtrb-None-VGG-BiLSTM-CTC-license-plate-recognizer.pth', help='Path of recognizer weight')
parser.add_argument('--input-img', type=str,
                    default='io/input/test.jpg', help='Path of image to test')
parser.add_argument('--threshold', type=float, default=.7,
                    help='Threshold of testing')
parser.add_argument('--show-output', type=bool,
                    default=True, help='Show the output image')
parser.add_argument('--save-output', type=bool,
                    default=True, help='Save the output image')
parser.add_argument('--db-name', type=str, default='license_plate_db.db', help='Path of your database')
parser.add_argument('--threshold-verification', type=float, default=.8, help='Threshold of verification')


opt = parser.parse_args()
plate_detector = YOLO(opt.detector_weight)
plate_recognizer = DTRB(opt.recognizer_weight, opt)

conn = sl.connect(opt.db_name)
curr = conn.cursor()
rows = curr.execute('select * from masters')

image = cv.imread(opt.input_img)
results: Results = plate_detector.predict(image)

classes = ['License Plate']

for result in results:
    boxes: Boxes = result.boxes
    for i, box in enumerate(boxes):
        conf = int(box.conf[0] * 100) / 100
        if conf > opt.threshold:
            x1, y1, x2, y2 = map(lambda x: int(x), box.xyxy[0])
            current_class = classes[int(box.cls[0])]

            plate_image = image[y1:y2, x1:x2].copy()

            if opt.save_output:
                cv.imwrite(
                    f"io/output/plate_image_result_{i}.jpg", plate_image)
            cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 4)
            cv.putText(image, f'{current_class} {conf}', (x1,
                       y2 + 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            plate_image = cv.resize(plate_image, (100, 32))
            plate_image = cv.cvtColor(plate_image, cv.COLOR_BGR2GRAY)
            text_pred, score = plate_recognizer.predict(plate_image)
            score = int(score * 100) / 100
            image = cv.putText(
                image, f'{text_pred} {score}', (x1, y1 - 10), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            
            master_name: str = ''

            for master in rows:
                distance = SequenceMatcher(None, master[2], text_pred).ratio()
                if distance >= opt.threshold_verification:
                    master_name = master[1]
                    break
            
            if master_name:
                print(f'👍  Master {master_name} Found.')
            else:
                print(f'👎  {text_pred} Not Found.')



if opt.save_output:
    cv.imwrite("io/output/image_result.jpg", image)


if opt.show_output:
    cv.imshow('output image', cv.resize(image, (0, 0), fx=.5, fy=.5))
    cv.waitKey()
