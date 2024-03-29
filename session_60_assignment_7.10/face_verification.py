from argparse import ArgumentParser, BooleanOptionalAction

from insightface.app import FaceAnalysis
import cv2 as cv

import utils as ul

arg = ArgumentParser()
arg.add_argument('--model', type=str, default='buffalo_s',
                 help='Enter the model')
arg.add_argument('--img1', type=str, required=True,
                 help='Enter the first image path')
arg.add_argument('--img2', type=str, required=True,
                 help='Enter the second image path')
arg.add_argument('--thresh', type=float, default=25.,
                 help='Threshold of compare')
arg.add_argument('--show', type=bool, default=False, action=BooleanOptionalAction,
                 help='Show the results')

opt = arg.parse_args()




def main():
    app = FaceAnalysis(name=opt.model, providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img1 = cv.imread(opt.img1)
    img2 = cv.imread(opt.img2)

    img1_rgb = cv.cvtColor(img1,  cv.COLOR_BGR2RGB)
    img3_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

    result1 = app.get(img1_rgb)[0]
    result2 = app.get(img3_rgb)[0]

    embedding1 = result1['embedding']
    embedding2 = result2['embedding']

    img1 = ul.draw_info(img1, result1)
    img2 = ul.draw_info(img2, result2)

    distance = round(ul.calc_distance(embedding1, embedding2), 4)
    text = ''
    if distance <= opt.thresh:
        text = f'Same Person With This Distance: {distance}'
        print('\n\n👍 ' + text)
    else:
        text = f'Different Person With This Distance: {distance}'
        print('\n\n👎 ' + text)
    
    if opt.show:
        img1 = cv.resize(img1, (640, 640))
        img2 = cv.resize(img2, (640, 640))

        main_img = ul.stack_images([img1, img2], 2, 1.)
        h, w = main_img.shape[:2]
        cv.putText(main_img, text, (w // 2 - 350, h // 2 + 150), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
        cv.imshow('result', main_img)
        cv.waitKey()

if __name__ == '__main__':
    main()
