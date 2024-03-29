import copy

import cv2 as cv
import numpy as np

def calc_distance(embedding1, embedding2):
    return float(np.sqrt(np.sum(
        (embedding1 - embedding2) ** 2
    )))

def stack_images(img_list, cols, scale):
    img_list = copy.deepcopy(img_list)

    total_images = len(img_list)
    rows = total_images // cols if total_images // cols * cols == total_images else total_images // cols + 1
    blank_images = cols * rows - total_images
    h, w = img_list[0].shape[:2]

    imgBlank = np.zeros((h, w, 3), np.uint8)
    img_list.extend([imgBlank] * blank_images)

    for i in range(cols * rows):
        img_list[i] = cv.resize(img_list[i], (0, 0), fx=scale, fy=scale)
        if len(img_list[i].shape) == 2:
            img_list[i] = cv.cvtColor(img_list[i], cv.COLOR_GRAY2BGR)

    hor = [imgBlank] * rows
    for y in range(rows):
        line = []
        for x in range(cols):
            line.append(img_list[y * cols + x])
        hor[y] = np.hstack(line)

    ver = np.vstack(hor)
    return ver

def draw_info(img, result):
    for lm in result.kps:
        cv.circle(img, (int(lm[0]), int(lm[1])), 5, (255, 0, 255), -1)
        
    x1, y1, x2, y2 = map(lambda x: int(x), result.bbox)
    cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img
