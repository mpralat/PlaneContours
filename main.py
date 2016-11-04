from skimage import data
from matplotlib import pyplot as plt
from skimage import io
import os
import glob
import argparse
import sys
import numpy as np

from skimage import morphology, feature, io, filters
import cv2

color_tab = [(255,0,0), (0,255,0), (0,0,255), (255,128,0), (255,255,0), ]
def parse(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o',
        '--output',
        default=os.path.join(os.getcwd(), "output")
    )
    parser.add_argument(
        '-i',
        '--input',
        default=os.path.join(os.getcwd(), "input")
    )
    return parser.parse_args()

def transform(img_path):


    img = cv2.imread(img_path,0)

    img_copy = cv2.imread(img_path)

    # Here we need to apply different functions to obtain a good base for getting the contours
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=np.ones((3,3), np.uint8))
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    # img = cv2.equalizeHist(img)
    # cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=np.ones((3,3), np.uint8))
    img = cv2.Canny(img, 100,200)


    # After preprocessing the picture we get the contours and centroids
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        if cv2.contourArea(contour):
            M = cv2.moments(contour)
            centroid_x = int(M['m10']/M['m00'])
            centroid_y = int(M['m01']/M['m00'])
            cv2.circle(img_copy, center=(centroid_x, centroid_y), radius=2, color=(255,255,255))
            cv2.drawContours(img_copy, contours, idx, (255,255,0), 3)
    # ctn = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    # cv2.drawContours(img_copy, contours, -1, (255,0,0), 3)

    return img_copy



if __name__ == "__main__":
    args = parse(sys.argv)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print(args.output)
    for img in glob.glob(os.path.join(args.input, "*.jpg")):
        print(os.path.join(args.output, os.path.basename(img)))
        cv2.imwrite(os.path.join(args.output, os.path.basename(img)), transform(img))