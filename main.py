import os
import glob
import argparse
import sys
import numpy as np

import cv2


def parse_args(arguments):
    """
    Parsing arguments given in the command line
    If there are none, we take pictures from input folder and saving to output

    :param arguments program arguments supplied via the command prompt
    """
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

def convert_to_black_white(pixel):
    r, g, b = pixel
    brightness = np.math.sqrt(0.299 * (r ** 2) + 0.587 * (g ** 2) + 0.114 * (b ** 2))
    if brightness < 0.5:
        return (0,0,0)
    else:
        return (255,255,255)


def transform(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image_copy = cv2.imread(img_path, 0)

    # Here we need to apply different functions to obtain a good base for getting the contours
    cv2.imwrite('test1.jpg', image)
    # Apply Black&White transformation
    _, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite('test_bw.jpg', image)



    # # Apply morphological transformation - opening (erosion and then dilation) - reduces noise
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8))
    # cv2.imwrite('test2_morph.jpg', image)
    # # Apply histogram equalisation based on the cumulative distribution of intensity with limited contrast
    # clahe = cv2.createCLAHE(clipLimit=0.7, tileGridSize=(8, 8))
    # image = clahe.apply(image)
    # cv2.imwrite('test3_clahe.jpg', image)
    #
    # # image = cv2.equalizeHist(image)
    # # cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # # cv2.imwrite('test3_5_threshold.jpg', image)
    #
    # # Apply morphological transformation - closing (dilation and the erosion) - smooths the image
    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), np.uint8))
    # cv2.imwrite('test4_morph2.jpg', image)
    image = cv2.Canny(image, 100, 200)
    cv2.imwrite('test5_canny2.jpg', image)

    # th, bw = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # image = cv2.Canny(image, th/2, th)
    # cv2.imwrite('test5_canny2.jpg', image)

    # After preprocessing the picture we get the contours and centroids
    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        if cv2.contourArea(contour):
            m = cv2.moments(contour)
            centroid_x = int(m['m10'] / m['m00'])
            centroid_y = int(m['m01'] / m['m00'])
            cv2.circle(image_copy, center=(centroid_x, centroid_y), radius=2, color=(255, 255, 255))
            cv2.drawContours(image_copy, contours, idx, (255, 255, 0), 3)
    # ctn = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    # cv2.drawContours(image_copy, contours, -1, (255,0,0), 3)

    return image


if __name__ == "__main__":
    args = parse_args(sys.argv)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print(args.output)
    for img in glob.glob(os.path.join(args.input, "*.jpg")):
        print(os.path.join(args.output, os.path.basename(img)))
        cv2.imwrite(os.path.join(args.output, os.path.basename(img)), transform(img))
        # exit(0)
