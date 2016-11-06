import os
import glob
import argparse
import sys
import numpy as np
import cv2

colors = [
    (204, 204, 0),
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 51, 153)
]
color_idx = 0


def parse(arguments):
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
        return (0, 0, 0)
    else:
        return (255, 255, 255)


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def transform(img_path):
    # Load image in color and grayscale
    global color_idx
    img_grayscale = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(img_path)
    cv2.imwrite('test1_orig.jpg', img_grayscale)

    # Here we need to apply different functions to obtain a good base for getting the contours
    # Apply morphological transformation - opening (erosion and then dilation) - reduces noise
    img_grayscale = cv2.morphologyEx(img_grayscale, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
    cv2.imwrite('test2_morph.jpg', img_grayscale)

    # Apply histogram equalisation based on the cumulative distribution of intensity with limited contrast
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5))
    # img_grayscale = clahe.apply(img_grayscale)
    # cv2.imwrite('test3_clahe.jpg', img_grayscale)

    # Adaptive thresholding
    # img_grayscale = cv2.equalizeHist(img_grayscale)
    # cv2.adaptiveThreshold(img_grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 5)
    # cv2.imwrite('test3_adaptiveThreshold.jpg', img_grayscale)

    # Gaussian blur
    # img_grayscale = cv2.GaussianBlur(img_grayscale, (0, 0), 1.0)
    # img_grayscale = cv2.addWeighted(img_grayscale, 1.5, blurred_image, -0.5, 0)
    cv2.imwrite('test4_sharpen.jpg', img_grayscale)

    # img_grayscale = cv2.morphologyEx(img_grayscale, cv2.MORPH_CLOSE, kernel=np.ones((15, 15), np.uint8))
    # cv2.imwrite('test5_morph2.jpg', img_grayscale)
    # img_grayscale = cv2.Canny(img_grayscale, 100, 200)
    img_grayscale = auto_canny(img_grayscale)
    cv2.imwrite('test6_canny.jpg', img_grayscale)
    img_grayscale = cv2.morphologyEx(img_grayscale, cv2.MORPH_DILATE, kernel=np.ones((2, 2), np.uint8))
    cv2.imwrite('test7_dillatation.jpg', img_grayscale)

# After preprocessing the picture we get the contours and centroids
    _, contours, hierarchy = cv2.findContours(img_grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        if cv2.contourArea(contour):
            M = cv2.moments(contour)
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            cv2.circle(img_color, center=(centroid_x, centroid_y), radius=2, color=(255, 255, 255))
            cv2.drawContours(img_color, contours, idx, colors[color_idx], 3)
            color_idx = (color_idx + 1) % len(colors)
    # ctn = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    # cv2.drawContours(img_color, contours, -1, (255,0,0), 3)

    return img_color


if __name__ == "__main__":
    args = parse(sys.argv)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print(args.output)
    for img in glob.glob(os.path.join(args.input, "*.jpg")):
        print(os.path.join(args.output, os.path.basename(img)))
        cv2.imwrite(os.path.join(args.output, os.path.basename(img)), transform(img))
