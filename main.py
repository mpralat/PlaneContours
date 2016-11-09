import os
import glob
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


colors = [
    (204, 0, 204),
    (204, 204, 0),
    (0, 204, 0),
    (102, 0, 204),
    (255, 128, 0),
    (255, 255, 0),
    (0, 204, 204),
    (0, 255, 0),
    (51, 255, 255),
    (204, 0, 0),
    (255, 102, 178),
    (153, 255, 153),
    (255, 153, 153)
]
color_idx = 0


def parse():
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


def auto_canny(image, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # Return the edged image
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
    clahe = cv2.createCLAHE(clipLimit=0.35, tileGridSize=(5, 5))
    img_grayscale = clahe.apply(img_grayscale)
    cv2.imwrite('test3_clahe.jpg', img_grayscale)

    # Using the Canny algorithm to detect the edges.
    img_grayscale = auto_canny(img_grayscale, 0.560)
    cv2.imwrite('test6_canny.jpg', img_grayscale)
    # Apply dilation to merge neighbouring contours.
    img_grayscale = cv2.morphologyEx(img_grayscale, cv2.MORPH_DILATE, kernel=np.ones((2, 2), np.uint8))
    cv2.imwrite('test7_dillatation.jpg', img_grayscale)

# After preprocessing the picture we get the contours and centroids
    _, contours, hierarchy = cv2.findContours(img_grayscale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for idx, contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)
        # Discard small contours that aren't planes for sure
        if contour_area > 590:
            M = cv2.moments(contour)
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            cv2.circle(img_color, center=(centroid_x, centroid_y), radius=2, color=(255, 255, 255))
            cv2.drawContours(img_color, contours, idx, colors[color_idx], 3)
            color_idx = (color_idx + 1) % len(colors)

    return img_color

def plot_plot():
    fig = plt.figure()
    fig.set_size_inches(15, 10)
    pics = glob.glob(os.path.join(os.getcwd(),  "steps/*.jpg"))
    for idx, img in enumerate(sorted(pics)):
        a = fig.add_subplot(2, 3, idx % 6 + 1)
        a.axis("off")
        imgplot = plt.imshow(cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2RGB))
    plt.savefig("steps.pdf")

if __name__ == "__main__":
    # plot_plot()
    args = parse()

    fig = plt.figure()
    # Changing the resolution here.
    fig.set_size_inches(100, 70)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print(args.output)

    for idx, img in enumerate(glob.glob(os.path.join(args.input, "*.jpg"))):
        print(os.path.join(args.output, os.path.basename(img)))
        a = fig.add_subplot(3, 3, idx % 9 + 1)
        a.axis("off")
        imgplot = plt.imshow(cv2.cvtColor(transform(img), cv2.COLOR_BGR2RGB))

        cv2.imwrite(os.path.join(args.output, os.path.basename(img)), transform(img))
        if (idx + 1) % 9 == 0:
            plt.savefig("high_res_results" + str(idx) + ".pdf")

