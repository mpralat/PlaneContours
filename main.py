from skimage import data
from matplotlib import pyplot as plt
from skimage import io
import os
import glob
import argparse
import sys

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
    print(img_path)

if __name__ == "__main__":
    args = parse(sys.argv)
    for img in glob.glob(os.path.join(args.input, "*.jpg")):
        transform(img)