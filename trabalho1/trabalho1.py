# Bruno Orsi Berton - RA 150573 - MC920

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from skimage import exposure


def rgb_to_gray(rgb):
    return np.dot(rgb, [0.299, 0.587, 0.114])


def main(arguments):
    # Open desired image
    image = None
    try:
        image = misc.imread(arguments.image_path)
    except FileNotFoundError:
        print("File {} not found.".format(args.image_path))
        exit(0)

    # print(image.shape)
    # print(image.shape[0])
    # print(image.shape[1])
    # print(image.shape[2])
    # print(image[0].shape)
    # print(image[40][200])

    gray_image = rgb_to_gray(image)
    misc.imsave('gray-image.png', gray_image)


if __name__ == "__main__":
    # Parse the image relative path argument
    parser = argparse.ArgumentParser(description='Process the image.')
    parser.add_argument('image_path', help='Relative image path that will be processed.')
    args = parser.parse_args()

    main(args)
