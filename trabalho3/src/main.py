import argparse

import numpy as np
from scipy import misc
from scipy.ndimage.interpolation import rotate
from skimage.feature import canny
from skimage.transform import (hough_line, hough_line_peaks)

alignment_algorithm = {
    0: 'horizontal',
    1: 'hough'
}


def main(arguments):
    print("Using {} mode.".format(alignment_algorithm[arguments.mode]))

    original_image = read_image(arguments.input_image_path)

    if alignment_algorithm[arguments.mode] == 'horizontal':
        print("TODO")
        angle = None
        exit(0)
    else:
        angle = hough_method(original_image)

    print("Angle found {:.4f}.".format(angle))

    fixed_image = rotate_image(original_image, angle)
    misc.imsave(arguments.output_image, fixed_image)


def hough_method(image):
    edges = canny(image, 2, 1, 25)
    h, theta, d = hough_line(edges)
    angles = []
    for _, angle, _ in zip(*hough_line_peaks(h, theta, d)):
        angles.append(np.rad2deg(angle))

    angle = np.median(angles)

    return angle - 90 if angle >= 0 else angle + 90


def read_image(path):
    # Open desired image
    input_image = None
    try:
        input_image = misc.imread(path, flatten=True)
    except FileNotFoundError:
        print("Image {} not found.".format(path))
        exit(0)

    return input_image


def rotate_image(image, angle):
    return rotate(image, angle, reshape=False, cval=255)


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    # TODO remove default and nargs parameters after
    parser.add_argument('input_image_path', default='../images/sample2.png', nargs='?')
    parser.add_argument('mode', type=int, choices=range(2), metavar="[0-1]", default=0, nargs='?')
    parser.add_argument('output_image', default='fixed_image.png', nargs='?')
    args = parser.parse_args()

    main(args)
