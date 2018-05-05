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
    original_gray_image = read_image(arguments.input_image_path, gray_scale=True)

    if alignment_algorithm[arguments.mode] == 'horizontal':
        angle = horizontal_projection_method(original_gray_image)
    else:
        angle = hough_method(original_gray_image)

    print("The image should be fixed by {:.4f} degrees.".format(angle))

    fixed_image = rotate_image(original_image, angle)
    misc.imsave(arguments.output_image, fixed_image)


def horizontal_projection_method(image):
    copy_image = image.copy()
    copy_image[copy_image <= 128] = 1
    copy_image[copy_image > 128] = 0
    original_hp = np.sum(copy_image, 1)

    best_angle = 0
    max_dist = -1
    for i in range(-90, 91):
        rotated_image = rotate_image(image, i)
        # binarize the image
        rotated_image[rotated_image <= 128] = 1
        rotated_image[rotated_image > 128] = 0

        horizontal_profile = np.sum(rotated_image, 1)
        result = ssd(original_hp, horizontal_profile)
        if result > max_dist:
            best_angle = i
            max_dist = result

    return best_angle


def ssd(a, b):
    return ((a - b) ** 2).sum()


def hough_method(image):
    edges = canny(image, 2, 1, 25)
    h, theta, d = hough_line(edges)
    angles = []
    for _, angle, _ in zip(*hough_line_peaks(h, theta, d)):
        angles.append(np.rad2deg(angle))

    angle = np.median(angles)

    return angle - 90 if angle >= 0 else angle + 90


def read_image(path, gray_scale=False):
    # Open desired image
    input_image = None
    try:
        input_image = misc.imread(path, flatten=gray_scale)
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
    parser.add_argument('mode', type=int, choices=range(2), metavar="[0-1]", default=1, nargs='?')
    parser.add_argument('output_image', default='fixed_image.png', nargs='?')
    args = parser.parse_args()

    main(args)
