import argparse
import math

import numpy as np
from scipy import misc


def main(arguments):
    print(arguments)

    image = read_image("../images/baboon.png")

    # resized = misc.imresize(image, size=0.5, interp='nearest')
    # misc.imsave("resized_nearest.png", resized)

    # resized1 = nearest_neighbor(image, 0.5)
    # misc.imsave("resized_nearest.png", resized1)
    #
    # resized2 = bilinear(image, 0.5)
    # misc.imsave("resized_bilinear.png", resized2)
    #
    # resized3 = bicubic(image, 0.5)
    # misc.imsave("resized_bicubic.png", resized3)

    resized4 = lagrange(image, 0.5)
    misc.imsave("resized_lagrange.png", resized4)

    rotated = rotate_image(resized4, 20)
    misc.imsave("rotated_image.png", rotated)


def nearest_neighbor(image, scale):
    print("Using nearest neighbor method...")

    new_dimension = int(image.shape[0] * scale)
    resized_image = np.full((new_dimension, new_dimension), 0, dtype=np.uint8)
    number_rows, number_cols = resized_image.shape

    # for each row
    for row in range(number_rows):
        # for each column
        for column in range(number_cols):
            x = int(row / scale)
            y = int(column / scale)
            resized_image[row][column] = image[x][y]

    print("Image resized!")
    return resized_image


def bilinear(image, scale):
    print("Using bilinear method...")

    new_dimension = int(image.shape[0] * scale)
    resized_image = np.full((new_dimension, new_dimension), 0, dtype=np.uint8)
    number_rows, number_cols = resized_image.shape

    # for each row
    for row in range(number_rows):
        # for each column
        for column in range(number_cols):
            x = int(row / scale)
            y = int(column / scale)
            dx = (row / scale) - x
            dy = (column / scale) - y
            x_plus_one = min(x + 1, image.shape[0] - 1)
            y_plus_one = min(y + 1, image.shape[1] - 1)

            new_value = (1 - dx) * (1 - dy) * image[x][y] + \
                        dx * (1 - dy) * image[x_plus_one][y] + \
                        (1 - dx) * dy * image[x][y_plus_one] + \
                        dx * dy * image[x_plus_one][y_plus_one]

            resized_image[row][column] = new_value

    print("Image resized!")
    return resized_image


def bicubic(image, scale):
    print("Using bicubic method...")

    new_dimension = int(image.shape[0] * scale)
    resized_image = np.full((new_dimension, new_dimension), 0, dtype=np.uint8)
    number_rows, number_cols = resized_image.shape

    # for each row
    for row in range(number_rows):
        # for each column
        for column in range(number_cols):
            x = int(row / scale)
            y = int(column / scale)
            dx = (row / scale) - x
            dy = (column / scale) - y

            new_value = 0
            for m in range(-1, 3):
                for n in range(-1, 3):
                    print(x + m)
                    normalized_x = min(x + m, image.shape[0] - 1)
                    normalized_y = min(y + n, image.shape[1] - 1)
                    new_value += image[normalized_x][normalized_y] * r_function(m - dx) * r_function(dy - n)

            resized_image[row][column] = new_value

    print("Image resized!")
    return resized_image


def r_function(s):
    v1 = p_function(s + 2) ** 3
    v2 = p_function(s + 1) ** 3
    v3 = p_function(s) ** 3
    v4 = p_function(s - 1) ** 3

    return (1 / 6) * (v1 - 4 * v2 + 6 * v3 - 4 * v4)


def p_function(t):
    if t > 0:
        return t
    else:
        return 0


def lagrange(image, scale):
    print("Using Lagrange method...")

    new_dimension = int(image.shape[0] * scale)
    resized_image = np.full((new_dimension, new_dimension), 0, dtype=np.uint8)
    number_rows, number_cols = resized_image.shape

    # for each row
    for row in range(number_rows):
        # for each column
        for column in range(number_cols):
            x = int(row / scale)
            y = int(column / scale)
            dx = (row / scale) - x
            dy = (column / scale) - y

            resized_image[row][column] = calculate_lagrange(x, y, dx, dy, image)

    print("Image resized!")
    return resized_image


def calculate_lagrange(x, y, dx, dy, image):
    v1 = (-dy) * (dy - 1) * (dy - 2) * l_function(1, x, y, dx, image)
    v2 = (dy + 1) * (dy - 1) * (dy - 2) * l_function(2, x, y, dx, image)
    v3 = (-dy) * (dy + 1) * (dy - 2) * l_function(3, x, y, dx, image)
    v4 = dy * (dy + 1) * (dy - 1) * l_function(4, x, y, dx, image)

    return (v1 / 6) + (v2 / 2) + (v3 / 2) + (v4 / 6)


def l_function(n, x, y, dx, image):
    normalized_y = min(y + n - 2, image.shape[1] - 1)
    v1 = (-dx) * (dx - 1) * (dx - 2) * image[x - 1, normalized_y]
    v2 = (dx + 1) * (dx - 1) * (dx - 2) * image[x, normalized_y]
    v3 = (-dx) * (dx + 1) * (dx - 2) * image[min(x + 1, image.shape[0] - 1), normalized_y]
    v4 = dx * (dx + 1) * (dx - 1) * image[min(x + 2, image.shape[0] - 1), normalized_y]

    return (v1 / 6) + (v2 / 2) + (v3 / 2) + (v4 / 6)


def rotate_image(image, angle):
    print("Rotating image...")

    rad_angle = np.deg2rad(angle)

    rotated_image = np.full(image.shape, 255, dtype=np.uint8)
    number_rows, number_cols = rotated_image.shape

    # for each row
    for row in range(number_rows):
        # for each column
        for column in range(number_cols):
            x = min(round(row * math.cos(rad_angle) - column * math.sin(rad_angle)), rotated_image.shape[0] - 1)
            y = min(round(row * math.sin(rad_angle) + column * math.cos(rad_angle)), rotated_image.shape[1] - 1)
            rotated_image[x][y] = image[row][column]

    print("Image rotated!")
    return rotated_image


def read_image(path, gray_scale=False):
    # Open desired image
    input_image = None
    try:
        input_image = misc.imread(path, flatten=gray_scale)
    except FileNotFoundError:
        print("Image {} not found.".format(path))
        exit(0)

    return input_image


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    # TODO parse arguments
    args = parser.parse_args()

    main(args)
