import argparse
import math

import numpy as np
from scipy import misc

"""
python3 trabalho4.py -a 98.75 -m 4 -i ../images/city.png -o transformed.png

python3 trabalho4.py -e 0.7 -m 1 -i ../images/city.png -o transformed.png

python3 trabalho4.py -d 700 400 -m 2 -i ../images/city.png -o transformed_new.png
"""


def main(arguments):
    image = read_image(arguments.input_image)

    scale = None
    angle = None
    if arguments.angle is not None:
        angle = arguments.angle
    elif arguments.scale is not None:
        scale = arguments.scale
    else:
        scale = (arguments.dimension[1], arguments.dimension[0])

    if arguments.method == 1:
        transformed_image = nearest_neighbor(image, scale=scale, angle=angle)
    elif arguments.method == 2:
        transformed_image = bilinear(image, scale=scale, angle=angle)
    elif arguments.method == 3:
        transformed_image = bicubic(image, scale=scale, angle=angle)
    else:
        transformed_image = lagrange(image, scale=scale, angle=angle)

    misc.imsave(arguments.output_image, transformed_image)


def nearest_neighbor(image, scale=None, angle=None):
    print("Using nearest neighbor method...")

    if scale:
        if isinstance(scale, tuple):
            scale = (scale[0] / image.shape[0], scale[1] / image.shape[1])
        else:
            scale = (scale, scale)

        new_dimension = (int(image.shape[0] * scale[0]), int(image.shape[1] * scale[1]))
        new_image = np.full(new_dimension, 0, dtype=np.uint8)
    else:
        rad_angle = np.deg2rad(360 - angle)
        new_image = np.full(image.shape, 255, dtype=np.uint8)

    number_rows, number_cols = new_image.shape

    x_center = number_rows / 2
    y_center = number_cols / 2

    # for each row
    for row in range(number_rows):
        # for each column
        for column in range(number_cols):

            if scale:
                x = min(round(row / scale[0]), image.shape[0] - 1)
                y = min(round(column / scale[1]), image.shape[1] - 1)
                new_image[row][column] = image[x][y]
            else:
                x = round((row - x_center) * math.cos(rad_angle) - (column - y_center) * math.sin(rad_angle) + x_center)
                y = round((row - x_center) * math.sin(rad_angle) + (column - y_center) * math.cos(rad_angle) + y_center)
                if 0 <= x <= number_rows - 1 and 0 <= y <= number_cols - 1:
                    new_image[row][column] = image[x][y]

    if scale:
        print("Image resized!")
    else:
        print("Image rotated!")
    return new_image


def bilinear(image, scale=None, angle=None):
    print("Using bilinear method...")

    if scale:
        if isinstance(scale, tuple):
            scale = (scale[0] / image.shape[0], scale[1] / image.shape[1])
        else:
            scale = (scale, scale)

        new_dimension = (int(image.shape[0] * scale[0]), int(image.shape[1] * scale[1]))
        new_image = np.full(new_dimension, 0, dtype=np.uint8)
    else:
        rad_angle = np.deg2rad(360 - angle)
        new_image = np.full(image.shape, 255, dtype=np.uint8)

    number_rows, number_cols = new_image.shape

    x_center = number_rows / 2
    y_center = number_cols / 2

    # for each row
    for row in range(number_rows):
        # for each column
        for column in range(number_cols):

            if scale:
                x = int(row / scale[0])
                y = int(column / scale[1])
                dx = (row / scale[0]) - x
                dy = (column / scale[1]) - y
                x_plus_one = min(x + 1, image.shape[0] - 1)
                y_plus_one = min(y + 1, image.shape[1] - 1)

                new_value = (1 - dx) * (1 - dy) * image[x][y] + \
                            dx * (1 - dy) * image[x_plus_one][y] + \
                            (1 - dx) * dy * image[x][y_plus_one] + \
                            dx * dy * image[x_plus_one][y_plus_one]

            else:
                x = int((row - x_center) * math.cos(rad_angle) - (column - y_center) * math.sin(rad_angle) + x_center)
                y = int((row - x_center) * math.sin(rad_angle) + (column - y_center) * math.cos(rad_angle) + y_center)
                dx = (row - x_center) * math.cos(rad_angle) - (column - y_center) * math.sin(rad_angle) + x_center - x
                dy = (row - x_center) * math.sin(rad_angle) + (column - y_center) * math.cos(rad_angle) + y_center - y

                x_plus_one = min(x + 1, image.shape[0] - 1)
                y_plus_one = min(y + 1, image.shape[1] - 1)

                if 0 <= x <= number_rows - 1 and 0 <= y <= number_cols - 1:
                    new_value = (1 - dx) * (1 - dy) * image[x][y] + \
                                dx * (1 - dy) * image[x_plus_one][y] + \
                                (1 - dx) * dy * image[x][y_plus_one] + \
                                dx * dy * image[x_plus_one][y_plus_one]

                    new_image[row][column] = new_value
                else:
                    new_value = 255

            new_image[row][column] = new_value

    if scale:
        print("Image resized!")
    else:
        print("Image rotated!")
    return new_image


def bicubic(image, scale=None, angle=None):
    print("Using bicubic method...")

    if scale:
        if isinstance(scale, tuple):
            scale = (scale[0] / image.shape[0], scale[1] / image.shape[1])
        else:
            scale = (scale, scale)

        new_dimension = (int(image.shape[0] * scale[0]), int(image.shape[1] * scale[1]))
        new_image = np.full(new_dimension, 0, dtype=np.uint8)
    else:
        rad_angle = np.deg2rad(360 - angle)
        new_image = np.full(image.shape, 255, dtype=np.uint8)

    number_rows, number_cols = new_image.shape
    x_center = number_rows / 2
    y_center = number_cols / 2

    # for each row
    for row in range(number_rows):
        # for each column
        for column in range(number_cols):

            if scale:
                x = int(row / scale[0])
                y = int(column / scale[1])
                dx = (row / scale[0]) - x
                dy = (column / scale[1]) - y

                new_value = calculate_new_value(image, x, y, dx, dy)
            else:
                x = int((row - x_center) * math.cos(rad_angle) - (column - y_center) * math.sin(rad_angle) + x_center)
                y = int((row - x_center) * math.sin(rad_angle) + (column - y_center) * math.cos(rad_angle) + y_center)
                dx = (row - x_center) * math.cos(rad_angle) - (column - y_center) * math.sin(rad_angle) + x_center - x
                dy = (row - x_center) * math.sin(rad_angle) + (column - y_center) * math.cos(rad_angle) + y_center - y

                if 0 <= x <= number_rows - 1 and 0 <= y <= number_cols - 1:
                    new_value = calculate_new_value(image, x, y, dx, dy)
                else:
                    new_value = 255

            new_image[row][column] = new_value

    if scale:
        print("Image resized!")
    else:
        print("Image rotated!")
    return new_image


def calculate_new_value(image, x, y, dx, dy):
    new_value = 0
    for m in range(-1, 3):
        for n in range(-1, 3):
            normalized_x = min(x + m, image.shape[0] - 1)
            normalized_y = min(y + n, image.shape[1] - 1)
            new_value += image[normalized_x][normalized_y] * r_function(m - dx) * r_function(dy - n)

    return new_value


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


def lagrange(image, scale=None, angle=None):
    print("Using Lagrange method...")

    if scale:
        if isinstance(scale, tuple):
            scale = (scale[0] / image.shape[0], scale[1] / image.shape[1])
        else:
            scale = (scale, scale)

        new_dimension = (int(image.shape[0] * scale[0]), int(image.shape[1] * scale[1]))
        new_image = np.full(new_dimension, 0, dtype=np.uint8)
    else:
        rad_angle = np.deg2rad(360 - angle)
        new_image = np.full(image.shape, 255, dtype=np.uint8)

    number_rows, number_cols = new_image.shape
    x_center = number_rows / 2
    y_center = number_cols / 2

    # for each row
    for row in range(number_rows):
        # for each column
        for column in range(number_cols):
            if scale:
                x = int(row / scale[0])
                y = int(column / scale[1])
                dx = (row / scale[0]) - x
                dy = (column / scale[1]) - y

                new_image[row][column] = calculate_lagrange(x, y, dx, dy, image)
            else:
                x = int((row - x_center) * math.cos(rad_angle) - (column - y_center) * math.sin(rad_angle) + x_center)
                y = int((row - x_center) * math.sin(rad_angle) + (column - y_center) * math.cos(rad_angle) + y_center)
                dx = (row - x_center) * math.cos(rad_angle) - (column - y_center) * math.sin(rad_angle) + x_center - x
                dy = (row - x_center) * math.sin(rad_angle) + (column - y_center) * math.cos(rad_angle) + y_center - y

                if 0 <= x <= number_rows - 1 and 0 <= y <= number_cols - 1:
                    new_image[row][column] = calculate_lagrange(x, y, dx, dy, image)

    if scale:
        print("Image resized!")
    else:
        print("Image rotated!")
    return new_image


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
    parser.add_argument("-a", "--angle", type=float, required=False)
    parser.add_argument("-e", "--scale", type=float, required=False)
    parser.add_argument("-d", "--dimension", type=int, nargs=2, required=False)
    parser.add_argument("-m", "--method", type=int, required=True, choices=range(1, 5), metavar="[1-4]",
                        help='1 - nearest_neighbor | 2 - bilinear | 3 - bicubic | 4 - lagrange')
    parser.add_argument("-i", "--input_image", required=True)
    parser.add_argument("-o", "--output_image", required=True)

    args = parser.parse_args()

    count_args = 0
    if args.angle is not None:
        count_args += 1
    if args.scale is not None:
        count_args += 1
    if args.dimension is not None:
        count_args += 1

    if count_args > 1:
        print("Choose only one between angle, scale and dimension")
        exit()
    elif count_args == 0:
        print("Choose one between angle, scale and dimension")
        exit()

    main(args)
