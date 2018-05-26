import argparse

import numpy as np
from scipy import misc


def main(arguments):
    print(arguments)

    image = read_image("../images/baboon.png")

    # resized = misc.imresize(image, size=0.5, interp='nearest')
    # misc.imsave("resized_nearest.png", resized)

    resized1 = nearest_neighbor(image, 0.5)
    misc.imsave("resized_nearest.png", resized1)

    resized2 = bilinear(image, 0.5)
    misc.imsave("resized_bilinear.png", resized2)


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
