import argparse

from scipy import misc
from scipy.ndimage.interpolation import rotate

alignment_algorithm = {
    0: 'horizontal',
    1: 'hough'
}


def main(arguments):
    print("Using {} mode.".format(alignment_algorithm[arguments.mode]))

    original_image = read_image(arguments.input_image_path)

    # TODO implementar os dois algoritmos

    # TODO depois de descobrir o angulo, substituir o valor
    fixed_image = rotate_image(original_image, -28)
    misc.imsave(arguments.output_image, fixed_image)


def read_image(path):
    # Open desired image
    input_image = None
    try:
        input_image = misc.imread(path)
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
    parser.add_argument('input_image_path', default='../images/neg_28.png', nargs='?')
    parser.add_argument('mode', type=int, choices=range(2), metavar="[0-1]", default=0, nargs='?')
    parser.add_argument('output_image', default='fixed_image.png', nargs='?')
    args = parser.parse_args()

    main(args)
