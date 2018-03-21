# Bruno Orsi Berton - RA 150573 - MC920

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from skimage import measure


def main(arguments):
    # Open desired image
    image = None
    try:
        image = misc.imread(arguments.image_path)
    except FileNotFoundError:
        print("File {} not found.".format(args.image_path))
        exit(0)

    gray_image = extract_gray_image(image)
    misc.imsave('gray-image.png', gray_image)

    contours = extract_contours(gray_image)

    contours_image = save_contours_image(contours, gray_image.shape)
    print(contours_image)
    labels, num_labels = measure.label(gray_image, return_num=True)
    print(labels)
    print(num_labels)
    props = measure.regionprops(labels)

    for prop in props:
        print("região: {} perímetro: {} área: {}".format(prop.perimeter, prop.label, prop.area))


def extract_gray_image(color_image):
    return np.round(rgb_to_gray(color_image))


def rgb_to_gray(rgb):
    return np.dot(rgb, [0.299, 0.587, 0.114])


def extract_contours(gray_image):
    return measure.find_contours(gray_image, 0.8)


def save_contours_image(contours, shape):
    empty_image = np.full(shape, 255, dtype=np.uint8)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    plt.imshow(empty_image, interpolation='nearest', cmap='gray', shape=shape, vmin=0, vmax=255)

    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='black')

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('contours.png', bbox_inches='tight', pad_inches=0, cmap='gray')

    return misc.imread('contours.png')


if __name__ == "__main__":
    # Parse the image relative path argument
    parser = argparse.ArgumentParser(description='Process the image.')
    parser.add_argument('image_path', help='Relative image path that will be processed.')
    args = parser.parse_args()

    main(args)
