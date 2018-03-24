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

    save_contours_image(contours, gray_image.shape)

    # anything that is not background(white - 255) is gonna change to black
    gray_image[gray_image < 255] = 0

    labels, num_labels = measure.label(gray_image, return_num=True, background=255)
    props = measure.regionprops(labels, intensity_image=gray_image)

    num_small_regions, num_medium_regions, num_big_regions = print_statistics(props)
    save_labeled_image(image, props)
    save_histogram_by_area(np.array([num_small_regions, num_medium_regions, num_big_regions]))


def extract_gray_image(color_image):
    return np.round(rgb_to_gray(color_image)).astype(int)


def rgb_to_gray(rgb):
    return np.dot(rgb, [0.299, 0.587, 0.114])


def extract_contours(gray_image):
    return measure.find_contours(gray_image, 0.8)


def save_contours_image(contours, shape):
    empty_image = np.full(shape, 255, dtype=np.uint8)

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(empty_image, interpolation='nearest', cmap='gray', shape=shape, vmin=0, vmax=255)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='black')

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig('contours.png', bbox_inches='tight', pad_inches=0, cmap='gray')

    return misc.imread('contours.png')


def print_statistics(props):
    num_small_regions = 0
    num_medium_regions = 0
    num_big_regions = 0

    print("número de regiões: {}".format(len(props)))
    print()
    for prop in props:
        print("região: {} perímetro: {:.2f} área: {}".format(prop.label - 1, prop.perimeter, prop.area))
        if prop.area < 1500:
            num_small_regions += 1

        if 1500 <= prop.area < 3000:
            num_medium_regions += 1

        if prop.area >= 3000:
            num_big_regions += 1

    print()
    print("número de regiões pequenas: {}".format(num_small_regions))
    print("número de regiões médias: {}".format(num_medium_regions))
    print("número de regiões grandes: {}".format(num_big_regions))
    return num_small_regions, num_medium_regions, num_big_regions


def save_labeled_image(image, props):
    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(image)

    for prop in props:
        ax.text(prop.centroid[1] - 5, prop.centroid[0] + 5, str(prop.label - 1))

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig('labeled-image.png', bbox_inches='tight', pad_inches=0)


def save_histogram_by_area(histogram_array):
    # Generates the histogram image
    fig, ax = plt.subplots()
    ax.set_ylabel('Número de regiões')
    ax.set_xlabel('Área(pequena, média e grande)')

    ax.bar(250, histogram_array[0], 750, align='edge', color='blue')
    ax.bar(1750, histogram_array[1], 750, align='edge', color='blue')
    ax.bar(3250, histogram_array[2], 750, align='edge', color='blue')
    plt.savefig('histogram.png')


if __name__ == "__main__":
    # Parse the image relative path argument
    parser = argparse.ArgumentParser(description='Process the image.')
    parser.add_argument('image_path', help='Relative image path that will be processed.')
    args = parser.parse_args()

    main(args)
