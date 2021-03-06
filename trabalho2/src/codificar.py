import argparse

import numpy as np
from scipy import misc


# python3 codificar.py ../images/baboon.png ../text/input-text.txt 0 output-image.png

def main(arguments):
    # Open desired image
    input_image = None
    try:
        input_image = misc.imread(arguments.input_image_path)
    except FileNotFoundError:
        print("File {} not found.".format(arguments.input_image_path))
        exit(0)

    with open(arguments.input_text_path, 'r') as input_text_file:
        lines = input_text_file.read()

    number_rows, number_cols, color_layer = input_image.shape

    if number_rows * number_cols * 3 < len(lines) * 8 + 7 * 8:
        print("Mensagem grande demais para ser escondida nesta imagem!")
        return

    print("Codificando: \n{}".format(lines))
    binary_message = []
    for c in lines:
        binary_message += get_binary_list_from_int(ord(c))

    # put in the message the end marker
    for c in "[#END#]":
        binary_message += get_binary_list_from_int(ord(c))

    output_image = encode_message(arguments.bits_layer, input_image, binary_message)
    misc.imsave(arguments.output_image, output_image)

    save_bit_layer_as_image(output_image, 7)
    save_bit_layer_as_image(output_image, 0)
    save_bit_layer_as_image(output_image, 1)
    save_bit_layer_as_image(output_image, 2)

    print()
    print("Codificação completa!")


def save_bit_layer_as_image(image, bit_layer):
    print("Gerando imagem {}-bit-layer".format(bit_layer))
    number_rows, number_cols, color_layer = image.shape
    layer_image = np.full(image.shape, 255, dtype=np.uint8)

    # for each row
    for row in range(number_rows):
        # for each column
        for column in range(number_cols):
            # for each layer of color
            for i in range(color_layer):
                layer_binary = get_binary_list_from_int(image[row][column][i])
                layer_bit = layer_binary[7 - bit_layer]
                layer_binary = [0] * len(layer_binary)
                layer_binary[7 - bit_layer] = layer_bit
                layer_image[row][column][i] = get_binary_list_to_int(layer_binary)

    misc.imsave("layer-{}.png".format(bit_layer), layer_image)
    print("Pronto!")


def encode_message(bits_layer, input_image, binary_message):
    number_rows, number_cols, color_layer = input_image.shape
    output_image = input_image.copy()
    binary_iterator = 0

    # for each row
    for row in range(number_rows):
        # for each column
        for column in range(number_cols):
            # for each layer of color
            for i in range(color_layer):
                layer_binary = get_binary_list_from_int(output_image[row][column][i])
                layer_binary[7 - bits_layer] = binary_message[binary_iterator]
                output_image[row][column][i] = get_binary_list_to_int(layer_binary)
                binary_iterator += 1

                if binary_iterator == len(binary_message):
                    return output_image


def get_binary_list_from_int(i):
    return [int(d) for d in bin(i)[2:].zfill(8)]


def get_binary_list_to_int(binary_list):
    return int(''.join(str(e) for e in binary_list), 2)


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('input_image_path')
    parser.add_argument('input_text_path')
    parser.add_argument('bits_layer', type=int, choices=range(3), metavar="[0-2]")
    parser.add_argument('output_image')
    args = parser.parse_args()

    main(args)
