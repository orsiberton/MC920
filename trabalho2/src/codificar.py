import argparse

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

    print("Codificando: \n{}".format(lines))
    binary_message = []
    for c in lines:
        binary_message += get_binary_list_from_int(ord(c))

    # put in the message the end marker
    for c in "[#END#]":
        binary_message += get_binary_list_from_int(ord(c))

    output_image = encode_message(int(arguments.bits_layer), input_image, binary_message)
    misc.imsave(arguments.output_image, output_image)

    print()
    print("Codificação completa!")


def encode_message(bits_layer, input_image, binary_message):
    number_rows, number_cols, color_layer = input_image.shape
    output_image = input_image.copy()
    binary_iterator = 0

    # for each row
    for row in range(number_rows):
        # for each column
        for column in range(256):
            # for each layer of color
            for i in range(3):
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
    parser.add_argument('bits_layer')
    parser.add_argument('output_image')
    args = parser.parse_args()

    main(args)
