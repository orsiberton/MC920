import argparse

from scipy import misc


# python3 decodificar.py output-image.png 0 output-text.txt


def main(arguments):
    # Open desired image
    output_image = None
    try:
        output_image = misc.imread(arguments.output_image_path)
    except FileNotFoundError:
        print("File {} not found.".format(arguments.output_image_path))
        exit(0)

    binary_message = []
    number_rows, number_cols, color_layer = output_image.shape

    print("Decodificando...")

    # for each row
    for row in range(number_rows):
        # for each column
        for column in range(256):
            # for each layer of color
            for i in range(3):
                binary_layer = get_binary_list_from_int(output_image[row][column][i])
                binary_message.append(binary_layer[7 - arguments.bits_layer])

    encoded_bits = list(chunks(binary_message, 8))

    decoded_text = get_decoded_text(encoded_bits).split('[#END#]')[0]
    with open(arguments.output_text, 'w') as output_text_file:
        output_text_file.write(decoded_text)

    print("Decodificação completa! Arquivo {} gerado.".format(arguments.output_text))


def get_binary_list_from_int(i):
    return [int(d) for d in bin(i)[2:].zfill(8)]


def get_binary_list_to_int(binary_list):
    return int(''.join(str(e) for e in binary_list), 2)


def get_decoded_text(encoded_bits):
    return ''.join(chr(get_binary_list_to_int(bits)) for bits in encoded_bits)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('output_image_path')
    parser.add_argument('bits_layer', type=int, choices=range(3), metavar="[0-2]")
    parser.add_argument('output_text')
    args = parser.parse_args()

    main(args)
