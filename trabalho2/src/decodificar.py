import argparse

from scipy import misc

def main(arguments):
    # Open desired image
    output_image = None
    try:
        output_image = misc.imread(arguments.input_image_path)
    except FileNotFoundError:
        print("File {} not found.".format(arguments.input_image_path))
        exit(0)


    return 0


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('output_image_path')
    parser.add_argument('bits_layer')
    parser.add_argument('output_text')
    args = parser.parse_args()

    main(args)
