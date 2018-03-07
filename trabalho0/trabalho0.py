# Bruno Orsi Berton - RA 150573 - MC920

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from skimage import exposure

# Parse the image relative path argument
parser = argparse.ArgumentParser(description='Process the image.')
parser.add_argument('image_path', help='Relative image path that will be processed.')
args = parser.parse_args()

# get image name from user

# Open desired image
try:
    image = misc.imread(args.image_path)
except FileNotFoundError:
    print("Arquivo {} não encontrado.".format(args.image_path))
    exit(0)

# Generates the image histogram
fig, ax = plt.subplots()
ax.set_ylabel('Frequência')
ax.set_xlabel('Níveis de cinza')

histogram = plt.hist(image.ravel(), bins=256, range=(0, 255), fc='g', ec='g')
plt.savefig('histogram.png')
print("Histograma gerado com sucesso! arquivo: histogram.png\n")

# Print statistics
number_rows, number_cols = image.shape
print("Estatísticas:")
print("largura: {}".format(number_cols))
print("altura: {}".format(number_rows))
print("intensidade mínima: {}".format(image.min()))
print("intensidade máxima: {}".format(image.max()))
print("intensidade média: {:.2f}\n".format(image.mean()))

# Generates the negative image, subtraction the image matrix from a full 255 matrix
full_matrix = np.full(image.shape, 255)
negative_image = full_matrix - image
misc.imsave('negative-image.png', negative_image)
print("Negativo da imagem gerado com sucesso! arquivo: negative-image.png\n")

filtered_image = exposure.rescale_intensity(image, out_range=(120, 180))
misc.imsave('filtered-image.png', filtered_image)
print("Intensidade da imagem alterada com sucesso! arquivo: filtered-image.png")
