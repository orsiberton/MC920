import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from scipy import ndimage

# open image file and stores it in a numpy array
img = misc.imread('../images/baboon.png')

print(img)

# print image dimensions and type
print(img.shape, img.dtype)
# show image
plt.imshow(img, cmap='gray')
plt.show()
# save image in PNG format
misc.imsave('face2.png', img)
# calculate some statistical information
print(img.min(), img.mean(), img.max())
# apply rotation transformation
f = np.flipud(img)
plt.imshow(f, cmap='gray')
plt.show()
# smooth image with Gaussian filter
g = ndimage.gaussian_filter(img, sigma=7)
h = ndimage.gaussian_filter(img, sigma=11)
plt.imshow(g, cmap='gray')
plt.show()
plt.imshow(h, cmap='gray')
plt.show()
