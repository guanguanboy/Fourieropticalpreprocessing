import skimage
from skimage import io
from skimage import util

import matplotlib.pyplot as plt

origin = skimage.io.imread('./lena.jpg')
print(origin.shape)
print(type(origin))

plt.imshow(origin)
plt.show()

noisy = skimage.util.random_noise(origin, mode='gaussian', var=0.01)
plt.imshow(noisy)
plt.show()