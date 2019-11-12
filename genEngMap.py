import numpy as np

# convert rgb to grayscale
def rgb2gray(rgb):
  return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def genEngMap(I):
  dim = I.ndim
  if dim == 3:
    Ig = rgb2gray(I)
  else:
    Ig = I

  Ig = Ig.astype(np.float64())

  [gradx, grady] = np.gradient(Ig);
  e = np.abs(gradx) + np.abs(grady)
  return e