import skimage.feature
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio


def corner_detector(img):

  img_gray=(0.3*img[:,:,0]+0.59*img[:,:,1]+0.11*img[:,:,2]).astype(np.uint8)
  cimg=skimage.feature.corner_harris(img_gray)
  return cimg