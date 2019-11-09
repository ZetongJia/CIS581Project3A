'''
  File name: corner_detector.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''

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

# I = np.array(Image.open('left.jpg').convert('RGB'))
# cimg = corner_detector(I)
# print(cimg)
# plt.imshow(I)
# plt.show()
