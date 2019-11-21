import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature

from getFeatureForOne import getFeatureForOne

def getFeatures(img,bbox):
    num_box = bbox.shape[0]
    res_x_temp = np.zeros((1,num_box),dtype = np.object)
    res_y_temp = np.zeros((1,num_box),dtype = np.object)

    N = 0

    for i in range((num_box)):
      row_start = int(bbox[i,0,0])
      row_end = int(bbox[i,2,0])
      col_start = int(bbox[i,0,1])
      col_end = int(bbox[i,1,1])
      img_temp = img[row_start:row_end,col_start:col_end]
      x, y, rmax = getFeatureForOne(img_temp)
      N = max(N, x.shape[0])
      res_x_temp[0,i] = x
      res_y_temp[0,i] = y

    res_x = np.zeros((N,num_box))
    res_y = np.zeros((N,num_box))

    for i in range((num_box)):
      size_temp = res_x_temp[0,i].shape[0]
      res_x[0:size_temp,i] = res_x_temp[0,i]
      res_y[0:size_temp,i] = res_y_temp[0,i]

    return res_x, res_y