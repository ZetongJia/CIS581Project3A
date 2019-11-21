import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
from helpers import rgb2gray
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature

from getFeatureForOne import getFeatureForOne

def getFeatures(img,bbox):
    num_box = bbox.shape[0];
    res_x_temp = np.zeros((1,num_box),dtype = np.object);
    res_y_temp = np.zeros((1,num_box),dtype = np.object);

    N = 0;

    for i in range((num_box)):
      # h = bbox[0,2,1] - bbox[0,1,1]
      # w = bbox[0,1,0] - bbox[0,0,0]
      row_start = int(bbox[i,1,1])
      row_end = int(bbox[i,2,1])
      col_start = int(bbox[i,0,0])
      col_end = int(bbox[i,1,0])
      img_temp = img[row_start:row_end,col_start:col_end]
      # plt.imshow(img_temp)
      # plt.show()
      x, y, rmax = getFeatureForOne(img_temp)
      x += col_start
      y += row_start
      N = max(N, x.shape[0])   
      res_x_temp[0,i] = x.reshape(-1)
      res_y_temp[0,i] = y.reshape(-1)

    res_x = np.zeros((N,num_box));
    res_y = np.zeros((N,num_box));

    for i in range((num_box)):
      size_temp = res_x_temp[0,i].shape[0];
      res_x[0:size_temp,i] = res_x_temp[0,i].reshape(-1,1);
      res_y[0:size_temp,i] = res_y_temp[0,i].reshape(-1,1);

    return res_x, res_y

# test
im = Image.open("1.jpg")
np_im = np.array(im)
img_gray = rgb2gray(np_im)
bbox = np.zeros((1,4,2))
#left top 
bbox[0,0,0] = 290
bbox[0,0,1] = 185
#right top 
bbox[0,1,0] = 400
bbox[0,1,1] = 185
#left bottom 
bbox[0,2,0] = 290
bbox[0,2,1] = 270
#right bottom 
bbox[0,3,0] = 400
bbox[0,3,1] = 270

res_x, res_y = getFeatures(img_gray, bbox)
plt.imshow(im)
plt.scatter(res_x,res_y)
plt.show()
