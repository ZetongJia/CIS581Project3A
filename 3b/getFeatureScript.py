from helpers import rgb2gray
import numpy as np
from PIL import Image
from getFeatures import getFeatures

im = Image.open("1.jpg")
np_im = np.array(im)

img_gray = rgb2gray(np_im)
bbox = np.zeros((1,4,2))
#left top 
bbox[0,0,0] = 185
bbox[0,0,1] = 290
#right top 
bbox[0,1,0] = 185
bbox[0,1,1] = 400
#left bottom 
bbox[0,2,0] = 270
bbox[0,2,1] = 290
#right bottom 
bbox[0,3,0] = 270
bbox[0,3,1] = 400

res_x, res_y = getFeatures(img_gray, bbox)
print(res_x.shape)
print(res_x)
print(res_y)