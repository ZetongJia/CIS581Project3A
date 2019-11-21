from helpers import rgb2gray
import numpy as np
from PIL import Image
from getFeatures import getFeatures
import matplotlib.pyplot as plt
import matplotlib.patches as patches

im = Image.open("1.jpg")
# np.array(Image.open('ski.jpg').convert('RGB'))
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

btlt = (bbox[0,0,0],bbox[0,0,1])
h = bbox[0,2,1] - bbox[0,1,1]
w = bbox[0,1,0] - bbox[0,0,0]

res_x, res_y = getFeatures(img_gray, bbox)

# show img with features
fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(img_gray)
bb_rect = patches.Rectangle(btlt, w, h, linewidth=2, fill=False, edgecolor='red')
ax.add_patch(bb_rect)
plt.scatter(res_x, res_y)
plt.show()
# print(res_x.shape)
# print(res_x)
# print(res_y)
