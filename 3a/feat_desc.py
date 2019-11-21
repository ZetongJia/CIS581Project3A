'''
  File name: feat_desc.py
  Author:
  Date created:
'''

'''
  File clarification:
    Extracting Feature Descriptor for each feature point. You should use the subsampled image around each point feature, 
    just extract axis-aligned 8x8 patches. Note that it’s extremely important to sample these patches from the larger 40x40 
    window to have a nice big blurred descriptor. 
    - Input img: H × W matrix representing the gray scale input image.
    - Input x: N × 1 vector representing the column coordinates of corners.
    - Input y: N × 1 vector representing the row coordinates of corners.
    - Outpuy descs: 64 × N matrix, with column i being the 64 dimensional descriptor (8 × 8 grid linearized) computed at location (xi , yi) in img.
'''
import numpy as np
from skimage import filters
from sklearn import preprocessing
from helpers import interp2

def feat_desc(img, x, y):
  N = x.size
  descs = np.zeros((64, N))
  edges = filters.sobel(img)
  
  for i in range(N):
    x_temp = np.arange(x[i]-19, x[i]+21, 1)
    x_i = np.tile(x_temp,(40,1))
    y_temp = np.arange(y[i]-19, y[i]+21, 1)
    y_temp = y_temp.reshape(-1,1)
    y_i = np.tile(y_temp,(1,40))
    interp = interp2(edges, x_i, y_i)
    final = np.zeros((320,5))
    final[0:40,0:5] = interp[0:40, 0:5]
    final[40:80,0:5] = interp[0:40, 5:10]
    final[80:120,0:5] = interp[0:40, 10:15]
    final[120:160,0:5] = interp[0:40, 15:20]
    final[160:200,0:5] = interp[0:40, 20:25]
    final[200:240,0:5] = interp[0:40, 25:30]
    final[240:280,0:5] = interp[0:40, 30:35]
    final[280:320,0:5] = interp[0:40, 35:40]
    final = final.reshape((64,25))
    final_max = np.amax(final, axis=1)
    descs_i = preprocessing.scale(final_max)
    descs[0:64,i:i+1] = descs_i.reshape((64,1))
  return descs
