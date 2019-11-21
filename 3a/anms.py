'''
  File name: anms.py
  Author:
  Date created:
'''

'''
  File clarification:
    Implement Adaptive Non-Maximal Suppression. The goal is to create an uniformly distributed 
    points given the number of feature desired:
    - Input cimg: H × W matrix representing the corner metric matrix.
    - Input max_pts: the number of corners desired.
    - Outpuy x: N × 1 vector representing the column coordinates of corners.
    - Output y: N × 1 vector representing the row coordinates of corners.
    - Output rmax: suppression radius used to get max pts corners.
'''

import numpy as np

from helpers import localMax

def anms(cimg, max_pts):
  # Your Code Here
  cimg[0:20],cimg[-1:-21:-1],cimg[:,0:20],cimg[:,-1:-21:-1] = 0,0,0,0;
#  y_ind,x_ind = np.where((cimg==localMax(cimg,5,5))*(cimg>0))
#  val = cimg[(cimg==localMax(cimg,5,5))*(cimg>0)]
  thresh = 0.012*cimg.max()
  y_ind,x_ind = np.where(cimg>thresh)
  val = cimg[cimg>thresh]
  
  if x_ind.size < max_pts:
    x,y = x_ind,y_ind
    rmax = 999999
    return x,y,rmax
  
  radius = np.zeros((x_ind.size,1))
  c = 0.9
  max_val = c * val.max()
  for i in range(x_ind.size):
    if val[i] > max_val:
      radius[i] = 999999
      continue
    else:
      dist = np.sqrt((x_ind - x_ind[i])**2 + (y_ind - y_ind[i])**2)
      dist = dist[val * c > val[i]]
      radius[i] = dist.min()
  radius = radius.reshape(-1)
  index = np.argsort(-radius)
  index = index[0:max_pts]
  
  x,y = x_ind[index].reshape(-1,1),y_ind[index].reshape(-1,1)
  rmax = radius[index[-1]]
  
  return x, y, rmax