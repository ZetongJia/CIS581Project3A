import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature


def getFeatureForOne(img):
    max_pts = 500
    cimg = feature.corner_harris(img)

    cimg[0:20],cimg[-1:-21:-1],cimg[:,0:20],cimg[:,-1:-21:-1] = 0,0,0,0
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