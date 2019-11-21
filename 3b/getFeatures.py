import numpy as np
from skimage import feature
from helpers import rgb2gray
from helpers import flipChannel

def getFeatures(img,bbox):
    num_box = bbox.shape[0];
    res_x_temp = np.zeros((1,num_box),dtype = np.object);
    res_y_temp = np.zeros((1,num_box),dtype = np.object);

    N = 0;

    for i in range((num_box)):
        row_start = int(bbox[i,1,1]);
        row_end = int(bbox[i,2,1]);
        col_start = int(bbox[i,0,0]);
        col_end = int(bbox[i,1,0]);
        img_temp = img[row_start:row_end,col_start:col_end];
        x, y, rmax = getBoxFeature(img_temp);
        x += col_start;
        y += row_start;
        N = max(N, x.shape[0]);
        res_x_temp[0,i] = x.reshape(-1);
        res_y_temp[0,i] = y.reshape(-1);

    res_x = np.zeros((N,num_box));
    res_y = np.zeros((N,num_box));

    for i in range((num_box)):
      size_temp = res_x_temp[0,i].shape[0];
      res_x[0:size_temp,i] = res_x_temp[0,i];
      res_y[0:size_temp,i] = res_y_temp[0,i];

    return res_x, res_y;

def getBoxFeature(img):
    max_pts = 5;
    
    if (img.ndim == 3):
        img = rgb2gray(flipChannel(img));
    cimg = feature.corner_harris(img);

    cimg[0:5],cimg[-1:-6:-1],cimg[:,0:5],cimg[:,-1:-6:-1] = 0,0,0,0;
    thresh = 0.012*cimg.max();
    y_ind,x_ind = np.where(cimg>thresh);
    val = cimg[cimg>thresh];
    
    if x_ind.size < max_pts:
        x,y = x_ind,y_ind;
        rmax = 999999;
        return x,y,rmax
    
    radius = np.zeros((x_ind.size,1));
    c = 0.9;
    max_val = c * val.max();
    for i in range(x_ind.size):
        if val[i] > max_val:
            radius[i] = 999999;
            continue;
        else:
            dist = np.sqrt((x_ind - x_ind[i])**2 + (y_ind - y_ind[i])**2);
            dist = dist[val * c > val[i]];
            radius[i] = dist.min();
    radius = radius.reshape(-1);
    index = np.argsort(-radius);
    index = index[0:max_pts];
    
    x,y = x_ind[index].reshape(-1,1),y_ind[index].reshape(-1,1);
    rmax = radius[index[-1]];
    return x, y, rmax;