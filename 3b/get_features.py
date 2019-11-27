import numpy as np
import cv2 as cv
from skimage import feature
from helpers import rgb2gray
from helpers import flipChannel

# Method 1:
# get feature points by processing bbox

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
        x, y = getBoxFeature(img_temp);
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

# get feature points by good feature to track
def getBoxFeature(img):
    max_pts = 20;
    row, col = img.shape[0], img.shape[1];
    
    if (img.ndim == 3):
        img = rgb2gray(flipChannel(img));
    corners = cv.goodFeaturesToTrack(img.astype(np.float32),max_pts,0.01,10);
    corners = np.int0(corners);
    x, y = corners[:,:,0], corners[:,:,1];
    inlier_ind = (x>5) * (x<col-5) * (y>5) * (y<row-5);
    x, y = x[inlier_ind], y[inlier_ind];
    
    return x, y;




# =============================================================================
# # Method 2
# # get feature points by harris
# 
# def getBoxFeature(img):
#     max_pts = 25;
#     
#     if (img.ndim == 3):
#         img = rgb2gray(flipChannel(img));
#     cimg = feature.corner_harris(img);
# 
#     cimg[0:5],cimg[-1:-6:-1],cimg[:,0:5],cimg[:,-1:-6:-1] = 0,0,0,0;
#     thresh = 0.012*cimg.max();
#     y_ind,x_ind = np.where(cimg>thresh);
#     val = cimg[cimg>thresh];
#     
#     if x_ind.size < max_pts:
#         x,y = x_ind,y_ind;
#         rmax = 999999;
#         return x,y,rmax
#     
#     radius = np.zeros((x_ind.size,1));
#     c = 0.9;
#     max_val = c * val.max();
#     for i in range(x_ind.size):
#         if val[i] > max_val:
#             radius[i] = 999999;
#             continue;
#         else:
#             dist = np.sqrt((x_ind - x_ind[i])**2 + (y_ind - y_ind[i])**2);
#             dist = dist[val * c > val[i]];
#             radius[i] = dist.min();
#     radius = radius.reshape(-1);
#     index = np.argsort(-radius);
#     index = index[0:max_pts];
#     
#     x,y = x_ind[index].reshape(-1,1),y_ind[index].reshape(-1,1);
#     return x, y;
# =============================================================================





# =============================================================================
# # Method 3
# # get feature points by processing the whole image and harris corner
# 
# def getFeatures(img,bbox):
#     max_pts = 30;
#     
#     if (img.ndim == 3):
#         img = rgb2gray(flipChannel(img));
#     cimg = feature.corner_harris(img);
# 
#     cimg[0:5],cimg[-1:-6:-1],cimg[:,0:5],cimg[:,-1:-6:-1] = 0,0,0,0;
#     thresh = 0.012*cimg.max();
#     y_ind,x_ind = np.where(cimg>thresh);
#     val = cimg[cimg>thresh];
#     
#     if x_ind.size < max_pts:
#         x,y = x_ind,y_ind;
#         rmax = 999999;
#         return x,y,rmax
#     
#     radius = np.zeros((x_ind.size,1));
#     c = 0.95;
#     max_val = c * val.max();
#     for i in range(x_ind.size):
#         if val[i] > max_val:
#             radius[i] = 999999;
#             continue;
#         else:
#             dist = np.sqrt((x_ind - x_ind[i])**2 + (y_ind - y_ind[i])**2);
#             dist = dist[val * c > val[i]];
#             radius[i] = dist.min();
#     radius = radius.reshape(-1);
#     index = np.argsort(-radius);
#     index = index[0:max_pts];
#     
#     x,y = x_ind[index].reshape(-1,1),y_ind[index].reshape(-1,1);
#     rmax = radius[index[-1]];
#     
#     F = bbox.shape[0];
#     in_bbox = np.zeros((x.shape[0],F),dtype = np.int32);
#     for k in range(x.shape[0]):
#         for f in range(F):
#             x_temp, y_temp = x[k][0], y[k][0];
#             in_bbox[k][f] = x_temp > bbox[f][0][0] and x_temp < bbox[f][3][0]\
#                         and y_temp > bbox[f][0][1] and y_temp < bbox[f][3][1];
#     
#     N = np.max(np.sum(in_bbox,axis = 0));
#     res_x = np.zeros((N,F),dtype = np.int32);
#     res_y = np.zeros((N,F),dtype = np.int32);
#     
#     for f in range(F):
#         n = 0;
#         for k in range(in_bbox.shape[0]):
#             if in_bbox[k][f] == True:
#                 res_x[n][f], res_y[n][f] = x[k], y[k];
#                 n = n + 1;
#                 if n == N:
#                     break;
#     return res_x, res_y;
# =============================================================================
