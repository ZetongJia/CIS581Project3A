'''
  File name: helpers.py
  Author: Jiatong Sun
  Date created: 11/08/2019
'''

import cv2 as cv
import numpy as np
from scipy import signal

#  Generate one dimension Gaussian distribution
#  - input mu: the mean of pdf
#  - input sigma: the standard derivation of pdf
#  - input length: the size of pdf
#  - output: a row vector represents one dimension Gaussian distribution

def GaussianPDF_1D(mu, sigma, length):
  # create an array
  half_len = length / 2

  if np.remainder(length, 2) == 0:
    ax = np.arange(-half_len, half_len, 1)
  else:
    ax = np.arange(-half_len, half_len + 1, 1)

  ax = ax.reshape([-1, ax.size])
  denominator = sigma * np.sqrt(2 * np.pi)
  nominator = np.exp( -np.square(ax - mu) / (2 * sigma * sigma) )

  return nominator / denominator


#  Generate two dimensional Gaussian distribution
#  - input mu: the mean of pdf
#  - input sigma: the standard derivation of pdf
#  - input row: length in row axis
#  - input column: length in column axis
#  - output: a 2D matrix represents two dimensional Gaussian distribution

def GaussianPDF_2D(mu, sigma, row, col):
  # create row vector as 1D Gaussian pdf
  g_row = GaussianPDF_1D(mu, sigma, row)
  # create column vector as 1D Gaussian pdf
  g_col = GaussianPDF_1D(mu, sigma, col).transpose()

  return signal.convolve2d(g_row, g_col, 'full')


#  Convert RGB image to gray one manually
#  - Input I_rgb: 3-dimensional rgb image
#  - Output I_gray: 2-dimensional grayscale image

def rgb2gray(I_rgb):
  r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
  I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return I_gray


# Interpolation:xq and yq are the coordinates of the interpolated location
#   - input v: the value lies on grid point which is corresponding to the meshgrid coordinates 
#   - input xq: the query points x coordinates
#   - input yq: the query points y coordinates
#   - output interpv: the interpolated value at querying coordinates xq, yq, it has the same size as xq and yq.

def interp2(v, xq, yq):

	if len(xq.shape) == 2 or len(yq.shape) == 2:
		dim_input = 2
		q_h = xq.shape[0]
		q_w = xq.shape[1]
		xq = xq.flatten()
		yq = yq.flatten()

	h = v.shape[0]
	w = v.shape[1]
	if xq.shape != yq.shape:
		raise 'query coordinates Xq Yq should have same shape'

	x_floor = np.floor(xq).astype(np.int32)
	y_floor = np.floor(yq).astype(np.int32)
	x_ceil = np.ceil(xq).astype(np.int32)
	y_ceil = np.ceil(yq).astype(np.int32)

	x_floor[x_floor < 0] = 0
	y_floor[y_floor < 0] = 0
	x_ceil[x_ceil < 0] = 0
	y_ceil[y_ceil < 0] = 0

	x_floor[x_floor >= w-1] = w-1
	y_floor[y_floor >= h-1] = h-1
	x_ceil[x_ceil >= w-1] = w-1
	y_ceil[y_ceil >= h-1] = h-1

	v1 = v[y_floor, x_floor]
	v2 = v[y_floor, x_ceil]
	v3 = v[y_ceil, x_floor]
	v4 = v[y_ceil, x_ceil]

	lh = yq - y_floor
	lw = xq - x_floor
	hh = 1 - lh
	hw = 1 - lw

	w1 = hh * hw
	w2 = hh * lw
	w3 = lh * hw
	w4 = lh * lw

	interp_val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4

	if dim_input == 2:
		return interp_val.reshape(q_h, q_w)
	return interp_val

def interp2_general(v, xq, yq):
    if len(xq.shape) == 2 or len(yq.shape) == 2:
        dim_input = 2
        q_h = xq.shape[0]
        q_w = xq.shape[1]
        xq = xq.flatten()
        yq = yq.flatten()
    
    h = v.shape[0]
    w = v.shape[1]
    if xq.shape != yq.shape:
        raise 'query coordinates Xq Yq should have same shape'
    
    x_floor = np.floor(xq).astype(np.int32)
    y_floor = np.floor(yq).astype(np.int32)
    x_ceil = np.ceil(xq).astype(np.int32)
    y_ceil = np.ceil(yq).astype(np.int32)

    left = min([min(x_floor),0])
    right = max([max(x_ceil),w])
    top = min([min(y_floor),0])
    bottom = max([max(y_ceil),h])
    
    newH = bottom - top + 1
    newW = right - left + 1
    X,Y = np.meshgrid(np.arange(left,right),np.arange(top,bottom))
    V = np.zeros((newH,newW),dtype = np.int32)
    yy,xx = np.where((X>=0)*(X<w)*(Y>=0)*(Y<h))
    V[min(yy):max(yy)+1,min(xx):max(xx)+1] = v
    
    V1 = V[y_floor-top, x_floor-left]
    V2 = V[y_floor-top, x_ceil-left]
    V3 = V[y_ceil-top, x_floor-left]
    V4 = V[y_ceil-top, x_ceil-left]

    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw
    
    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw
    
    interp_val = w1 * V1 + w2 * V2 + w3 * V3 + w4 * V4
    if dim_input == 2:
        return interp_val.reshape(q_h, q_w)
    return interp_val

def drawPoints(img,x,y,color):
    x,y = x.reshape(-1),y.reshape(-1)
    for i in range(x.size):
        point = (x[i],y[i])
        cv.circle(img,point,2,color,2)
        
def drawLines(img,x1,y1,x2,y2,color):
    x1,y1 = x1.reshape(-1),y1.reshape(-1)
    x2,y2 = x2.reshape(-1),y2.reshape(-1)
    for i in range(x1.size):
        point1 = (x1[i],y1[i])
        point2 = (x2[i],y2[i])
        cv.line(img,point1,point2,color,2)
        
def localMax(img,h,w):
    row,col=img.shape[0],img.shape[1]
    maxImage = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            maxImage[i,j] = img[(np.int32(max(i-(h-1)/2,0))):(np.int32(min(i+(h+1)/2,row))),
                    (np.int32(max(j-(w-1)/2,0))):(np.int32(min(j+(w+1)/2,col)))].max()
    return maxImage

def dist2(p,q):
    col_p = p.shape[1]
    col_q = q.shape[1]
    p2 = np.dot((p**2).sum(axis=0).reshape(-1,1),np.ones((1,col_q)))
    q2 = np.dot(np.ones((col_p,1)),(q**2).sum(axis=0).reshape(1,-1))
    pq = np.dot(p.transpose(),q)
    dist = p2 + q2 - 2 * pq
    return dist

def sort2(A):
    row,col = A.shape[0],A.shape[1]
    val = np.zeros((row,col))
    ind = np.zeros((row,col))
    for i in range(row):
        val[i] = sorted(A[i])
        ind[i] = np.argsort(A[i])
    return val,ind  

def flipChannel(img):
    new_image = np.zeros(img.shape)
    new_image[:,:,0] = img[:,:,2]
    new_image[:,:,1] = img[:,:,1]
    new_image[:,:,2] = img[:,:,0]
    new_image = new_image.astype(np.uint8)
    return new_image

def getNewSize(H1, H2, imgA, imgB, imgC):
    h_left,w_left,_ = imgA.shape
    h_middle,w_middle,_ = imgB.shape
    h_right,w_right,_ = imgC.shape
    
# Warp Left Picture    
    [X_left, Y_left] = np.meshgrid(np.arange(w_left),np.arange(h_left))
    coor_left = np.ones((3,h_left*w_left))
    coor_left[0,:] = np.reshape(X_left, (1,h_left*w_left))
    coor_left[1,:] = np.reshape(Y_left, (1,h_left*w_left))
    coor_left_warp = np.dot(H1,coor_left)
    coor_left_warp[0,:] = np.divide(coor_left_warp[0,:],coor_left_warp[2,:])
    coor_left_warp[1,:] = np.divide(coor_left_warp[1,:],coor_left_warp[2,:])
    
# Warp Right Picture 
    [X_right, Y_right] = np.meshgrid(np.arange(w_right),np.arange(h_right))
    coor_right = np.ones((3,h_right*w_right))
    coor_right[0,:] = np.reshape(X_right, (1,h_right*w_right))
    coor_right[1,:] = np.reshape(Y_right, (1,h_right*w_right))
    coor_right_warp = np.dot(H2,coor_right)
    coor_right_warp[0,:] = np.divide(coor_right_warp[0,:],coor_right_warp[2,:])
    coor_right_warp[1,:] = np.divide(coor_right_warp[1,:],coor_right_warp[2,:])
    
# Decide New Image Size
    new_left_edge = int(np.fix(np.min([np.min(coor_left_warp[0,:]),0\
                                        ,np.min(coor_right_warp[0,:])])))
    new_right_edge = int(np.fix(np.max([np.max(coor_left_warp[0,:]),w_middle\
                                        ,np.max(coor_right_warp[0,:])])))
    new_top_edge = int(np.fix(np.min([np.min(coor_left_warp[1,:]),0\
                                        ,np.min(coor_right_warp[1,:])])))
    new_bottom_edge = int(np.fix(np.max([np.max(coor_left_warp[1,:]),h_middle\
                                        ,np.max(coor_right_warp[1,:])])))

    newH = new_bottom_edge - new_top_edge + 1
    newW = new_right_edge - new_left_edge + 1
    x_offset = new_left_edge
    y_offset = new_top_edge
    
# Middle Image      
    new_middle = np.zeros((newH,newW,3))
    new_middle[-y_offset:-y_offset + h_middle\
                   , -x_offset: -x_offset + w_middle,:] = imgB
    new_middle = new_middle.astype(np.uint8)
    
# Left Image & Right Image
    [X, Y] = np.meshgrid(np.arange(w_left),np.arange(h_left))
    [XX,YY] = np.meshgrid(np.arange(x_offset,x_offset+newW)\
                , np.arange(y_offset,y_offset+newH))
    offset_img = np.ones((3,newH*newW))
    offset_img[0,:] = np.reshape(XX,(1,newH*newW))
    offset_img[1,:] = np.reshape(YY,(1,newH*newW))
    
    left_inv = np.linalg.solve(H1,offset_img)
    left_XX = np.reshape(np.divide(left_inv[0,:],left_inv[2,:]), (newH, newW))
    left_YY = np.reshape(np.divide(left_inv[1,:],left_inv[2,:]), (newH, newW))
    new_left = np.zeros((newH, newW, 3))
    new_left[:,:,0] = interp2_general(imgA[:,:,0].astype(np.double), left_XX, left_YY)
    new_left[:,:,1] = interp2_general(imgA[:,:,1].astype(np.double), left_XX, left_YY)
    new_left[:,:,2] = interp2_general(imgA[:,:,2].astype(np.double), left_XX, left_YY)
    new_left = new_left.astype(np.uint8)
    
    right_inv = np.linalg.solve(H2,offset_img)
    right_XX = np.reshape(np.divide(right_inv[0,:],right_inv[2,:]), (newH, newW))
    right_YY = np.reshape(np.divide(right_inv[1,:],right_inv[2,:]), (newH, newW))
    new_right = np.zeros((newH, newW, 3))
    new_right[:,:,0] = interp2_general(imgC[:,:,0].astype(np.double), right_XX, right_YY)
    new_right[:,:,1] = interp2_general(imgC[:,:,1].astype(np.double), right_XX, right_YY)
    new_right[:,:,2] = interp2_general(imgC[:,:,2].astype(np.double), right_XX, right_YY)
    new_right = new_right.astype(np.uint8)

    return new_left, new_middle, new_right

def alphaBlend(imgA,imgB):
    maskA = np.logical_or(imgA[:,:,0]>0,imgA[:,:,1]>0,imgA[:,:,2]>0)
    maskB = np.logical_or(imgB[:,:,0]>0,imgB[:,:,1]>0,imgB[:,:,2]>0)
    maskAB = np.logical_and(maskA>0,maskB>0)
    
    _,col = np.where(maskAB>0)
    left = col.min()
    right = col.max()
    maskA = np.ones(maskAB.shape,dtype = np.float32)
    maskA[:,left:right+1] = np.tile(np.linspace(1,0,right-left+1),(maskAB.shape[0],1))
    maskB = np.ones(maskAB.shape,dtype = np.float32)
    maskB[:,left:right+1] = np.tile(np.linspace(0,1,right-left+1),(maskAB.shape[0],1))
    blend_img = np.zeros(imgA.shape,dtype = np.float32)
    blend_img[:,:,0] = imgA[:,:,0] * maskA + imgB[:,:,0] * maskB
    blend_img[:,:,1] = imgA[:,:,1] * maskA + imgB[:,:,1] * maskB
    blend_img[:,:,2] = imgA[:,:,2] * maskA + imgB[:,:,2] * maskB
    blend_img = blend_img.astype(np.uint8)
    
    return blend_img  