'''
  File name: helpers.py
  Author: Jiatong Sun
  Date created: 11/08/2019
'''

import cv2 as cv
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


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

	interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

	if dim_input == 2:
		return interp_val.reshape(q_h, q_w)
	return interp_val

def drawPoints(img,x,y):
    x,y = x.reshape(-1),y.reshape(-1)
    for i in range(x.size):
        point = (x[i],y[i])
        cv.circle(img,point,2,(i*127/x.size,i*127/x.size,255-i*127/x.size),2)

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