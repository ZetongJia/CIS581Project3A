import cv2 as cv
import numpy as np

# Convert RGB image to gray one manually
def rgb2gray(I_rgb):
  r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
  I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return I_gray


# Interpolation:xq and yq are the coordinates of the interpolated location
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

# draw a vector of points in certain color
def drawPoints(img,x,y,color):
    x, y = x.reshape(-1).astype(np.int32),y.reshape(-1).astype(np.int32)
    for i in range(x.size):
        point = (x[i],y[i])
        cv.circle(img,point,2,color,2)

# flip the channel of img (conversion between BGR and RGB)
def flipChannel(img):
    new_image = np.zeros(img.shape)
    new_image[:,:,0] = img[:,:,2]
    new_image[:,:,1] = img[:,:,1]
    new_image[:,:,2] = img[:,:,0]
    new_image = new_image.astype(np.uint8)
    return new_image

# generate a 5*5 square patch area
def generatePatch(x,y):
    x_temp = np.array([x-2,x-1,x,x+1,x+2]);
    y_temp = np.array([y-2,y-1,y,y+1,y+2]);
    [X,Y] = np.meshgrid(x_temp,y_temp);
    return X,Y

# get the corners of a rectange
def getBoxPoints(x,y,w,h):
    box_pts = np.zeros((4,2),dtype=np.int32);
    box_pts[0][0] = x;
    box_pts[1][0] = x + w;
    box_pts[2][0] = x;
    box_pts[3][0] = x + w;
    box_pts[0][1] = y;
    box_pts[1][1] = y;
    box_pts[2][1] = y + h;
    box_pts[3][1] = y + h;
    return box_pts

def resetBox(bbox,last_bbox):
    for f in range(bbox.shape[0]):
        cur_row = bbox[f,3,1] - bbox[f,0,1];
        cur_col = bbox[f,3,0] - bbox[f,0,0];
        last_row = last_bbox[f,3,1] - last_bbox[f,0,1];
        last_col = last_bbox[f,3,0] - last_bbox[f,0,0];
        if cur_row < last_row - 10:
            bbox[f,0,1] = bbox[f,0,1] - 10;
            bbox[f,1,1] = bbox[f,1,1] - 10;
            bbox[f,2,1] = bbox[f,2,1] + 10;
            bbox[f,3,1] = bbox[f,3,1] + 10;
        if cur_col < last_col - 10:
            bbox[f,0,0] = bbox[f,0,0] - 10;
            bbox[f,1,0] = bbox[f,1,0] - 10;
            bbox[f,2,0] = bbox[f,2,0] + 10;
            bbox[f,3,0] = bbox[f,3,0] + 10;
    return bbox;
