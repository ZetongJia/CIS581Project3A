import numpy as np
import matplotlib.pyplot as plt
from cumMinEngVer import *
from genEngMap import *
from PIL import Image

def rgb2gray(I_rgb):
  r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
  I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return I_gray

def getVerSeam(I, Mx, Tbx):
  [x, y] = I.shape
  E = Mx.min(1)[x-1]

  idxs = np.zeros((x), dtype='int')
  idx = np.argmin(Mx[x-1,:])
  idxs[-1] = idx
  for j in range(1,x+1):
    c = x - j
    idxs[c] = idx
    idx += Tbx[c, idx]
  return idxs

def get_overlapping_mask(imgA, imgB):
	maskA = np.where(imgA>0, imgA, 0)
	maskB = np.where(imgB>0, imgB, 0)
	maskAB = np.logical_and(maskA>0,maskB>0)
	return maskAB

def seam_carving_blend_left(imgA, imgB):
	maskAB = get_overlapping_mask(imgA, imgB)
	top = min(np.argwhere(maskAB>0), key=lambda x: x[0])[0]
	bottom = max(np.argwhere(maskAB>0), key=lambda x: x[0])[0]
	left = min(np.argwhere(imgB>0), key=lambda x: x[1])[1]
	right = max(np.argwhere(imgB>0), key=lambda x: x[1])[1]
	print(left, right, top, bottom)

	overlap_img = imgB[top:bottom, left:right]
	e = genEngMap(overlap_img)
	Mx, Tbx = cumMinEngVer(e)
	seam_idxs = getVerSeam(overlap_img, Mx, Tbx)
	blended_img = np.copy(imgA)

	for i in range(len(seam_idxs)):
		idx = seam_idxs[i]
		offset_y = left + idx
		offset_x = top + i
		blended_img[offset_x,offset_y:] = imgB[offset_x,offset_y:]
	
	return blended_img

def seam_carving_blend_right(imgB, imgC):
	maskBC = get_overlapping_mask(imgB, imgC)
	top = min(np.argwhere(maskBC>0), key=lambda x: x[0])[0]
	bottom = max(np.argwhere(maskBC>0), key=lambda x: x[0])[0]
	left = min(np.argwhere(imgC>0), key=lambda x: x[1])[1]
	right = max(np.argwhere(imgB>0), key=lambda x: x[1])[1]
	print(left, right, top, bottom)

	overlap_img = imgB[top:bottom, left:right]
	e = genEngMap(overlap_img)
	Mx, Tbx = cumMinEngVer(e)
	seam_idxs = getVerSeam(overlap_img, Mx, Tbx)
	blended_img = np.copy(imgB) + imgC - np.multiply(maskBC,imgC)

	for i in range(len(seam_idxs)):
		idx = seam_idxs[i]
		offset_y = left + idx
		offset_x = top + i
		blended_img[offset_x,offset_y:] = imgC[offset_x,offset_y:]

	return blended_img

# # test
# # imgA_name = 'blend_left.jpg'
# # imgB_name = 'blend_middle.jpg'
# # imgC_name = 'blend_right.jpg'
# imgA = np.array(Image.open('blend_left.jpg').convert('RGB'))
# imgB = np.array(Image.open('blend_middle.jpg').convert('RGB'))
# imgC = np.array(Image.open('blend_right.jpg').convert('RGB'))
# imgA_gray = rgb2gray(imgA)
# imgB_gray = rgb2gray(imgB)
# imgC_gray = rgb2gray(imgC)
# mask = get_overlapping_mask(imgA, imgB)
# imgAB_blend = seam_carving_blend_left(imgA_gray, imgB_gray)
# imgABC_blend = seam_carving_blend_right(imgAB_blend, imgC_gray)
# # plt.imshow(imgAB_blend,cmap='gray')
# # plt.show()
# plt.imshow(imgABC_blend,cmap='gray')
# plt.show()
