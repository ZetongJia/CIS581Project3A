import numpy as np
import matplotlib.pyplot as plts
from PIL import Image

def rgb2gray(I_rgb):
  r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
  I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return I_gray

def genEngMap(I):
  dim = I.ndim
  if dim == 3:
    Ig = rgb2gray(I)
  else:
    Ig = I

  Ig = Ig.astype(np.float64())

  [gradx, grady] = np.gradient(Ig);
  e = np.abs(gradx) + np.abs(grady)
  return e

def cumMinEngHor(e):
  My = np.zeros_like(e)
  Tby = np.zeros_like(e, dtype=int)

  My[:,0] = e[:,0]
  for j in range(1,e.shape[1]):
    for i in range(e.shape[0]):
      if i == 0: prev = [float('inf'), My[i,j-1], My[i+1,j-1]]
      if i == e.shape[0] - 1: prev = [My[i-1,j-1], My[i,j-1], float('inf')]
      else: prev = [My[i-1,j-1], My[i,j-1], My[i+1,j-1]]
      My[i,j] = min(prev) + e[i,j]
      Tby[i, j] = np.argmin(prev)-1
  return My, Tby

def cumMinEngVer(e):
  My, Tby = cumMinEngHor(np.transpose(e))
  return np.transpose(My), np.transpose(Tby)

def getVerSeam(I, Mx, Tbx):
  [x, y, _] = I.shape
  E = Mx.min(1)[x-1]

  idxs = np.zeros((x), dtype='int')
  idx = np.argmin(Mx[x-1,:])
  idxs[-1] = idx
  for j in range(1,x+1):
    c = x - j
    idxs[c] = idx
    idx += Tbx[c, idx]
  return idxs

def getOverlappingMask(imgA, imgB):
	maskA = np.logical_or(imgA[:,:,0]>0,imgA[:,:,1]>0,imgA[:,:,2]>0)
	maskB = np.logical_or(imgB[:,:,0]>0,imgB[:,:,1]>0,imgB[:,:,2]>0)
	maskAB = np.logical_and(maskA>0,maskB>0)
	return maskAB

def seamBlendLeft(imgA, imgB):
	imgA_gray = rgb2gray(imgA)
	imgB_gray = rgb2gray(imgB)
	top = min(np.argwhere(imgB_gray>0), key=lambda x: x[0])[0]
	bottom = max(np.argwhere(imgB_gray>0), key=lambda x: x[0])[0]
	left = min(np.argwhere(imgB_gray>0), key=lambda x: x[1])[1]
	right = max(np.argwhere(imgA_gray>0), key=lambda x: x[1])[1]

	overlap_img = imgB[top:bottom, left:right]
	e = genEngMap(overlap_img)
	Mx, Tbx = cumMinEngVer(e)
	seam_idxs = getVerSeam(overlap_img, Mx, Tbx)
	blended_img = np.copy(imgA)

	for i in range(len(seam_idxs)):
		idx = seam_idxs[i]
		offset_y = left + idx
		offset_x = top + i
		blended_img[offset_x,offset_y:,:] = imgB[offset_x,offset_y:,:]
	
	return blended_img

def seamBlendRight(imgB, imgC):
	imgB_gray = rgb2gray(imgB)
	imgC_gray = rgb2gray(imgC)
	maskBC = getOverlappingMask(imgB, imgC)
	top = min(np.argwhere(imgB_gray>0), key=lambda x: x[0])[0]
	bottom = max(np.argwhere(imgB_gray>0), key=lambda x: x[0])[0]
	left = min(np.argwhere(imgC_gray>0), key=lambda x: x[1])[1]
	right = max(np.argwhere(imgB_gray>0), key=lambda x: x[1])[1]

	overlap_img = imgB[top:bottom, left:right]
	e = genEngMap(overlap_img)
	Mx, Tbx = cumMinEngVer(e)
	seam_idxs = getVerSeam(overlap_img, Mx, Tbx)
	blended_img = np.copy(imgB)
	blended_img[:,:,0] = imgB[:,:,0] + imgC[:,:,0] - np.multiply(maskBC,imgC[:,:,0])
	blended_img[:,:,1] = imgB[:,:,1] + imgC[:,:,1] - np.multiply(maskBC,imgC[:,:,1])
	blended_img[:,:,2] = imgB[:,:,2] + imgC[:,:,2] - np.multiply(maskBC,imgC[:,:,2])

	for i in range(len(seam_idxs)):
		idx = seam_idxs[i]
		offset_y = left + idx
		offset_x = top + i
		blended_img[offset_x,offset_y:,:] = imgC[offset_x,offset_y:,:]

	return blended_img

def seamBlend(imgA,imgB,imgC):
    seam_blend = seamBlendRight(seamBlendLeft(imgA,imgB),imgC)
    seam_blend = seam_blend.astype(np.uint8)
    return seam_blend