'''
  File name: corner_detector.py
  Author:
  Date created:
'''

'''
  File clarification:
    Detects corner features in an image. You can probably find free “harris” corner detector on-line, 
    and you are allowed to use them.
    - Input img: H × W matrix representing the gray scale input image.
    - Output cimg: H × W matrix representing the corner metric matrix.
'''

import numpy as np
import cv2 as cv
import scipy.ndimage as ndimage
import scipy.signal as signal
import matplotlib.pyplot as plt
from skimage import filters

from helpers import GaussianPDF_2D

def corner_detector(img):
  # Your Code Here 
# Filter Method 1
    img = np.float32(img) 
    cimg = cv.cornerHarris(img,3,5,0.06)
    return cimg
# =============================================================================
# # Filter Method 2
#     img = ndimage.gaussian_filter(img, sigma = 7);
#     sobel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
#    
#     Ix = ndimage.correlate(img, sobel, mode='nearest')
#     Iy = ndimage.correlate(img, sobel.transpose(), mode='nearest')
#    
#     Ix2 = ndimage.gaussian_filter(Ix**2, sigma = 2);
#     Iy2 = ndimage.gaussian_filter(Iy**2, sigma = 2);
#     Ixy = ndimage.gaussian_filter(Ix*Iy, sigma = 2);
#     k = 0.04
#     cimg = (Ix2*Iy2 - Ixy**2)-k*(Ix2 + Iy2)**2;
# =============================================================================
  
# =============================================================================
# # Filter Method 3
#     G1 = GaussianPDF_2D(0,7,4,4)
#     [dx,dy] =  np.gradient(G1, axis = (1,0))
#     Ix = signal.convolve2d(img,dx,'same')
#     Iy = signal.convolve2d(img,dy,'same')
#     G2 = GaussianPDF_2D(0,2,4,4)
#     Ix2 = signal.convolve2d(Ix**2,G2,'same')
#     Iy2 = signal.convolve2d(Iy**2,G2,'same')
#     Ixy = signal.convolve2d(Ix*Iy,G2,'same')
# =============================================================================
   
# =============================================================================
# # R Method 2
#     cimg = (Ix2*Iy2 - Ixy**2)/(Ix2 + Iy2 + 1e-10);
#     return cimg
# =============================================================================
    