# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:26:46 2019

@author: Jiatong Sun
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.io import loadmat
from PIL import Image

from helpers import rgb2gray
from helpers import drawPoints

from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography
from mymosaic import mymosaic

if __name__ == "__main__": 
    imgA_name = 'franklin_left_small.jpg'
    imgB_name = 'franklin_middle_small.jpg'
    imgC_name = 'franklin_right_small.jpg'
#    imgA_name = 'franklin_left_small.jpg'
#    imgB_name = 'franklin_middle_small.jpg'
#    imgC_name = 'franklin_right_small.jpg'
    imgA = mpimg.imread(imgA_name)
    imgB = mpimg.imread(imgB_name)
    imgC = mpimg.imread(imgC_name)
#    plt.imshow(imgA)
#    plt.imshow(imgB)
#    plt.imshow(imgC)
    img_input = np.zeros((1,3),dtype = np.object)
    img_input[0,0] = imgA
    img_input[0,1] = imgB
    img_input[0,2] = imgC
    
    img_mosaic = mymosaic(img_input)
    Image.fromarray(img_mosaic).save('lololo.jpg')