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

from helpers import rgb2gray
from helpers import drawPoints

from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography

if __name__ == "__main__": 
    imgA_name = 'left_small.jpg'
    imgB_name = 'middle_small.jpg'
#    imgA_name = 'left.jpg'
#    imgB_name = 'middle.jpg'
    imgA = mpimg.imread(imgA_name)
    imgB = mpimg.imread(imgB_name)
#    plt.imshow(imgA)
#    plt.imshow(imgB)
    imgA_gray = rgb2gray(imgA)
    imgB_gray = rgb2gray(imgB)
#    plt.imshow(imgA_gray, cmap="gray")
#    plt.imshow(imgB_gray, cmap="gray")
     
    cimgA = corner_detector(imgA_gray)
    cimgB = corner_detector(imgB_gray)
    
#    plt.imshow(cimgA, cmap="gray")
    
    max_pts = 500
     
    xA,yA,rmaxA = anms(cimgA, max_pts)
    xB,yB,rmaxB = anms(cimgB, max_pts)
    
    descsA = feat_desc(imgA_gray, xA, yA)
    descsB = feat_desc(imgB_gray, xB, yB)
    
    match = feat_match(descsA, descsB)
    match_num = match[match>0].size
    
    xA,yA = xA[match>0].reshape(-1,1),yA[match>0].reshape(-1,1)
    xB,yB = xB[match[match>0]].reshape(-1,1),yB[match[match>0]].reshape(-1,1)
    
    thresh = 10
    H,inlier_ind = ransac_est_homography(xA, yA, xB, yB, thresh)
    
    IA = cv.imread(imgA_name)
    drawPoints(IA,xA,yA)
    cv.imshow('IA',IA)
    IB = cv.imread(imgB_name)
    drawPoints(IB,xB,yB)
    cv.imshow('IB',IB)
    cv.waitKey(0)
    