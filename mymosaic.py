'''
  File name: mymosaic.py
  Author:
  Date created:
'''

'''
  File clarification:
    Produce a mosaic by overlaying the pairwise aligned images to create the final mosaic image. If you want to implement
    imwarp (or similar function) by yourself, you should apply bilinear interpolation when you copy pixel values. 
    As a bonus, you can implement smooth image blending of the final mosaic.
    - Input img_input: M elements numpy array or list, each element is a input image.
    - Outpuy img_mosaic: H × W × 3 matrix representing the final mosaic image.
'''
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
from helpers import interp2
from helpers import getNewSize
from helpers import blend

def mymosaic(img_input):
  # Your Code Here
    imgA = img_input[0,0]
    imgB = img_input[0,1]
    imgC = img_input[0,2]
    
    imgA_gray = rgb2gray(imgA)
    imgB_gray = rgb2gray(imgB)
    imgC_gray = rgb2gray(imgC)
#    plt.imshow(imgA_gray, cmap="gray")
#    plt.imshow(imgB_gray, cmap="gray")
#    plt.imshow(imgC_gray, cmap="gray")
     
    cimgA = corner_detector(imgA_gray)
    cimgB = corner_detector(imgB_gray)
    cimgC = corner_detector(imgC_gray)
    
#    plt.imshow(cimgA, cmap="gray")
    
    max_pts = 500
     
    xA,yA,rmaxA = anms(cimgA, max_pts)
    xB,yB,rmaxB = anms(cimgB, max_pts)
    xC,yC,rmaxC = anms(cimgC, max_pts)
    
    descsA = feat_desc(imgA_gray, xA, yA)
    descsB = feat_desc(imgB_gray, xB, yB)
    descsC = feat_desc(imgC_gray, xC, yC)
    
    match1 = feat_match(descsA, descsB)
    match2 = feat_match(descsC, descsB)
    
    ransac_thresh = 10
    
    xA1,yA1 = xA[match1>0].reshape(-1,1),yA[match1>0].reshape(-1,1)
    xB1,yB1 = xB[match1[match1>0]].reshape(-1,1),yB[match1[match1>0]].reshape(-1,1)
    H1,inlier_ind1 = ransac_est_homography(xA1, yA1, xB1, yB1, ransac_thresh)
    
    xC2,yC2 = xC[match2>0].reshape(-1,1),yC[match2>0].reshape(-1,1)
    xB2,yB2 = xB[match2[match2>0]].reshape(-1,1),yB[match2[match2>0]].reshape(-1,1)
    H2,inlier_ind2 = ransac_est_homography(xC2, yC2, xB2, yB2, ransac_thresh)
    
    newImageLeft, newImageMiddle, newImageRight = getNewSize(H1,H2,imgA,imgB,imgC)
    blend1 = blend(newImageLeft,newImageMiddle)
    img_mosaic = blend1
#    output = blend(newImageRight,blend(newImageLeft,newImageMiddle))
#    img_mosaic = output
        
#    IA1 = cv.imread('left_small.jpg')
#    drawPoints(IA1,xA1,yA1)
#    cv.imshow('IA1',IA1)
#    
#    IB1 = cv.imread('middle_small.jpg')
#    drawPoints(IB1,xB1,yB1)
#    cv.imshow('IB1',IB1)
#    
#    IB2 = cv.imread('middle_small.jpg')
#    drawPoints(IB2,xB2,yB2)
#    cv.imshow('IB2',IB2)
#    
#    IC2 = cv.imread('right_small.jpg')
#    drawPoints(IC2,xC2,yC2)
#    cv.imshow('IC2',IC2)
#    
#    cv.waitKey(0)
    
    return img_mosaic