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

from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography
from helpers import rgb2gray
from helpers import getNewSize
from helpers import drawPoints
from helpers import drawLines
from helpers import alphaBlend
from helpers import flipChannel
from seam_carving_blend import seamBlend

def mymosaic(img_input):
  # Your Code Here
    img_mosaic = np.zeros((img_input.shape[0],1),dtype=object)
    for i in range(img_input.shape[0]):
        imgA = img_input[i,0]
        imgB = img_input[i,1]
        imgC = img_input[i,2]
        
        imgA_gray = rgb2gray(imgA)
        imgB_gray = rgb2gray(imgB)
        imgC_gray = rgb2gray(imgC)
        
        cimgA = corner_detector(imgA_gray)
        cimgB = corner_detector(imgB_gray)
        cimgC = corner_detector(imgC_gray)
        
        max_pts = 500
         
        xA,yA,rmaxA = anms(cimgA, max_pts)
        xB,yB,rmaxB = anms(cimgB, max_pts)
        xC,yC,rmaxC = anms(cimgC, max_pts)
        
# =============================================================================
# # Demonstrate ANMS result
#         IA = flipChannel(imgA)
#         drawPoints(IA,xA,yA,(0,0,255))
#         cv.imwrite('A'+str(i+1)+'.jpg',IA)
#         
#         IB = flipChannel(imgB)
#         drawPoints(IB,xB,yB,(0,0,255))
#         cv.imwrite('B'+str(i+1)+'.jpg',IB)
#         
#         IC = flipChannel(imgC)
#         drawPoints(IC,xC,yC,(0,0,255))
#         cv.imwrite('C'+str(i+1)+'.jpg',IC)
# =============================================================================
        
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
# =============================================================================
# # Demonstrating RANSAC match result
#         row,col,_ = imgA.shape
#         
#         outlier_ind1 = np.delete(np.arange(len(xA1)),inlier_ind1)
#         IA1 = flipChannel(imgA)
#         drawPoints(IA1,xA1[inlier_ind1],yA1[inlier_ind1],(0,0,255))
#         drawPoints(IA1,xA1[outlier_ind1],yA1[outlier_ind1],(255,0,0))
#         IB1 = flipChannel(imgB)
#         drawPoints(IB1,xB1[inlier_ind1],yB1[inlier_ind1],(0,0,255))
#         drawPoints(IB1,xB1[outlier_ind1],yB1[outlier_ind1],(255,0,0))
#         imgAB = np.zeros((row,2*col,3))
#         imgAB[:,0:col,:] = IA1
#         imgAB[:,col:2*col,:] = IB1
#         drawLines(imgAB,xA1[inlier_ind1],yA1[inlier_ind1]\
#                   ,xB1[inlier_ind1]+col,yB1[inlier_ind1],(0,255,0))
#         cv.imwrite('left_match'+str(i+1)+'.jpg',imgAB)
#         
#         outlier_ind2 = np.delete(np.arange(len(xC2)),inlier_ind2)
#         IC2 = flipChannel(imgC)
#         drawPoints(IC2,xC2[inlier_ind2],yC2[inlier_ind2],(0,0,255))
#         drawPoints(IC2,xC2[outlier_ind2],yC2[outlier_ind2],(255,0,0))
#         IB2 = flipChannel(imgB)
#         drawPoints(IB2,xB2[inlier_ind2],yB2[inlier_ind2],(0,0,255))
#         drawPoints(IB2,xB2[outlier_ind2],yB2[outlier_ind2],(255,0,0))
#         imgBC = np.zeros((row,2*col,3))
#         imgBC[:,0:col,:] = IB2
#         imgBC[:,col:2*col,:] = IC2
#         drawLines(imgBC,xB2[inlier_ind2],yB2[inlier_ind2]\
#                   ,xC2[inlier_ind2]+col,yC2[inlier_ind2],(0,255,0))
#         cv.imwrite('right_match'+str(i+1)+'.jpg',imgBC)
# =============================================================================
        
        new_left, new_middle, new_right = getNewSize(H1,H2,imgA,imgB,imgC)

#   Blend Images by  Seam Carving or Alpha Blending
        img_mosaic[i,0] = seamBlend(new_left, new_middle, new_right)
#        img_mosaic[i,0] = alphaBlend(alphaBlend(new_left,new_middle),new_right)
        
    return img_mosaic