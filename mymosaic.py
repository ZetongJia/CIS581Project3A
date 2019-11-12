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

from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc
from feat_match import feat_match
from ransac_est_homography import ransac_est_homography
from helpers import rgb2gray
from helpers import getNewSize
from helpers import alphaBlend
from seam_carving_blend import *

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
        
        new_left, new_middle, new_right = getNewSize(H1,H2,imgA,imgB,imgC)
        img_mosaic[i,0] = seam_carving_blend_right(seam_carving_blend_left(new_left,new_middle),new_right)
        # img_mosaic[i,0] = alphaBlend(alphaBlend(new_left,new_middle),new_right)
        
    return img_mosaic