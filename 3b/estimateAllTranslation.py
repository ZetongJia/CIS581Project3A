# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:28:25 2019

@author: Jiatong Sun
"""

import numpy as np
import scipy.signal as signal
from estimateFeatureTranslation import estimateFeatureTranslation
from helpers import rgb2gray
from helpers import GaussianPDF_2D

def estimateAllTranslation(startXs,startYs,img1,img2):
    row,col = img1.shape[0], img1.shape[1];
    N,F = startXs.shape[0], startXs.shape[1];
    I_gray_1 = rgb2gray(img1);
    G = GaussianPDF_2D(0,3,4,4);
    [dx,dy] =  np.gradient(G, axis = (1,0));
    Ix = signal.convolve2d(I_gray_1,dx,'same');
    Iy = signal.convolve2d(I_gray_1,dy,'same');
    newXs = np.zeros((N,F),dtype = np.int32);
    newYs = np.zeros((N,F),dtype = np.int32);
    for i in range(N):
        for j in range(F):
            newX, newY = estimateFeatureTranslation(startXs[i][j],\
                                startYs[i][j], Ix, Iy, img1, img2)
            if newX>=0 and newX<col and newY>=0 and newY<row>0:
                newXs[i][j] = newX;
                newYs[i][j] = newY;
                
    return newXs, newYs