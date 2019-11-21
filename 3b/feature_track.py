# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 00:28:25 2019

@author: Jiatong Sun
"""

import numpy as np
import cv2 as cv
import scipy.signal as signal
import skimage.transform as tf

from helpers import rgb2gray
from helpers import GaussianPDF_2D
from helpers import flipChannel
from helpers import generatePatch
from helpers import interp2

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
                
    return newXs, newYs;

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
    I_gray_1, I_gray_2 = rgb2gray(img1), rgb2gray(img2);
    X_old, Y_old = generatePatch(startX,startY);
    X_old, Y_old = X_old.astype(np.int32),Y_old.astype(np.int32)
    Ix_temp, Iy_temp = Ix[Y_old,X_old], Iy[Y_old,X_old];
    x0, y0 = startX, startY;
    min_error = 999999;
    newX, newY = 0, 0;
    
    iteration = 10;
    for i in range(iteration):
        X_new,Y_new = generatePatch(x0,y0);
        old_coor = np.array((x0,y0)).reshape(-1,1);
        It_temp = interp2(I_gray_2,X_new,Y_new) - I_gray_1[Y_old,X_old];
        error = np.linalg.norm(It_temp);
        if error < min_error:
            min_error = error;
            newX, newY = x0,y0;
        
        A = np.hstack((Ix_temp.reshape(-1,1),Iy_temp.reshape(-1,1)));
        b = It_temp.reshape(-1,1);
        flow_temp = np.linalg.solve(np.dot(A.T,A),-np.dot(A.T,b));
        new_coor = old_coor + flow_temp;
        x0, y0 = new_coor[0], new_coor[1];
        
    return newX, newY;

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    for j in range(startXs.shape[1]):
        old_coor = np.hstack((startXs[:,j].reshape(-1,1),\
                              startYs[:,j].reshape(-1,1)));
        new_coor = np.hstack((newXs[:,j].reshape(-1,1),\
                              newXs[:,j].reshape(-1,1)));
        tform = tf.SimilarityTransform();
        res =tform.estimate(new_coor, old_coor);
        M = tform.params;
        if res:
            

img1 = flipChannel(cv.imread('1.jpg'));
img2 = flipChannel(cv.imread('17.jpg'));
newXs, newYs = estimateAllTranslation(feat_x,feat_y,img1,img2);
old_coor = np.hstack((feat_x[:,1].reshape(-1,1),feat_y[:,1].reshape(-1,1)));
new_coor = np.hstack((newXs[:,1].reshape(-1,1),newXs[:,1].reshape(-1,1)));
tform = tf.SimilarityTransform();
res =tform.estimate(new_coor, old_coor)
M = tform.params

