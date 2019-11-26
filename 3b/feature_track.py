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
from helpers import getBoxPoints

def estimateAllTranslation(startXs,startYs,img1,img2):
    row,col = img1.shape[0], img1.shape[1];
    N,F = startXs.shape[0], startXs.shape[1];
    I_gray_1 = rgb2gray(img1);
#    G = GaussianPDF_2D(0,1,4,4);
#    [dx,dy] =  np.gradient(G, axis = (1,0));
#    Ix = signal.convolve2d(I_gray_1,dx,'same');
#    Iy = signal.convolve2d(I_gray_1,dy,'same');
    Ix,Iy = np.gradient(I_gray_1,axis = (1,0));
    newXs = np.zeros((N,F),dtype = np.int32);
    newYs = np.zeros((N,F),dtype = np.int32);
    for i in range(N):
        for j in range(F):
            if startXs[i][j] == 0 or startYs[i][j] == 0:
                continue;
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
    error_thresh = 1;
    
    iteration = 5;
    for i in range(iteration):
        X_new,Y_new = generatePatch(x0,y0);
        old_coor = np.array((x0,y0)).reshape(-1,1);
        It_temp = interp2(I_gray_2,X_new,Y_new) - I_gray_1[Y_old,X_old];
        error = np.linalg.norm(It_temp);
            
        if error < min_error:
            min_error = error;
            newX, newY = x0,y0;
            if error < error_thresh:
                break;
        
        A = np.hstack((Ix_temp.reshape(-1,1),Iy_temp.reshape(-1,1)));
        b = -It_temp.reshape(-1,1);
        flow_temp = np.linalg.solve(np.dot(A.T,A),np.dot(A.T,b));
        new_coor = old_coor + flow_temp;
        x0, y0 = new_coor[0,0], new_coor[1,0];
    
    return newX, newY;

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    newbbox = np.zeros((bbox.shape),dtype=np.float32);
    dist_thresh = 4;
    N, F = startXs.shape[0], startXs.shape[1];
    Xs = np.zeros((N,F),dtype = np.float32);
    Ys = np.zeros((N,F),dtype = np.float32);
    N1 = 0;
    for f in range(F):
        x_old, y_old = startXs[:,f:f+1], startYs[:,f:f+1];
        x_new, y_new = newXs[:,f:f+1], newYs[:,f:f+1];
#        delete zero
        non_zero_ind = np.nonzero(x_old * y_old);
        x_old, y_old = x_old[non_zero_ind], y_old[non_zero_ind];
        x_new, y_new = x_new[non_zero_ind], y_new[non_zero_ind];
#        delete deviation
        error = np.sqrt((x_new-x_old)**2 + (y_new-y_old)**2);
        inlier_ind = np.where(error < dist_thresh);
        x_old, y_old = x_old[inlier_ind], y_old[inlier_ind];
        x_new, y_new = x_new[inlier_ind], y_new[inlier_ind];
#        calculate similarity matrix
        coor_old = np.vstack((x_old,y_old));
        coor_new = np.vstack((x_new,y_new));
        tform = tf.estimate_transform('similarity', coor_old.T, coor_new.T);
        tformp = np.asmatrix(tform.params);
#        transform
        corner_old_temp = np.vstack((bbox[f,:,:].T,np.ones(4,dtype = np.int32)));
        corner_new_temp = tformp.dot(corner_old_temp);
        corner_new = np.matrix.transpose(corner_new_temp[0:2, :]);
        newbbox[f,:,:] = corner_new;
#        obtain Xs, Ys
#        TO DO: delete bbox outlier
        # x_new = x_new[np.where(x_new[])]
        # N1 = max(x_old.shape[0], N1);
        # Xs[0:x_old.size,f:f+1] = x_new.reshape(-1,1);
        # Ys[0:y_old.size,f:f+1] = y_new.reshape(-1,1);
        x_in_bbox = x_new[np.where(x_new <= newbbox[f,3,0])] # and x_new >= newbbox[f, 0,0])]
        x_in_bbox = x_new[np.where(x_new >= newbbox[f,0,0])] # and x_new >= newbbox[f, 0,0])]
        y_in_bbox = y_new[np.where(y_new <= newbbox[f,3,1])] # and y_new >= newbbox[f, 0,1])]
        y_in_bbox = y_new[np.where(y_new >= newbbox[f,0,1])] # and y_new >= newbbox[f, 0,1])]
        N1 = max(x_in_bbox.shape[0], y_in_bbox.shape[0], N1)
        Xs[0:x_in_bbox.size,f:f+1] = x_in_bbox.reshape(-1,1);
        Ys[0:y_in_bbox.size,f:f+1] = y_in_bbox.reshape(-1,1);
    
    Xs = Xs[0:N1,:];
    Ys = Ys[0:N1,:];
        
    return Xs, Ys, newbbox;

