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
    G = GaussianPDF_2D(0,3,4,4);
    [dx,dy] =  np.gradient(G, axis = (1,0));
    Ix = signal.convolve2d(I_gray_1,dx,'same');
    Iy = signal.convolve2d(I_gray_1,dy,'same');
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
            if error < error_thresh:
                break;
        
        A = np.hstack((Ix_temp.reshape(-1,1),Iy_temp.reshape(-1,1)));
        b = It_temp.reshape(-1,1);
        flow_temp = np.linalg.solve(np.dot(A.T,A),-np.dot(A.T,b));
        new_coor = old_coor + flow_temp;
        x0, y0 = new_coor[0], new_coor[1];
        
    return newX, newY;

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    newbbox = np.zeros((bbox.shape),dtype=np.int32);
    for j in range(startXs.shape[1]):
        old_coor = np.hstack((startXs[:,j].reshape(-1,1),\
                              startYs[:,j].reshape(-1,1)));
        new_coor = np.hstack((newXs[:,j].reshape(-1,1),\
                              newYs[:,j].reshape(-1,1)));
        tform = tf.SimilarityTransform();
        res =tform.estimate(new_coor, old_coor);
        M = tform.params;
        if res:
            x_temp = bbox[j,:,0].reshape(-1);
            y_temp = bbox[j,:,1].reshape(-1);
            old_corners = np.vstack((x_temp,y_temp,np.ones(4,dtype = np.int32)));
            new_corners_temp = np.dot(M,old_corners);
            corner_1 = new_corners_temp[0:2,0].reshape(1,-1);
            corner_2 = new_corners_temp[0:2,1].reshape(1,-1);
            corner_3 = new_corners_temp[0:2,2].reshape(1,-1);
            corner_4 = new_corners_temp[0:2,3].reshape(1,-1);
            new_corners_temp = np.array([corner_1,corner_2,corner_3,corner_4],\
                                        dtype = np.float32);
            x,y,w,h = cv.boundingRect(new_corners_temp);
            new_corners = getBoxPoints(x,y,w,h);
            newbbox[j,:,:] = new_corners;
    error = np.sqrt((newXs-startXs)**2 + (newYs-startYs)**2);
    newXs[error>4] = 0;
    newYs[error>4] = 0;
    N1 = np.max(np.sum(newXs>0,axis = 0));
    Xs = np.zeros((N1,bbox.shape[0]),dtype = np.int32);
    Ys = np.zeros((N1,bbox.shape[0]),dtype = np.int32);
    for j in range(bbox.shape[0]):
        X_coor_temp = newXs[:,0];
        Y_coor_temp = newYs[:,0];
        n = np.sum(X_coor_temp>0);
        Xs[0:n,j] = X_coor_temp[X_coor_temp!=0];
        Ys[0:n,j] = Y_coor_temp[Y_coor_temp!=0];
    
    return Xs, Ys, newbbox;

#img1 = flipChannel(cv.imread('2.jpg'));
#img2 = flipChannel(cv.imread('3.jpg'));
#newXs, newYs = estimateAllTranslation(feat_x,feat_y,img1,img2);
#Xs,Ys,newbbox = applyGeometricTransformation(feat_x,feat_y,newXs,newYs,bbox);
