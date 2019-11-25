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
    newbbox = np.zeros((bbox.shape),dtype=np.int32);
    dist_thresh = 5;
    eliminate_mat = np.ones(startXs.shape,dtype = np.int32);
    for j in range(startXs.shape[1]):
        eliminate_ind = np.arange(eliminate_mat.shape[0]);
        
        inlier_ind = (startXs[:,j] != 0);
        
        old_coor_x = startXs[:,j].reshape(-1);
        old_coor_y = startYs[:,j].reshape(-1);
        new_coor_x = newXs[:,j].reshape(-1);
        new_coor_y = newYs[:,j].reshape(-1);
        
        old_coor_x = old_coor_x[inlier_ind];
        old_coor_y = old_coor_y[inlier_ind];
        new_coor_x = new_coor_x[inlier_ind];
        new_coor_y = new_coor_y[inlier_ind];
        
        point_error = np.sqrt((new_coor_x-old_coor_x)**2 +\
                              (new_coor_y-old_coor_y)**2);
        point_error_ind = (point_error<dist_thresh);
        
        old_coor_x = old_coor_x[point_error_ind];
        old_coor_y = old_coor_y[point_error_ind];
        new_coor_x = new_coor_x[point_error_ind];
        new_coor_y = new_coor_y[point_error_ind];
        
        old_coor = np.vstack((old_coor_x,old_coor_y));
        new_coor = np.vstack((new_coor_x,new_coor_y));
        
        tform = tf.estimate_transform('similarity', old_coor.T, new_coor.T);
        tformp = np.asmatrix(tform.params);
        old_coor_mat = np.vstack((old_coor, np.ones(old_coor.shape[1])));
        new_coor_mat = tformp.dot(old_coor_mat);
        error = np.linalg.norm(new_coor_mat[0:2,:] - old_coor,axis=0);
        error_ind = (error<dist_thresh);
        
        filter_old_x = old_coor[0,:][error_ind];
        filter_old_y = old_coor[1,:][error_ind];
        filter_new_x = new_coor[0,:][error_ind];
        filter_new_y = new_coor[1,:][error_ind];
        
        eliminate_temp = np.zeros(eliminate_mat.shape[0],dtype = np.int32);
        eliminate_temp[eliminate_ind[inlier_ind==True]\
                       [point_error_ind==True][error_ind==True]] = True;
        eliminate_mat[:,j:j+1] = eliminate_temp.reshape(-1,1);

        
        filter_old_coor = np.vstack((filter_old_x,filter_old_y));
        filter_new_coor = np.vstack((filter_new_x,filter_new_y));
        
        filter_tform = tf.estimate_transform('similarity',\
                                     filter_old_coor.T, filter_new_coor.T);
        filter_tformp = np.asmatrix(filter_tform.params);
        
        x_temp = bbox[j,:,0].reshape(-1);
        y_temp = bbox[j,:,1].reshape(-1);
        old_corners = np.vstack((x_temp,y_temp,np.ones(4,dtype = np.int32)));
        
        new_corners_temp = filter_tformp.dot(old_corners);
        corner_1 = new_corners_temp[0:2,0].reshape(1,-1);
        corner_2 = new_corners_temp[0:2,1].reshape(1,-1);
        corner_3 = new_corners_temp[0:2,2].reshape(1,-1);
        corner_4 = new_corners_temp[0:2,3].reshape(1,-1);
        new_corners_temp = np.array([corner_1,corner_2,corner_3,corner_4],\
                                    dtype = np.float32);
        x,y,w,h = cv.boundingRect(new_corners_temp);
        new_corners = getBoxPoints(x,y,w,h);
        newbbox[j,:,:] = new_corners;
        
#        new_corners = filter_tformp.dot(old_corners);
#        newbbox[j, :, :] = np.matrix.transpose(new_corners[0:2, :]);

    newXs = newXs * eliminate_mat;
    newYs = newYs * eliminate_mat;
    N1 = np.max(np.sum(newXs>0,axis = 0));
    Xs = np.zeros((N1,bbox.shape[0]),dtype = np.int32);
    Ys = np.zeros((N1,bbox.shape[0]),dtype = np.int32);
    for j in range(bbox.shape[0]):
        X_coor_temp = newXs[:,j];
        Y_coor_temp = newYs[:,j];
        n = np.sum(X_coor_temp>0);
        Xs[0:n,j] = X_coor_temp[X_coor_temp!=0];
        Ys[0:n,j] = Y_coor_temp[Y_coor_temp!=0];
    
    return Xs, Ys, newbbox;

