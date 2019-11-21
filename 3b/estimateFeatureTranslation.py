# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 01:58:31 2019

@author: Jiatong Sun
"""

import numpy as np
from scipy.interpolate import interp2d
from helpers import rgb2gray
from helpers import generatePatch

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
    I_gray_1 = rgb2gray(img1);
    I_gray_2 = rgb2gray(img2);
    X_old, Y_old = generatePatch(startX,startY);
    x0, y0 = startX, startY;
    min_error = 999999;
    newX, newY = 0, 0;
    
    iteration = 10;
    for i in range(iteration):
        X_new,Y_new = generatePatch(x0,y0);
        old_coor = np.array((x0,y0)).reshape(-1,1);
        It_temp = interp2d(X_new,Y_new,I_gray_2) - I_gray_1[Y_old,X_old];
        error = np.linalg.norm(It_temp);
        if error < min_error:
            min_error = error;
            newX, newY = x0,y0;
        
        Ix_temp = Ix[Y_old,X_old];
        Iy_temp = Iy[Y_old,X_old];
        A = np.hstack((Ix_temp.reshape(-1,1),Iy_temp.reshape(-1,1)));
        b = It_temp.reshape(-1,1);
        flow_temp = np.linalg.solve(np.dot(A.T,A),-np.dot(A.T,b));
        new_coor = old_coor + flow_temp;
        x0, y0 = new_coor[0], new_coor[1];
        
    return newX, newY
        