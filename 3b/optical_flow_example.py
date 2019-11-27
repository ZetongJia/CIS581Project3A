# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:24:29 2019

@author: Jiatong Sun
"""

import numpy as np
import skimage.transform as tf

from helpers import interp2

if __name__ == "__main__":
    
    i1 = np.vstack((np.arange(0,10,1).reshape(1,-1),\
                   np.arange(0,20,2).reshape(1,-1),\
                   np.arange(0,30,3).reshape(1,-1)));
    i2 = np.vstack((np.arange(2,12,1).reshape(1,-1),\
                   np.arange(4,24,2).reshape(1,-1),\
                   np.arange(6,36,3).reshape(1,-1)));
    print("tiny image 1:");
    print(i1);
    print("");
    print("tiny image 2:");
    print(i2);
    print("");
    
    ix,iy = np.gradient(i1,axis = (1,0));
    
    xs1, ys1 = 5, 1;
    xs2, ys2 = 8, 1;
    
    print("start point 1: ", (xs1,ys1));
    print("start point 2: ", (xs2,ys2));
    print("");
    
    x1, y1 = xs1, ys1;
    x2, y2 = xs2, ys2;
    
    X_old = np.array([[x1-1,x1,x1+1],[x1-1,x1,x1+1],[x1-1,x1,x1+1]],dtype = np.int32);
    Y_old = np.array([[0,0,0],[1,1,1],[2,2,2]],dtype = np.int32);
    ix_temp = ix[Y_old,X_old];
    iy_temp = iy[Y_old,X_old];
    for i in range(5):
        X_new = np.array([[x1-1,x1,x1+1],[x1-1,x1,x1+1],[x1-1,x1,x1+1]]);
        Y_new = np.array([[0,0,0],[1,1,1],[2,2,2]]);
        old_coor = np.array((x1,y1)).reshape(-1,1);
        it_temp = interp2(i2,X_new,Y_new) - i1[Y_old,X_old];
        error = np.linalg.norm(it_temp);
        A = np.hstack((ix_temp.reshape(-1,1),iy_temp.reshape(-1,1)));
        b = -it_temp.reshape(-1,1);
        flow_temp = np.linalg.solve(np.dot(A.T,A),np.dot(A.T,b));
        new_coor = old_coor + flow_temp;
        x1, y1 = new_coor[0,0], new_coor[1,0];
    xe1, ye1 = x1, y1;
    print("result point 1:",(xe1,ye1));
    
    X_old = np.array([[x2-1,x2,x2+1],[x2-1,x2,x2+1],[x2-1,x2,x2+1]],dtype = np.int32);
    Y_old = np.array([[0,0,0],[1,1,1],[2,2,2]],dtype = np.int32);
    ix_temp = ix[Y_old,X_old];
    iy_temp = iy[Y_old,X_old];
    for i in range(5):
        X_new = np.array([[x2-1,x2,x2+1],[x2-1,x2,x2+1],[x2-1,x2,x2+1]]);
        Y_new = np.array([[0,0,0],[1,1,1],[2,2,2]]);
        old_coor = np.array((x2,y2)).reshape(-1,1);
        it_temp = interp2(i2,X_new,Y_new) - i1[Y_old,X_old];
        error = np.linalg.norm(it_temp);
        A = np.hstack((ix_temp.reshape(-1,1),iy_temp.reshape(-1,1)));
        b = -it_temp.reshape(-1,1);
        flow_temp = np.linalg.solve(np.dot(A.T,A),np.dot(A.T,b));
        new_coor = old_coor + flow_temp;
        x2, y2 = new_coor[0,0], new_coor[1,0];
    xe2, ye2 = x2, y2;
    print("result point 2:",(xe2,ye2));
    print("")
    
    old_coor = np.array([[xs1,ys1],[xs2,ys2]]);
    new_coor = np.array([[xe1,ye1],[xe2,ye2]]);
    tform = tf.estimate_transform('similarity', old_coor, new_coor);
    tformp = np.asmatrix(tform.params);
    print("similirity matrix:");
    print(tform.params);
    
    corres = tformp.dot(np.array([[xs1,xs2],[ys1,ys2],[1,1]]));
    
    