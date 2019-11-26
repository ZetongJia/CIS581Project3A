# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:20:56 2019

@author: Jiatong Sun
 q"""

import numpy as np
import cv2 as cv

from feature_track import estimateAllTranslation
from feature_track import applyGeometricTransformation
from get_features import getFeatures
from helpers import getBoxPoints
from helpers import drawPoints
from helpers import flipChannel

if __name__ == "__main__": 
    cap = cv.VideoCapture("Easy.mp4");
    fourcc = cv.VideoWriter_fourcc(*'MJPG');
    out = cv.VideoWriter('output.avi',fourcc, 10, (640,360));
    frame_cnt = 0;
    max_object = 1;
    bbox = np.zeros((max_object,4,2),dtype = np.int32);
    F = 0;
    while True: 
        ret,frame = cap.read();
        frame_cnt = frame_cnt + 1;
        row,col = frame.shape[0],frame.shape[1];
#        print(frame.shape);
        if frame_cnt == 1:
            cv.imwrite(str(frame_cnt)+".jpg",frame);
            last_frame = frame;
            while True:
                x,y,w,h = np.int32(cv.selectROI("roi", frame, fromCenter=False));
                cv.destroyAllWindows();
                cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2);
                bbox[F,:,:] = getBoxPoints(x,y,w,h);
                F = F + 1;
                if F >= max_object:
                    cv.destroyAllWindows();
                    break;
            feat_x, feat_y = getFeatures(frame, bbox);
            drawPoints(frame,feat_x,feat_y,(0,0,255));
            cv.imshow("feature points", frame);
            cv.waitKey(0);
            cv.destroyAllWindows();
#        elif frame_cnt % 5 == 1:
        else:
            newXs, newYs = estimateAllTranslation(feat_x,feat_y,\
                            flipChannel(last_frame),flipChannel(frame));
            feat_x,feat_y,bbox = applyGeometricTransformation(feat_x,\
                                                feat_y,newXs,newYs,bbox);
#            print("");
#            print(feat_x);
#            print(feat_y);
            last_frame = frame; 
#            inlier_ind =  (feat_x >= 0) * (feat_x < col) *\
#                      (feat_y >= 0) * (feat_y < row);
#            feat_x = feat_x * inlier_ind;
#            feat_y = feat_y * inlier_ind;
#            bbox_orth = np.zeros(bbox.shape,dtype = np.int32);
#            for f in range(bbox.shape[0]):
#                new_corners_temp = bbox[f,:,:];
#                corner_1 = new_corners_temp[0,0:2].reshape(1,-1);
#                corner_2 = new_corners_temp[1,0:2].reshape(1,-1);
#                corner_3 = new_corners_temp[2,0:2].reshape(1,-1);
#                corner_4 = new_corners_temp[3,0:2].reshape(1,-1);
#                new_corners_temp = np.array([corner_1,corner_2,corner_3,\
#                                             corner_4],dtype = np.float32);
#                x,y,w,h = cv.boundingRect(new_corners_temp);
#                bbox_orth[f,:,:] = getBoxPoints(x,y,w,h);
#                cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2);
#                drawPoints(frame,feat_x[:,f],feat_y[:,f],(0,0,255));
#            print(bbox);
            for f in range(bbox.shape[0]):
                cv.rectangle(frame,(bbox[f,0,0],bbox[f,0,1]),
                            (bbox[f,3,0],bbox[f,3,1]),(0,255,0),2);
                drawPoints(frame,feat_x[:,f],feat_y[:,f],(0,0,255));
#        else:
#            for f in range(bbox.shape[0]): 
#                cv.rectangle(frame,(bbox[f,0,0],bbox[f,0,1]),
#                            (bbox[f,3,0],bbox[f,3,1]),(0,255,0),2);
#                drawPoints(frame,feat_x[:,f],feat_y[:,f],(0,0,255));
                                    
        out.write(frame);
        cv.imshow("capture",frame);
        if cv.waitKey(30) & 0xff == ord('q'):
            cv.destroyAllWindows();
            break;
        elif cv.waitKey(30) & 0xff == ord('s'):
            cv.imwrite(str(frame_cnt)+".jpg",frame);
    cap.release();
    out.release();