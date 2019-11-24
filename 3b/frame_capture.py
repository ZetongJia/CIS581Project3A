# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:20:56 2019

@author: Jiatong Sun
"""

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
    frame_cnt = 0;
    max_object = 1;
    bbox = np.zeros((max_object,4,2),dtype = np.int32);
    F = 0;
    while True: 
        ret,frame = cap.read();
        frame_cnt = frame_cnt + 1;
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
#        elif frame_cnt == 3:
#            cv.imwrite(str(frame_cnt)+".jpg",frame);
        else:
            newXs, newYs = estimateAllTranslation(feat_x,feat_y,\
                            flipChannel(last_frame),flipChannel(frame));
            feat_x,feat_y,bbox = applyGeometricTransformation(feat_x,\
                                                feat_y,newXs,newYs,bbox);
            last_frame = frame; 
            for f in range(bbox.shape[0]): 
                cv.rectangle(frame,(bbox[f,0,0],bbox[f,0,1]),
                            (bbox[f,3,0],bbox[f,3,1]),(0,255,0),2);
            drawPoints(frame,feat_x,feat_y,(0,0,255));
                                    
        cv.imshow("capture",frame);
        if cv.waitKey(30) & 0xff == ord('q'):
            cv.destroyAllWindows();
            break;
        elif cv.waitKey(30) & 0xff == ord('s'):
            cv.imwrite(str(frame_cnt)+".jpg",frame);