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
    out = cv.VideoWriter('output.avi',fourcc, 30, (640,360));
    frame_cnt = 0;
    max_object = 2 ;
    bbox = np.zeros((max_object,4,2),dtype = np.int32); 
    trace_x = list();
    trace_y = list();
    while True: 
        ret,frame = cap.read();
        frame_show = frame.copy();
        if ret == True:
            frame_cnt = frame_cnt + 1;
            row,col = frame.shape[0],frame.shape[1];
            if frame_cnt == 1:
                cv.imwrite(str(frame_cnt)+".jpg",frame);
                last_frame = frame;
                F = 0;
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
                trace_x.append(feat_x);
                trace_y.append(feat_y);
                drawPoints(frame,feat_x,feat_y,(0,0,255));
                cv.imshow("feature points", frame);
                cv.waitKey(0);
                cv.destroyAllWindows();
            else:
                newXs, newYs = estimateAllTranslation(feat_x,feat_y,\
                                flipChannel(last_frame),flipChannel(frame));
                feat_x,feat_y,bbox = applyGeometricTransformation(feat_x,\
                                                    feat_y,newXs,newYs,bbox);
                last_frame = frame; 
                trace_x.append(feat_x);
                trace_y.append(feat_y);
                f = 0;
                for f in range(bbox.shape[0]):
                    if np.sum(np.isnan(bbox[f,:,:])) > 0:
                        bbox[f,:,:] = 0;
                        feat_x[:,f:f+1] = 0;
                        feat_y[:,f:f+1] = 0;
                
                for f in range(bbox.shape[0]):
                    if np.sum(bbox[f,:,:]) == 0:
                        continue;
                    cv.rectangle(frame,(bbox[f,0,0],bbox[f,0,1]),\
                                (bbox[f,3,0],bbox[f,3,1]),(0,255,0),2);
                    cv.rectangle(frame_show,(bbox[f,0,0],bbox[f,0,1]),\
                            (bbox[f,3,0],bbox[f,3,1]),((2-f)*127,(f+1)*127,0),2);
                    drawPoints(frame,feat_x[:,f],feat_y[:,f],(0,0,255));
                for k in range(len(trace_x)):
                    for f in range(trace_x[k].shape[1]):
                        drawPoints(frame_show,trace_x[k][:,f],\
                                   trace_y[k][:,f],((2-f)*127,0,(f+1)*127));
                        
            out.write(frame_show);
            cv.imshow("show", frame_show);
            cv.imshow("capture",frame);
            if cv.waitKey(30) & 0xff == ord('q'):
                cv.destroyAllWindows();
                break;
            elif cv.waitKey(30) & 0xff == ord('s'):
                cv.imwrite(str(frame_cnt)+".jpg",frame);
    cap.release();
#    out.release();