# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:20:56 2019

@author: Jiatong Sun
"""

import numpy as np
import cv2 as cv

from helpers import getBoxPoints

if __name__ == "__main__": 
    cap = cv.VideoCapture("Easy.mp4");
    frame_cnt = 0;
    max_object = 2;
    bbox = np.zeros((max_object,4,2),dtype = np.int32);
    F = 0;
    while True:
        ret,frame = cap.read();
        frame_cnt = frame_cnt + 1;
        if frame_cnt == 1:
            cv.imwrite("1.jpg",frame);
            while True:
                x,y,w,h = np.int32(cv.selectROI("roi", frame, fromCenter=False));
                cv.destroyAllWindows();
                cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2);
                bbox[F,:,:] = getBoxPoints(x,y,w,h);
                F = F + 1;
                if F >= max_object:
                    cv.destroyAllWindows();
                    break;
        cv.imshow("capture",frame);
        if cv.waitKey(30) & 0xff == ord('q'):
            cv.destroyAllWindows();
            break;
        elif cv.waitKey(30) & 0xff == ord('s'):
            cv.imwrite(str(frame_cnt)+".jpg",frame);