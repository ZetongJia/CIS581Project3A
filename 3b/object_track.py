import numpy as np
import cv2 as cv

from feature_track import estimateAllTranslation
from feature_track import applyGeometricTransformation
from get_features import getFeatures
from helpers import getBoxPoints
from helpers import drawPoints
from helpers import flipChannel

# get output video from input video
def objectTracking(rawVideo):
    
#    read video
    cap = cv.VideoCapture(rawVideo);
#    initialize a video write
    trackVideo = 'Output_' + rawVideo;
    fourcc = cv.VideoWriter_fourcc(*'MJPG');
    fps =cap.get(cv.CAP_PROP_FPS);
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),\
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)));
    out = cv.VideoWriter(trackVideo, fourcc, fps, size);
#    other initial settings
    max_object = 2 ;
    frame_cnt = 0;
    bbox = np.zeros((max_object,4,2),dtype = np.int32); 
    trace_x = list();
    trace_y = list();
    
#    start process frames one by one
    while True: 
        ret,frame = cap.read();
        trace_frame = frame.copy();
        if ret == True:
            frame_cnt = frame_cnt + 1;
#            select objects on the first frame
            if frame_cnt == 1:
                cv.imwrite(str(frame_cnt)+".jpg",frame);
                last_frame = frame;
                F = 0;
                while True:
#                    select objects manually
                    x,y,w,h = np.int32(cv.selectROI("roi", frame, fromCenter=False));
                    cv.destroyAllWindows();
                    cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2);
                    bbox[F,:,:] = getBoxPoints(x,y,w,h);
                    F = F + 1;
                    if F >= max_object:
                        cv.destroyAllWindows();
                        break;
                feat_x, feat_y = getFeatures(frame, bbox);
                
#                save new feature points to trace
                trace_x.append(feat_x);
                trace_y.append(feat_y);
#           process other frames in the video
            else:
                newXs, newYs = estimateAllTranslation(feat_x,feat_y,\
                                flipChannel(last_frame),flipChannel(frame));
                feat_x,feat_y,bbox = applyGeometricTransformation(feat_x,\
                                                    feat_y,newXs,newYs,bbox);
                last_frame = frame;
                
#                save new feature points to trace
                trace_x.append(feat_x);
                trace_y.append(feat_y);
                
#                delete NaN bbox
                for f in range(bbox.shape[0]):
                    if np.sum(np.isnan(bbox[f,:,:])) > 0:
                        bbox[f,:,:] = 0;
                        feat_x[:,f:f+1] = 0;
                        feat_y[:,f:f+1] = 0;
                
#                draw bbox and feature points
                for f in range(bbox.shape[0]):
                    if np.sum(bbox[f,:,:]) == 0:
                        continue;
                    cv.rectangle(frame,(bbox[f,0,0],bbox[f,0,1]),\
                                (bbox[f,3,0],bbox[f,3,1]),(0,255,0),2);
                    cv.rectangle(trace_frame,(bbox[f,0,0],bbox[f,0,1]),\
                            (bbox[f,3,0],bbox[f,3,1]),((2-f)*127,(f+1)*127,0),2);
                    drawPoints(frame,feat_x[:,f],feat_y[:,f],(0,0,255));
                    
#                draw trace
                for k in range(len(trace_x)):
                    for f in range(trace_x[k].shape[1]):
                        drawPoints(trace_frame,trace_x[k][:,f],\
                                   trace_y[k][:,f],((2-f)*127,0,(f+1)*127));
#            save video with bbox and all feature points
            out.write(trace_frame);
            cv.imshow("trace", trace_frame);
            cv.imshow("optical flow",frame);
            
#            press 'q' on the keyboard to exit
            if cv.waitKey(30) & 0xff == ord('q'):
                cv.destroyAllWindows();
                break;
#            press 's' on the keyboard to save image
            elif cv.waitKey(30) & 0xff == ord('s'):
                cv.imwrite(str(frame_cnt)+".jpg",frame);
#    release video reader and video writer
    cap.release();
    out.release();
    
    return trackVideo;