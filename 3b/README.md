# Team 30 Project3B

**Member: Zetong Jia, Qiao Han, Jiatong Sun**

## File

1. Code: 
   1. Simply run ***optical_flow_test_script.py*** to obtain result videos.

      The first frame will pop up for users to select the objects of interest, where users should use the mouse to select several rectangle areas (currently two objects for **"Easy.mp4"** and one for **"Medium.mp4"**) and press whichever button on the keyboard to view the tracking process. 

      Window **"trace"** shows  current boxes and all features points.

      Window **"optical flow"** shows current boxes and current feature points.

      The output video will be name as **"Output_" + input name**.

   2. ***get_features.py*** is used for obtaining feature points in the selected bounding box. 

      The method is to use `cv2.goodFeaturesToTrack` in every box, and additional methods including `cv2.corner_harris()` and finding feature points in the whole image are also implemented and commented below.

   3. ***feature_track.py*** is used for calculating optical flow for all feature points and obtaining new bounding boxes. 

      In the function `estimateAllTranslation()`, we discard the points near the edges so that we can generate 5×5 patch for each points successfully.

      In the function `estimateFeatureTranslation()`, we implement the core algorithm of optical flow to get corresponding points for all feature points. Each feature point will be calculated via five iterations.

      In the function `applyGeometricTransformation()`, we discard zero coordinate points (which demonstrated invalid points), deviating points (threshold = 4 pixels) and out-of-box points.

   4. ***object_track.py*** reads all frames in the input video and generate the output video with updated bounding boxes and all trace points. 

      We discard every nan box, which usually means a feature has already moved out of the image.

      For **"Medium.mp4"**, we invoke `getFeatures()` if feature points is fewer than 8. We also increase x (y) length of a bounding box by 20 if this box's x(y) length is shorter than last bounding box minus 10 so that we can keep the bounding box's area as long as possible. 

   5. ***helpers.py*** contains several tool functions.	
2. Input Videos
   1. **"Easy.mp4"**
   2. **"Medium.mp4"**
3. Output Videos
   1. **"Output_Easy.mp4"**
   2. **"Output_Medium.mp4"**

## Comment

1. We use `cv2.VideoCapture()` to read video and `cv2.VideoWriter()` to write video. Since the instruction requires the format of object tracking function to be `[trackedVideo] = objectTracking(rawVideo)` , we implement both read and write functions in the function itself, and set the type of `rawVideo` and `trackedVideo` to be string to represent the video names.

   

