# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:26:46 2019

@author: Jiatong Sun
"""

import numpy as np
import matplotlib.image as mpimg
from PIL import Image
from mymosaic import mymosaic

if __name__ == "__main__": 
    img_name = np.array(['franklin','cabin','sign'],dtype = np.object)
    M = img_name.shape[0]       # numbers of image sets
    N = 3                    	# numbers of images per set
    img_input_name = np.tile(img_name.reshape(-1,1),(1,N))
    img_input_name[:,0] = img_input_name[:,0] + '_left.jpg'
    img_input_name[:,1] = img_input_name[:,1] + '_middle.jpg'
    img_input_name[:,2] = img_input_name[:,2] + '_right.jpg'
    
    img_input = np.zeros((M,N),dtype = np.object)
    for i in range(M):
        for j in range(N):
            img_input[i,j] = mpimg.imread(img_input_name[i,j])
    
    img_mosaic = mymosaic(img_input)
    img_output_name = img_name.reshape(-1,1) + '_output.jpg'
    for i in range(M):
        Image.fromarray(img_mosaic[i,0]).save(img_output_name[i,0])