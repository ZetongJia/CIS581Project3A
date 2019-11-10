'''
  File name: feat_match.py
  Author:
  Date created:
'''

'''
  File clarification:
    Matching feature descriptors between two images. You can use k-d tree to find the k nearest neighbour. 
    Remember to filter the correspondences using the ratio of the best and second-best match SSD. You can set the threshold to 0.6.
    - Input descs1: 64 × N1 matrix representing the corner descriptors of first image.
    - Input descs2: 64 × N2 matrix representing the corner descriptors of second image.
    - Outpuy match: N1 × 1 vector where match i points to the index of the descriptor in descs2 that matches with the
                    feature i in descriptor descs1. If no match is found, you should put match i = −1.
'''
import pyflann
import numpy as np

from helpers import dist2
from helpers import sort2

def feat_match(descs1, descs2):
#  [c,N1] = descs1.shape
#  flann = pyflann.FLANN()
#  result, dists = flann.nn(
#      np.transpose(descs2), np.transpose(descs1), 2, algorithm="kmeans", branching=32, iterations=7, checks=16)
#  match = np.zeros((N1, 1))
#  match = result[:,0]
#  match[dists[:,0]/dists[:,1] > 0.8] = -1
    dist = dist2(descs1,descs2)
    ord_dist,index = sort2(dist)
    ratio = ord_dist[:,0]/(ord_dist[:,1] + 1e-10)
    thresh = 0.5
    match = index[:,0].astype(np.int32)
    match[ratio > thresh] = -1

    return match