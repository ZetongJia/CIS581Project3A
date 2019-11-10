'''
  File name: ransac_est_homography.py
  Author:
  Date created:
'''

'''
  File clarification:
    Use a robust method (RANSAC) to compute a homography. Use 4-point RANSAC as 
    described in class to compute a robust homography estimate:
    - Input x1, y1, x2, y2: N × 1 vectors representing the correspondences feature coordinates in the first and second image. 
                            It means the point (x1_i , y1_i) in the first image are matched to (x2_i , y2_i) in the second image.
    - Input thresh: the threshold on distance used to determine if transformed points agree.
    - Outpuy H: 3 × 3 matrix representing the homograph matrix computed in final step of RANSAC.
    - Output inlier_ind: N × 1 vector representing if the correspondence is inlier or not. 1 means inlier, 0 means outlier.
'''

import numpy as np
from est_homography import est_homography

def ransac_est_homography(x1, y1, x2, y2, thresh):
  # Your Code Here
  ninlier = 0
  iteration = 1000
  min_consensus = 0.95
  fpoints = 4
  npoints = len(x1)
  src_P = np.hstack((x1,y1,np.ones((npoints,1),dtype = np.int32))).transpose() 
  tar_P = np.hstack((x2,y2,np.ones((npoints,1),dtype = np.int32))).transpose()
  for i in range(iteration):
      rd = [np.random.randint(0, npoints) for __ in range(fpoints)]

      src_x, src_y = x1[rd], y1[rd]
      tar_x, tar_y = x2[rd], y2[rd]
      h = est_homography(src_x, src_y, tar_x, tar_y)
#      h = est_homography(tar_x, tar_y, src_x, src_y)
      src_warp = np.dot(h,src_P)
      src_warp[2,src_warp[2,:]==0] = 1e-10
      src_warp[0,:] = src_warp[0,:] / src_warp[2,:]
      src_warp[1,:] = src_warp[1,:] / src_warp[2,:]
      error = np.sqrt((src_warp[0,:] - tar_P[0,:])**2 + (src_warp[1,:] - tar_P[1,:])**2)
      n = (error < thresh).sum()
      if n >= min_consensus * npoints:
          H = h
          inlier_ind,_ = np.where(error.reshape(-1,1) < thresh)
          break
      elif n > ninlier:
          ninlier = n
          H = h
          inlier_ind,_ = np.where(error.reshape(-1,1) < thresh)
  return H, inlier_ind