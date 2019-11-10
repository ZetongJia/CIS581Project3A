'''
  File name: est_homography.py
  Author: Haoyuan(Steve) Zhang
  Date created: 10/15/2017
'''

'''
  File clarification:
    Estimate homography for source and target image, given correspondences in source image (x, y) and target image (X, Y) respectively
    - Input x, y: the coordinates of source correspondences
    - Input X, Y: the coordinates of target correspondences
      (X/Y/x/y , each is a numpy array of n x 1, n >= 4)

    - Output H: the homography output which is 3 x 3
      ([X,Y,1]^T ~ H * [x,y,1]^T)
'''

import numpy as np
import scipy

def est_homography(x, y, X, Y):
  N = x.size
  A = np.zeros([2 * N, 8])
  
#  x0,y0,X0,Y0 = x.reshape(-1),y.reshape(-1),X.reshape(-1),y.reshape(-1)
  A[0::2,0:3] = np.hstack((x,y,np.ones((N,1),dtype=np.int32)))
  A[1::2,3:6] = np.hstack((x,y,np.ones((N,1),dtype=np.int32)))
  A[0::2,6:8] = np.hstack((-x*X,-y*X))
  A[1::2,6:8] = np.hstack((-x*Y,-y*Y))
  
  B = np.hstack((X,Y)).reshape(-1,1)
  
  if np.linalg.det(A) != 0:
      h = np.linalg.solve(A,B)
      H = np.vstack((h,1)).reshape(3,3)
  else:
      H = np.identity(3)        # A is noninvertable  


#  i = 0
#  while i < N:
#    a = np.hstack((x[i], y[i], 1)).reshape(-1, 3).astype(np.int32)
#    c = np.vstack((X[i], Y[i])).astype(np.int32)
#    d = - c * a
#
#    A[2 * i, 0 : 3], A[2 * i + 1, 3 : 6]= a, a
#    A[2 * i : 2 * i + 2, 6 : ] = d
#
#    i += 1
#  
#  # compute the solution of A
#  U, s, V = np.linalg.svd(A, full_matrices=True)
#  h = V[8, :]
#  H = h.reshape(3, 3)

  return H