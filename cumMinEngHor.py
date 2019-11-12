import numpy as np

def cumMinEngHor(e):
  My = np.zeros_like(e)
  Tby = np.zeros_like(e, dtype=int)

  My[:,0] = e[:,0]
  for j in range(1,e.shape[1]):
    for i in range(e.shape[0]):
      if i == 0: prev = [float('inf'), My[i,j-1], My[i+1,j-1]]
      if i == e.shape[0] - 1: prev = [My[i-1,j-1], My[i,j-1], float('inf')]
      else: prev = [My[i-1,j-1], My[i,j-1], My[i+1,j-1]]
      My[i,j] = min(prev) + e[i,j]
      Tby[i, j] = np.argmin(prev)-1
  return My, Tby