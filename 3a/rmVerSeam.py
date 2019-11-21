import numpy as np

def rmVerSeam(I, Mx, Tbx):
  [x, y, ch] = I.shape
  E = Mx.min(1)[x-1]

  Ix = np.zeros((x, y-1, ch))
  idx = np.argmin(Mx[x-1,:])
  for j in range(1,x+1):
    c = x - j
    Ix[c, 0:idx, :] = I[c, 0:idx, :]
    Ix[c, idx:y-1, :] = I[c, idx+1:, :]
    idx += Tbx[c, idx]
    Ix = Ix.astype('uint8')
  return Ix, E

def getVerSeam(I, Mx, Tbx):
  [x, y, ch] = I.shape
  E = Mx.min(1)[x-1]

  idxs = np.zeros((x))
  idx = np.argmin(Mx[x-1,:])
  idxs[-1] = idx
  for j in range(1,x+1):
    c = x - j
    idxs[c] = idx
    idx += Tbx[c, idx]
  return idxs

def rmHorSeam(I, My, Tby):
  [x, y, ch] = I.shape
  E = My.min(0)[y-1]

  Iy = np.zeros((x-1, y, ch))
  idx = np.argmin(My[x-1,:])
  for j in range(1,y+1):
    c = y - j
    Iy[0:idx, c, :] = I[0:idx, c, :]
    Iy[idx:x-1, c, :] = I[idx+1:x, c, :]
    idx += Tby[idx, c]
    Iy = Iy.astype('uint8')
  return Iy, E

def getHorSeam(I, My, Tby):
  [x, y] = I.shape
  E = My.min(0)[y-1]

  idxs = np.zeros((y), dtype='int')
  idx = np.argmin(My[x-1,:])
  idxs[-1] = idx
  for j in range(1,y+1):
    c = y - j
    idxs[c] = idx
    idx += Tby[c, idx]
  return idxs