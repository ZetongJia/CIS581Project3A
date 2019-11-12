import numpy as np
from cumMinEngHor import cumMinEngHor

def cumMinEngVer(e):
  My, Tby = cumMinEngHor(np.transpose(e))
  return np.transpose(My), np.transpose(Tby)