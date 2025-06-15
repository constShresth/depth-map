from tarfile import BLOCKSIZE
import numpy as np 
import cv2 as cv
from matplotlib import pyplot as plt  
import os

from scipy.__config__ import show

class DepthMap:
  def __init__(self, showImages) :
    # Load Images
    root = os.getcwd()
    imgLeftPath = os.path.join(root, 'left.png')
    imgRightPath = os.path.join(root, 'right.png')
    self.imgLeft = cv.imread(imgLeftPath, cv.IMREAD_GRAYSCALE)
    self.imgRight = cv.imread(imgRightPath, cv.IMREAD_GRAYSCALE)
    if showImages:
      plt.figure()
      plt.subplot (121)
      plt.imshow(self.imgLeft)
      plt.subplot (122)
      plt.imshow(self.imgRight)
      plt.show()


  def computeDepthMapSGBM(self) :
    # window_size = 7
    # min_disp =16
    # nDispFactor = 14 # adjust this (14 is good)
    # num_disp = 16*nDispFactor-min_disp
    blockSize = 5
    stereo = cv.StereoSGBM_create(
      minDisparity = 0,
      numDisparities = 128,  # must be divisible by 16, try 64 or 128
      blockSize = 5  ,       # between 3 and 11; 5 is a good start
      P1 = 8 * 3 * blockSize**2,
      P2 = 32 * 3 * blockSize**2,
      disp12MaxDiff = 1,
      uniquenessRatio = 10,
      speckleWindowSize = 100,
      speckleRange = 2,
      preFilterCap=63,
      mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)



    # Compute disparity map
    disparity = stereo.compute(self.imgLeft, self.imgRight).astype(np.
    float32) / 16.0
    
    # Display the disparity map
    plt.imshow(disparity, 'gray')
    plt.colorbar ()
    plt.show()



def demoStereoSGBM():
  dp = DepthMap(showImages=False)
  dp.computeDepthMapSGBM()

# if __name__== '_main_':
  # demoStereoBM()

demoStereoSGBM()