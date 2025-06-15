import cv2
import numpy as np

# Load stereo pair
left = cv2.imread('left.png', 0)
right = cv2.imread('right.png', 0)

# Create stereo matcher
stereo = cv2.StereoSGBM_create(minDisparity=0,
                               numDisparities=16*5,
                               blockSize=5,
                               P1=8*3*5**2,
                               P2=32*3*5**2)

# Compute disparity map
disparity = stereo.compute(left, right).astype(np.float32) / 16.0

# Normalize and display
disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('Disparity', disp_vis.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
