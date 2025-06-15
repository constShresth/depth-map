import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

class DepthMap:
    def __init__(self, show_images=False):
        # Load stereo image pair
        root = os.getcwd()
        img_left_path = os.path.join(root, 'left.png')
        img_right_path = os.path.join(root, 'right.png')
        self.imgLeft = cv.imread(img_left_path, cv.IMREAD_GRAYSCALE)
        self.imgRight = cv.imread(img_right_path, cv.IMREAD_GRAYSCALE)

        if show_images:
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.imgLeft, cmap='gray')
            plt.title("Left Image")
            plt.subplot(122)
            plt.imshow(self.imgRight, cmap='gray')
            plt.title("Right Image")
            plt.show()

    def compute_depth_map_sgbm(self, show_disparity=True):
        block_size = 5
        stereo = cv.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
        )

        disparity = stereo.compute(self.imgLeft, self.imgRight).astype(np.float32) / 16.0

        # Convert disparity to depth map
        f = 721.5377  # focal length (pixels)
        B = 0.54      # baseline (meters)
        depth = (f * B) / (disparity + 1e-6)  # depth in meters

        # Filter noise with median blur
        depth_filtered = cv.medianBlur(depth.astype(np.uint8), 5)

        if show_disparity:
            plt.figure(figsize=(10, 5))
            plt.imshow(depth_filtered, cmap='gray')
            plt.title("Filtered Depth Map")
            plt.colorbar(label='Relative Depth')
            plt.show()

        # Analyze zones
        self.analyze_zones(depth_filtered)

    def analyze_zones(self, depth_map):
      h, w = depth_map.shape
      h_start = int(h * 0.2)
      h_end = int(h * 0.8)

      left_zone = depth_map[h_start:h_end, :w // 3]
      center_zone = depth_map[h_start:h_end, w // 3:2 * w // 3]
      right_zone = depth_map[h_start:h_end, 2 * w // 3:]

      def robust_median(zone):
          zone_valid = zone[(zone > 0.5) & (zone < 8.0)]
          return np.percentile(zone_valid, 25) if zone_valid.size > 0 else float('inf')

      avg_left = robust_median(left_zone)
      avg_center = robust_median(center_zone)
      avg_right = robust_median(right_zone)

      print(f"Average Distances (meters): Left: {avg_left:.2f}, Center: {avg_center:.2f}, Right: {avg_right:.2f}")

      threshold = 4.0  # More sensitive now
      if avg_center < threshold:
          direction = "Move Left" if avg_left > avg_right else "Move Right"
      else:
          direction = "Move Straight"

      print(f"Recommended Action: {direction}")


def demo_stereo_sgbm():
    dp = DepthMap(show_images=False)
    dp.compute_depth_map_sgbm()


# Run the demo
demo_stereo_sgbm()
