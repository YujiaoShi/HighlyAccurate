import os
import numpy as np

TRAIN_FILE = "/mnt/workspace/users/leekt/HighlyAccurate/dataLoader/kitti_360_train.txt"
# TRAIN_START = 0
# TRAIN_END = 99
TRAIN_INTERVALS = [(0, 99), (5000, 5099)]


TEST_FILE = "/mnt/workspace/users/leekt/HighlyAccurate/dataLoader/kitti_360_test.txt"
# TEST_START = 100
# TEST_END = 199
TEST_INTERVALS = [(100, 199), (5100, 5199)]
TEST_NOISE_SCALE = 1

DRIVE_DIR = "2013_05_28_drive_0000_sync"
CAMERA_DIRS = ["image_01/data_rect"]

if __name__ == '__main__':
    np.random.seed(2023)

    # Generate train data
    with open(TRAIN_FILE, 'w') as f:
        for camera_dir in CAMERA_DIRS:
            # for i in range(TRAIN_START, TRAIN_END + 1):
            for start, end in TRAIN_INTERVALS:
                for i in range(start, end + 1):
                    img_index = f"{i:010}" + ".png\n"
                    content = os.path.join(DRIVE_DIR, camera_dir, img_index)
                    f.writelines(content)

    with open(TEST_FILE, 'w') as f:
        for camera_dir in CAMERA_DIRS:
            # for i in range(TEST_START, TEST_END + 1):
            for start, end in TEST_INTERVALS:
                for i in range(start, end + 1):
                    gt_shift_x = np.random.uniform(-TEST_NOISE_SCALE,
                                                TEST_NOISE_SCALE)
                    gt_shift_y = np.random.uniform(-TEST_NOISE_SCALE,
                                                TEST_NOISE_SCALE)
                    gt_shift_theta = np.random.uniform(
                        -TEST_NOISE_SCALE, TEST_NOISE_SCALE)
                    img_index = f"{i:010}" + ".png"
                    shift_data = f" {gt_shift_x:.4f} {gt_shift_y:.4f} {gt_shift_theta:.4f}\n"
                    content = os.path.join(DRIVE_DIR, camera_dir, img_index) + shift_data 
                    f.writelines(content)
