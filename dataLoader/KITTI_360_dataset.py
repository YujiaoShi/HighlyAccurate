import random

import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

import torch
import pandas as pd
import utils
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

root_dir = '/mnt/workspace/datasets/kitti-360-SLAM' # '../../data/Kitti' # '../Data' #'..\\Data' #

satmap_dir = 'satmap'
calibration_dir = 'KITTI-360/calibration'
grdimage_dir = 'KITTI-360/data_2d_raw'
left_color_camera_dir = 'image_00/data_rect'  # 'image_02\\data' #
right_color_camera_dir = 'image_01/data_rect'  # 'image_03\\data' #
pose_dir = 'KITTI-360/data_poses'
oxts_dir = 'oxts/data'

GrdImg_H = 256
GrdImg_W = 1024
GrdOriImg_H = 376
GrdOriImg_W = 1408
num_thread_workers = 2

# train_file = './dataLoader/train_kitti_360.txt'
# test_file = './dataLoader/test_kitti_360.txt'
train_file = '../../../dataLoader/kitti_360_train.txt'
test_file = '../../../dataLoader/kitti_360_test.txt'



class SatGrdDataset(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = utils.get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

        # self.shift_range_meters = shift_range  # in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()
        self.file_name = [file[:-1] for file in file_name]

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        file_name = self.file_name[idx]
        # day_dir = file_name[:10]
        # drive_dir = file_name[:38]
        # image_no = file_name[38:]
        drive_dir = file_name[:26]   # 2013_05_28_drive_0000_sync/
        image_no = file_name[46:]    # 0000000000.png

        # goroyeh
        # extrinsics = cam2imu @ imu2world (pose.txt)
        # =================== read camera to imu transform for two front cams and left/right fishcams
        cam2pose_file_name = os.path.join(self.root, calibration_dir, 'calib_cam_to_pose.txt') # From cam to GPS/IMU
        cam2imus = []
        with open(cam2pose_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(':')
                # print(f'read line: {line}')
                values = items[1].strip().split(' ')
                values = [float(val) for val in values]
                
                cam2imu = torch.tensor([
                            [values[0],values[1],values[2], values[3]],
                            [values[4],values[5],values[6], values[7]],
                            [values[8],values[9],values[10], values[11]],
                            [        0,       0,         0,         1]])
                cam2imus.append(cam2imu)

        imu2world_file_name = os.path.join(self.root, pose_dir, drive_dir, 'poses.txt')
        # Get pose.txt raw[idx]
        with open(imu2world_file_name, 'r') as f:
            lines = f.readlines()
            target_row = lines[idx]
            values = target_row.strip().split(' ')
            values = [float(val) for val in values]
            # print(f'target_row {target_row}')
  
            imu2world_matrix = torch.tensor([
                [values[1], values[2], values[3], values[4]],
                [values[5], values[6], values[7], values[8]],
                [values[9], values[10], values[11], values[12]],
                [       0,         0,         0,            1]
            ])

        extrinsics = torch.zeros([4,4,4]) # Goal: (4, 4, 4) 4 cameras, (4x4)
        for i, cam2imu in  enumerate(cam2imus):
            extrinsic = cam2imu @ imu2world_matrix
            extrinsics[i,:,:] = extrinsic
            # print(f'extrinsics: {extrinsic}')

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, calibration_dir, 'perspective.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # print("line = ", line)
                # left color camera k matrix
                if 'P_rect_00' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
                    # if not self.stereo:

                    # print("left_camera_k = ", left_camera_k)
                    break

        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, drive_dir, image_no.lower())
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, pose_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
                content = f.readline().split(' ')
                # get heading
                heading = float(content[5])
                heading = torch.from_numpy(np.asarray(heading))

                left_img_name = os.path.join(self.root, grdimage_dir, drive_dir, left_color_camera_dir,
                                             image_no.lower())
                with Image.open(left_img_name, 'r') as GrdImg:
                    grd_img_left = GrdImg.convert('RGB')
                    if self.grdimage_transform is not None:
                        grd_img_left = self.grdimage_transform(grd_img_left)
                grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)

        sat_rot = sat_map.rotate(-heading / np.pi * 180)
        sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
                                          (1, 0, utils.CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, utils.CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=Image.BILINEAR)
        # the homography is defined on: from target pixel to source pixel
        # now east direction is the real vehicle heading direction

        # randomly generate shift
        gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
        gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction

        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=Image.BILINEAR)

        # randomly generate roation
        theta = np.random.uniform(-1, 1)
        sat_rand_shift_rand_rot = \
            sat_rand_shift.rotate(theta * self.rotation_range)

        sat_map =TF.center_crop(sat_rand_shift_rand_rot, utils.SatMap_process_sidelength)
        # sat_map = np.array(sat_map, dtype=np.float32)

        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        # grd_left_imgs[0] : shape (3, 256, 1024) (C, H, W)
        return sat_map, left_camera_k, grd_left_imgs[0], \
               torch.tensor(-gt_shift_x, dtype=torch.float32).reshape(1), \
               torch.tensor(-gt_shift_y, dtype=torch.float32).reshape(1), \
               torch.tensor(theta, dtype=torch.float32).reshape(1), \
               extrinsics, \
               file_name



class SatGrdDatasetTest(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = utils.get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

        # self.shift_range_meters = shift_range  # in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()
        self.file_name = [file[:-1] for file in file_name]

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        line = self.file_name[idx]
        file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
        drive_dir = file_name[:26]
        image_no = file_name[46:]
        # print("drive_dir = ", drive_dir)
        # print("image_no = ", image_no)

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, calibration_dir, 'perspective.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_00' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
                    # if not self.stereo:
                    break

        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, drive_dir, image_no.lower())
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, pose_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            heading = float(content[5])
            heading = torch.from_numpy(np.asarray(heading))

            left_img_name = os.path.join(self.root, grdimage_dir, drive_dir, left_color_camera_dir,
                                         image_no.lower())
            with Image.open(left_img_name, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB')
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)

        sat_rot = sat_map.rotate(-heading / np.pi * 180)
        sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
                                          (1, 0, utils.CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, utils.CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=Image.BILINEAR)
        # the homography is defined on: from target pixel to source pixel
        # now east direction is the real vehicle heading direction

        # randomly generate shift
        # gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
        # gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction
        gt_shift_x = -float(gt_shift_x)  # --> right as positive, parallel to the heading direction
        gt_shift_y = -float(gt_shift_y)  # --> up as positive, vertical to the heading direction

        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=Image.BILINEAR)

        # randomly generate roation
        # theta = np.random.uniform(-1, 1)
        theta = float(theta)
        sat_rand_shift_rand_rot = \
            sat_rand_shift.rotate(theta * self.rotation_range)

        sat_map = TF.center_crop(sat_rand_shift_rand_rot, utils.SatMap_process_sidelength)
        # sat_map = np.array(sat_map, dtype=np.float32)

        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        return sat_map, left_camera_k, grd_left_imgs[0], \
               torch.tensor(-gt_shift_x, dtype=torch.float32).reshape(1), \
               torch.tensor(-gt_shift_y, dtype=torch.float32).reshape(1), \
               torch.tensor(theta, dtype=torch.float32).reshape(1), \
               file_name


class SatGrdDatasetLocalize(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = utils.get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

        # self.shift_range_meters = shift_range  # in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        self.skip_in_seq = 2  # skip 2 in sequence: 6,3,1~
        if transform != None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()
        self.file_name = [file[:-1] for file in file_name]

    def __len__(self):
        return len(self.file_name)

    def get_file_list(self):
        return self.file_name

    def __getitem__(self, idx):
        '''
        For the localize dataset, we return ground img at t and satellite img at t + 1
        '''
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        line = self.file_name[idx]
        file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
        drive_dir = file_name[:26]
        image_no = file_name[46:]
        image_no_next = f"{int(image_no[:-4])+1:010}" + ".png"

        # # Check image_no
        # print("image_no = ", image_no)
        # image_no_next = f"{int(image_no[:-4])+1:010}" + ".png"
        # print("image_no + 1 = ", image_no_next)

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, calibration_dir, 'perspective.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_00' in line:
                    # get 3*3 matrix from P_rect_**:
                    items = line.split(':')
                    valus = items[1].strip().split(' ')
                    fx = float(valus[0]) * GrdImg_W / GrdOriImg_W
                    cx = float(valus[2]) * GrdImg_W / GrdOriImg_W
                    fy = float(valus[5]) * GrdImg_H / GrdOriImg_H
                    cy = float(valus[6]) * GrdImg_H / GrdOriImg_H
                    left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                    left_camera_k = torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
                    # if not self.stereo:
                    break

        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, drive_dir, image_no_next.lower())
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])

        # oxt: such as 0000000000.txt
        # Use the next image since heading is applied to satellite image (t + 1)
        oxts_file_name = os.path.join(self.root, pose_dir, drive_dir, oxts_dir,
                                      image_no_next.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            heading = float(content[5])
            heading = torch.from_numpy(np.asarray(heading))

            left_img_name = os.path.join(self.root, grdimage_dir, drive_dir, left_color_camera_dir,
                                         image_no.lower())
            with Image.open(left_img_name, 'r') as GrdImg:
                grd_img_left = GrdImg.convert('RGB')
                if self.grdimage_transform is not None:
                    grd_img_left = self.grdimage_transform(grd_img_left)

            grd_left_imgs = torch.cat([grd_left_imgs, grd_img_left.unsqueeze(0)], dim=0)

        sat_rot = sat_map.rotate(-heading / np.pi * 180)
        sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
                                          (1, 0, utils.CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, utils.CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=Image.BILINEAR)
        # the homography is defined on: from target pixel to source pixel
        # now east direction is the real vehicle heading direction

        # randomly generate shift
        # gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
        # gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction

        # leekt: Here I ignored to ground truth shift since we will localize using the ground image from the previous timestamp
        # gt_shift_x = -float(gt_shift_x)  # --> right as positive, parallel to the heading direction
        # gt_shift_y = -float(gt_shift_y)  # --> up as positive, vertical to the heading direction

        # sat_rand_shift = \
        #     sat_align_cam.transform(
        #         sat_align_cam.size, Image.AFFINE,
        #         (1, 0, gt_shift_x * self.shift_range_pixels_lon,
        #          0, 1, -gt_shift_y * self.shift_range_pixels_lat),
        #         resample=Image.BILINEAR)

        # # randomly generate roation
        # # theta = np.random.uniform(-1, 1)

        # theta = float(theta)
        # sat_rand_shift_rand_rot = \
        #     sat_rand_shift.rotate(theta * self.rotation_range)

        # sat_map = TF.center_crop(sat_rand_shift_rand_rot, utils.SatMap_process_sidelength)
        # # sat_map = np.array(sat_map, dtype=np.float32)

        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        return sat_map, left_camera_k, grd_left_imgs[0]
    
        # return sat_map, left_camera_k, grd_left_imgs[0], \
        #        torch.tensor(-gt_shift_x, dtype=torch.float32).reshape(1), \
        #        torch.tensor(-gt_shift_y, dtype=torch.float32).reshape(1), \
        #        torch.tensor(theta, dtype=torch.float32).reshape(1), \
        #        file_name
    

def load_train_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    # cwd = os.getcwd()

    # # Print the current working directory
    # print("Current working directory: {0}".format(cwd))  
    # /home/goroyeh/Yujiao/leekt/HighlyAccurate/outputs/2023-02-21/16-09-48

    train_set = SatGrdDataset(root=root_dir, file=train_file,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)
    return train_loader


def load_test_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    # # Plz keep the following two lines!!! These are for fair test comparison.
    # np.random.seed(2022)
    # torch.manual_seed(2022)

    test1_set = SatGrdDatasetTest(root=root_dir, file=test_file,
                            transform=(satmap_transform, grdimage_transform),
                            shift_range_lat=shift_range_lat,
                            shift_range_lon=shift_range_lon,
                            rotation_range=rotation_range)

    test1_loader = DataLoader(test1_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=num_thread_workers, drop_last=False)
    return test1_loader


# def load_test2_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
#     SatMap_process_sidelength = utils.get_process_satmap_sidelength()

#     satmap_transform = transforms.Compose([
#         transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
#         transforms.ToTensor(),
#     ])

#     Grd_h = GrdImg_H
#     Grd_w = GrdImg_W

#     grdimage_transform = transforms.Compose([
#         transforms.Resize(size=[Grd_h, Grd_w]),
#         transforms.ToTensor(),
#     ])

#     # # Plz keep the following two lines!!! These are for fair test comparison.
#     # np.random.seed(2022)
#     # torch.manual_seed(2022)

#     test2_set = SatGrdDatasetTest(root=root_dir, file=test2_file,
#                               transform=(satmap_transform, grdimage_transform),
#                               shift_range_lat=shift_range_lat,
#                               shift_range_lon=shift_range_lon,
#                               rotation_range=rotation_range)

#     test2_loader = DataLoader(test2_set, batch_size=batch_size, shuffle=False, pin_memory=True,
#                               num_workers=num_thread_workers, drop_last=False)
#     return test2_loader

def load_localize_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
    ])

    Grd_h = GrdImg_H
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])

    # # Plz keep the following two lines!!! These are for fair test comparison.
    # np.random.seed(2022)
    # torch.manual_seed(2022)

    localize_set = SatGrdDatasetLocalize(root=root_dir, file=test_file,
                            transform=(satmap_transform, grdimage_transform),
                            shift_range_lat=shift_range_lat,
                            shift_range_lon=shift_range_lon,
                            rotation_range=rotation_range)

    localize_loader = DataLoader(localize_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=num_thread_workers, drop_last=False)
    return localize_loader





