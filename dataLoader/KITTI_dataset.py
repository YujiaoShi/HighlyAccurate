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

root_dir = '/media/yujiao/6TB/dataset/Kitti1' # '../../data/Kitti' # '../Data' #'..\\Data' #

test_csv_file_name = 'test.csv'
ignore_csv_file_name = 'ignore.csv'
satmap_dir = 'satmap'
grdimage_dir = 'raw_data'
left_color_camera_dir = 'image_02/data'  # 'image_02\\data' #
right_color_camera_dir = 'image_03/data'  # 'image_03\\data' #
oxts_dir = 'oxts/data'  # 'oxts\\data' #

GrdImg_H = 256  # 256 # original: 375 #224, 256
GrdImg_W = 1024  # 1024 # original:1242 #1248, 1024
GrdOriImg_H = 375
GrdOriImg_W = 1242
num_thread_workers = 2

# train_file = './dataLoader/train_files.txt'
train_file = './dataLoader/train_files.txt'
test1_file = './dataLoader/test1_files.txt'
test2_file = './dataLoader/test2_files.txt'



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
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_02' in line:
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
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])
        image_no = file_name[38:]

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
                content = f.readline().split(' ')
                # get heading
                heading = float(content[5])
                heading = torch.from_numpy(np.asarray(heading))

                left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
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

        return sat_map, left_camera_k, grd_left_imgs[0], \
               torch.tensor(-gt_shift_x, dtype=torch.float32).reshape(1), \
               torch.tensor(-gt_shift_y, dtype=torch.float32).reshape(1), \
               torch.tensor(theta, dtype=torch.float32).reshape(1), \
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
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # =================== read camera intrinsice for left and right cameras ====================
        calib_file_name = os.path.join(self.root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
        with open(calib_file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # left color camera k matrix
                if 'P_rect_02' in line:
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
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        # =================== initialize some required variables ============================
        grd_left_imgs = torch.tensor([])

        # oxt: such as 0000000000.txt
        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            # get heading
            heading = float(content[5])
            heading = torch.from_numpy(np.asarray(heading))

            left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
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

    train_set = SatGrdDataset(root=root_dir, file=train_file,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)
    return train_loader


def load_test1_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
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

    test1_set = SatGrdDatasetTest(root=root_dir, file=test1_file,
                            transform=(satmap_transform, grdimage_transform),
                            shift_range_lat=shift_range_lat,
                            shift_range_lon=shift_range_lon,
                            rotation_range=rotation_range)

    test1_loader = DataLoader(test1_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                            num_workers=num_thread_workers, drop_last=False)
    return test1_loader


def load_test2_data(batch_size, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
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

    test2_set = SatGrdDatasetTest(root=root_dir, file=test2_file,
                              transform=(satmap_transform, grdimage_transform),
                              shift_range_lat=shift_range_lat,
                              shift_range_lon=shift_range_lon,
                              rotation_range=rotation_range)

    test2_loader = DataLoader(test2_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)
    return test2_loader







