#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os

import cv2
import torchvision.utils

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
# from dataLoader.DataLoad import load_train_data, load_test_data, load_val_data
from dataLoader.Ford_dataset import SatGrdDatasetFord, SatGrdDatasetFordTest, train_logs, train_logs_img_inds, test_logs, test_logs_img_inds
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import scipy.io as scio

import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights

from models_ford import LM_S2GP_Ford

import numpy as np
import os
import argparse

from torch.utils.data import DataLoader
import time

from utils import gps2distance
torch.autograd.set_detect_anomaly(True)

########################### ranking test ############################
def test1(net, args, save_path, best_rank_result, test_log_ind=1, epoch=0, device=torch.device("cuda:0")):
    ### net evaluation state
    net.eval()
    mini_batch = args.batch_size

    # Plz keep the following two lines!!! These are for fair test comparison.
    np.random.seed(2022)
    torch.manual_seed(2022)

    test_set = SatGrdDatasetFordTest(logs=test_logs[test_log_ind:test_log_ind+1],
                                 logs_img_inds=test_logs_img_inds[test_log_ind:test_log_ind+1],
                                  shift_range_lat=args.shift_range_lat, shift_range_lon=args.shift_range_lon,
                                  rotation_range=args.rotation_range, whole=args.test_whole)
    testloader = DataLoader(test_set, batch_size=mini_batch, shuffle=False, pin_memory=True,
                             num_workers=2, drop_last=False)

    satmap_sidelength_meters = test_set.satmap_sidelength_meters

    pred_shifts = []
    pred_headings = []
    gt_shifts = []
    gt_headings = []

    start_time = time.time()
    for i, data in enumerate(testloader, 0):
        sat_map, grd_img, gt_shift_u, gt_shift_v, gt_heading, R_FL, T_FL = \
            [item.to(device) for item in data[:-1]]

        shifts_u, shifts_v, theta = \
            net(sat_map, grd_img, satmap_sidelength_meters, R_FL, T_FL,
                mode='test', level_first=args.level_first)
        # shifts: [B, 2]
        # headings: [B, 1]

        shifts = torch.stack([shifts_u, shifts_v], dim=-1)
        headings = theta.unsqueeze(dim=-1)

        gt_shift = torch.stack([gt_shift_u, gt_shift_v], dim=-1)  # [B, 2]

        loss = torch.mean(shifts_u - gt_shift_u)
        loss.backward()  # just to release graph

        pred_shifts.append(shifts.data.cpu().numpy())
        pred_headings.append(headings.data.cpu().numpy())
        gt_shifts.append(gt_shift.data.cpu().numpy())
        gt_headings.append(gt_heading.reshape(-1, 1).data.cpu().numpy())

        if i % 20 == 0:
            print(i)

    end_time = time.time()
    duration = (end_time - start_time)/len(testloader)

    pred_shifts = np.concatenate(pred_shifts, axis=0) * np.array([args.shift_range_lat, args.shift_range_lon]).reshape(1, 2)
    pred_headings = np.concatenate(pred_headings, axis=0) * args.rotation_range
    gt_shifts = np.concatenate(gt_shifts, axis=0) * np.array([args.shift_range_lat, args.shift_range_lon]).reshape(1, 2)
    gt_headings = np.concatenate(gt_headings, axis=0) * args.rotation_range

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    scio.savemat(os.path.join(save_path, str(test_log_ind) + '_result.mat'), {'gt_shifts': gt_shifts, 'gt_headings': gt_headings,
                                                         'pred_shifts': pred_shifts, 'pred_headings': pred_headings})

    distance = np.sqrt(np.sum((pred_shifts - gt_shifts) ** 2, axis=1))  # [N]
    angle_diff = np.remainder(np.abs(pred_headings - gt_headings), 360)
    idx0 = angle_diff > 180
    angle_diff[idx0] = 360 - angle_diff[idx0]
    # angle_diff = angle_diff.numpy()

    init_dis = np.sqrt(np.sum(gt_shifts ** 2, axis=1))
    init_angle = np.abs(gt_headings)

    metrics = [1, 3, 5]
    angles = [1, 3, 5]

    f = open(os.path.join(save_path, str(test_log_ind) + '_results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    f.write('Time per image (second): ' + str(duration) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    print(str(test_log_ind) + ' Validation results:')
    print('Init distance average: ', np.mean(init_dis))
    print('Pred distance average: ', np.mean(distance))
    print('Init angle average: ', np.mean(init_angle))
    print('Pred angle average: ', np.mean(angle_diff))
    dis_recalls = []
    angle_recalls = []
    init_dis_recalls = []
    init_angle_recalls = []
    for idx in range(len(metrics)):
        dis_recalls.append(np.sum(distance < metrics[idx]) / distance.shape[0] * 100)
        print('within ' + str(metrics[idx]) + ' meters pred: ' + str(dis_recalls[idx]))
        f.write('within ' + str(metrics[idx]) + ' meters pred: ' + str(dis_recalls[idx]) + '\n')

        init_dis_recalls.append(np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100)
        print('within ' + str(metrics[idx]) + ' meters init: ' + str(init_dis_recalls[idx]))
        f.write('within ' + str(metrics[idx]) + ' meters init: ' + str(init_dis_recalls[idx]) + '\n')

    print('-------------------------')
    f.write('------------------------\n')
    x_recalls = []
    init_x_recalls = []
    y_recalls = []
    init_y_recalls = []
    diff_shifts = np.abs(pred_shifts - gt_shifts)
    for idx in range(len(metrics)):
        x_recalls.append(np.sum(diff_shifts[:, 0] < metrics[idx]) / diff_shifts.shape[0] * 100)
        print('lateral within ' + str(metrics[idx]) + ' meters pred: ' + str(x_recalls[idx]))
        f.write('lateral within ' + str(metrics[idx]) + ' meters pred: ' + str(x_recalls[idx]) + '\n')
        init_x_recalls.append(np.sum(np.abs(gt_shifts[:, 0]) < metrics[idx]) / init_dis.shape[0] * 100)
        print('lateral within ' + str(metrics[idx]) + ' meters init: ' + str(init_x_recalls[idx]))
        f.write('lateral within ' + str(metrics[idx]) + ' meters init: ' + str(init_x_recalls[idx]) + '\n')

        y_recalls.append(np.sum(diff_shifts[:, 1] < metrics[idx]) / diff_shifts.shape[0] * 100)
        print('longitudinal within ' + str(metrics[idx]) + ' meters pred: ' + str(y_recalls[idx]))
        f.write('longitudinal within ' + str(metrics[idx]) + ' meters pred: ' + str(y_recalls[idx]) + '\n')
        init_y_recalls.append(np.sum(np.abs(gt_shifts[:, 1]) < metrics[idx]) / diff_shifts.shape[0] * 100)
        print('longitudinal within ' + str(metrics[idx]) + ' meters init: ' + str(init_y_recalls[idx]))
        f.write('longitudinal within ' + str(metrics[idx]) + ' meters init: ' + str(init_y_recalls[idx]) + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(angles)):
        angle_recalls.append(np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100)
        print('within ' + str(angles[idx]) + ' degrees pred: ' + str(angle_recalls[idx]))
        f.write('within ' + str(angles[idx]) + ' degrees pred: ' + str(angle_recalls[idx]) + '\n')

        init_angle_recalls.append(np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100)
        print('within ' + str(angles[idx]) + ' degrees init: ' + str(init_angle_recalls[idx]))
        f.write('within ' + str(angles[idx]) + ' degrees init: ' + str(init_angle_recalls[idx]) + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()
    result = np.sum((distance < metrics[2]) & (angle_diff < angles[0])) / distance.shape[0] * 100

    net.train()

    ### save the best params
    if (result > best_rank_result):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(net.state_dict(), os.path.join(save_path, 'Model_best.pth'))

    return result


###### learning criterion assignment #######
def train(lr, args, save_path, train_log_start=1, train_log_end=2):
    bestRankResult = 0.0  # current best
    # loop over the dataset multiple times
    for epoch in range(args.resume, args.epochs):
        net.train()

        # base_lr = 0
        base_lr = lr
        base_lr = base_lr * ((1.0 - float(epoch) / 100.0) ** (1.0))

        print(base_lr)

        optimizer = optim.Adam(net.parameters(), lr=base_lr)

        optimizer.zero_grad()

        ### feeding A and P into train loader
        train_set = SatGrdDatasetFord(logs=train_logs[train_log_start:train_log_end],
                                      logs_img_inds=train_logs_img_inds[train_log_start:train_log_end],
                 shift_range_lat=args.shift_range_lat, shift_range_lon=args.shift_range_lon,
                                     rotation_range=args.rotation_range, whole=args.train_whole)
        trainloader = DataLoader(train_set, batch_size=mini_batch, shuffle=(args.visualize==0), pin_memory=True,
                              num_workers=1, drop_last=False)

        satmap_sidelength_meters = train_set.satmap_sidelength_meters

        loss_vec = []

        print('batch_size:', mini_batch, '\n num of batches:', len(trainloader))

        for Loop, Data in enumerate(trainloader, 0):
            # get the inputs
            sat_map, grd_img, gt_shift_u, gt_shift_v, theta, R_FL, T_FL = \
                [item.to(device) for item in Data[:-1]]
            file_name = Data[-1]

            # zero the parameter gradients
            optimizer.zero_grad()

            if args.estimate_depth:
                loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                shift_lat_last, shift_lon_last, theta_last, \
                L1_loss, L2_loss, L3_loss, L4_loss, grd_conf_list, grd_depth_list = \
                    net(sat_map, grd_img, satmap_sidelength_meters, R_FL, T_FL, gt_shift_u, gt_shift_v, theta,
                        mode='train', file_name=file_name, level_first=args.level_first)
            else:
                loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                shift_lat_last, shift_lon_last, theta_last, \
                L1_loss, L2_loss, L3_loss, L4_loss, grd_conf_list = \
                    net(sat_map, grd_img, satmap_sidelength_meters, R_FL, T_FL, gt_shift_u, gt_shift_v, theta,
                        mode='train', file_name=file_name, level_first=args.level_first, loop=Loop)

            # if 1:
            #     from visualize import show_cam_on_image
            #     from PIL import Image
            #     for conf_idx, grd_conf in enumerate(grd_conf_list):
            #         img_conf = show_cam_on_image(grd_img[0].permute(1, 2, 0).data.cpu().numpy(),
            #                                      grd_conf[0, 0].data.cpu().numpy(), use_rgb=True)
            #         Image.fromarray(img_conf).save('./visualize/conf' + str(Loop) + '_' + str(conf_idx) + '.png')
            #
            #     conf0 = cv2.resize(grd_conf_list[0][0, 0].data.cpu().numpy(), (grd_img.shape[-1], grd_img.shape[-2]))
            #     conf1 = cv2.resize(grd_conf_list[1][0, 0].data.cpu().numpy(), (grd_img.shape[-1], grd_img.shape[-2]))
            #     conf2 = cv2.resize(grd_conf_list[2][0, 0].data.cpu().numpy(), (grd_img.shape[-1], grd_img.shape[-2]))
            #
            #     conf = (conf0 * conf1 * conf2)
            #     conf = (conf - np.min(conf))/(np.max(conf) - np.min(conf)) * 255
            #     import matplotlib.pyplot as plt
            #     fig, ax = plt.subplots()
            #     im = ax.imshow(conf, cmap='jet')
            #     from mpl_toolkits.axes_grid1 import make_axes_locatable
            #     divider = make_axes_locatable(ax)
            #     cax = divider.append_axes("right", size="5%", pad=0.05)
            #     plt.colorbar(im, cax)
            #     ax.get_yaxis().set_ticks([])
            #     ax.get_xaxis().set_ticks([])
            #     ax.set_axis_off()
            #     fig.tight_layout()
            #     plt.show()
            #     # Image.fromarray(conf.astype(np.uint8)).convert('RGB').save('./visualize/conf' + str(Loop) + '.png')

            loss.backward()

            optimizer.step()  # This step is responsible for updating weights
            optimizer.zero_grad()

            ### record the loss
            loss_vec.append(loss.item())

            if Loop % 10 == 9:  #
                level = args.level - 1
                # for level in range(len(shifts_decrease)):
                print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) + ' Delta: Level-' + str(level) +
                      ' loss: ' + str(np.round(loss_decrease[level].item(), decimals=4)) +
                      ' lat: ' + str(np.round(shift_lat_decrease[level].item(), decimals=2)) +
                      ' lon: ' + str(np.round(shift_lon_decrease[level].item(), decimals=2)) +
                      ' rot: ' + str(np.round(thetas_decrease[level].item(), decimals=2)))

                if args.loss_method == 3:
                    print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) + ' Last: Level-' + str(level) +
                          ' loss: ' + str(np.round(loss_last[level].item(), decimals=4)) +
                          ' lat: ' + str(np.round(shift_lat_last[level].item(), decimals=2)) +
                          ' lon: ' + str(np.round(shift_lon_last[level].item(), decimals=2)) +
                          ' rot: ' + str(np.round(theta_last[level].item(), decimals=2)) +
                          ' L1: ' + str(np.round(torch.sum(L1_loss).item(), decimals=2)) +
                          ' L2: ' + str(np.round(torch.sum(L2_loss).item(), decimals=2)) +
                          ' L3: ' + str(np.round(torch.sum(L3_loss).item(), decimals=2)) +
                          ' L4: ' + str(np.round(torch.sum(L4_loss).item(), decimals=2)))
                elif args.loss_method == 1 or args.loss_method == 2:
                    print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) + ' Last: Level-' + str(level) +
                          ' loss: ' + str(np.round(loss_last[level].item(), decimals=4)) +
                          ' lat: ' + str(np.round(shift_lat_last[level].item(), decimals=4)) +
                          ' lon: ' + str(np.round(shift_lon_last[level].item(), decimals=4)) +
                          ' rot: ' + str(np.round(theta_last[level].item(), decimals=4)) +
                          ' L1: ' + str(np.round(torch.sum(L1_loss).item(), decimals=2)))
                else:
                    print('Epoch: ' + str(epoch) + ' Loop: ' + str(Loop) + ' Last: Level-' + str(level) +
                          ' loss: ' + str(np.round(loss_last[level].item(), decimals=4)) +
                          ' lat: ' + str(np.round(shift_lat_last[level].item(), decimals=2)) +
                          ' lon: ' + str(np.round(shift_lon_last[level].item(), decimals=2)) +
                          ' rot: ' + str(np.round(theta_last[level].item(), decimals=2))
                          )

                # writer.add_scalar('training loss', loss,
                #                   epoch * len(trainloader) + Loop)
                # for level_i, grd_depth in enumerate(grd_depth_list):
                #     depth_grid = torchvision.utils.make_grid(grd_depth, normalize=True)
                #     writer.add_image('grd_depth_level_' + str(level_i), depth_grid)

                # grd_img_grid = torchvision.utils.make_grid(grd_img, normalize=True)
                # writer.add_image('grd_img', grd_img_grid)
                # del loss, loss_decrease, shifts_decrease, heading_decrease, loss_last, shifts_last, heading_last, \
                #     sat_map, left_camera_k, grd_left_imgs, gt_shift, gt_heading

        ### save modelget_similarity_fn
        compNum = epoch % 100
        print('taking snapshot ...')
        # save_path = save_path + '_Start' + str(train_log_start) + '_End' + str(train_log_end)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(net.state_dict(), os.path.join(save_path, 'model_' + str(compNum) + '.pth'))

        ### ranking test
        current = test1(net, args, save_path, bestRankResult, test_log_ind=train_log_start, epoch=epoch)
        if (current > bestRankResult):
            bestRankResult = current

        # test1(args, save_path, bestRankResult, test_log_ind=0)

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')
    parser.add_argument('--debug', type=int, default=0, help='debug to dump middle processing images')

    parser.add_argument('--epochs', type=int, default=2, help='number of training epochs')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  # 1e-2

    parser.add_argument('--stereo', type=int, default=0, help='use left and right ground image')
    parser.add_argument('--sequence', type=int, default=1, help='use n images merge to 1 ground image')

    parser.add_argument('--rotation_range', type=float, default=10., help='degree')
    parser.add_argument('--shift_range_lat', type=float, default=20., help='meters')
    parser.add_argument('--shift_range_lon', type=float, default=20., help='meters')

    parser.add_argument('--coe_shift_lat', type=float, default=100., help='meters')
    parser.add_argument('--coe_shift_lon', type=float, default=100., help='meters')
    parser.add_argument('--coe_heading', type=float, default=100., help='degree')
    parser.add_argument('--coe_L1', type=float, default=100., help='feature')
    parser.add_argument('--coe_L2', type=float, default=100., help='meters')
    parser.add_argument('--coe_L3', type=float, default=100., help='degree')
    parser.add_argument('--coe_L4', type=float, default=100., help='feature')

    parser.add_argument('--metric_distance', type=float, default=5., help='meters')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--loss_method', type=int, default=0, help='0, 1, 2, 3')

    parser.add_argument('--level', type=int, default=3, help='2, 3, 4, -1, -2, -3, -4')
    parser.add_argument('--N_iters', type=int, default=5, help='any integer')
    parser.add_argument('--using_weight', type=int, default=0, help='weighted LM or not')
    parser.add_argument('--damping', type=float, default=0.1, help='coefficient in LM optimization')
    parser.add_argument('--train_damping', type=int, default=0, help='coefficient in LM optimization')

    # parameters below are used for the first-step metric learning traning
    parser.add_argument('--negative_samples', type=int, default=32, help='number of negative samples '
                                                                         'for the metric learning training')
    parser.add_argument('--use_conf_metric', type=int, default=0, help='0  or 1 ')

    parser.add_argument('--direction', type=str, default='S2GP', help='G2SP' or 'S2GP')
    parser.add_argument('--Load', type=int, default=0, help='0 or 1, load_metric_learning_weight or not')
    parser.add_argument('--Optimizer', type=str, default='LM', help='LM or SGD')

    parser.add_argument('--train_log_start', type=int, default=0, help='')
    parser.add_argument('--train_log_end', type=int, default=1, help='')
    parser.add_argument('--test_log_ind', type=int, default=0, help='')

    parser.add_argument('--transformer', type=int, default=0, help='0 or 1, use or not use transformer')
    parser.add_argument('--estimate_depth', type=int, default=0, help='0 or 1, estimate grd depth or not')

    parser.add_argument('--level_first', type=int, default=0, help='0 or 1, estimate grd depth or not')
    parser.add_argument('--proj', type=str, default='geo', help='geo, polar, nn')
    parser.add_argument('--use_gt_depth', type=int, default=0, help='0 or 1')

    parser.add_argument('--dropout', type=int, default=0, help='0 or 1')
    parser.add_argument('--use_hessian', type=int, default=0, help='0 or 1')

    parser.add_argument('--visualize', type=int, default=0, help='0 or 1')

    parser.add_argument('--beta1', type=float, default=0.9, help='coefficients for adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='coefficients for adam optimizer')

    parser.add_argument('--train_whole', type=int, default=0, help='0 or 1')
    parser.add_argument('--test_whole', type=int, default=0, help='0 or 1')

    parser.add_argument('--use_default_model', type=int, default=0, help='0 or 1')

    args = parser.parse_args()

    return args


def getSavePath(args):
    path_prefix = ''
    if args.use_default_model:
        path_prefix = '/mnt/workspace/datasets/yujiao_data/Models/ModelsFord/LM_'
    else:
        path_prefix = './ModelsFord/LM_'
    save_path = path_prefix + str(args.direction) \
                + '/lat' + str(args.shift_range_lat) + 'm_lon' + str(args.shift_range_lon) + 'm_rot' + str(args.rotation_range) \
                + '_Lev' + str(args.level) + '_Nit' + str(args.N_iters) \
                + '_Wei' + str(args.using_weight) \
                + '_Dam' + str(args.train_damping) \
                + '_Load' + str(args.Load) + '_' + str(args.Optimizer) \
                + '_loss' + str(args.loss_method) \
                + '_' + str(args.coe_shift_lat) + '_' + str(args.coe_shift_lon) + '_' + str(args.coe_heading) \
                + '_' + str(args.coe_L1) + '_' + str(args.coe_L2) + '_' + str(args.coe_L3) + '_' + str(args.coe_L4) \
                + '_Start' + str(args.train_log_start) + '_End' + str(args.train_log_end)

    if args.transformer:
        restore_path = save_path
        save_path += '_transformer'
    else:
        restore_path = None

    if args.estimate_depth:
        save_path += '_Depth1'

    if args.level_first:
        save_path += '_Level1st'

    if args.proj != 'geo':
        save_path += '_' + args.proj

    if args.use_hessian:
        save_path += '_Hess'
    if args.dropout > 0:
        save_path += '_Dropout' + str(args.dropout)

    if args.train_whole:
        save_path += '_Whole'

    # restore_path = './Models/' + str(args.direction) + '/shift' + str(20.0) + 'm_rotate' + str(10.0) \
    #                + '_level4'  # + str(args.level)

    print('save_path:', save_path)

    return restore_path, save_path


if __name__ == '__main__':
    # test to load 1 data
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    if torch.cuda.is_available():
        print("Training ford using GPU")
        device = torch.device("cuda:0")
        # device = torch.device("cpu")

    else:
        print("Training ford using CPU")
        device = torch.device("cpu")

    np.random.seed(2022)

    args = parse_args()

    mini_batch = args.batch_size

    restore_path, save_path = getSavePath(args)

    # writer = SummaryWriter(save_path)

    # net = LM_G2SP(args)
    net = eval('LM_' + args.direction + '_Ford')(args)
    # net = LM_S2GP_Ford(args)

    ### cudaargs.epochs, args.debug)
    net.to(device)
    ###########################

    if args.test:
        net.load_state_dict(torch.load(os.path.join(save_path, 'Model_best.pth')))
        # net.load_state_dict(torch.load('./Models/sequence4_stereo0_fuse0_corrTrue_batch8_loss1_GRUdir2_GRUlayers1/stage_1/Model_best.pth'))
        test1(net, args, save_path, 0., args.train_log_start)


    else:

        if args.resume:
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_' + str(args.resume - 1) + '.pth')))
            print("resume from " + 'model_' + str(args.resume - 1) + '.pth')

        else:
            # if args.Load:
            if restore_path:
                save_dict = torch.load(os.path.join(restore_path, 'Model_best.pth'))
                net_dict = net.state_dict()
                state_dict = {k: v for k, v in save_dict.items() if
                              k in net_dict.keys() and net_dict[k].size() == save_dict[k].size()}
                net.load_state_dict(state_dict, strict=False)
                print('Restore model from ', restore_path + ' done ...')

                for param in net.SatFeatureNet.parameters():
                    param.requires_grad = False

                for param in net.GrdFeatureNet.parameters():
                    param.requires_grad = False

        lr = args.lr

        if args.visualize:
            net.load_state_dict(torch.load(os.path.join(save_path, 'Model_best.pth')))

        train(lr, args, save_path, train_log_start=args.train_log_start, train_log_end=args.train_log_end)

        # writer.flush()
        # writer.close()

