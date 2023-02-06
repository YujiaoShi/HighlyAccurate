#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os

import torchvision.utils

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataLoader.KITTI_dataset import load_train_data, load_test1_data, load_test2_data
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import scipy.io as scio

import ssl

ssl._create_default_https_context = ssl._create_unverified_context  # for downloading pretrained VGG weights

from models_kitti import LM_G2SP, loss_func, LM_S2GP

import numpy as np
import os
import argparse

from utils import gps2distance
import time

########################### ranking test ############################
def test1(net_test, args, save_path, best_rank_result, epoch):
    ### net evaluation state
    net_test.eval()

    dataloader = load_test1_data(mini_batch, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    pred_shifts = []
    pred_headings = []
    gt_shifts = []
    gt_headings = []

    start_time = time.time()
    for i, data in enumerate(dataloader, 0):
        sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [item.to(device) for item in data[:-1]]

        if args.direction == 'S2GP':
            shifts_lat, shifts_lon, theta = net_test(sat_map, grd_left_imgs, mode='test')
        elif args.direction == 'G2SP':
            shifts_lat, shifts_lon, theta = net_test(sat_map, grd_left_imgs, left_camera_k, mode='test')

        shifts = torch.stack([shifts_lat, shifts_lon], dim=-1)
        headings = theta.unsqueeze(dim=-1)
        # shifts: [B, 2]
        # headings: [B, 1]

        gt_shift = torch.cat([gt_shift_v, gt_shift_u], dim=-1)  # [B, 2]

        if args.shift_range_lat==0 and args.shift_range_lon==0:
            loss = torch.mean(headings - gt_heading)
        else:
            loss = torch.mean(shifts_lat - gt_shift_u)
        loss.backward()  # just to release graph

        pred_shifts.append(shifts.data.cpu().numpy())
        pred_headings.append(headings.data.cpu().numpy())
        gt_shifts.append(gt_shift.data.cpu().numpy())
        gt_headings.append(gt_heading.data.cpu().numpy())

        if i % 20 == 0:
            print(i)

    end_time = time.time()
    duration = (end_time - start_time)/len(dataloader)

    pred_shifts = np.concatenate(pred_shifts, axis=0) * np.array([args.shift_range_lat, args.shift_range_lon]).reshape(1, 2)
    pred_headings = np.concatenate(pred_headings, axis=0) * args.rotation_range
    gt_shifts = np.concatenate(gt_shifts, axis=0) * np.array([args.shift_range_lat, args.shift_range_lon]).reshape(1, 2)
    gt_headings = np.concatenate(gt_headings, axis=0) * args.rotation_range

    scio.savemat(os.path.join(save_path, 'Test1_results.mat'), {'gt_shifts': gt_shifts, 'gt_headings': gt_headings,
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

    f = open(os.path.join(save_path, 'Test1_results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    f.write('Time per image (second): ' + str(duration) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    print('Time per image (second): ' + str(duration) + '\n')
    print('Validation results:')
    print('Init distance average: ', np.mean(init_dis))
    print('Pred distance average: ', np.mean(distance))
    print('Init angle average: ', np.mean(init_angle))
    print('Pred angle average: ', np.mean(angle_diff))


    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    diff_shifts = np.abs(pred_shifts - gt_shifts)
    for idx in range(len(metrics)):
        pred = np.sum(diff_shifts[:, 0] < metrics[idx]) / diff_shifts.shape[0] * 100
        init = np.sum(np.abs(gt_shifts[:, 0]) < metrics[idx]) / init_dis.shape[0] * 100

        line = 'lateral      within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

        pred = np.sum(diff_shifts[:, 1] < metrics[idx]) / diff_shifts.shape[0] * 100
        init = np.sum(np.abs(gt_shifts[:, 1]) < metrics[idx]) / diff_shifts.shape[0] * 100

        line = 'longitudinal within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        line = 'angle within ' + str(angles[idx]) + ' degrees (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(angles)):
        pred = np.sum((angle_diff[:, 0] < angles[idx]) & (diff_shifts[:, 0] < metrics[idx])) / angle_diff.shape[0] * 100
        init = np.sum((init_angle[:, 0] < angles[idx]) & (np.abs(gt_shifts[:, 0]) < metrics[idx])) / angle_diff.shape[0] * 100
        line = 'lat within ' + str(metrics[idx]) + ' & angle within ' + str(angles[idx]) + \
               ' (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()
    result = np.sum((distance < metrics[0]) & (angle_diff < angles[0])) / distance.shape[0] * 100

    net_test.train()

    ### save the best params
    if (result > best_rank_result):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(net_test.state_dict(), os.path.join(save_path, 'Model_best.pth'))

    return result


def test2(net_test, args, save_path, best_rank_result, epoch):
    ### net evaluation state
    net_test.eval()

    dataloader = load_test2_data(mini_batch, args.shift_range_lat, args.shift_range_lon, args.rotation_range)
    pred_shifts = []
    pred_headings = []
    gt_shifts = []
    gt_headings = []

    start_time = time.time()

    for i, data in enumerate(dataloader, 0):
        sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [item.to(device) for item in data[:-1]]

        # shifts_lat, shifts_lon, theta = net_test(sat_map, grd_left_imgs, mode='test')
        if args.direction == 'S2GP':
            shifts_lat, shifts_lon, theta = net_test(sat_map, grd_left_imgs, mode='test', level_first=args.level_first)
        elif args.direction == 'G2SP':
            shifts_lat, shifts_lon, theta = net_test(sat_map, grd_left_imgs, left_camera_k, mode='test')

        shifts = torch.stack([shifts_lat, shifts_lon], dim=-1)
        headings = theta.unsqueeze(dim=-1)
        # shifts: [B, 2]
        # headings: [B, 1]

        gt_shift = torch.cat([gt_shift_v, gt_shift_u], dim=-1)  # [B, 2]

        if args.shift_range_lat==0 and args.shift_range_lon==0:
            loss = torch.mean(headings - gt_heading)
        else:
            loss = torch.mean(shifts_lat - gt_shift_u)
        loss.backward()  # just to release graph

        pred_shifts.append(shifts.data.cpu().numpy())
        pred_headings.append(headings.data.cpu().numpy())
        gt_shifts.append(gt_shift.data.cpu().numpy())
        gt_headings.append(gt_heading.data.cpu().numpy())

        if i % 20 == 0:
            print(i)

    end_time = time.time()
    duration = (end_time - start_time)/len(dataloader)

    pred_shifts = np.concatenate(pred_shifts, axis=0) * np.array([args.shift_range_lat, args.shift_range_lon]).reshape(1, 2)
    pred_headings = np.concatenate(pred_headings, axis=0) * args.rotation_range
    gt_shifts = np.concatenate(gt_shifts, axis=0) * np.array([args.shift_range_lat, args.shift_range_lon]).reshape(1, 2)
    gt_headings = np.concatenate(gt_headings, axis=0) * args.rotation_range

    scio.savemat(os.path.join(save_path, 'Test2_results.mat'), {'gt_shifts': gt_shifts, 'gt_headings': gt_headings,
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

    f = open(os.path.join(save_path, 'Test2_results.txt'), 'a')
    f.write('====================================\n')
    f.write('       EPOCH: ' + str(epoch) + '\n')
    f.write('Time per image (second): ' + str(duration) + '\n')
    print('====================================')
    print('       EPOCH: ' + str(epoch))
    print('Time per image (second): ' + str(duration) + '\n')
    print('Test results:')
    print('Init distance average: ', np.mean(init_dis))
    print('Pred distance average: ', np.mean(distance))
    print('Init angle average: ', np.mean(init_angle))
    print('Pred angle average: ', np.mean(angle_diff))


    for idx in range(len(metrics)):
        pred = np.sum(distance < metrics[idx]) / distance.shape[0] * 100
        init = np.sum(init_dis < metrics[idx]) / init_dis.shape[0] * 100

        line = 'distance within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    diff_shifts = np.abs(pred_shifts - gt_shifts)
    for idx in range(len(metrics)):
        pred = np.sum(diff_shifts[:, 0] < metrics[idx]) / diff_shifts.shape[0] * 100
        init = np.sum(np.abs(gt_shifts[:, 0]) < metrics[idx]) / init_dis.shape[0] * 100

        line = 'lateral      within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

        pred = np.sum(diff_shifts[:, 1] < metrics[idx]) / diff_shifts.shape[0] * 100
        init = np.sum(np.abs(gt_shifts[:, 1]) < metrics[idx]) / diff_shifts.shape[0] * 100

        line = 'longitudinal within ' + str(metrics[idx]) + ' meters (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(angles)):
        pred = np.sum(angle_diff < angles[idx]) / angle_diff.shape[0] * 100
        init = np.sum(init_angle < angles[idx]) / angle_diff.shape[0] * 100
        line = 'angle within ' + str(angles[idx]) + ' degrees (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('-------------------------')
    f.write('------------------------\n')

    for idx in range(len(angles)):
        pred = np.sum((angle_diff[:, 0] < angles[idx]) & (diff_shifts[:, 0] < metrics[idx])) / angle_diff.shape[0] * 100
        init = np.sum((init_angle[:, 0] < angles[idx]) & (np.abs(gt_shifts[:, 0]) < metrics[idx])) / angle_diff.shape[0] * 100
        line = 'lat within ' + str(metrics[idx]) + ' & angle within ' + str(angles[idx]) + \
               ' (pred, init): ' + str(pred) + ' ' + str(init)
        print(line)
        f.write(line + '\n')

    print('====================================')
    f.write('====================================\n')
    f.close()
    # result = np.sum((distance < metrics[0]) & (angle_diff < angles[0])) / distance.shape[0] * 100

    net_test.train()

    # ### save the best params
    # if (result > best_rank_result):
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     torch.save(net_test.state_dict(), os.path.join(save_path, 'Model_best.pth'))

    return


###### learning criterion assignment #######
def train(net, lr, args, save_path):
    bestRankResult = 0.0  # current best, Siam-FCANET18
    # loop over the dataset multiple times
    print(args.resume)
    print(args.epochs)
    for epoch in range(args.resume, args.epochs):
        net.train()

        # base_lr = 0
        base_lr = lr
        base_lr = base_lr * ((1.0 - float(epoch) / 100.0) ** (1.0))

        print(base_lr)

        optimizer = optim.Adam(net.parameters(), lr=base_lr)

        optimizer.zero_grad()

        ### feeding A and P into train loader
        trainloader = load_train_data(mini_batch, args.shift_range_lat, args.shift_range_lon, args.rotation_range)

        loss_vec = []

        print('batch_size:', mini_batch, '\n num of batches:', len(trainloader))

        for Loop, Data in enumerate(trainloader, 0):
            # get the inputs

            sat_map, left_camera_k, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading = [item.to(device) for item in Data[:-1]]
            file_name = Data[-1]

            # zero the parameter gradients
            optimizer.zero_grad()

            if args.direction == 'S2GP':
                loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                shift_lat_last, shift_lon_last, theta_last, \
                L1_loss, L2_loss, L3_loss, L4_loss, grd_conf_list = \
                    net(sat_map, grd_left_imgs, gt_shift_u, gt_shift_v, gt_heading, mode='train', file_name=file_name,
                        loop=Loop, level_first=args.level_first)
            elif args.direction =='G2SP':
                loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                shift_lat_last, shift_lon_last, theta_last, \
                L1_loss, L2_loss, L3_loss, L4_loss, grd_conf_list = \
                    net(sat_map, grd_left_imgs, left_camera_k, gt_shift_u, gt_shift_v, gt_heading, mode='train', file_name=file_name)

            loss.backward()

            optimizer.step()  # This step is responsible for updating weights
            optimizer.zero_grad()

            ### record the loss
            loss_vec.append(loss.item())

            if Loop % 10 == 9:  #
                level = args.level - 1
                # for level in range(len(shifts_decrease)):
                # print(loss_decrease[level].shape)
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

        ### save modelget_similarity_fn
        compNum = epoch % 100
        print('taking snapshot ...')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(net.state_dict(), os.path.join(save_path, 'model_' + str(compNum) + '.pth'))

        ### ranking test
        current = test1(net, args, save_path, bestRankResult, epoch)
        if (current > bestRankResult):
            bestRankResult = current

        test2(net, args, save_path, bestRankResult, epoch)

    print('Finished Training')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')
    parser.add_argument('--debug', type=int, default=0, help='debug to dump middle processing images')

    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')

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

    parser.add_argument('--batch_size', type=int, default=3, help='batch size')
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
    parser.add_argument('--Optimizer', type=str, default='LM', help='LM or SGD or ADAM')

    parser.add_argument('--level_first', type=int, default=0, help='0 or 1, estimate grd depth or not')
    parser.add_argument('--proj', type=str, default='geo', help='geo, polar, nn')
    parser.add_argument('--use_gt_depth', type=int, default=0, help='0 or 1')

    parser.add_argument('--dropout', type=int, default=0, help='0 or 1')
    parser.add_argument('--use_hessian', type=int, default=0, help='0 or 1')

    parser.add_argument('--visualize', type=int, default=0, help='0 or 0')

    parser.add_argument('--beta1', type=float, default=0.9, help='coefficients for adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='coefficients for adam optimizer')

    args = parser.parse_args()

    return args


def getSavePath(args):
    save_path = './ModelsKitti/LM_' + str(args.direction) \
                + '/lat' + str(args.shift_range_lat) + 'm_lon' + str(args.shift_range_lon) + 'm_rot' + str(
        args.rotation_range) \
                + '_Lev' + str(args.level) + '_Nit' + str(args.N_iters) \
                + '_Wei' + str(args.using_weight) \
                + '_Dam' + str(args.train_damping) \
                + '_Load' + str(args.Load) + '_' + str(args.Optimizer) \
                + '_loss' + str(args.loss_method) \
                + '_' + str(args.coe_shift_lat) + '_' + str(args.coe_shift_lon) + '_' + str(args.coe_heading) \
                + '_' + str(args.coe_L1) + '_' + str(args.coe_L2) + '_' + str(args.coe_L3) + '_' + str(args.coe_L4)

    if args.level_first:
        save_path += '_Level1st'

    if args.proj != 'geo':
        save_path += '_' + args.proj

    if args.use_gt_depth:
        save_path += '_depth'

    if args.use_hessian:
        save_path += '_Hess'

    if args.dropout > 0:
        save_path += '_Dropout' + str(args.dropout)

    if args.damping != 0.1:
        save_path += '_Damping' + str(args.damping)


    print('save_path:', save_path)

    return save_path


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    np.random.seed(2022)

    args = parse_args()

    mini_batch = args.batch_size

    save_path = getSavePath(args)

    net = eval('LM_' + args.direction)(args)

    ### cudaargs.epochs, args.debug)
    net.to(device)
    ###########################

    if args.test:
        net.load_state_dict(torch.load(os.path.join(save_path, 'model_1.pth')))
        test1(net, args, save_path, 0., epoch=0)
        test2(net, args, save_path, 0., epoch=0)

    else:

        if args.resume:
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_' + str(args.resume - 1) + '.pth')))
            print("resume from " + 'model_' + str(args.resume - 1) + '.pth')

        if args.visualize:
            net.load_state_dict(torch.load(os.path.join(save_path, 'model_1.pth')))

        lr = args.lr

        train(net, lr, args, save_path)

