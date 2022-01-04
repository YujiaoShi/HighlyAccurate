#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

import utils


class VGGUnet(nn.Module):
    def __init__(self, level, estimate_depth=0):
        super(VGGUnet, self).__init__()
        # print('estimate_depth: ', estimate_depth)

        self.level = level

        vgg16 = torchvision.models.vgg16(pretrained=True)

        # load CNN from VGG16, the first three block
        self.conv0 = vgg16.features[0]
        self.conv2 = vgg16.features[2]  # \\64
        self.conv5 = vgg16.features[5]  #
        self.conv7 = vgg16.features[7]  # \\128
        self.conv10 = vgg16.features[10]
        self.conv12 = vgg16.features[12]
        self.conv14 = vgg16.features[14]  # \\256

        self.conv_dec1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv14.out_channels + self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False),
        )

        self.conv_dec2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv7.out_channels + self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False)
        )

        self.conv_dec3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv2.out_channels + self.conv2.out_channels, 32, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False)
        )

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False,
                                     return_indices=True)

        self.conf0 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.conf1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.conf2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.conf3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid(),
        )

        self.estimate_depth = estimate_depth

        if estimate_depth:
            self.depth0 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Tanh(),
            )
            self.depth1 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Tanh(),
            )
            self.depth2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Tanh(),
            )
            self.depth3 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Tanh(),
            )

            self.depth0[-2].weight.data.zero_()
            self.depth1[-2].weight.data.zero_()
            self.depth2[-2].weight.data.zero_()
            self.depth3[-2].weight.data.zero_()


    def forward(self, x):
        # block0
        x0 = self.conv0(x)
        x1 = self.relu(x0)
        x2 = self.conv2(x1)
        x3, ind3 = self.max_pool(x2)  # [H/2, W/2]

        x4 = self.relu(x3)
        x5 = self.conv5(x4)
        x6 = self.relu(x5)
        x7 = self.conv7(x6)
        x8, ind8 = self.max_pool(x7)  # [H/4, W/4]

        # block2
        x9 = self.relu(x8)
        x10 = self.conv10(x9)
        x11 = self.relu(x10)
        x12 = self.conv12(x11)
        x13 = self.relu(x12)
        x14 = self.conv14(x13)
        x15, ind15 = self.max_pool(x14)  # [H/8, W/8]

        # dec1
        x16 = F.interpolate(x15, [x8.shape[2], x9.shape[3]], mode="nearest")
        x17 = torch.cat([x16, x8], dim=1)
        x18 = self.conv_dec1(x17)  # [H/4, W/4]

        # dec2
        x19 = F.interpolate(x18, [x3.shape[2], x3.shape[3]], mode="nearest")
        x20 = torch.cat([x19, x3], dim=1)
        x21 = self.conv_dec2(x20)  # [H/2, W/2]

        x22 = F.interpolate(x21, [x2.shape[2], x2.shape[3]], mode="nearest")
        x23 = torch.cat([x22, x2], dim=1)
        x24 = self.conv_dec3(x23)  # [H, W]

        # c0 = 1 / (1 + self.conf0(x15))
        # c1 = 1 / (1 + self.conf1(x18))
        # c2 = 1 / (1 + self.conf2(x21))
        c0 = nn.Sigmoid()(-self.conf0(x15))
        c1 = nn.Sigmoid()(-self.conf1(x18))
        c2 = nn.Sigmoid()(-self.conf2(x21))
        c3 = nn.Sigmoid()(-self.conf3(x24))

        if self.estimate_depth:
            # its actually height, not depth
            d0 = process_depth(self.depth0(x15))
            d1 = process_depth(self.depth1(x18))
            d2 = process_depth(self.depth2(x21))
            d3 = process_depth(self.depth3(x24))

        x15 = L2_norm(x15)
        x18 = L2_norm(x18)
        x21 = L2_norm(x21)
        x24 = L2_norm(x24)


        if self.estimate_depth:
            if self.level == -1:
                return [x15], [c0], [d0]
            elif self.level == -2:
                return [x18], [c1], [d1]
            elif self.level == -3:
                return [x21], [c2], [d2]
            elif self.level == 2:
                return [x18, x21], [c1, c2], [d1, d2]
            elif self.level == 3:
                return [x15, x18, x21], [c0, c1, c2], [d0, d1, d2]
            elif self.level == 4:
                return [x15, x18, x21, x24], [c0, c1, c2, c3], [d0, d1, d2, d3]
        else:
            if self.level == -1:
                return [x15], [c0]
            elif self.level == -2:
                return [x18], [c1]
            elif self.level == -3:
                return [x21], [c2]
            elif self.level == 2:
                return [x18, x21], [c1, c2]
            elif self.level == 3:
                return [x15, x18, x21], [c0, c1, c2]
            elif self.level == 4:
                return [x15, x18, x21, x24], [c0, c1, c2, c3]


class VGGUnet_G2S(nn.Module):
    def __init__(self, level):
        super(VGGUnet_G2S, self).__init__()
        # print('estimate_depth: ', estimate_depth)

        self.level = level

        vgg16 = torchvision.models.vgg16(pretrained=True)

        # load CNN from VGG16, the first three block
        self.conv0 = vgg16.features[0]
        self.conv2 = vgg16.features[2]  # \\64
        self.conv5 = vgg16.features[5]  #
        self.conv7 = vgg16.features[7]  # \\128
        self.conv10 = vgg16.features[10]
        self.conv12 = vgg16.features[12]
        self.conv14 = vgg16.features[14]  # \\256

        self.conv_dec1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv14.out_channels + self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False),
        )

        self.conv_dec2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv7.out_channels + self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False)
        )

        self.conv_dec3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv2.out_channels + self.conv2.out_channels, 32, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False)
        )

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False,
                                     return_indices=True)

        self.conf0 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.conf1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.conf2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.conf3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # block0
        x0 = self.conv0(x)
        x1 = self.relu(x0)
        x2 = self.conv2(x1)
        x3, ind3 = self.max_pool(x2)  # [H/2, W/2]

        B, C, H2, W2 = x2.shape
        x2_ = x2.reshape(B, C, H2 * 2, W2 // 2)

        B, C, H3, W3 = x3.shape
        x3_ = x3.reshape(B, C, H3*2, W3//2)

        x4 = self.relu(x3)
        x5 = self.conv5(x4)
        x6 = self.relu(x5)
        x7 = self.conv7(x6)
        x8, ind8 = self.max_pool(x7)  # [H/4, W/4]

        B, C, H8, W8 = x8.shape
        x8_ = x8.reshape(B, C, H8 * 2, W8 // 2)

        # block2
        x9 = self.relu(x8)
        x10 = self.conv10(x9)
        x11 = self.relu(x10)
        x12 = self.conv12(x11)
        x13 = self.relu(x12)
        x14 = self.conv14(x13)
        x15, ind15 = self.max_pool(x14)  # [H/8, W/8]

        B, C, H15, W15 = x15.shape
        x15_ = x15.reshape(B, C, H15 * 2, W15 // 2)

        # dec1
        x16 = F.interpolate(x15_, [x8_.shape[2], x8_.shape[3]], mode="nearest")
        x17 = torch.cat([x16, x8_], dim=1)
        x18 = self.conv_dec1(x17)  # [H/4, W/4]

        # dec2
        x19 = F.interpolate(x18, [x3_.shape[2], x3_.shape[3]], mode="nearest")
        x20 = torch.cat([x19, x3_], dim=1)
        x21 = self.conv_dec2(x20)  # [H/2, W/2]

        x22 = F.interpolate(x21, [x2_.shape[2], x2_.shape[3]], mode="nearest")
        x23 = torch.cat([x22, x2_], dim=1)
        x24 = self.conv_dec3(x23)  # [H, W]

        c0 = nn.Sigmoid()(-self.conf0(x15))
        c1 = nn.Sigmoid()(-self.conf1(x18))
        c2 = nn.Sigmoid()(-self.conf2(x21))
        c3 = nn.Sigmoid()(-self.conf3(x24))

        x15 = L2_norm(x15_)
        x18 = L2_norm(x18)
        x21 = L2_norm(x21)
        x24 = L2_norm(x24)

        if self.level == -1:
            return [x15], [c0]
        elif self.level == -2:
            return [x18], [c1]
        elif self.level == -3:
            return [x21], [c2]
        elif self.level == 2:
            return [x18, x21], [c1, c2]
        elif self.level == 3:
            return [x15, x18, x21], [c0, c1, c2]
        elif self.level == 4:
            return [x15, x18, x21, x24], [c0, c1, c2, c3]


def process_depth(d):
    B, _, H, W = d.shape
    d = (d + 1)/2
    d1 = torch.cat([d[:, :, :H//2, :] * 10, d[:, :, H//2 :, :] * 1.6], dim=2)
    return d1

# class VGGUnet(nn.Module):
#     def __init__(self, level):
#         super(VGGUnet, self).__init__()
#
#         self.level = level
#
#         vgg16 = torchvision.models.vgg16(pretrained=True)
#
#         # load CNN from VGG16, the first three block
#         self.conv0 = vgg16.features[0]
#         self.conv2 = vgg16.features[2]  # \\64
#         self.conv5 = vgg16.features[5]  #
#         self.conv7 = vgg16.features[7]  # \\128
#         self.conv10 = vgg16.features[10]
#         self.conv12 = vgg16.features[12]
#         self.conv14 = vgg16.features[14]  # \\256
#
#         self.bottleneck = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3),
#                       stride=(1, 1), padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
#                       bias=False),
#         )
#
#         self.conv_dec1 = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.conv14.out_channels + self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3),
#                       stride=(1, 1), padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
#                       bias=False),
#         )
#
#         self.conv_dec2 = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.conv14.out_channels + self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3),
#                       stride=(1, 1), padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
#                       bias=False)
#         )
#
#         self.conv_dec3 = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.conv7.out_channels + self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3),
#                       stride=(1, 1), padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
#                       bias=False)
#         )
#
#         self.relu = nn.ReLU(inplace=True)
#         self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False,
#                                      return_indices=True)
#
#         self.conf0 = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
#             nn.Sigmoid())
#         self.conf1 = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
#             nn.Sigmoid())
#         self.conf2 = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
#             nn.Sigmoid())
#         self.conf3 = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
#             nn.Sigmoid())
#
#         self.feat0 = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(256, 32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
#             nn.Sigmoid())
#         self.feat1 = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(256, 32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
#             nn.Sigmoid())
#         self.feat2 = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
#             nn.Sigmoid())
#         self.feat3 = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
#             nn.Sigmoid())
#
#     def forward(self, x):
#         # block0
#         x0 = self.conv0(x)
#         x1 = self.relu(x0)
#         x2 = self.conv2(x1)  # [H, W, 64]
#
#         x3, ind3 = self.max_pool(x2)
#         x4 = self.relu(x3)
#         x5 = self.conv5(x4)
#         x6 = self.relu(x5)
#         x7 = self.conv7(x6) # [H/2, W/2, 128]
#
#         x8, ind8 = self.max_pool(x7)
#         x9 = self.relu(x8)
#         x10 = self.conv10(x9)
#         x11 = self.relu(x10)
#         x14 = self.conv12(x11)
#         # x13 = self.relu(x12)
#         # x14 = self.conv14(x11)  # [H/4, W/4, 256]
#
#         x15, ind15 = self.max_pool(x14)
#         x16 = self.bottleneck(x15)   # [H/8, W/8, 256]
#
#         # dec1
#         x17 = F.interpolate(x16, [x14.shape[2], x14.shape[3]], mode="nearest")
#         x18 = torch.cat([x17, x14], dim=1)
#         x19 = self.conv_dec1(x18)  # [H/4, W/4, 256]
#
#         # dec2
#         x20 = F.interpolate(x19, [x7.shape[2], x7.shape[3]], mode="nearest")
#         x21 = torch.cat([x20, x7], dim=1)
#         x22 = self.conv_dec2(x21)  # [H/2, H/2, 128]
#
#         x23 = F.interpolate(x22, [x2.shape[2], x2.shape[3]], mode="nearest")
#         x24 = torch.cat([x23, x2], dim=1)
#         x25 = self.conv_dec3(x24)
#
#         # c0 = 1 / (1 + self.conf0(x15))
#         # c1 = 1 / (1 + self.conf1(x18))
#         # c2 = 1 / (1 + self.conf2(x21))
#         c0 = self.conf0(x16)
#         c1 = self.conf1(x19)
#         c2 = self.conf2(x22)
#         c3 = self.conf3(x25)
#
#         x16 = L2_norm(self.feat0(x16))
#         x19 = L2_norm(self.feat1(x19))
#         x22 = L2_norm(self.feat2(x22))
#         x25 = L2_norm(self.feat3(x25))
#
#         if self.level == -1:
#             return [x16], [c0]
#         elif self.level == -2:
#             return [x19], [c1]
#         elif self.level == -3:
#             return [x22], [c2]
#         elif self.level == -4:
#             return [x25], [c3]
#         elif self.level == 2:
#             return [x16, x19], [c0, c1]
#         elif self.level == 3:
#             return [x16, x19, x22], [c0, c1, c2]
#         elif self.level == 4:
#             return [x16, x19, x22, x25], [c0, c1, c2, c3]


def L2_norm(x):
    B, C, H, W = x.shape
    y = F.normalize(x.reshape(B, C*H*W))
    return y.reshape(B, C, H, W)


# class VGGUNet(nn.Module):
#     def __init__(
#         self, net="vgg16", pool="max", n_encoder_stages=3, n_decoder_convs=2, last_feature_channel=32
#     ):
#         super().__init__()
#
#         if net == "vgg16":
#             vgg = torchvision.models.vgg16(pretrained=True).features
#         elif net == "vgg19":
#             vgg = torchvision.models.vgg19(pretrained=True).features
#         else:
#             raise Exception("invalid vgg net")
#
#         encs = []
#         enc = []
#         encs_channels = []
#         channels = -1
#         for mod in vgg:
#             if isinstance(mod, nn.Conv2d):
#                 channels = mod.out_channels
#
#             if isinstance(mod, nn.MaxPool2d):
#                 encs.append(nn.Sequential(*enc))
#                 encs_channels.append(channels)
#                 n_encoder_stages -= 1
#                 if n_encoder_stages <= 0:
#                     break
#                 if pool == "average":
#                     enc = [
#                         nn.AvgPool2d(
#                             kernel_size=2, stride=2, padding=0, ceil_mode=False
#                         )
#                     ]
#                 elif pool == "max":
#                     enc = [
#                         nn.MaxPool2d(
#                             kernel_size=2, stride=2, padding=0, ceil_mode=False
#                         )
#                     ]
#                 else:
#                     raise Exception("invalid pool")
#             else:
#                 enc.append(mod)
#         self.encs = nn.ModuleList(encs)
#         self.conv_conf0 = nn.Sequential(
#             nn.Conv2d(encs_channels[-1], 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
#             nn.ReLU(inplace=True),
#         )
#
#         cin = encs_channels[-1] + encs_channels[-2]
#         decs = []
#         for idx, cout in enumerate(reversed(encs_channels[:-1])):
#             decs.append(self._dec(cin, cout, n_convs=n_decoder_convs))
#             cin = cout + encs_channels[max(-idx - 3, -len(encs_channels))]
#
#         cin = cout
#         decs.append(nn.Conv2d(cin, last_feature_channel + 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False))
#         self.decs = nn.ModuleList(decs)
#
#     def _dec(self, channels_in, channels_out, n_convs=2):
#         mods = []
#         for idx in range(n_convs):
#
#             mods.append(
#                 nn.Conv2d(
#                     channels_in,
#                     channels_out,
#                     kernel_size=(3, 3),
#                     stride=(1, 1),
#                     padding=1,
#                     bias=False,
#                 )
#             )
#             mods.append(nn.ReLU())
#             channels_in = channels_out
#         return nn.Sequential(*mods)
#
#     def forward(self, x):
#         feats = []
#         for enc in self.encs:
#             x = enc(x)
#             feats.append(x)
#
#         for dec in self.decs:
#             x0 = feats.pop()
#             x1 = feats.pop()
#             x0 = F.interpolate(
#                 x0, size=(x1.shape[2], x1.shape[3]), mode="nearest"
#             )
#             x = torch.cat((x0, x1), dim=1)
#             x = dec(x)
#             feats.append(x)
#
#         x = feats.pop()
#         uncertainty, feat = torch.split(x, [1, x.shape[1] - 1], dim=1)
#         uncertainty = nn.ReLU()(uncertainty)
#         confidence = 1/(1+uncertainty)
#
#         y = torch.cat([confidence, feat], dim=1)
#
#         return y


