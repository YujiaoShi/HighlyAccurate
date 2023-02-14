import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
import utils
import os
import torchvision.transforms.functional as TF

from VGG import VGGUnet
from jacobian import grid_sample

# from models_kitti import normalize_feature
# from transformer import LocalFeatureTransformer
# from position_encoding import PositionEncoding, PositionEncodingSine
from RNNs import NNrefine

EPS = utils.EPS


class LM_S2GP_Ford(nn.Module):
    def __init__(self, args):  # device='cuda:0',
        super(LM_S2GP_Ford, self).__init__()
        '''
        loss_method: 0: direct R T loss 1: feat loss 2: noise aware feat loss
        '''
        self.args = args

        self.level = args.level
        self.N_iters = args.N_iters
        self.using_weight = args.using_weight
        self.loss_method = args.loss_method

        self.estimate_depth = args.estimate_depth

        self.SatFeatureNet = VGGUnet(self.level)
        self.GrdFeatureNet = VGGUnet(self.level, self.estimate_depth)

        self.damping = nn.Parameter(
            torch.zeros(size=(1, 3), dtype=torch.float32, requires_grad=True))

        ori_grdH, ori_grdW = 256, 1024
        self.ori_grdH = 256
        self.ori_grdW = 1024
        xyz_grds = []
        if self.level ==3 or self.level == 4:
            for level in range(4):
                grd_H, grd_W = ori_grdH / (2 ** (3 - level)), ori_grdW / (2 ** (3 - level))
                if self.estimate_depth:
                    xyz_grd, mask, xyz_raw = self.grd_img2cam(grd_H, grd_W, ori_grdH, ori_grdW)
                    # [1, grd_H, grd_W, 3] under the grd camera coordinates without depth multiplicated
                    xyz_grds.append((xyz_grd, mask, xyz_raw))
                else:
                    if self.args.proj == 'geo':
                        xyz_grd, mask = self.grd_img2cam(grd_H, grd_W, ori_grdH, ori_grdW)  # [1, grd_H, grd_W, 3] under the grd camera coordinates
                    else:
                        xyz_grd, mask = self.grd_img2cam_polar(grd_H, grd_W, ori_grdH, ori_grdW)
                    xyz_grds.append((xyz_grd, mask))
        elif self.level == 2:
            for level in range(2):
                grd_H, grd_W = ori_grdH / (2 ** (2 - level)), ori_grdW / (2 ** (2 - level))
                # print(grd_H, grd_W)
                xyz_grd, mask = self.grd_img2cam(grd_H, grd_W, ori_grdH,
                                                 ori_grdW)  # [1, grd_H, grd_W, 3] under the grd camera coordinates
                xyz_grds.append((xyz_grd, mask))

        self.xyz_grds = xyz_grds

        # self.confs = nn.ModuleList()
        #
        # self.confs.extend([
        #     nn.Sequential(
        #         nn.ReLU(),
        #         nn.Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
        #         nn.Sigmoid(),
        #     ),
        #     nn.Sequential(
        #         nn.ReLU(),
        #         nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
        #         nn.Sigmoid(),
        #     ),
        #     nn.Sequential(
        #         nn.ReLU(),
        #         nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
        #         nn.Sigmoid(),
        #     ),
        #     nn.Sequential(
        #         nn.ReLU(),
        #         nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
        #         nn.Sigmoid(),
        #     )
        # ])
        #
        # self.transformers = nn.ModuleList()
        # self.transformers.extend([
        #     LocalFeatureTransformer(d_model=256, n_head=8),
        #     LocalFeatureTransformer(d_model=128, n_head=8),
        #     LocalFeatureTransformer(d_model=64, n_head=8),
        # ])
        # # self.pe = PositionEncoding(d_model=256, max_len=128)
        # self.pe = PositionEncodingSine(d_model=256, max_shape=(128, 512))

        if self.args.Optimizer == 'NN':
            self.NNrefine = NNrefine()

        torch.autograd.set_detect_anomaly(True)
        # Running the forward pass with detection enabled will allow the backward pass to print the traceback of the forward operation that created the failing backward function.
        # Any backward computation that generate “nan” value will raise an error.

    def grd_img2cam(self, grd_H, grd_W, ori_grdH, ori_grdW):

        # ori_camera_k = torch.tensor([[[582.9802, 0.0000, 496.2420],
        #                               [0.0000, 482.7076, 125.0034],
        #                               [0.0000, 0.0000, 1.0000]]],
        #                             dtype=torch.float32, requires_grad=True)  # [1, 3, 3]
        K_FL = torch.tensor([945.391406, 0.0, 855.502825, 0.0, 945.668274, 566.372868, 0.0, 0.0, 1.0],
                                    dtype=torch.float32, requires_grad=True).reshape(1, 3, 3)
        # Original image resolution
        H_FL = 860
        W_FL = 1656

        # Network input image resolution
        H = 256
        W = 1024

        ori_camera_k = torch.zeros_like(K_FL)

        ori_camera_k[0, 0] = K_FL[0, 0] / W_FL * W
        ori_camera_k[0, 1] = K_FL[0, 1] / H_FL * H
        ori_camera_k[0, 2] = K_FL[0, 2]

        camera_height = utils.get_camera_height()  # question mark. How to determine?
        # camera_height = 2
        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1,
                             :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        camera_k_inv = torch.inverse(camera_k)  # [B, 3, 3]

        v, u = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32),
                              torch.arange(0, grd_W, dtype=torch.float32))
        uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).unsqueeze(dim=0)  # [1, grd_H, grd_W, 3]
        xyz_w = torch.sum(camera_k_inv[:, None, None, :, :] * uv1[:, :, :, None, :], dim=-1)  # [1, grd_H, grd_W, 3]

        w = camera_height / torch.where(torch.abs(xyz_w[..., 1:2]) > utils.EPS, xyz_w[..., 1:2],
                                        utils.EPS * torch.ones_like(xyz_w[..., 1:2]))  # [BN, grd_H, grd_W, 1]
        xyz_grd = xyz_w * w  # [1, grd_H, grd_W, 3] under the grd camera coordinates

        mask = (xyz_grd[..., -1] > 0).float()  # # [1, grd_H, grd_W]

        assert torch.sum(mask) > 0
        if self.estimate_depth:
            return xyz_grd, mask, xyz_w
        else:
            return xyz_grd, mask

    def grd_img2cam_polar(self, grd_H, grd_W, ori_grdH, ori_grdW):

        v, u = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32),
                              torch.arange(0, grd_W, dtype=torch.float32))
        theta = u/grd_W * np.pi/4
        radius = (1 - v / grd_H) * 30  # set radius as 30 meters

        z = radius * torch.cos(np.pi/4 - theta)
        x = -radius * torch.sin(np.pi/4 - theta)
        y = utils.get_camera_height() * torch.ones_like(z)
        xyz_grd = torch.stack([x, y, z], dim=-1).unsqueeze(dim=0) # [1, grd_H, grd_W, 3] under the grd camera coordinates

        mask = torch.ones_like(z).unsqueeze(dim=0)  # [1, grd_H, grd_W]

        return xyz_grd, mask

    def cam2body2world2sat(self, R_FL, T_FL, shift_u, shift_v, theta, level,
                           satmap_sidelength_meters, satmap_sidelength_pixels, require_jac=False, depth=None):
        '''
        Args:
            R_FL: [B, 3, 3]
            T_FL: [B, 3]
            shift_u: [B, 1] within [-1, 1], initialize as 0
            shift_v: [B, 1] within [-1, 1], initialize as 0
            theta: [B, 1] within [-1, 1], initialize as 0
            yaw_init: [B, 1]
            level: scalar
            satmap_sidelength:  scalar
            require_jac: bool
            depth: [B, 1, H, W]

        Returns:

        '''
        B = shift_u.shape[0]
        if self.estimate_depth and depth is not None:
            xyz_w = self.xyz_grds[level][2].detach().to(shift_u.device).repeat(B, 1, 1, 1)  # [B, grd_H, grd_W, 3]

            camera_height = utils.get_camera_height()
            w = (camera_height - depth.permute(0, 2, 3, 1)) / torch.where(torch.abs(xyz_w[..., 1:2]) > utils.EPS, xyz_w[..., 1:2],
                                            utils.EPS * torch.ones_like(xyz_w[..., 1:2]))  # [BN, grd_H, grd_W, 1]
            Xc = xyz_w * w  # [1, grd_H, grd_W, 3] under the grd camera coordinates

            mask = (Xc[..., -1] > 0).float()  # # [1, grd_H, grd_W]
            assert torch.sum(mask) > 0

            # Xc = Xraw * depth.permute(0, 2, 3, 1)
            # mask = (Xc[..., -1] > 0).float()
            # assert torch.sum(mask) > 0
        else:
            Xc = self.xyz_grds[level][0].detach().to(shift_u.device).repeat(B, 1, 1, 1)  # [B, grd_H, grd_W, 3]
            mask = self.xyz_grds[level][1].detach().to(shift_u.device).repeat(B, 1, 1)  # [B, grd_H, grd_W]
        Xb = torch.sum(R_FL[:, None, None, :, :] * Xc[:, :, :, None, :], dim=-1) + T_FL[:, None, None, :] # [B, grd_H, grd_W, 3]
        grd_H, grd_W = Xb.shape[1:3]

        shift_u_meters = self.args.shift_range_lat * shift_u
        shift_v_meters = self.args.shift_range_lon * shift_v
        Tw = torch.cat([shift_v_meters, -shift_u_meters, torch.zeros_like(shift_v_meters)], dim=-1)  # [B, 3]

        yaw = theta * self.args.rotation_range / 180 * np.pi
        cos = torch.cos(yaw)
        sin = torch.sin(yaw)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        Rw = torch.cat([cos, sin, zeros, -sin, cos, zeros, zeros, zeros, ones], dim=-1)  # shape = [B, 9]
        Rw = Rw.view(B, 3, 3)  # shape = [B, 3, 3]
        Xw = torch.sum(Rw[:, None, None, :, :] * (Xb[:, :, :, None, :] + Tw[:, None, None, None, :]), dim=-1)
        # [B, grd_H, grd_W, 3]

        Rs = torch.tensor([0, 1, 0, -1, 0, 0, 0, 0, 1], dtype=torch.float32, device=yaw.device).reshape(3, 3)
        Rs = Rs.unsqueeze(dim=0).repeat(B, 1, 1)
        Xs = torch.sum(Rs[:, None, None, :, :] * Xw[:, :, :, None, :], dim=-1)

        meters_per_pixel = satmap_sidelength_meters / satmap_sidelength_pixels
        sat_uv = Xs[..., :2] / meters_per_pixel + satmap_sidelength_pixels // 2

        if require_jac:
            dRw_dtheta = self.args.rotation_range / 180 * np.pi * \
                         torch.cat([-sin, cos, zeros, -cos, -sin, zeros, zeros, zeros, zeros], dim=-1).view(B, 3, 3)
            dTw_dshiftu = self.args.shift_range_lat * \
                          torch.tensor([0., -1., 0.], dtype=torch.float32, device=shift_u.device, requires_grad=True).view(1, 3).repeat(B, 1)
            dTw_dshiftv = self.args.shift_range_lon * \
                          torch.tensor([1., 0., 0.], dtype=torch.float32, device=shift_u.device, requires_grad=True).view(1, 3).repeat(B, 1)

            dXw_dtheta = torch.sum(dRw_dtheta[:, None, None, :, :] * (Xb[:, :, :, None, :] + Tw[:, None, None, None, :]), dim=-1)
            # [B, grd_H, grd_W, 3]
            dXw_dshiftu = torch.sum(Rw * dTw_dshiftu[:, None, :], dim=-1)  # [B, 3]
            dXw_dshiftv = torch.sum(Rw * dTw_dshiftv[:, None, :], dim=-1)  # [B, 3]

            dXs_dtheta = torch.sum(Rs[:, None, None, :, :] * dXw_dtheta[:, :, :, None, :], dim=-1) # [B, grd_H, grd_W, 3]
            dXs_dshiftu = torch.sum(Rs * dXw_dshiftu[:, None, :], dim=-1)[:, None, None, :].repeat(1, grd_H, grd_W, 1)
            dXs_dshiftv = torch.sum(Rs * dXw_dshiftv[:, None, :], dim=-1)[:, None, None, :].repeat(1, grd_H, grd_W, 1)

            jac_theta = dXs_dtheta[..., 0:2] / meters_per_pixel
            # [B, grd_H, grd_W, 2] the last "2" refers to "uv"
            jac_shiftu = dXs_dshiftu[..., 0:2] / meters_per_pixel
            jac_shiftv = dXs_dshiftv[..., 0:2] / meters_per_pixel

            if torch.any(torch.isnan(jac_shiftu)):
                print('Nan occurs')
            if torch.any(torch.isnan(jac_shiftv)):
                print('Nan occurs')
            if torch.any(torch.isnan(jac_theta)):
                print('Nan occurs')

            return sat_uv, mask, jac_shiftu, jac_shiftv, jac_theta

        return sat_uv, mask

    def project_map_to_grd(self, sat_f, sat_c, R_FL, T_FL, shift_u, shift_v, theta, level,
                           satmap_sidelength_meters, require_jac=True, depth=None):
        '''
        Args:
            sat_f: [B, C, H, W]
            sat_c: [B, 1, H, W]
            R_FL: [B, 3, 3] fixed for all the images in the Ford dataset
            T_FL: [B, 3] fixed for all the images in the Ford dataset
            shift_u: [B, 1]
            shift_v: [B, 1]
            theta: [B, 1]
            level: scalar, feature level
            satmap_sidelength_meters: scalar, the coverage of satellite maps, fixed
            require_jac:

        Returns:

        '''

        B, C, satmap_sidelength_pixels, _ = sat_f.size()
        A = satmap_sidelength_pixels

        uv, mask, jac_shiftu, jac_shiftv, jac_theta = self.cam2body2world2sat(R_FL, T_FL, shift_u, shift_v, theta, level,
                           satmap_sidelength_meters, satmap_sidelength_pixels, require_jac=True, depth=depth)
        # [B, H, W, 2], [B, H, W], [B, H, W, 2], [B, H, W, 2], [B, H, W, 2]
        # # --------------------------------------------------------------------------------------------------
        # def cam2body2world2sat(shift_u, shift_v, theta):
        #     '''
        #     Args:
        #         shift_u: [B, 1] within [-1, 1], initialize as 0
        #         shift_v: [B, 1] within [-1, 1], initialize as 0
        #         theta: [B, 1] within [-1, 1], initialize as 0
        #     Returns:
        #
        #     '''
        #     B = shift_u.shape[0]
        #     Xc = self.xyz_grds[level][0].detach().to(shift_u.device).repeat(B, 1, 1, 1)  # [B, grd_H, grd_W, 3]
        #     Xb = torch.sum(R_FL[:, None, None, :, :] * Xc[:, :, :, None, :], dim=-1) + T_FL  # [B, grd_H, grd_W, 3]
        #     grd_H, grd_W = Xb.shape[1:3]
        #
        #     shift_u_meters = self.args.shift_range_lat * shift_u
        #     shift_v_meters = self.args.shift_range_lon * shift_v
        #     Tw = torch.cat([shift_v_meters, -shift_u_meters, torch.zeros_like(shift_v_meters)], dim=-1)  # [B, 3]
        #
        #     yaw = theta * self.args.rotation_range / 180 * np.pi
        #     cos = torch.cos(yaw)
        #     sin = torch.sin(yaw)
        #     zeros = torch.zeros_like(cos)
        #     ones = torch.ones_like(cos)
        #     Rw = torch.cat([cos, sin, zeros, -sin, cos, zeros, zeros, zeros, ones], dim=-1)  # shape = [B, 9]
        #     Rw = Rw.view(B, 3, 3)  # shape = [B, 3, 3]
        #     Xw = torch.sum(Rw[:, None, None, :, :] * (Xb[:, :, :, None, :] + Tw[:, None, None, None, :]), dim=-1)
        #     # [B, grd_H, grd_W, 3]
        #
        #     Rs = torch.tensor([0, 1, 0, -1, 0, 0, 0, 0, 1], dtype=torch.float32, device=yaw.device).reshape(3, 3)
        #     Rs = Rs.unsqueeze(dim=0).repeat(B, 1, 1)
        #     Xs = torch.sum(Rs[:, None, None, :, :] * Xw[:, :, :, None, :], dim=-1)
        #
        #     meters_per_pixel = satmap_sidelength_meters / satmap_sidelength_pixels
        #     sat_uv = Xs[..., :2] / meters_per_pixel - satmap_sidelength_pixels // 2
        #
        #     return sat_uv
        #
        # auto_jac = torch.autograd.functional.jacobian(cam2body2world2sat, (shift_u, shift_v, theta))
        # auto_jac_shiftu = auto_jac[0][:, :, :, :, 0, 0]  # [B(1), H, W, 2]
        # diffu = torch.abs(auto_jac_shiftu - jac_shiftu)
        # auto_jac_shiftv = auto_jac[1][:, :, :, :, 0, 0]  # [B(1), H, W, 2]
        # diffv = torch.abs(auto_jac_shiftv - jac_shiftv)
        #
        # auto_jac_heading = auto_jac[2][:, :, :, :, 0, 0]
        # diffttheta = torch.abs(auto_jac_heading - jac_theta)
        # theta_np = jac_theta[0].data.cpu().numpy()
        # auto_theta_np = auto_jac_heading[0].data.cpu().numpy()
        # diffu_np = diffu.data.cpu().numpy()
        # diffv_np = diffv.data.cpu().numpy()
        # diff_theta_np = diffttheta.data.cpu().numpy()
        # # --------------------------------------------------------------------------------------------------

        B, grd_H, grd_W, _ = uv.shape
        if require_jac:
            jac = torch.stack([jac_shiftu, jac_shiftv, jac_theta], dim=0)  # [3, B, H, W, 2]

            if torch.any(torch.isnan(jac)):
                print('nan occurs')
        else:
            jac = None

        try:
            sat_f_trans, new_jac = grid_sample(sat_f,
                                               uv,
                                               jac)
            # [B, C, H, W], [3, B, C, H, W]
        except Exception as e:
            print("exception happened: ", e)
            print('shift u: ', shift_u)
            print('shift v: ', shift_v)
            print('heading: ', theta)
            print('satmap_sidelength_meters: ', satmap_sidelength_meters)
            print('satmap_sidelength_pixels: ', satmap_sidelength_pixels)
            import scipy.io as scio
            scio.savemat('data.mat', {'R_FL': R_FL, 'T_FL': T_FL, 'uv': uv})

        sat_f_trans = sat_f_trans * mask[:, None, :, :]
        if require_jac:
            new_jac = new_jac * mask[None, :, None, :, :]

        if sat_c is not None:
            sat_c_trans, _ = grid_sample(sat_c,
                                         uv)  # [B, 1, H, W]
            sat_c_trans = sat_c_trans * mask[:, None, :, :]
        else:
            sat_c_trans = None

        return sat_f_trans, sat_c_trans, new_jac, uv * mask[:, :, :, None], mask

    def LM_update(self, shift_u, shift_v, theta, sat_feat_proj, sat_conf_proj, grd_feat, grd_conf, dfeat_dpose):
        '''
        Args:
            shift_u: [B, 1]
            shift_v: [B, 1]
            theta: [B, 1]
            sat_feat_proj: [B, C, H, W]
            sat_conf_proj: [B, 1, H, W]
            grd_feat: [B, C, H, W]
            grd_conf: [B, 1, H, W]
            dfeat_dpose: [3, B, C, H, W]

        Returns:

        '''

        if self.args.train_damping:
            # damping = self.damping
            min_, max_ = -6, 5
            damping = 10.**(min_ + self.damping.sigmoid()*(max_ - min_))
        else:
            damping = (self.args.damping * torch.ones(size=(1, 3), dtype=torch.float32, requires_grad=True)).to(
                dfeat_dpose.device)

        N, B, C, H, W = dfeat_dpose.shape

        if self.args.dropout > 0:
            inds = np.random.permutation(np.arange(H * W))[: H*W//2]
            dfeat_dpose = dfeat_dpose.reshape(N, B, C, -1)[:, :, :, inds].reshape(N, B, -1)
            sat_feat_proj = sat_feat_proj.reshape(B, C, -1)[:, :, inds].reshape(B, -1)
            grd_feat = grd_feat.reshape(B, C, -1)[:, :, inds].reshape(B, -1)
            sat_conf_proj = sat_conf_proj.reshape(B, -1)[:, inds]
            grd_conf = grd_conf.reshape(B, -1)[:, inds]
        else:
            dfeat_dpose = dfeat_dpose.reshape(N, B, -1)
            sat_feat_proj = sat_feat_proj.reshape(B, -1)
            grd_feat = grd_feat.reshape(B, -1)
            sat_conf_proj = sat_conf_proj.reshape(B, -1)
            grd_conf = grd_conf.reshape(B, -1)

        sat_feat_norm = torch.norm(sat_feat_proj, p=2, dim=-1)
        sat_feat_norm = torch.maximum(sat_feat_norm, 1e-6 * torch.ones_like(sat_feat_norm))
        sat_feat_proj = sat_feat_proj / sat_feat_norm[:, None]
        dfeat_dpose = dfeat_dpose / sat_feat_norm[None, :, None]  # [N, B, D]

        grd_feat_norm = torch.norm(grd_feat, p=2, dim=-1)
        grd_feat_norm = torch.maximum(grd_feat_norm, 1e-6 * torch.ones_like(grd_feat_norm))
        grd_feat = grd_feat / grd_feat_norm[:, None]


        r = sat_feat_proj - grd_feat  # [B, D]

        if self.using_weight:
            # weight = (sat_conf_proj * grd_conf).repeat(1, C, 1, 1).reshape(B, C * H * W)
            weight = (grd_conf[:, None, :]).repeat(1, C, 1).reshape(B, -1)
        else:
            weight = torch.ones([B, grd_feat.shape[-1]], dtype=torch.float32, device=shift_u.device, requires_grad=True)

        J = dfeat_dpose.permute(1, 2, 0)  # [B, C*H*W, #pose]
        temp = J.transpose(1, 2) * weight.unsqueeze(dim=1)
        Hessian = temp @ J  # [B, #pose, #pose]
        if self.args.use_hessian:
            diag_H = torch.diag_embed(torch.diagonal(Hessian, dim1=1, dim2=2))  # [B, 3, 3]
        else:
            diag_H = torch.eye(Hessian.shape[-1], requires_grad=True).unsqueeze(dim=0).repeat(B, 1, 1).to(
                Hessian.device)
        delta_pose = - torch.inverse(Hessian + damping * diag_H) \
                     @ temp @ r.reshape(B, -1, 1)

        shift_u_new = shift_u + delta_pose[:, 0:1, 0]
        shift_v_new = shift_v + delta_pose[:, 1:2, 0]
        theta_new = theta + delta_pose[:, 2:3, 0]

        rand_u = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_v = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_u.requires_grad = True
        rand_v.requires_grad = True
        shift_u_new = torch.where((shift_u_new > -2.5) & (shift_u_new < 2.5), shift_u_new, rand_u)
        shift_v_new = torch.where((shift_v_new > -2.5) & (shift_v_new < 2.5), shift_v_new, rand_v)
        # shift_u_new = torch.where((shift_u_new > -2) & (shift_u_new < 2), shift_u_new, rand_u)
        # shift_v_new = torch.where((shift_v_new > -2) & (shift_v_new < 2), shift_v_new, rand_v)

        if torch.any(torch.isnan(theta_new)):
            print('theta_new is nan')
            print(theta, delta_pose[:, 2:3, 0], Hessian)

        return shift_u_new, shift_v_new, theta_new

    # def LM_update(self, shift_u, shift_v, theta, sat_feat_proj, sat_conf_proj, grd_feat, grd_conf, dfeat_dpose):
    #     '''
    #     Args:
    #         shift_u: [B, 1]
    #         shift_v: [B, 1]
    #         theta: [B, 1]
    #         sat_feat_proj: [B, C, H, W]
    #         sat_conf_proj: [B, 1, H, W]
    #         grd_feat: [B, C, H, W]
    #         grd_conf: [B, 1, H, W]
    #         dfeat_dpose: [3, B, C, H, W]
    #
    #     Returns:
    #
    #     '''
    #     N, B, C, H, W = dfeat_dpose.shape
    #
    #     sat_feat_norm = torch.norm(sat_feat_proj.reshape([B, -1]), p=2, dim=-1)
    #     sat_feat_proj = sat_feat_proj / sat_feat_norm[:, None, None, None]
    #     dfeat_dpose = dfeat_dpose / sat_feat_norm[None, :, None, None, None]
    #
    #     if self.args.train_damping:
    #         # damping = self.damping
    #         min_, max_ = -6, 5
    #         damping = 10.**(min_ + self.damping.sigmoid()*(max_ - min_))
    #     else:
    #         damping = (self.args.damping * torch.ones(size=(1, 3), dtype=torch.float32, requires_grad=True)).to(
    #             dfeat_dpose.device)
    #     r = sat_feat_proj - grd_feat  # [B, C, H, W]
    #
    #     if self.using_weight:
    #         # weight = (sat_conf_proj * grd_conf).repeat(1, C, 1, 1).reshape(B, C * H * W)
    #         weight = (grd_conf).repeat(1, C, 1, 1).reshape(B, C * H * W)
    #     else:
    #         weight = torch.ones([B, C * H * W], dtype=torch.float32, device=shift_u.device, requires_grad=True)
    #
    #     J = dfeat_dpose.flatten(start_dim=2).permute(1, 2, 0)  # [B, C*H*W, #pose]
    #     temp = J.transpose(1, 2) * weight.unsqueeze(dim=1)
    #     Hessian = temp @ J  # [B, #pose, #pose]
    #     if self.args.use_hessian:
    #         diag_H = torch.diag_embed(torch.diagonal(Hessian, dim1=1, dim2=2))  # [B, 3, 3]
    #     else:
    #         diag_H = torch.eye(Hessian.shape[-1], requires_grad=True).unsqueeze(dim=0).repeat(B, 1, 1).to(
    #             Hessian.device)
    #     delta_pose = - torch.inverse(Hessian + damping * diag_H) \
    #                  @ temp @ r.reshape(B, C * H * W, 1)
    #
    #     shift_u_new = shift_u + delta_pose[:, 0:1, 0]
    #     shift_v_new = shift_v + delta_pose[:, 1:2, 0]
    #     theta_new = theta + delta_pose[:, 2:3, 0]
    #
    #     rand_u = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
    #     rand_v = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
    #     rand_u.requires_grad = True
    #     rand_v.requires_grad = True
    #     shift_u_new = torch.where((shift_u_new > -2.5) & (shift_u_new < 2.5), shift_u_new, rand_u)
    #     shift_v_new = torch.where((shift_v_new > -2.5) & (shift_v_new < 2.5), shift_v_new, rand_v)
    #     # shift_u_new = torch.where((shift_u_new > -2) & (shift_u_new < 2), shift_u_new, rand_u)
    #     # shift_v_new = torch.where((shift_v_new > -2) & (shift_v_new < 2), shift_v_new, rand_v)
    #
    #     if torch.isnan(theta_new):
    #         print('theta_new is nan')
    #         print(theta, delta_pose[:, 2:3, 0], Hessian)
    #
    #     return shift_u_new, shift_v_new, theta_new

    def GN_update(self, shift_u, shift_v, theta, sat_feat_proj, sat_conf_proj, grd_feat, grd_conf, dfeat_dpose):
        '''
        Args:
            shift_u: [B, 1]
            shift_v: [B, 1]
            theta: [B, 1]
            sat_feat_proj: [B, C, H, W]
            sat_conf_proj: [B, 1, H, W]
            grd_feat: [B, C, H, W]
            grd_conf: [B, 1, H, W]
            dfeat_dpose: [3, B, C, H, W]

        Returns:

        '''
        N, B, C, H, W = dfeat_dpose.shape

        sat_feat_norm = torch.norm(sat_feat_proj.reshape([B, -1]), p=2, dim=-1)
        sat_feat_proj = sat_feat_proj / sat_feat_norm[:, None, None, None]
        dfeat_dpose = dfeat_dpose / sat_feat_norm[None, :, None, None, None]

        if self.args.train_damping:
            # damping = self.damping
            min_, max_ = -6, 5
            damping = 10.**(min_ + self.damping.sigmoid()*(max_ - min_))
        else:
            damping = (self.args.damping * torch.ones(size=(1, 3), dtype=torch.float32, requires_grad=True)).to(
                dfeat_dpose.device)
        r = sat_feat_proj - grd_feat  # [B, C, H, W]

        if self.using_weight:
            # weight = (sat_conf_proj * grd_conf).repeat(1, C, 1, 1).reshape(B, C * H * W)
            weight = (grd_conf).repeat(1, C, 1, 1).reshape(B, C * H * W)
        else:
            weight = torch.ones([B, C * H * W], dtype=torch.float32, device=shift_u.device, requires_grad=True)

        J = dfeat_dpose.flatten(start_dim=2).permute(1, 2, 0)  # [B, C*H*W, #pose]
        temp = J.transpose(1, 2) * weight.unsqueeze(dim=1)
        Hessian = temp @ J  # [B, #pose, #pose]
        # diag_H = torch.diag_embed(torch.diagonal(Hessian, dim1=1, dim2=2))  # [B, 3, 3]
        diag_H = torch.eye(Hessian.shape[-1], requires_grad=True).unsqueeze(dim=0).repeat(B, 1, 1).to(
            Hessian.device)
        # delta_pose = - torch.inverse(Hessian + damping * diag_H) \
        #              @ temp @ r.reshape(B, C * H * W, 1)
        delta_pose = - torch.inverse(Hessian) \
                     @ temp @ r.reshape(B, C * H * W, 1)

        shift_u_new = shift_u + delta_pose[:, 0:1, 0]
        shift_v_new = shift_v + delta_pose[:, 1:2, 0]
        theta_new = theta + delta_pose[:, 2:3, 0]

        rand_u = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_v = torch.distributions.uniform.Uniform(-1, 1).sample([B, 1]).to(shift_u.device)
        rand_u.requires_grad = True
        rand_v.requires_grad = True
        shift_u_new = torch.where((shift_u_new > -2.5) & (shift_u_new < 2.5), shift_u_new, rand_u)
        shift_v_new = torch.where((shift_v_new > -2.5) & (shift_v_new < 2.5), shift_v_new, rand_v)
        # shift_u_new = torch.where((shift_u_new > -2) & (shift_u_new < 2), shift_u_new, rand_u)
        # shift_v_new = torch.where((shift_v_new > -2) & (shift_v_new < 2), shift_v_new, rand_v)

        if torch.isnan(theta_new):
            print('theta_new is nan')
            print(theta, delta_pose[:, 2:3, 0], Hessian)

        return shift_u_new, shift_v_new, theta_new

    def NN_update(self, shift_u, shift_v, theta, sat_feat_proj, sat_conf_proj, grd_feat, grd_conf, dfeat_dpose):

        delta = self.NNrefine(sat_feat_proj, grd_feat)  # [B, 3]

        shift_u_new = shift_u + delta[:, 0]
        shift_v_new = shift_v + delta[:, 1]
        theta_new = theta + delta[:, 2]
        return shift_u_new, shift_v_new, theta_new

    def SGD_update(self, shift_u, shift_v, theta, sat_feat_proj, sat_conf_proj, grd_feat, grd_conf, dfeat_dpose):
        '''
        Args:
            shift: [B, 2]
            heading: [B, 1]
            sat_feat_proj: [B, C, H, W]
            sat_conf_proj: [B, 1, H, W]
            grd_feat: [B, C, H, W]
            grd_conf: [B, 1, H, W]
            dfeat_dpose: [3, B, C, H, W]
        Returns:
        '''

        B, C, H, W = grd_feat.shape
        r = sat_feat_proj - grd_feat  # [B, C, H, W]

        idx0 = torch.le(r, 0)
        idx1 = torch.greater(r, 0)
        mask = idx0 * (-1) + idx1
        dr_dfeat = mask.float() / (C * H * W)  # [B, C, H, W]
        delta_pose = torch.sum(dr_dfeat[None, ...] * dfeat_dpose, dim=[2, 3, 4]).transpose(0, 1)  # [B, #pose]

        shift_u_new = shift_u - 0.001 * delta_pose[:, 0, 0]
        shift_v_new = shift_v - 0.001 * delta_pose[:, 1, 0]
        theta_new = theta - 0.001 * delta_pose[:, 2, 0]
        return shift_u_new, shift_v_new, theta_new

    # def SelfTransformer(self, feat, feat_grd, level):
    #     '''
    #     Args:
    #         feat: [B, C, H, W]
    #     Returns:
    #     '''
    #     B, C, H, W = feat.shape
    #     # feat0 = self.pe(feat.permute(0, 3, 2, 1).reshape(B * W, H, C))
    #     feat0 = self.pe(feat).reshape(B, C, H * W).transpose(1, 2)
    #
    #     feat1 = self.transformers[level](feat0, feat_grd.reshape(B, C, H*W).transpose(1, 2))
    #     feat2 = feat1.reshape(B, H, W, C).permute(0, 3, 1, 2)
    #     # feat2 = feat1.reshape(B, W, H, C).permute(0, 3, 2, 1)
    #
    #     return feat2

    def forward_iters_level(self, sat_map, grd_img_left, satmap_sidelength_meters, R_FL, T_FL,
                gt_shift_u=None, gt_shift_v=None, gt_theta=None, mode='train',
                file_name=None, loop=0):
        '''
        :param sat_map: [B, C, A, A] A--> sidelength
        :param left_camera_k: [B, 3, 3]
        :param grd_img_left: [B, C, H, W]
        :return:
        '''

        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        # sat_img_proj, _, _, sat_uv, _ = self.project_map_to_grd(
        #     sat_map, None, R_FL, T_FL, gt_shift_u[:, None], gt_shift_v[:, None], gt_theta[:, None], 3, satmap_sidelength_meters, require_jac=False)
        # # [B, C, H, W],  [B, H, W, 2]
        # print(gt_shift_u, gt_shift_v)
        #
        # sat_img = transforms.ToPILImage()(sat_img_proj[0])
        # sat_img.save('sat_proj.png')
        # grd = transforms.ToPILImage()(grd_img_left[0])
        # grd.save('grd.png')
        # sat = transforms.ToPILImage()(sat_map[0])
        # sat.save('sat.png')

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)

        if self.estimate_depth:
            grd_feat_list, grd_conf_list, grd_depth_list = self.GrdFeatureNet(grd_img_left)
        else:
            grd_feat_list, grd_conf_list = self.GrdFeatureNet(grd_img_left)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        theta = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        gt_uv_dict = {}
        gt_feat_dict = {}
        pred_uv_dict = {}
        pred_feat_dict = {}
        shift_us_all = []
        shift_vs_all = []
        thetas_all = []
        for iter in range(self.N_iters):
            shift_us = []
            shift_vs = []
            thetas = []
            for level in range(len(sat_feat_list)):
                sat_feat = sat_feat_list[level]
                sat_conf = sat_conf_list[level]
                grd_feat = grd_feat_list[level]
                grd_conf = grd_conf_list[level]
                # grd_conf = 1. / (1. + grd_conf_list[level])

                if self.estimate_depth:
                    grd_depth = grd_depth_list[level]
                else:
                    grd_depth = None

                grd_H, grd_W = grd_feat.shape[-2:]
                sat_feat_proj, sat_conf_proj, dfeat_dpose, sat_uv, mask = self.project_map_to_grd(
                    sat_feat, sat_conf, R_FL, T_FL, shift_u, shift_v, theta, level, satmap_sidelength_meters,
                    require_jac=True, depth=grd_depth)
                # [B, C, H, W], [B, 1, H, W], [3, B, C, H, W], [B, H, W, 2]
                # print('sat_proj.shape ', sat_feat_proj.shape)
                sat_conf_proj = 1 / (1 + sat_conf_proj)  # [B, 1, H, W]
                # sat_feat_proj, _, dfeat_dpose, sat_uv, mask = self.project_map_to_grd(
                #     sat_feat, None, R_FL, T_FL, shift_u, shift_v, theta, level, satmap_sidelength_meters,
                #     require_jac=True)
                # # [B, C, H, W], [B, 1, H, W], [3, B, C, H, W], [B, H, W, 2]
                # sat_conf_proj = nn.Sigmoid()(-self.confs[level](sat_feat_proj))

                grd_feat = grd_feat * mask[:, None, :, :]
                grd_conf = grd_conf * mask[:, None, :, :]

                # if self.args.transformer:
                #     # sat_feat_new, grd_feat_new = self.SatGrdTransformer(sat_feat_proj, grd_feat, level)
                #     # sat_feat_new = sat_feat_new * mask[:, None, :, :]
                #     # grd_feat_new = grd_feat_new * mask[:, None, :, :]
                #     sat_feat_proj = self.SelfTransformer(sat_feat_proj[:, :, grd_H // 2:, :],
                #                                          grd_feat[:, :, grd_H // 2:, :], level)
                #     sat_feat_proj = sat_feat_proj * mask[:, None, grd_H // 2:, :]

                # else:
                #     sat_feat_new = sat_feat_proj * mask[:, None, :, :]
                #     grd_feat_new = grd_feat * mask[:, None, :, :]

                if self.args.proj == 'geo':
                    sat_feat_new = sat_feat_proj[:, :, grd_H // 2:, :]
                    sat_conf_new = sat_conf_proj[:, :, grd_H // 2:, :]
                    grd_feat_new = grd_feat[:, :, grd_H // 2:, :]
                    grd_conf_new = grd_conf[:, :, grd_H // 2:, :]
                    dfeat_dpose_new = dfeat_dpose[:, :, :, grd_H // 2:, :]
                else:
                    sat_feat_new = sat_feat_proj
                    sat_conf_new = sat_conf_proj
                    grd_feat_new = grd_feat
                    grd_conf_new = grd_conf
                    dfeat_dpose_new = dfeat_dpose

                if self.args.Optimizer == 'LM':
                    shift_u_new, shift_v_new, theta_new = self.LM_update(shift_u, shift_v, theta,
                                                            sat_feat_new,
                                                            sat_conf_new,
                                                            grd_feat_new,
                                                            grd_conf_new,
                                                            dfeat_dpose_new)  # only need to compare bottom half
                elif self.args.Optimizer == 'SGD':
                    # r = sat_feat_proj[:, :, grd_H // 2:, :] - grd_feat[:, :, grd_H // 2:, :]
                    # p = torch.mean(torch.abs(r), dim=[1, 2, 3])  # *100 #* 256 * 256 * 3
                    # dp_dshift = torch.autograd.grad(p, shift, retain_graph=True, create_graph=True,
                    #                              only_inputs=True)[0]
                    # dp_dheading = torch.autograd.grad(p, heading, retain_graph=True, create_graph=True,
                    #                                 only_inputs=True)[0]

                    shift_u_new, shift_v_new, theta_new = self.SGD_update(shift_u, shift_v, theta,
                                                             sat_feat_new[:, :, grd_H // 2:, :],
                                                             sat_conf_proj[:, :, grd_H // 2:, :],
                                                             grd_feat_new[:, :, grd_H // 2:, :],
                                                             grd_conf[:, :, grd_H // 2:, :],
                                                             dfeat_dpose[:, :, :, grd_H // 2:,
                                                             :])  # only need to compare bottom half
                    # print(shift_new - (shift - 0.001 * dp_dshift))
                    # print(heading_new - (heading - 0.001 * dp_dheading))
                elif self.args.Optimizer == 'GN':
                    shift_u_new, shift_v_new, theta_new = self.GN_update(shift_u, shift_v, theta,
                                                                         sat_feat_new,
                                                                         sat_conf_new,
                                                                         grd_feat_new,
                                                                         grd_conf_new,
                                                                         dfeat_dpose_new)
                elif self.args.Optimizer == 'NN':
                    shift_u_new, shift_v_new, theta_new = self.NN_update(shift_u, shift_v, theta,
                                                                         sat_feat_new,
                                                                         sat_conf_new,
                                                                         grd_feat_new,
                                                                         grd_conf_new,
                                                                         dfeat_dpose_new)

                shift_us.append(shift_u_new[:, 0])  # [B]
                shift_vs.append(shift_v_new[:, 0])  # [B]
                thetas.append(theta_new[:, 0])  # [B]

                shift_u = shift_u_new.clone()
                shift_v = shift_v_new.clone()
                theta = theta_new.clone()

                if level not in pred_feat_dict.keys():
                    pred_feat_dict[level] = [sat_feat_proj]
                    pred_uv_dict[level] = [
                        sat_uv / torch.tensor([sat_feat.shape[-1], sat_feat.shape[-2]], dtype=torch.float32).reshape(1, 1, 1, 2).to(sat_feat.device)
                    ]
                else:
                    pred_feat_dict[level].append(sat_feat_proj)
                    pred_uv_dict[level].append(
                        sat_uv / torch.tensor([sat_feat.shape[-1], sat_feat.shape[-2]], dtype=torch.float32).reshape(1, 1, 1, 2).to(sat_feat.device))

                if level not in gt_uv_dict.keys() and mode == 'train':
                    gt_sat_feat_proj, _, _, gt_uv, _ = self.project_map_to_grd(
                        sat_feat, None, R_FL, T_FL, gt_shift_u[:, None], gt_shift_v[:, None], gt_theta[:, None], level, satmap_sidelength_meters,
                        require_jac=False)
                    # [B, N, C, H, W], [B, N, H, W, 2]
                    gt_feat_dict[level] = gt_sat_feat_proj[:, 0, ...]  # [B, C, H, W]
                    gt_uv_dict[level] = gt_uv[:, 0, ...] / torch.tensor([sat_feat.shape[-1], sat_feat.shape[-2]],
                                                                        dtype=torch.float32).reshape(1, 1, 1, 2).to(
                        sat_feat.device)
                    # [B, H, W, 2]

            shift_us_all.append(torch.stack(shift_us, dim=1))  # [B, Level]
            shift_vs_all.append(torch.stack(shift_vs, dim=1))  # [B, Level]
            thetas_all.append(torch.stack(thetas, dim=1))  # [B, Level]

        shift_lats = torch.stack(shift_us_all, dim=1)  # [B, N_iters, Level]
        shift_lons = torch.stack(shift_vs_all, dim=1)  # [B, N_iters, Level]
        thetas = torch.stack(thetas_all, dim=1)  # [B, N_iters, Level]

        if self.args.visualize:
            from visualize_utils import features_to_RGB, RGB_iterative_pose_ford
            save_dir = './visualize_rot' + str(self.args.rotation_range) + '_ford'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            # features_to_RGB(sat_feat_list, grd_feat_list, pred_feat_dict, gt_feat_dict, loop,
            #                 save_dir)
            RGB_iterative_pose_ford(sat_map, grd_img_left, shift_lats, shift_lons, thetas, gt_shift_u, gt_shift_v, gt_theta,
                               0.22, self.args, loop, save_dir)

        if mode == 'train':

            # gt_shift = torch.stack([gt_shiftu, gt_shiftv], dim=-1)  # [B, 2]
            # gt_heading = gt_theta.reshape(-1, 1)  # [B, 1]
            if self.args.rotation_range == 0:
                coe_heading = 0
            else:
                coe_heading = self.args.coe_heading

            loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
            shift_lat_last, shift_lon_last, theta_last, \
            L1_loss, L2_loss, L3_loss, L4_loss \
                = loss_func(self.args.loss_method, grd_feat_list, pred_feat_dict, gt_feat_dict,
                            shift_lats, shift_lons, thetas, gt_shift_u, gt_shift_v, gt_theta,
                            pred_uv_dict, gt_uv_dict, 
                            self.args.coe_shift_lat, self.args.coe_shift_lon, coe_heading,
                            self.args.coe_L1, self.args.coe_L2, self.args.coe_L3, self.args.coe_L4)

            if self.estimate_depth:
                return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last,\
                       shift_lat_last, shift_lon_last, theta_last, \
                       L1_loss, L2_loss, L3_loss, L4_loss, grd_conf_list, grd_depth_list
            else:
                return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                       shift_lat_last, shift_lon_last, theta_last, \
                       L1_loss, L2_loss, L3_loss, L4_loss, grd_conf_list
        else:
            return torch.stack(shift_us_all, dim=1)[:, -1, -1], torch.stack(shift_vs_all, dim=1)[:, -1, -1], \
                   torch.stack(thetas_all, dim=1)[:, -1, -1]
            # [B], [B], [B]

    def forward_level_iters(self, sat_map, grd_img_left, satmap_sidelength_meters, R_FL, T_FL,
                gt_shift_u=None, gt_shift_v=None, gt_theta=None, mode='train',
                file_name=None):
        '''
        :param sat_map: [B, C, A, A] A--> sidelength
        :param left_camera_k: [B, 3, 3]
        :param grd_img_left: [B, C, H, W]
        :return:
        '''

        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)

        if self.estimate_depth:
            grd_feat_list, grd_conf_list, grd_depth_list = self.GrdFeatureNet(grd_img_left)
        else:
            grd_feat_list, grd_conf_list = self.GrdFeatureNet(grd_img_left)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        theta = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        gt_uv_dict = {}
        gt_feat_dict = {}
        pred_uv_dict = {}
        pred_feat_dict = {}
        shift_us_all = []
        shift_vs_all = []
        thetas_all = []

        for level in range(len(sat_feat_list)):
            shift_us = []
            shift_vs = []
            thetas = []

            sat_feat = sat_feat_list[level]
            sat_conf = sat_conf_list[level]
            grd_feat = grd_feat_list[level]
            grd_conf = grd_conf_list[level]
            # grd_conf = 1. / (1. + grd_conf_list[level])

            if self.estimate_depth:
                grd_depth = grd_depth_list[level]
            else:
                grd_depth = None

            for iter in range(self.N_iters):

                grd_H, grd_W = grd_feat.shape[-2:]
                sat_feat_proj, sat_conf_proj, dfeat_dpose, sat_uv, mask = self.project_map_to_grd(
                    sat_feat, sat_conf, R_FL, T_FL, shift_u, shift_v, theta, level, satmap_sidelength_meters,
                    require_jac=True, depth=grd_depth)
                # [B, C, H, W], [B, 1, H, W], [3, B, C, H, W], [B, H, W, 2]
                # print('sat_proj.shape ', sat_feat_proj.shape)
                sat_conf_proj = 1 / (1 + sat_conf_proj)  # [B, 1, H, W]
                # sat_feat_proj, _, dfeat_dpose, sat_uv, mask = self.project_map_to_grd(
                #     sat_feat, None, R_FL, T_FL, shift_u, shift_v, theta, level, satmap_sidelength_meters,
                #     require_jac=True)
                # # [B, C, H, W], [B, 1, H, W], [3, B, C, H, W], [B, H, W, 2]
                # sat_conf_proj = nn.Sigmoid()(-self.confs[level](sat_feat_proj))

                grd_feat = grd_feat * mask[:, None, :, :]
                grd_conf = grd_conf * mask[:, None, :, :]

                if self.args.Optimizer == 'LM':
                    shift_u_new, shift_v_new, theta_new = self.LM_update(shift_u, shift_v, theta,
                                                            sat_feat_proj[:, :, grd_H // 2:, :],
                                                            sat_conf_proj[:, :, grd_H // 2:, :],
                                                            grd_feat[:, :, grd_H // 2:, :],
                                                            grd_conf[:, :, grd_H // 2:, :],
                                                            dfeat_dpose[:, :, :, grd_H // 2:, :])  # only need to compare bottom half
                elif self.args.Optimizer == 'SGD':
                    # r = sat_feat_proj[:, :, grd_H // 2:, :] - grd_feat[:, :, grd_H // 2:, :]
                    # p = torch.mean(torch.abs(r), dim=[1, 2, 3])  # *100 #* 256 * 256 * 3
                    # dp_dshift = torch.autograd.grad(p, shift, retain_graph=True, create_graph=True,
                    #                              only_inputs=True)[0]
                    # dp_dheading = torch.autograd.grad(p, heading, retain_graph=True, create_graph=True,
                    #                                 only_inputs=True)[0]

                    shift_u_new, shift_v_new, theta_new = self.SGD_update(shift_u, shift_v, theta,
                                                             sat_feat_proj[:, :, grd_H // 2:, :],
                                                             sat_conf_proj[:, :, grd_H // 2:, :],
                                                             grd_feat[:, :, grd_H // 2:, :],
                                                             grd_conf[:, :, grd_H // 2:, :],
                                                             dfeat_dpose[:, :, :, grd_H // 2:,
                                                             :])  # only need to compare bottom half

                    # print(shift_new - (shift - 0.001 * dp_dshift))
                    # print(heading_new - (heading - 0.001 * dp_dheading))

                # shifts.append(torch.cat([shift_v_new, shift_u_new], dim=-1))
                shift_us.append(shift_u_new[:, 0])  # [B]
                shift_vs.append(shift_v_new[:, 0])  # [B]
                thetas.append(theta_new[:, 0])  # [B]

                shift_u = shift_u_new.clone()
                shift_v = shift_v_new.clone()
                theta = theta_new.clone()

                if level not in pred_feat_dict.keys():
                    pred_feat_dict[level] = [sat_feat_proj]
                    pred_uv_dict[level] = [
                        sat_uv / torch.tensor([sat_feat.shape[-1], sat_feat.shape[-2]], dtype=torch.float32).reshape(1, 1, 1, 2).to(sat_feat.device)
                    ]
                else:
                    pred_feat_dict[level].append(sat_feat_proj)
                    pred_uv_dict[level].append(
                        sat_uv / torch.tensor([sat_feat.shape[-1], sat_feat.shape[-2]], dtype=torch.float32).reshape(1, 1, 1, 2).to(sat_feat.device))

                if level not in gt_uv_dict.keys() and mode == 'train':
                    gt_sat_feat_proj, _, _, gt_uv, _ = self.project_map_to_grd(
                        sat_feat, None, R_FL, T_FL, gt_shift_u[:, None], gt_shift_v[:, None], gt_theta[:, None], level, satmap_sidelength_meters,
                        require_jac=False)
                    # [B, N, C, H, W], [B, N, H, W, 2]
                    gt_feat_dict[level] = gt_sat_feat_proj[:, 0, ...]  # [B, C, H, W]
                    gt_uv_dict[level] = gt_uv[:, 0, ...] / torch.tensor([sat_feat.shape[-1], sat_feat.shape[-2]],
                                                                        dtype=torch.float32).reshape(1, 1, 1, 2).to(
                        sat_feat.device)
                    # [B, H, W, 2]

            shift_us_all.append(torch.stack(shift_us, dim=1))  # [B, N_iters]
            shift_vs_all.append(torch.stack(shift_vs, dim=1))  # [B, N_iters]
            thetas_all.append(torch.stack(thetas, dim=1))  # [B, N_iters]

        shift_lats = torch.stack(shift_us_all, dim=2)  # [B, N_iters, Level]
        shift_lons = torch.stack(shift_vs_all, dim=2)  # [B, N_iters, Level]
        thetas = torch.stack(thetas_all, dim=2)  # [B, N_iters, Level]

        if mode == 'train':

            # gt_shift = torch.stack([gt_shiftu, gt_shiftv], dim=-1)  # [B, 2]
            # gt_heading = gt_theta.reshape(-1, 1)  # [B, 1]
            if self.args.rotation_range == 0:
                coe_heading = 0
            else:
                coe_heading = self.args.coe_heading

            loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
            shift_lat_last, shift_lon_last, theta_last, \
            L1_loss, L2_loss, L3_loss, L4_loss \
                = loss_func(self.args.loss_method, grd_feat_list, pred_feat_dict, gt_feat_dict,
                            shift_lats, shift_lons, thetas, gt_shift_u, gt_shift_v, gt_theta,
                            pred_uv_dict, gt_uv_dict,
                            self.args.coe_shift_lat, self.args.coe_shift_lon, coe_heading,
                            self.args.coe_L1, self.args.coe_L2, self.args.coe_L3, self.args.coe_L4)

            if self.estimate_depth:
                return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last,\
                       shift_lat_last, shift_lon_last, theta_last, \
                       L1_loss, L2_loss, L3_loss, L4_loss, grd_conf_list, grd_depth_list
            else:
                return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                       shift_lat_last, shift_lon_last, theta_last, \
                       L1_loss, L2_loss, L3_loss, L4_loss, grd_conf_list
        else:
            return torch.stack(shift_us_all, dim=1)[:, -1, -1], torch.stack(shift_vs_all, dim=1)[:, -1, -1], \
                   torch.stack(thetas_all, dim=1)[:, -1, -1]
            # [B], [B], [B]

    def forward(self, sat_map, grd_img_left, satmap_sidelength_meters, R_FL, T_FL,
                gt_shift_u=None, gt_shift_v=None, gt_theta=None, mode='train',
                file_name=None, level_first=0, loop=0):
        if level_first==0:
            return self.forward_iters_level(sat_map, grd_img_left, satmap_sidelength_meters, R_FL, T_FL,
                gt_shift_u, gt_shift_v, gt_theta, mode, file_name, loop)
        else:
            return self.forward_level_iters(sat_map, grd_img_left, satmap_sidelength_meters, R_FL, T_FL,
                gt_shift_u, gt_shift_v, gt_theta, mode, file_name)



# plz only use loss_method=0, other loss methods (1, 2, 3) are failure trials.
def loss_func(loss_method, ref_feat_list, pred_feat_dict, gt_feat_dict, shift_lats, shift_lons, thetas,
              gt_shift_lat, gt_shift_lon, gt_theta, pred_uv_dict, gt_uv_dict,
              coe_shift_lat=100, coe_shift_lon=100, coe_theta=100, coe_L1=100, coe_L2=100, coe_L3=100, coe_L4=100):
    '''
    Args:
        loss_method:
        ref_feat_list:
        pred_feat_dict:
        gt_feat_dict:
        shift_lats: [B, N_iters, Level]
        shift_lons: [B, N_iters, Level]
        thetas: [B, N_iters, Level]
        gt_shift_lat: [B]
        gt_shift_lon: [B]
        gt_theta: [B]
        pred_uv_dict:
        gt_uv_dict:
        coe_shift_lat:
        coe_shift_lon:
        coe_theta:
        coe_L1:
        coe_L2:
        coe_L3:
        coe_L4:

    Returns:

    '''
    B = gt_shift_lat.shape[0]
    # shift_lats = torch.stack(shift_lats_all, dim=1)  # [B, N_iters, Level]
    # shift_lons = torch.stack(shift_lons_all, dim=1)  # [B, N_iters, Level]
    # thetas = torch.stack(thetas_all, dim=1)  # [B, N_iters, Level]

    shift_lat_delta0 = torch.abs(shift_lats - gt_shift_lat[:, None, None])  # [B, N_iters, Level]
    shift_lon_delta0 = torch.abs(shift_lons - gt_shift_lon[:, None, None])  # [B, N_iters, Level]
    thetas_delta0 = torch.abs(thetas - gt_theta[:, None, None])  # [B, N_iters, level]

    shift_lat_delta = torch.mean(shift_lat_delta0, dim=0)  # [N_iters, Level]
    shift_lon_delta = torch.mean(shift_lon_delta0, dim=0)  # [N_iters, Level]
    thetas_delta = torch.mean(thetas_delta0, dim=0)  # [N_iters, level]

    shift_lat_decrease = shift_lat_delta[0] - shift_lat_delta[-1]  # [level]
    shift_lon_decrease = shift_lon_delta[0] - shift_lon_delta[-1]  # [level]
    thetas_decrease = thetas_delta[0] - thetas_delta[-1]  # [level]

    if loss_method == 0:
        losses = coe_shift_lat * shift_lat_delta + coe_shift_lon * shift_lon_delta + coe_theta * thetas_delta  # [N_iters, level]
        loss_decrease = losses[0] - losses[-1]  # [level]
        loss = torch.mean(losses)  # mean or sum
        loss_last = losses[-1]

        return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
               shift_lat_delta[-1], shift_lon_delta[-1], thetas_delta[-1], None, None, None, None
        # scalar, [Level] either tensor or list, [Level], [level], [level], [level], 

    elif loss_method == 1:
        losses = coe_shift_lat * shift_lat_delta + coe_shift_lon * shift_lon_delta + coe_theta * thetas_delta  # [N_iters, level]
        loss_decrease = losses[0] - losses[-1]  # [level]
        loss0 = torch.mean(losses)  # mean or sum
        loss_last = losses[-1]

        triplet_loss = []
        masks = (shifts_delta0[..., 0] > 0.001) & (shifts_delta0[..., 1] > 0.001) & (heading_delta0[..., 0] > 0.01)
        # [B, N_iters, Level]
        for level in range(len(ref_feat_list)):
            ref_feat = ref_feat_list[level]  # [B, C, H, W]  # already normalized
            pred_feat_list = pred_feat_dict[level]  # list, len=iters, item.shape=[B, C, H, W]
            pred_feat = normalize_feature(torch.stack(pred_feat_list, dim=1))  # [B, Niters, C, H, W]
            gt_feat = normalize_feature(gt_feat_dict[level])  # [B, C, H, W]

            pos_dis = 2 - 2 * torch.sum(ref_feat * gt_feat, dim=[-3, -2, -1])  # [B]
            neg_dis = 2 - 2 * torch.sum(ref_feat[:, None, ...] * pred_feat, dim=[-3, -2, -1])  # [B, Niters]

            temp = torch.log(1 + torch.exp(10 * masks[..., level] * (pos_dis[:, None] - neg_dis))) * masks[..., level]  # [B, Niters]
            # temp = torch.sum(temp, dim=0) #[Niters]

            triplet_loss.append(temp)
            # triplet_loss_decrease.append(temp[0] - temp[-1])

        triplet_loss = torch.stack(triplet_loss, dim=-1) # [B, N_iters, Level]
        L1_loss = coe_L1 * torch.sum(triplet_loss)/torch.sum(masks)

        loss = loss0 + L1_loss

        return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
               shift_lat_delta[-1], shift_lon_delta[-1], thetas_delta[-1], L1_loss, None, None, None
        # scalar, [Level] either tensor or list, [Level], [level], [level], [level], 

    elif loss_method == 2:
        losses = coe_shift_lat * shift_lat_delta + coe_shift_lon * shift_lon_delta + coe_theta * thetas_delta  # [N_iters, level]
        loss_decrease = losses[0] - losses[-1]  # [level]
        loss0 = torch.mean(losses)  # mean or sum
        loss_last = losses[-1]

        gt_feat_loss = []
        masks = (shifts_delta0[..., 0] > 0.001) & (shifts_delta0[..., 1] > 0.001) & (heading_delta0[..., 0] > 0.01)
        # [B, N_iters, Level]
        for level in range(len(ref_feat_list)):
            ref_feat = ref_feat_list[level]  # [B, C, H, W]  # already normalized
            # pred_feat_list = pred_feat_dict[level]  # list, len=iters, item.shape=[B, C, H, W]
            # pred_feat = normalize_feature(torch.stack(pred_feat_list, dim=1))  # [B, Niters, C, H, W]
            gt_feat = normalize_feature(gt_feat_dict[level])  # [B, C, H, W]

            pos_dis = 2 - 2 * torch.sum(ref_feat * gt_feat, dim=[-3, -2, -1])  # [B]

            gt_feat_loss.append(pos_dis)
            # triplet_loss_decrease.append(temp[0] - temp[-1])

        gt_feat_loss = torch.stack(gt_feat_loss, dim=-1)  # [B, Level]
        L1_loss = coe_L1 * torch.sum(gt_feat_loss) / B

        loss = loss0 + L1_loss

        return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
               shift_lat_delta[-1], shift_lon_delta[-1], thetas_delta[-1], L1_loss, None, None, None
        # scalar, [Level] either tensor or list, [Level], [level], [level], [level], 
    elif loss_method == 3:
        L1_list = []
        L2_list = []
        L3_list = []
        L4_list = []
        for level in range(len(ref_feat_list)):
            ref_feat = ref_feat_list[level]    # [B, C, H, W]  # already normalized
            pred_feat_list = pred_feat_dict[level]  # list, len=iters, item.shape=[B, C, H, W]
            pred_feat = normalize_feature(torch.stack(pred_feat_list, dim=1))  # [B, Niters, C, H, W]
            gt_feat = normalize_feature(gt_feat_dict[level])    # [B, C, H, W]

            pos_dis = 2 - 2 * torch.sum(ref_feat * gt_feat, dim=[-3, -2, -1])  # [B]
            neg_dis = 2 - 2 * torch.sum(ref_feat[:, None, ...] * pred_feat, dim=[-3, -2, -1])  # [B, Niters]
            neg_dis_update = neg_dis[:, 1:] - neg_dis[:, 0:-1]  # [B, Niters-1]

            pred_uv = pred_uv_dict[level]      # list, len=iters, item.shape=[B, H, W, 2], already normalized within [0, 1]
            gt_uv = gt_uv_dict[level]          # [B, H, W, 2]
            uv_diff = torch.mean(torch.sqrt(torch.sum(torch.square(torch.stack(pred_uv, dim=1) - gt_uv[:, None]),
                                                      dim=-1)), dim=[2, 3])  # [B, Niters]
            mask_neg = torch.greater(uv_diff, 0.002)  # approximately one pixel difference in the original satellite image
            L1 = coe_L1 * torch.log(1 + torch.exp(10 * mask_neg * (pos_dis[:, None] - neg_dis))) * mask_neg # [B, Niters]

            L2 = coe_L2 * uv_diff  # [B, Niters]

            uv_diff_update = uv_diff[:, 1:] - uv_diff[:, 0:-1]  # [B, Niters-1]
            L3 = coe_L3 * torch.log(1 + torch.exp(100 * uv_diff_update))  # [B, Niters-1]

            mask = torch.where(torch.le(uv_diff_update, 0.0), torch.ones_like(uv_diff_update), -1 * torch.ones_like(uv_diff_update))
            L4 = coe_L4 * torch.log(1 + torch.exp(10 * mask * neg_dis_update)) # [B, Niters-1]

            L1_list.append(torch.mean(L1, dim=0))  # [Niters] location-aware feature level triplet loss
            L2_list.append(torch.mean(L2, dim=0))  # [Niters] uv difference, directly reflect pose difference
            L3_list.append(torch.mean(L3, dim=0))  # [Niters-1] uv update triplet loss, we expect each LM iteration will put the pose to the right direction
            L4_list.append(torch.mean(L4, dim=0))  # [Niters-1] distance-aware feature level triplet loss

        L1_loss = torch.stack(L1_list, dim=-1)  # [Niters, Level]
        L2_loss = torch.stack(L2_list, dim=-1)  # [Niters, level]
        L3_loss = torch.stack(L3_list, dim=-1)  # [Niters - 1, level]
        L4_loss = torch.stack(L4_list, dim=-1)  # [Niters - 1, level]

        loss = torch.sum(L1_loss) + torch.sum(L2_loss) + torch.sum(L3_loss) + torch.sum(L4_loss)
        loss_decrease = (L2_loss[0] - L2_loss[-1])
        loss_last = (L2_loss[-1])
        return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
               shift_lat_delta[-1], shift_lon_delta[-1], thetas_delta[-1], \
               L1_loss, L2_loss, L3_loss, L4_loss
 


def normalize_feature(x):
    C, H, W = x.shape[-3:]
    norm = torch.norm(x.flatten(start_dim=-3), dim=-1)
    return x / norm[..., None, None, None]

