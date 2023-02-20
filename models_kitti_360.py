
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from torchvision import transforms
import utils
import os
import torchvision.transforms.functional as TF

# from GRU1 import ElevationEsitimate,VisibilityEsitimate,VisibilityEsitimate2,GRUFuse
from VGG import VGGUnet, VGGUnet_G2S
from jacobian import grid_sample

# from ConvLSTM import VE_LSTM3D, VE_LSTM2D, VE_conv, S_LSTM2D
from models_ford import loss_func
from RNNs import NNrefine

EPS = utils.EPS


class LM_G2SP(nn.Module):
    def __init__(self, args):  # device='cuda:0',
        super(LM_G2SP, self).__init__()
        '''
        loss_method: 0: direct R T loss 1: feat loss 2: noise aware feat loss
        '''
        self.args = args
        
        self.level = args.level
        self.N_iters = args.N_iters
        self.using_weight = args.using_weight
        self.loss_method = args.loss_method

        self.SatFeatureNet = VGGUnet(self.level)
        if self.args.proj == 'nn':
            self.GrdFeatureNet = VGGUnet_G2S(self.level)
        else:
            self.GrdFeatureNet = VGGUnet(self.level)

        self.damping = nn.Parameter(self.args.damping * torch.ones(size=(1, 3), dtype=torch.float32, requires_grad=True))

        self.meters_per_pixel = []
        meter_per_pixel = utils.get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel.append(meter_per_pixel * (2 ** (3 - level)))



        torch.autograd.set_detect_anomaly(True)
        # Running the forward pass with detection enabled will allow the backward pass to print the traceback of the forward operation that created the failing backward function.
        # Any backward computation that generate “nan” value will raise an error.

    def get_warp_sat2real(self, satmap_sidelength):
        # satellite: u:east , v:south from bottomleft and u_center: east; v_center: north from center
        # realword: X: south, Y:down, Z: east   origin is set to the ground plane

        # meshgrid the sat pannel
        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        ii, jj = torch.meshgrid(i, j)  # i:h,j:w

        # uv is coordinate from top/left, v: south, u:east
        uv = torch.stack([jj, ii], dim=-1).float()  # shape = [satmap_sidelength, satmap_sidelength, 2]

        # sat map from top/left to center coordinate
        u0 = v0 = satmap_sidelength // 2
        uv_center = uv - torch.tensor(
            [u0, v0]).cuda()  # .to(self.device) # shape = [satmap_sidelength, satmap_sidelength, 2]

        # affine matrix: scale*R
        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength
        R = torch.tensor([[0, 1], [1, 0]]).float().cuda()  # to(self.device) # u_center->z, v_center->x
        Aff_sat2real = meter_per_pixel * R  # shape = [2,2]

        # Trans matrix from sat to realword
        XZ = torch.einsum('ij, hwj -> hwi', Aff_sat2real,
                          uv_center)  # shape = [satmap_sidelength, satmap_sidelength, 2]

        Y = torch.zeros_like(XZ[..., 0:1])
        ones = torch.ones_like(Y)
        sat2realwap = torch.cat([XZ[:, :, :1], Y, XZ[:, :, 1:], ones], dim=-1)  # [sidelength,sidelength,4]

        return sat2realwap

    def seq_warp_real2camera(self, ori_shift_u, ori_shift_v, ori_heading, XYZ_1, ori_camera_k, grd_H, grd_W, ori_grdH, ori_grdW, require_jac=True):
        # realword: X: south, Y:down, Z: east
        # camera: u:south, v: down from center (when heading east, need to rotate heading angle)
        # XYZ_1:[H,W,4], heading:[B,1], camera_k:[B,3,3], shift:[B,2]
        B = ori_heading.shape[0]
        shift_u_meters = self.args.shift_range_lon * ori_shift_u
        shift_v_meters = self.args.shift_range_lat * ori_shift_v
        heading = ori_heading * self.args.rotation_range / 180 * np.pi

        cos = torch.cos(-heading)
        sin = torch.sin(-heading)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B,9]
        R = R.view(B, 3, 3)  # shape = [B,3,3]

        camera_height = utils.get_camera_height()
        # camera offset, shift[0]:east,Z, shift[1]:north,X
        height = camera_height * torch.ones_like(shift_u_meters)
        T = torch.cat([shift_v_meters, height, -shift_u_meters], dim=-1)  # shape = [B, 3]
        T = torch.unsqueeze(T, dim=-1)  # shape = [B,3,1]
        # T = torch.einsum('bij, bjk -> bik', R, T0)
        # T = R @ T0

        # P = K[R|T]
        camera_k = ori_camera_k.clone()
        camera_k[:, :1, :] = ori_camera_k[:, :1, :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        # P = torch.einsum('bij, bjk -> bik', camera_k, torch.cat([R, T], dim=-1)).float()  # shape = [B,3,4]
        P = camera_k @ torch.cat([R, T], dim=-1)

        # uv1 = torch.einsum('bij, hwj -> bhwi', P, XYZ_1)  # shape = [B, H, W, 3]
        uv1 = torch.sum(P[:, None, None, :, :] * XYZ_1[None, :, :, None, :], dim=-1)
        # only need view in front of camera ,Epsilon = 1e-6
        uv1_last = torch.maximum(uv1[:, :, :, 2:], torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)
        uv = uv1[:, :, :, :2] / uv1_last  # shape = [B, H, W, 2]

        mask = torch.greater(uv1_last, torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)

        # ------ start computing jacobian ----- denote shift[:, 0] as x, shift[:, 1] as y below ----
        if require_jac:
            dT_dx = self.args.shift_range_lon * torch.tensor([0., 0., -1.], dtype=torch.float32, device=ori_shift_u.device, requires_grad=True).view(1, 3, 1).repeat(B, 1, 1)
            dT_dy = self.args.shift_range_lat * torch.tensor([1., 0., 0.], dtype=torch.float32, device=ori_shift_u.device, requires_grad=True).view(1, 3, 1).repeat(B, 1, 1)
            T_zeros = torch.zeros([B, 3, 1], dtype=torch.float32, device=ori_shift_u.device, requires_grad=True)
            dR_dtheta = self.args.rotation_range / 180 * np.pi * torch.cat([sin, zeros, cos, zeros, zeros, zeros, -cos, zeros, sin], dim=-1).view(B, 3, 3)
            R_zeros = torch.zeros([B, 3, 3], dtype=torch.float32, device=ori_shift_u.device, requires_grad=True)
            dP_dx = camera_k @ torch.cat([R_zeros, dT_dx], dim=-1) # [B, 3, 4]
            dP_dy = camera_k @ torch.cat([R_zeros, dT_dy], dim=-1) # [B, 3, 4]
            dP_dtheta = camera_k @ torch.cat([dR_dtheta, T_zeros], dim=-1) # [B, 3, 4]
            duv1_dx = torch.sum(dP_dx[:, None, None, :, :] * XYZ_1[None, :, :, None, :], dim=-1)
            duv1_dy = torch.sum(dP_dy[:, None, None, :, :] * XYZ_1[None, :, :, None, :], dim=-1)
            duv1_dtheta = torch.sum(dP_dtheta[:, None, None, :, :] * XYZ_1[None, :, :, None, :], dim=-1)
            # duv1_dx = torch.einsum('bij, hwj -> bhwi', camera_k @ torch.cat([R_zeros, R @ dT0_dx], dim=-1), XYZ_1)
            # duv1_dy = torch.einsum('bij, hwj -> bhwi', camera_k @ torch.cat([R_zeros, R @ dT0_dy], dim=-1), XYZ_1)
            # duv1_dtheta = torch.einsum('bij, hwj -> bhwi', camera_k @ torch.cat([dR_dtheta, dR_dtheta @ T0], dim=-1), XYZ_1)

            duv_dx = duv1_dx[..., 0:2]/uv1_last - uv1[:, :, :, :2] * duv1_dx[..., 2:] /(uv1_last**2)
            duv_dy = duv1_dy[..., 0:2]/uv1_last - uv1[:, :, :, :2] * duv1_dy[..., 2:] /(uv1_last**2)
            duv_dtheta = duv1_dtheta[..., 0:2]/uv1_last - uv1[:, :, :, :2]* duv1_dtheta[..., 2:] /(uv1_last**2)

            duv_dx1 = torch.where(mask, duv_dx, torch.zeros_like(duv_dx))
            duv_dy1 = torch.where(mask, duv_dy, torch.zeros_like(duv_dy))
            duv_dtheta1 = torch.where(mask, duv_dtheta, torch.zeros_like(duv_dtheta))

            return uv, duv_dx1, duv_dy1, duv_dtheta1, mask
            
            # duv_dshift = torch.stack([duv_dx1, duv_dy1], dim=0)  # [ 2(pose_shift), B, H, W, 2(coordinates)]
            # duv_dtheta1 = duv_dtheta1.unsqueeze(dim=0) # [ 1(pose_heading), B, H, W, 2(coordinates)]
            # return uv, duv_dshift, duv_dtheta1, mask

            # duv1_dshift = torch.stack([duv1_dx, duv1_dy], dim=0)
            # duv1_dtheta = duv1_dtheta.unsqueeze(dim=0)
            # return uv1, duv1_dshift, duv1_dtheta, mask
        else:
            return uv, mask
            # return uv1

    def project_grd_to_map(self, grd_f, grd_c, shift_u, shift_v, heading, camera_k, satmap_sidelength, ori_grdH, ori_grdW):
        # inputs:
        #   grd_f: ground features: B,C,H,W
        #   shift: B, S, 2
        #   heading: heading angle: B,S
        #   camera_k: 3*3 K matrix of left color camera : B*3*3
        # return:
        #   grd_f_trans: B,S,E,C,satmap_sidelength,satmap_sidelength

        B, C, H, W = grd_f.size()

        XYZ_1 = self.get_warp_sat2real(satmap_sidelength)  # [ sidelength,sidelength,4]

        if self.args.proj == 'geo':
            uv, jac_shiftu, jac_shiftv, jac_heading, mask = self.seq_warp_real2camera(shift_u, shift_v, heading, XYZ_1, camera_k, H, W, ori_grdH, ori_grdW, require_jac=True)  # [B, S, E, H, W,2]
            # [B, H, W, 2], [2, B, H, W, 2], [1, B, H, W, 2]
            # # --------------------------------------------------------------------------------------------------
            # def seq_warp_real2camera(ori_shift, ori_heading, ori_camera_k):
            #     # realword: X: south, Y:down, Z: east
            #     # camera: u:south, v: down from center (when heading east, need to rotate heading angle)
            #     # XYZ_1:[H,W,4], heading:[B,1], camera_k:[B,3,3], shift:[B,2]
            #     B = ori_heading.shape[0]
            #
            #     shift = ori_shift * self.args.shift_range
            #     heading = ori_heading * self.args.rotation_range / 180. * np.pi
            #
            #     cos = torch.cos(-heading)
            #     sin = torch.sin(-heading)
            #     zeros = torch.zeros_like(cos)
            #     ones = torch.ones_like(cos)
            #     R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B,9]
            #     R = R.view(B, 3, 3)  # shape = [B,3,3]
            #
            #     camera_height = utils.get_camera_height()
            #     # camera offset, shift[0]:east,Z, shift[1]:north,X
            #     height = camera_height * torch.ones_like(shift[:, :1])
            #     T = torch.cat([shift[:, 1:], height, -shift[:, :1]], dim=-1)  # shape = [B, 3]
            #     T = torch.unsqueeze(T, dim=-1)  # shape = [B,3,1]
            #     # T = torch.einsum('bij, bjk -> bik', R, T0)
            #
            #     # P = K[R|T]
            #     camera_k = ori_camera_k.clone()
            #     camera_k[:, :1, :] = ori_camera_k[:, :1,
            #                          :] * W / ori_grdW  # original size input into feature get network/ output of feature get network
            #     camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * H / ori_grdH
            #     P = torch.einsum('bij, bjk -> bik', camera_k, torch.cat([R, T], dim=-1)).float()  # shape = [B,3,4]
            #
            #     uv1 = torch.einsum('bij, hwj -> bhwi', P, XYZ_1)  # shape = [B, H, W, 3]
            #     # only need view in front of camera ,Epsilon = 1e-6
            #     uv1_last = torch.maximum(uv1[:, :, :, 2:], torch.ones_like(uv1[:, :, :, 2:]) * 1e-6)
            #     uv = uv1[:, :, :, :2] / uv1_last  # shape = [B, H, W, 2]
            #     return uv
            #
            # auto_jac = torch.autograd.functional.jacobian(seq_warp_real2camera, (shift, heading, camera_k))
            # auto_jac_shift = torch.where(mask.unsqueeze(dim=0), auto_jac[0][:, :, :, :, 0, :].permute(4, 0, 1, 2, 3),
            #                              torch.zeros_like(jac_shift))
            # # auto_jac_shift = auto_jac[0][:, :, :, :, 0, :].permute(4, 0, 1, 2, 3)
            # diff = torch.abs(auto_jac_shift - jac_shift)
            # auto_jac_heading = torch.where(mask.unsqueeze(dim=0), auto_jac[1][:, :, :, :, 0, :].permute(4, 0, 1, 2, 3),
            #                                torch.zeros_like(jac_heading))
            # # auto_jac_heading = auto_jac[1][:, :, :, :, 0, :].permute(4, 0, 1, 2, 3)
            # diff1 = torch.abs(auto_jac_heading - jac_heading)
            # heading_np = jac_heading[0, 0].data.cpu().numpy()
            # auto_heading_np = auto_jac_heading[0, 0].data.cpu().numpy()
            # diff1_np = diff1.data.cpu().numpy()
            # diff_np = diff.data.cpu().numpy()
            # mask_np = mask[0, ..., 0].float().data.cpu().numpy()
            # # --------------------------------------------------------------------------------------------------

        elif self.args.proj == 'nn':
            uv, jac_shiftu, jac_shiftv, jac_heading, mask = self.inplane_grd_to_map(shift_u, shift_v, heading, satmap_sidelength, require_jac=True)
            # # --------------------------------------------------------------------------------------------------
            # def inplane_grd_to_map(ori_shift_u, ori_shift_v, ori_heading):
            #
            #     meter_per_pixel = utils.get_meter_per_pixel()
            #     meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength
            #
            #     B = ori_heading.shape[0]
            #     shift_u_pixels = self.args.shift_range_lon * ori_shift_u / meter_per_pixel
            #     shift_v_pixels = self.args.shift_range_lat * ori_shift_v / meter_per_pixel
            #     T = torch.cat([-shift_u_pixels, shift_v_pixels], dim=-1)  # [B, 2]
            #
            #     heading = ori_heading * self.args.rotation_range / 180 * np.pi
            #     cos = torch.cos(heading)
            #     sin = torch.sin(heading)
            #     R = torch.cat([cos, -sin, sin, cos], dim=-1).view(B, 2, 2)
            #
            #     i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
            #     v, u = torch.meshgrid(i, j)  # i:h,j:w
            #     uv_2 = torch.stack([u, v], dim=-1).unsqueeze(dim=0).repeat(B, 1, 1, 1).float()  # [B, H, W, 2]
            #     uv_2 = uv_2 - satmap_sidelength / 2
            #
            #     uv_1 = torch.einsum('bij, bhwj->bhwi', R, uv_2)
            #     uv_0 = uv_1 + T[:, None, None, :]  # [B, H, W, 2]
            #
            #     uv = uv_0 + satmap_sidelength / 2
            #
            #     return uv
            #
            # auto_jac = torch.autograd.functional.jacobian(inplane_grd_to_map, (shift_u, shift_v, heading))
            #
            # auto_jac_shiftu = auto_jac[0][:, :, :, :, 0, 0]
            # diff_u = torch.abs(auto_jac_shiftu - jac_shiftu)
            #
            # auto_jac_shiftv = auto_jac[1][:, :, :, :, 0, 0]
            # diff_v = torch.abs(auto_jac_shiftv - jac_shiftv)
            #
            # auto_jac_heading = auto_jac[2][:, :, :, :, 0, 0]
            # diff_h = torch.abs(auto_jac_heading - jac_heading)
            #
            # # diff1_np = diff1.data.cpu().numpy()
            # # diff_np = diff.data.cpu().numpy()
            # # mask_np = mask[0, ..., 0].float().data.cpu().numpy()
            # # --------------------------------------------------------------------------------------------------

        jac = torch.stack([jac_shiftu, jac_shiftv, jac_heading], dim=0) # [3, B, H, W, 2]

        grd_f_trans, new_jac = grid_sample(grd_f, uv, jac)
        # [B,C,sidelength,sidelength], [3, B, C, sidelength, sidelength]
        if grd_c is not None:
            grd_c_trans, _ = grid_sample(grd_c, uv)
        else:
            grd_c_trans = None

        return grd_f_trans, grd_c_trans, new_jac

    def inplane_grd_to_map(self, ori_shift_u, ori_shift_v, ori_heading, satmap_sidelength, require_jac=True):

        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength

        B = ori_heading.shape[0]
        shift_u_pixels = self.args.shift_range_lon * ori_shift_u / meter_per_pixel
        shift_v_pixels = self.args.shift_range_lat * ori_shift_v / meter_per_pixel
        T = torch.cat([-shift_u_pixels, shift_v_pixels], dim=-1)  # [B, 2]

        heading = ori_heading * self.args.rotation_range / 180 * np.pi
        cos = torch.cos(heading)
        sin = torch.sin(heading)
        R = torch.cat([cos, -sin, sin, cos], dim=-1).view(B, 2, 2)

        i = j = torch.arange(0, satmap_sidelength).cuda()  # to(self.device)
        v, u = torch.meshgrid(i, j)  # i:h,j:w
        uv_2 = torch.stack([u, v], dim=-1).unsqueeze(dim=0).repeat(B, 1, 1, 1).float()  # [B, H, W, 2]
        uv_2 = uv_2 - satmap_sidelength/2

        uv_1 = torch.einsum('bij, bhwj->bhwi', R, uv_2)
        uv_0 = uv_1 + T[:, None, None, :]   # [B, H, W, 2]

        uv = uv_0 + satmap_sidelength/2
        mask = torch.ones_like(uv[..., 0])

        if require_jac:
            dT_dshiftu = self.args.shift_range_lon / meter_per_pixel\
                         * torch.tensor([-1., 0], dtype=torch.float32, device=ori_shift_u.device,
                                        requires_grad=True).view(1, 2).repeat(B, 1)
            dT_dshiftv = self.args.shift_range_lat / meter_per_pixel\
                         * torch.tensor([0., 1], dtype=torch.float32, device=ori_shift_u.device,
                                        requires_grad=True).view(1, 2).repeat(B, 1)
            dR_dtheta = self.args.rotation_range / 180 * np.pi * torch.cat(
                [-sin, -cos, cos, -sin], dim=-1).view(B, 2, 2)

            duv_dshiftu = dT_dshiftu[:, None, None, :].repeat(1, satmap_sidelength, satmap_sidelength, 1)
            duv_dshiftv = dT_dshiftv[:, None, None, :].repeat(1, satmap_sidelength, satmap_sidelength, 1)
            duv_dtheta = torch.einsum('bij, bhwj->bhwi', dR_dtheta, uv_2)

            return uv, duv_dshiftu, duv_dshiftv, duv_dtheta, mask
        else:
            return uv, mask

    def LM_update(self, shift_u, shift_v, heading, grd_feat_proj, grd_conf_proj, sat_feat, sat_conf, dfeat_dpose):
        '''
        Args:
            shift_u: [B, 1]
            shift_v: [B, 1]
            heading: [B, 1]
            grd_feat_proj: [B, C, H, W]
            grd_conf_proj: [B, 1, H, W]
            sat_feat: [B, C, H, W]
            sat_conf: [B, 1, H, W]
            dfeat_dpose: [3, B, C, H, W]

        Returns:
        '''

        N, B, C, H, W = dfeat_dpose.shape

        # grd_feat_proj_norm = torch.norm(grd_feat_proj.reshape(B, -1), p=2, dim=-1)
        # grd_feat_proj = grd_feat_proj / grd_feat_proj_norm[:, None, None, None]
        # dfeat_dpose = dfeat_dpose / grd_feat_proj_norm[None, :, None, None, None]

        r = grd_feat_proj - sat_feat  # [B, C, H, W]

        if self.args.train_damping:
            damping = self.damping
        else:
            damping = (self.args.damping * torch.ones(size=(1, 3), dtype=torch.float32, requires_grad=True)).to(dfeat_dpose.device)

        if self.using_weight:
            weight = (grd_conf_proj).repeat(1, C, 1, 1).reshape(B, C * H * W)
        else:
            weight = torch.ones([B, C*H*W], dtype=torch.float32, device=sat_feat.device, requires_grad=True)

        J = dfeat_dpose.flatten(start_dim=2).permute(1, 2, 0)  # [B, C*H*W, #pose]
        temp = J.transpose(1, 2) * weight.unsqueeze(dim=1)
        Hessian = temp @ J  # [B, #pose, #pose]
        # diag_H = torch.diag_embed(torch.diagonal(Hessian, dim1=1, dim2=2))  # [B, 3, 3]
        diag_H = torch.eye(Hessian.shape[-1], requires_grad=True).unsqueeze(dim=0).repeat(B, 1, 1).to(
            Hessian.device)
        delta_pose = - torch.inverse(Hessian + damping * diag_H) \
                     @ temp @ r.reshape(B, C * H * W, 1)

        shift_u_new = shift_u + delta_pose[:, 0:1, 0]
        shift_v_new = shift_v + delta_pose[:, 1:2, 0]
        heading_new = heading + delta_pose[:, 2:, 0]

        return shift_u_new, shift_v_new, heading_new

    def forward(self, sat_map, grd_img_left, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
                mode='train', file_name=None, gt_depth=None):
        '''
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            left_camera_k: [B, 3, 3]
            grd_img_left: [B, C, H, W]
            gt_shift_u: [B, 1] u->longitudinal
            gt_shift_v: [B, 1] v->lateral
            gt_heading: [B, 1] east as 0-degree
            mode:
            file_name:

        Returns:

        '''
        '''
        :param sat_map: [B, C, A, A] A--> sidelength
        :param left_camera_k: [B, 3, 3]
        :param grd_img_left: [B, C, H, W]
        :return:
        '''

        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        # A = sat_map.shape[-1]
        # sat_align_cam_trans, _, dimg_dpose = self.project_grd_to_map(
        #     sat_align_cam, None, gt_shift_u, gt_shift_v, gt_heading, left_camera_k, A, ori_grdH, ori_grdW)
        # grd_img = transforms.ToPILImage()(sat_align_cam_trans[0])
        # grd_img.save('sat_align_cam_trans.png')
        # sat_align_cam = transforms.ToPILImage()(sat_align_cam[0])
        # sat_align_cam.save('sat_align_cam.png')
        # sat = transforms.ToPILImage()(sat_map[0])
        # sat.save('sat.png')

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)

        grd_feat_list, grd_conf_list = self.GrdFeatureNet(grd_img_left)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        pred_feat_dict = {}
        shift_us_all = []
        shift_vs_all = []
        headings_all = []
        for iter in range(self.N_iters):
            shift_us = []
            shift_vs = []
            headings = []
            for level in range(len(sat_feat_list)):
                sat_feat = sat_feat_list[level]
                sat_conf = sat_conf_list[level]
                grd_feat = grd_feat_list[level]
                grd_conf = grd_conf_list[level]

                A = sat_feat.shape[-1]
                grd_feat_proj, grd_conf_proj, dfeat_dpose = self.project_grd_to_map(
                    grd_feat, grd_conf, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)
                # grd_conf_proj = 1 / (1 + grd_conf_proj)

                # def project_grd_to_map(shift, heading):
                #
                #     B, C, H, W = grd_feat.size()
                #
                #     XYZ_1 = self.get_warp_sat2real(A)  # [ sidelength,sidelength,4]
                #
                #     uv = self.seq_warp_real2camera(shift, heading, XYZ_1, left_camera_k, H, W, ori_grdH, ori_grdW,
                #                                    require_jac=False)  # [B, S, E, H, W,2]
                #     # [B, H, W, 2]
                #
                #     grd_f_trans = grid_sample(grd_feat, uv)
                #     # [B,C,sidelength,sidelength]
                #
                #     return grd_f_trans
                #
                # auto_jac = torch.autograd.functional.jacobian(project_grd_to_map, (shift, heading))

                shift_u_new, shift_v_new, heading_new = self.LM_update(
                    shift_u, shift_v, heading, grd_feat_proj, grd_conf_proj, sat_feat, sat_conf, dfeat_dpose)

                shift_us.append(shift_u_new[:, 0])  # [B]
                shift_vs.append(shift_v_new[:, 0])  # [B]
                headings.append(heading_new[:, 0])

                shift_u = shift_u_new.clone()
                shift_v = shift_v_new.clone()
                heading = heading_new.clone()

                if level not in pred_feat_dict.keys():
                    pred_feat_dict[level] = [grd_feat_proj]
                else:
                    pred_feat_dict[level].append(grd_feat_proj)

            shift_us_all.append(torch.stack(shift_us, dim=1))  # [B, Level]
            shift_vs_all.append(torch.stack(shift_vs, dim=1))  # [B, Level]
            headings_all.append(torch.stack(headings, dim=1)) # [B, Level]

        shift_lats = torch.stack(shift_vs_all, dim=1)  # [B, N_iters, Level]
        shift_lons = torch.stack(shift_us_all, dim=1)  # [B, N_iters, Level]
        thetas = torch.stack(headings_all, dim=1)  # [B, N_iters, Level]

        if mode == 'train':
            loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
            shift_lat_last, shift_lon_last, theta_last, \
            L1_loss, L2_loss, L3_loss, L4_loss \
                = loss_func(self.args.loss_method, grd_feat_list, pred_feat_dict, None,
                            shift_lats, shift_lons, thetas, gt_shift_v[:, 0], gt_shift_u[:, 0], gt_heading[:, 0],
                            None, None,
                            self.args.coe_shift_lat, self.args.coe_shift_lon, self.args.coe_heading,
                            self.args.coe_L1, self.args.coe_L2, self.args.coe_L3, self.args.coe_L4)

            return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                    shift_lat_last, shift_lon_last, theta_last, \
                    L1_loss, L2_loss, L3_loss, L4_loss, grd_conf_list

        else:
            return shift_lats[:, -1, -1], shift_lons[:, -1, -1], thetas[:, -1, -1]

    def corr(self, sat_map, grd_img_left, left_camera_k, gt_shift_u=None, gt_shift_v=None, gt_heading=None,
                mode='train', file_name=None, gt_depth=None):
        '''
        Args:
            sat_map: [B, C, A, A] A--> sidelength
            left_camera_k: [B, 3, 3]
            grd_img_left: [B, C, H, W]
            gt_shift_u: [B, 1] u->longitudinal
            gt_shift_v: [B, 1] v->lateral
            gt_heading: [B, 1] east as 0-degree
            mode:
            file_name:

        Returns:

        '''
        '''
        :param sat_map: [B, C, A, A] A--> sidelength
        :param left_camera_k: [B, 3, 3]
        :param grd_img_left: [B, C, H, W]
        :return:
        '''

        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)

        grd_feat_list, grd_conf_list = self.GrdFeatureNet(grd_img_left)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        corr_maps = []

        for level in range(len(sat_feat_list)):
            meter_per_pixel = self.meters_per_pixel[level]

            sat_feat = sat_feat_list[level]
            grd_feat = grd_feat_list[level]

            A = sat_feat.shape[-1]
            grd_feat_proj, _, dfeat_dpose = self.project_grd_to_map(
                grd_feat, None, shift_u, shift_v, heading, left_camera_k, A, ori_grdH, ori_grdW)

            crop_H = int(A - self.args.shift_range_lat * 2 / meter_per_pixel)
            crop_W = int(A - self.args.shift_range_lon * 2 / meter_per_pixel)
            g2s_feat = TF.center_crop(grd_feat_proj, [crop_H, crop_W])
            g2s_feat = F.normalize(g2s_feat.reshape(B, -1)).reshape(B, -1, crop_H, crop_W)

            s_feat = sat_feat.reshape(1, -1, A, A) # [B, C, H, W]->[1, B*C, H, W]
            corr = F.conv2d(s_feat, g2s_feat, groups=B)[0]  #[B, H, W]

            denominator = F.avg_pool2d(sat_feat.pow(2), (crop_H, crop_W), stride=1, divisor_override=1)  # [B, 4W]
            denominator = torch.sum(denominator, dim=1)  # [B, H, W]
            denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
            corr = 2 - 2 * corr / denominator

            B, corr_H, corr_W = corr.shape

            corr_maps.append(corr)

            max_index = torch.argmin(corr.reshape(B, -1), dim=1)
            pred_u = (max_index % corr_W - corr_W / 2) * meter_per_pixel # / self.args.shift_range_lon
            pred_v = -(max_index // corr_W - corr_H/2) * meter_per_pixel # / self.args.shift_range_lat

            # corr0 = []
            # for b in range(B):
            #     corr0.append(F.conv2d(s_feat[b:b+1, :, :, :], g2s_feat[b:b+1, :, :, :]))  # [1, 1, H, W]
            # corr0 = torch.cat(corr0, dim=1)
            # print(torch.sum(torch.abs(corr0 - corr)))

        if mode == 'train':
            return self.triplet_loss(corr_maps, gt_shift_u, gt_shift_v)
        else:
            return pred_u, pred_v  # [B], [B]


    def triplet_loss(self, corr_maps, gt_shift_u, gt_shift_v):
        losses = []
        for level in range(len(corr_maps)):
            meter_per_pixel = self.meters_per_pixel[level]

            corr = corr_maps[level]
            B, corr_H, corr_W = corr.shape

            w = torch.round(corr_W / 2 + gt_shift_u[:, 0] * self.args.shift_range_lon / meter_per_pixel)
            h = torch.round(corr_H / 2 - gt_shift_v[:, 0] * self.args.shift_range_lat / meter_per_pixel)

            pos = corr[range(B), h.long(), w.long()]  # [B]
            pos_neg = pos.reshape(-1, 1, 1) - corr  # [B, H, W]
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (corr_H * corr_W - 1))
            losses.append(loss)

        return torch.sum(torch.stack(losses, dim=0))


class LM_S2GP(nn.Module):
    def __init__(self, args):  # device='cuda:0',
        super(LM_S2GP, self).__init__()
        '''
        loss_method: 0: direct R T loss 1: feat loss 2: noise aware feat loss
        '''
        self.args = args

        self.level = args.level
        self.N_iters = args.N_iters
        self.using_weight = args.using_weight
        self.loss_method = args.loss_method

        self.SatFeatureNet = VGGUnet(self.level)
        self.GrdFeatureNet = VGGUnet(self.level)


        if args.rotation_range > 0:
            self.damping = nn.Parameter(
                torch.zeros(size=(1, 3), dtype=torch.float32, requires_grad=True))
        else:
            self.damping = nn.Parameter(
            torch.zeros(size=(), dtype=torch.float32, requires_grad=True))

        ori_grdH, ori_grdW = 256, 1024
        xyz_grds = []
        for level in range(4):
            grd_H, grd_W = ori_grdH/(2**(3-level)), ori_grdW/(2**(3-level))
            if self.args.proj == 'geo':
                xyz_grd, mask, xyz_w = self.grd_img2cam(grd_H, grd_W, ori_grdH,
                                                 ori_grdW)  # [1, grd_H, grd_W, 3] under the grd camera coordinates
                xyz_grds.append((xyz_grd, mask, xyz_w))

            else:
                xyz_grd, mask = self.grd_img2cam_polar(grd_H, grd_W, ori_grdH, ori_grdW)
                xyz_grds.append((xyz_grd, mask))

        self.xyz_grds = xyz_grds

        self.meters_per_pixel = []
        meter_per_pixel = utils.get_meter_per_pixel()
        for level in range(4):
            self.meters_per_pixel.append(meter_per_pixel * (2 ** (3 - level)))

        polar_grids = []
        for level in range(4):
            grids = self.polar_coordinates(level)
            polar_grids.append(grids)
        self.polar_grids = polar_grids

        if self.args.Optimizer=='NN':
            self.NNrefine = NNrefine()

        torch.autograd.set_detect_anomaly(True)
        # Running the forward pass with detection enabled will allow the backward pass to print the traceback of the forward operation that created the failing backward function.
        # Any backward computation that generate “nan” value will raise an error.

    def grd_img2cam(self, grd_H, grd_W, ori_grdH, ori_grdW):
        
        ori_camera_k = torch.tensor([[[582.9802,   0.0000, 496.2420],
                                      [0.0000, 482.7076, 125.0034],
                                      [0.0000,   0.0000,   1.0000]]], 
                                    dtype=torch.float32, requires_grad=True)  # [1, 3, 3]
        
        camera_height = utils.get_camera_height()

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
        # xyz_grd = xyz_grd.reshape(B, N, grd_H, grd_W, 3)

        mask = (xyz_grd[..., -1] > 0).float()  # # [1, grd_H, grd_W]

        return xyz_grd, mask, xyz_w

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

    def grd2cam2world2sat(self, ori_shift_u, ori_shift_v, ori_heading, level,
                          satmap_sidelength, require_jac=False, gt_depth=None):
        '''
        realword: X: south, Y:down, Z: east
        camera: u:south, v: down from center (when heading east, need to rotate heading angle)
        Args:
            ori_shift_u: [B, 1]
            ori_shift_v: [B, 1]
            heading: [B, 1]
            XYZ_1: [H,W,4]
            ori_camera_k: [B,3,3]
            grd_H:
            grd_W:
            ori_grdH:
            ori_grdW:

        Returns:
        '''
        B, _ = ori_heading.shape
        heading = ori_heading * self.args.rotation_range / 180 * np.pi
        shift_u = ori_shift_u * self.args.shift_range_lon
        shift_v = ori_shift_v * self.args.shift_range_lat

        cos = torch.cos(heading)
        sin = torch.sin(heading)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B, 9]
        R = R.view(B, 3, 3)  # shape = [B, N, 3, 3]
        # this R is the inverse of the R in G2SP

        camera_height = utils.get_camera_height()
        # camera offset, shift[0]:east,Z, shift[1]:north,X
        height = camera_height * torch.ones_like(shift_u[:, :1])
        T0 = torch.cat([shift_v, height, -shift_u], dim=-1)  # shape = [B, 3]
        # T0 = torch.unsqueeze(T0, dim=-1)  # shape = [B, N, 3, 1]
        # T = torch.einsum('bnij, bnj -> bni', -R, T0) # [B, N, 3]
        T = torch.sum(-R * T0[:, None, :], dim=-1)   # [B, 3]

        # The above R, T define transformation from camera to world

        if self.args.use_gt_depth and gt_depth!=None:
            xyz_w = self.xyz_grds[level][2].detach().to(ori_shift_u.device).repeat(B, 1, 1, 1)
            H, W = xyz_w.shape[1:-1]
            depth = F.interpolate(gt_depth[:, None, :, :], (H, W))
            xyz_grd = xyz_w * depth.permute(0, 2, 3, 1)
            mask = (gt_depth != -1).float()
            mask = F.interpolate(mask[:, None, :, :], (H, W), mode='nearest')
            mask = mask[:, 0, :, :]
        else:
            xyz_grd = self.xyz_grds[level][0].detach().to(ori_shift_u.device).repeat(B, 1, 1, 1)
            mask = self.xyz_grds[level][1].detach().to(ori_shift_u.device).repeat(B, 1, 1)  # [B, grd_H, grd_W]
        grd_H, grd_W = xyz_grd.shape[1:3]

        xyz = torch.sum(R[:, None, None, :, :] * xyz_grd[:, :, :, None, :], dim=-1) + T[:, None, None, :]
        # [B, grd_H, grd_W, 3]
        # zx0 = torch.stack([xyz[..., 2], xyz[..., 0]], dim=-1)  # [B, N, grd_H, grd_W, 2]
        R_sat = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.float32, device=ori_shift_u.device, requires_grad=True)\
            .reshape(2, 3)
        zx = torch.sum(R_sat[None, None, None, :, :] * xyz[:, :, :, None, :], dim=-1)
        # [B, grd_H, grd_W, 2]
        # assert zx == zx0

        meter_per_pixel = utils.get_meter_per_pixel()
        meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength
        sat_uv = zx/meter_per_pixel + satmap_sidelength / 2  # [B, grd_H, grd_W, 2] sat map uv

        if require_jac:
            dR_dtheta = self.args.rotation_range / 180 * np.pi * \
                        torch.cat([-sin, zeros, -cos, zeros, zeros, zeros, cos, zeros, -sin], dim=-1)  # shape = [B, N, 9]
            dR_dtheta = dR_dtheta.view(B, 3, 3)
            # R_zeros = torch.zeros_like(dR_dtheta)

            dT0_dshiftu = self.args.shift_range_lon * torch.tensor([0., 0., -1.], dtype=torch.float32, device=shift_u.device,
                                                         requires_grad=True).view(1, 3).repeat(B, 1)
            dT0_dshiftv = self.args.shift_range_lat * torch.tensor([1., 0., 0.], dtype=torch.float32, device=shift_u.device,
                                                         requires_grad=True).view(1, 3).repeat(B, 1)
            # T0_zeros = torch.zeros_like(dT0_dx)

            dxyz_dshiftu = torch.sum(-R * dT0_dshiftu[:, None, :], dim=-1)[:, None, None, :].\
                repeat([1, grd_H, grd_W, 1])   # [B, grd_H, grd_W, 3]
            dxyz_dshiftv = torch.sum(-R * dT0_dshiftv[:, None, :], dim=-1)[:, None, None, :].\
                repeat([1, grd_H, grd_W, 1])   # [B, grd_H, grd_W, 3]
            dxyz_dtheta = torch.sum(dR_dtheta[:, None, None, :, :] * xyz_grd[:, :, :, None, :], dim=-1) + \
                          torch.sum(-dR_dtheta * T0[:, None, :], dim=-1)[:, None, None, :]

            duv_dshiftu = 1 / meter_per_pixel * \
                     torch.sum(R_sat[None, None, None, :, :] * dxyz_dshiftu[:, :, :, None, :], dim=-1)
            # [B, grd_H, grd_W, 2]
            duv_dshiftv = 1 / meter_per_pixel * \
                     torch.sum(R_sat[None, None, None, :, :] * dxyz_dshiftv[:, :, :, None, :], dim=-1)
            # [B, grd_H, grd_W, 2]
            duv_dtheta = 1 / meter_per_pixel * \
                     torch.sum(R_sat[None, None, None, :, :] * dxyz_dtheta[:, :, :, None, :], dim=-1)
            # [B, grd_H, grd_W, 2]

            # duv_dshift = torch.stack([duv_dx, duv_dy], dim=0)
            # duv_dtheta = duv_dtheta.unsqueeze(dim=0)

            return sat_uv, mask, duv_dshiftu, duv_dshiftv, duv_dtheta

        return sat_uv, mask, None, None, None

    def project_map_to_grd(self, sat_f, sat_c, shift_u, shift_v, heading, level, require_jac=True, gt_depth=None):
        '''
        Args:
            sat_f: [B, C, H, W]
            sat_c: [B, 1, H, W]
            shift_u: [B, 2]
            shift_v: [B, 2]
            heading: [B, 1]
            camera_k: [B, 3, 3]

            ori_grdH:
            ori_grdW:

        Returns:

        '''
        B, C, satmap_sidelength, _ = sat_f.size()
        A = satmap_sidelength

        uv, mask, jac_shiftu, jac_shiftv, jac_heading = self.grd2cam2world2sat(shift_u, shift_v, heading, level,
                                    satmap_sidelength, require_jac, gt_depth)
        # [B, H, W, 2], [B, H, W], [B, H, W, 2], [B, H, W, 2], [B,H, W, 2]
        # # --------------------------------------------------------------------------------------------------
        # def grd2cam2world2sat(ori_shift, ori_heading, ori_camera_k):
        #     '''
        #     realword: X: south, Y:down, Z: east
        #     camera: u:south, v: down from center (when heading east, need to rotate heading angle)
        #     Args:
        #         shift: [B, N, 2]
        #         heading: [B, N, 1]
        #         XYZ_1: [H,W,4]
        #         ori_camera_k: [B,3,3]
        #         grd_H:
        #         grd_W:
        #         ori_grdH:
        #         ori_grdW:
        #
        #     Returns:
        #     '''
        #     B, N, _ = ori_heading.shape
        #     heading = ori_heading * self.args.rotation_range / 180 * np.pi
        #     shift = ori_shift * self.args.shift_range
        #
        #     cos = torch.cos(heading)
        #     sin = torch.sin(heading)
        #     zeros = torch.zeros_like(cos)
        #     ones = torch.ones_like(cos)
        #     R = torch.cat([cos, zeros, -sin, zeros, ones, zeros, sin, zeros, cos], dim=-1)  # shape = [B, N, 9]
        #     R = R.view(B, N, 3, 3)  # shape = [B, N, 3, 3]
        #     # this R is the inverse of the R in G2SP
        #
        #     camera_height = utils.get_camera_height()
        #     # camera offset, shift[0]:east,Z, shift[1]:north,X
        #     height = camera_height * torch.ones_like(shift[:, :, :1])
        #     T0 = torch.cat([shift[:, :, 1:], height, -shift[:, :, :1]], dim=-1)  # shape = [B, N, 3]
        #     # T0 = torch.unsqueeze(T0, dim=-1)  # shape = [B, N, 3, 1]
        #     # T = torch.einsum('bnij, bnj -> bni', -R, T0) # [B, N, 3]
        #     T = torch.sum(-R * T0[:, :, None, :], dim=-1)  # [B, N, 3]
        #
        #     # The above R, T define transformation from camera to world
        #
        #     camera_k = ori_camera_k.clone()
        #     camera_k[:, :1, :] = ori_camera_k[:, :1,
        #                          :] * grd_W / ori_grdW  # original size input into feature get network/ output of feature get network
        #     camera_k[:, 1:2, :] = ori_camera_k[:, 1:2, :] * grd_H / ori_grdH
        #     camera_k_inv = torch.inverse(camera_k)  # [B, 3, 3]
        #
        #     v, u = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32, device=ori_shift.device),
        #                           torch.arange(0, grd_W, dtype=torch.float32, device=ori_shift.device))
        #     uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1).unsqueeze(dim=0).repeat(B * N, 1, 1,
        #                                                                                   1)  # [BN, grd_H, grd_W, 3]
        #     xyz_w = torch.sum(camera_k_inv[:, None, None, :, :] * uv1[:, :, :, None, :],
        #                       dim=-1)  # [BN, grd_H, grd_W, 3]
        #     w = camera_height / torch.where(torch.abs(xyz_w[..., 1:2]) > utils.EPS, xyz_w[..., 1:2],
        #                                     utils.EPS * torch.ones_like(xyz_w[..., 1:2]))  # [BN, grd_H, grd_W, 1]
        #     xyz_grd = xyz_w * w  # [BN, grd_H, grd_W, 3] under the grd camera coordinates
        #     xyz_grd = xyz_grd.reshape(B, N, grd_H, grd_W, 3)
        #
        #     xyz = torch.sum(R[:, :, None, None, :, :] * xyz_grd[:, :, :, :, None, :], dim=-1) + T[:, :, None, None, :]
        #     # [B, N, grd_H, grd_W, 3]
        #     # zx0 = torch.stack([xyz[..., 2], xyz[..., 0]], dim=-1)  # [B, N, grd_H, grd_W, 2]
        #     R_sat = torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.float32, device=ori_shift.device, requires_grad=True) \
        #         .reshape(2, 3)
        #     zx = torch.sum(R_sat[None, None, None, None, :, :] * xyz[:, :, :, :, None, :], dim=-1)
        #     # [B, N, grd_H, grd_W, 2]
        #     # assert zx == zx0
        #
        #     meter_per_pixel = utils.get_meter_per_pixel()
        #     meter_per_pixel *= utils.get_process_satmap_sidelength() / satmap_sidelength
        #     sat_uv = zx / meter_per_pixel + satmap_sidelength / 2  # [B, N, grd_H, grd_W, 2] sat map uv
        #
        #     return sat_uv
        #
        # auto_jac = torch.autograd.functional.jacobian(grd2cam2world2sat, (shift, heading, camera_k))
        # # auto_jac_shift = torch.where(mask.unsqueeze(dim=0), auto_jac[0][:, :, :, :, 0, :].permute(4, 0, 1, 2, 3),
        # #                              torch.zeros_like(jac_shift))
        # auto_jac_shift = auto_jac[0][:, :, :, :, :, 0, 0, :].permute(5, 0, 1, 2, 3, 4) # [2, B(1), N(1), H, W, 2]
        # diff = torch.abs(auto_jac_shift - jac_shift)
        # # auto_jac_heading = torch.where(mask.unsqueeze(dim=0), auto_jac[1][:, :, :, :, 0, :].permute(4, 0, 1, 2, 3),
        # #                                torch.zeros_like(jac_heading))
        # auto_jac_heading = auto_jac[1][:, :, :, :, :, 0, 0, :].permute(5, 0, 1, 2, 3, 4)
        # diff1 = torch.abs(auto_jac_heading - jac_heading)
        # heading_np = jac_heading[0, 0, 0].data.cpu().numpy()
        # auto_heading_np = auto_jac_heading[0, 0, 0].data.cpu().numpy()
        # diff1_np = diff1.data.cpu().numpy()
        # diff_np = diff.data.cpu().numpy()
        # mask_np = mask[0, ..., 0].float().data.cpu().numpy()
        # # --------------------------------------------------------------------------------------------------

        B, grd_H, grd_W, _ = uv.shape
        if require_jac:
            jac = torch.stack([jac_shiftu, jac_shiftv, jac_heading], dim=0)  # [3, B, H, W, 2]

            # jac = jac.reshape(3, -1, grd_H, grd_W, 2)
        else:
            jac = None

        # print('==================')
        # print(jac.shape)
        # print('==================')

        sat_f_trans, new_jac = grid_sample(sat_f,
                                           uv,
                                           jac)
        sat_f_trans = sat_f_trans * mask[:, None, :, :]
        if require_jac:
            new_jac = new_jac * mask[None, :, None, :, :]

        if sat_c is not None:
            sat_c_trans, _ = grid_sample(sat_c, uv)
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
        if self.args.rotation_range == 0:
            dfeat_dpose = dfeat_dpose[:2, ...]
        elif self.args.shift_range_lat == 0 and self.args.shift_range_lon == 0:
            dfeat_dpose = dfeat_dpose[2:, ...]

        N, B, C, H, W = dfeat_dpose.shape
        if self.args.train_damping:
            # damping = self.damping
            min_, max_ = -6, 5
            damping = 10.**(min_ + self.damping.sigmoid()*(max_ - min_))
        else:
            damping = (self.args.damping * torch.ones(size=(1, N), dtype=torch.float32, requires_grad=True)).to(
                dfeat_dpose.device)

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
        # print('===================')
        # print('Hessian.shape', Hessian.shape)
        if self.args.use_hessian:
            diag_H = torch.diag_embed(torch.diagonal(Hessian, dim1=1, dim2=2))  # [B, 3, 3]
            # print('diag_H.shape', diag_H.shape)
        else:
            diag_H = torch.eye(Hessian.shape[-1], requires_grad=True).unsqueeze(dim=0).repeat(B, 1, 1).to(
                Hessian.device)
        # print('Hessian + damping * diag_H.shape ', (Hessian + damping * diag_H).shape)
        delta_pose = - torch.inverse(Hessian + damping * diag_H) \
                     @ temp @ r.reshape(B, -1, 1)

        if self.args.rotation_range == 0:
            shift_u_new = shift_u + delta_pose[:, 0:1, 0]
            shift_v_new = shift_v + delta_pose[:, 1:2, 0]
            theta_new = theta
        elif self.args.shift_range_lat == 0 and self.args.shift_range_lon == 0:
            theta_new = theta + delta_pose[:, 0:1, 0]
            shift_u_new = shift_u
            shift_v_new = shift_v
        else:
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

    def NN_update(self, shift_u, shift_v, theta, sat_feat_proj, sat_conf_proj, grd_feat, grd_conf, dfeat_dpose):

        delta = self.NNrefine(sat_feat_proj, grd_feat)  # [B, 3]
        # print('=======================')
        # print('delta.shape: ', delta.shape)
        # print('shift_u.shape', shift_u.shape)
        # print('=======================')

        shift_u_new = shift_u + delta[:, 0:1]
        shift_v_new = shift_v + delta[:, 1:2]
        theta_new = theta + delta[:, 2:3]
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

        # idx0 = torch.le(r, 0)
        # idx1 = torch.greater(r, 0)
        # mask = idx0 * (-1) + idx1
        # dr_dfeat = mask.float() / (C * H * W)  # [B, C, H, W]
        dr_dfeat = 2 * r #/ (C * H * W)  # this is grad for l2 loss, above is grad for l1 loss
        delta_pose = torch.sum(dr_dfeat[None, ...] * dfeat_dpose, dim=[2, 3, 4]).transpose(0, 1)  # [B, #pose]

        # print(delta_pose)

        shift_u_new = shift_u - 0.01 * delta_pose[:, 0:1]
        shift_v_new = shift_v - 0.01 * delta_pose[:, 1:2]
        theta_new = theta - 0.01 * delta_pose[:, 2:3]
        return shift_u_new, shift_v_new, theta_new

    def ADAM_update(self, shift_u, shift_v, theta, sat_feat_proj, sat_conf_proj, grd_feat, grd_conf, dfeat_dpose, m, v, t):
        '''
        Args:
            shift: [B, 2]
            heading: [B, 1]
            sat_feat_proj: [B, C, H, W]
            sat_conf_proj: [B, 1, H, W]
            grd_feat: [B, C, H, W]
            grd_conf: [B, 1, H, W]
            dfeat_dpose: [3, B, C, H, W]
            m: [B, #pose], accumulator in ADAM
            v: [B, #pose], accumulator in ADAM
            t: scalar, current iteration number
        Returns:
        '''

        B, C, H, W = grd_feat.shape
        r = sat_feat_proj - grd_feat  # [B, C, H, W]

        # idx0 = torch.le(r, 0)
        # idx1 = torch.greater(r, 0)
        # mask = idx0 * (-1) + idx1
        # dr_dfeat = mask.float() / (C * H * W)  # [B, C, H, W]
        dr_dfeat = 2 * r #/ (C * H * W)  # this is grad for l2 loss, above is grad for l1 loss
        delta_pose = torch.sum(dr_dfeat[None, ...] * dfeat_dpose, dim=[2, 3, 4]).transpose(0, 1)  # [B, #pose]

        # adam optimizer
        m = self.args.beta1 * m + (1- self.args.beta1) * delta_pose
        v = self.args.beta2 * v + (1- self.args.beta2) * (delta_pose * delta_pose)
        m_hat = m / (1 - self.args.beta1 ** (t+1))
        v_hat = v / (1 - self.args.beta2 ** (t+1))
        delta_final = m_hat / (v_hat ** 0.5 + 1e-8)

        # print(delta_pose)

        shift_u_new = shift_u - 0.01 * delta_final[:, 0:1]
        shift_v_new = shift_v - 0.01 * delta_final[:, 1:2]
        theta_new = theta - 0.01 * delta_final[:, 2:3]
        return shift_u_new, shift_v_new, theta_new, m, v

    def forward(self, sat_map, grd_img_left, gt_shiftu=None, gt_shiftv=None, gt_heading=None, mode='train',
                file_name=None, gt_depth=None, loop=0, level_first=0):
        '''
        :param sat_map: [B, C, A, A] A--> sidelength
        :param grd_img_left: [B, C, H, W]
        :return:
        '''
        if level_first:
            return self.forward_level_first(sat_map, grd_img_left, gt_shiftu, gt_shiftv, gt_heading, mode,
                file_name, gt_depth, loop)
        else:
            return self.forward_iter_first(sat_map, grd_img_left, gt_shiftu, gt_shiftv, gt_heading, mode,
                file_name, gt_depth, loop)


    def forward_iter_first(self, sat_map, grd_img_left, gt_shiftu=None, gt_shiftv=None, gt_heading=None, mode='train',
                file_name=None, gt_depth=None, loop=0):
        '''
        :param sat_map: [B, C, A, A] A--> sidelength
        :param grd_img_left: [B, C, H, W]
        :return:
        '''

        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        # A = sat_map.shape[-1]
        # sat_img_proj, _, _, _, _ = self.project_map_to_grd(
        #     grd_img_left, None, gt_shiftu, gt_shiftv, gt_heading, level=3, require_jac=True, gt_depth=gt_depth)
        # sat_img = transforms.ToPILImage()(sat_img_proj[0])
        # sat_img.save('sat_proj.png')
        # grd = transforms.ToPILImage()(grd_img_left[0])
        # grd.save('grd.png')
        # sat = transforms.ToPILImage()(sat_map[0])
        # sat.save('sat.png')

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)

        grd_feat_list, grd_conf_list = self.GrdFeatureNet(grd_img_left)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        gt_uv_dict = {}
        gt_feat_dict = {}
        pred_uv_dict = {}
        pred_feat_dict = {}
        shift_us_all = []
        shift_vs_all = []
        headings_all = []
        for iter in range(self.N_iters):
            shift_us = []
            shift_vs = []
            headings = []
            for level in range(len(sat_feat_list)):
                sat_feat = sat_feat_list[level]
                sat_conf = sat_conf_list[level]
                grd_feat = grd_feat_list[level]
                grd_conf = grd_conf_list[level]

                grd_H, grd_W = grd_feat.shape[-2:]
                sat_feat_proj, sat_conf_proj, dfeat_dpose, sat_uv, mask = self.project_map_to_grd(
                    sat_feat, sat_conf, shift_u, shift_v, heading, level, gt_depth=gt_depth)
                # [B, C, H, W], [B, 1, H, W], [3, B, C, H, W], [B, H, W, 2]

                grd_feat = grd_feat * mask[:, None, :, :]
                grd_conf = grd_conf * mask[:, None, :, :]

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
                    shift_u_new, shift_v_new, heading_new = self.LM_update(shift_u, shift_v, heading,
                                                            sat_feat_new,
                                                            sat_conf_new,
                                                            grd_feat_new,
                                                            grd_conf_new,
                                                            dfeat_dpose_new)  # only need to compare bottom half
                elif self.args.Optimizer == 'SGD':
                    # r = sat_feat_proj[:, :, grd_H // 2:, :] - grd_feat[:, :, grd_H // 2:, :]
                    # p = torch.mean(torch.abs(r), dim=[1, 2, 3])  # *100 #* 256 * 256 * 3
                    # dp_dshiftu = torch.autograd.grad(p, shift_u, retain_graph=True, create_graph=True,
                    #                              only_inputs=True)[0]
                    # dp_dshiftv = torch.autograd.grad(p, shift_v, retain_graph=True, create_graph=True,
                    #                                  only_inputs=True)[0]
                    # dp_dheading = torch.autograd.grad(p, heading, retain_graph=True, create_graph=True,
                    #                                 only_inputs=True)[0]
                    # print(dp_dshiftu)
                    # print(dp_dshiftv)
                    # print(dp_dheading)

                    shift_u_new, shift_v_new, heading_new = self.SGD_update(shift_u, shift_v, heading,
                                                                           sat_feat_new,
                                                                           sat_conf_new,
                                                                           grd_feat_new,
                                                                           grd_conf_new,
                                                                           dfeat_dpose_new)
                elif self.args.Optimizer == 'NN':
                    shift_u_new, shift_v_new, heading_new = self.NN_update(shift_u, shift_v, heading,
                                                                         sat_feat_new,
                                                                         sat_conf_new,
                                                                         grd_feat_new,
                                                                         grd_conf_new,
                                                                         dfeat_dpose_new)
                elif self.args.Optimizer == 'ADAM':
                    t = iter * self.args.level + level
                    if t==0:
                        m = 0
                        v = 0
                    shift_u_new, shift_v_new, heading_new, m, v = self.ADAM_update(shift_u, shift_v, heading,
                                                                         sat_feat_new,
                                                                         sat_conf_new,
                                                                         grd_feat_new,
                                                                         grd_conf_new,
                                                                         dfeat_dpose_new,
                                                                         m, v, t)


                shift_us.append(shift_u_new[:, 0])  # [B]
                shift_vs.append(shift_v_new[:, 0])  # [B]
                headings.append(heading_new[:, 0])  # [B]

                shift_u = shift_u_new.clone()
                shift_v = shift_v_new.clone()
                heading = heading_new.clone()

                if level not in pred_feat_dict.keys():
                    pred_feat_dict[level] = [sat_feat_proj]
                    pred_uv_dict[level] = [sat_uv / torch.tensor([sat_feat.shape[-1], sat_feat.shape[-2]], dtype=torch.float32).reshape(1, 1, 1, 2).to(sat_feat.device)]
                else:
                    pred_feat_dict[level].append(sat_feat_proj)
                    pred_uv_dict[level].append(sat_uv / torch.tensor([sat_feat.shape[-1], sat_feat.shape[-2]], dtype=torch.float32).reshape(1, 1, 1, 2).to(sat_feat.device))

                if level not in gt_uv_dict.keys() and mode == 'train':
                    gt_sat_feat_proj, _, _, gt_uv, _ = self.project_map_to_grd(
                        sat_feat, None, gt_shiftu, gt_shiftv, gt_heading, level, require_jac=False, gt_depth=gt_depth)
                    # [B, C, H, W], [B, H, W, 2]
                    gt_feat_dict[level] = gt_sat_feat_proj # [B, C, H, W]
                    gt_uv_dict[level] = gt_uv / torch.tensor([sat_feat.shape[-1], sat_feat.shape[-2]], dtype=torch.float32).reshape(1, 1, 1, 2).to(sat_feat.device)
                    # [B, H, W, 2]

            shift_us_all.append(torch.stack(shift_us, dim=1))  # [B, Level]
            shift_vs_all.append(torch.stack(shift_vs, dim=1))  # [B, Level]
            headings_all.append(torch.stack(headings, dim=1))  # [B, Level]

        shift_lats = torch.stack(shift_vs_all, dim=1)  # [B, N_iters, Level]
        shift_lons = torch.stack(shift_us_all, dim=1)  # [B, N_iters, Level]
        thetas = torch.stack(headings_all, dim=1)  # [B, N_iters, Level]

        if self.args.visualize:
            from visualize_utils import features_to_RGB, RGB_iterative_pose
            save_dir = './visualize_rot' + str(self.args.rotation_range)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            features_to_RGB(sat_feat_list, grd_feat_list, pred_feat_dict, gt_feat_dict, loop,
                            save_dir)
            RGB_iterative_pose(sat_map, grd_img_left, shift_lats, shift_lons, thetas, gt_shiftu, gt_shiftv, gt_heading,
                               self.meters_per_pixel[-1], self.args, loop, save_dir)


        if mode == 'train':

            if self.args.rotation_range == 0:
                coe_heading = 0
            else:
                coe_heading = self.args.coe_heading

            loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
            shift_lat_last, shift_lon_last, theta_last, \
            L1_loss, L2_loss, L3_loss, L4_loss \
                = loss_func(self.args.loss_method, grd_feat_list, pred_feat_dict, gt_feat_dict,
                            shift_lats, shift_lons, thetas, gt_shiftv[:, 0], gt_shiftu[:, 0], gt_heading[:, 0],
                            pred_uv_dict, gt_uv_dict,
                            self.args.coe_shift_lat, self.args.coe_shift_lon, coe_heading,
                            self.args.coe_L1, self.args.coe_L2, self.args.coe_L3, self.args.coe_L4)

            return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                       shift_lat_last, shift_lon_last, theta_last, \
                       L1_loss, L2_loss, L3_loss, L4_loss, grd_conf_list
        else:
            return shift_lats[:, -1, -1], shift_lons[:, -1, -1], thetas[:, -1, -1]

    def forward_level_first(self, sat_map, grd_img_left, gt_shiftu=None, gt_shiftv=None, gt_heading=None, mode='train',
                file_name=None, gt_depth=None, loop=0):
        '''
        :param sat_map: [B, C, A, A] A--> sidelength
        :param grd_img_left: [B, C, H, W]
        :return:
        '''

        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        # A = sat_map.shape[-1]
        # sat_img_proj, _, _, _, _ = self.project_map_to_grd(
        #     grd_img_left, None, gt_shiftu, gt_shiftv, gt_heading, level=3, require_jac=True, gt_depth=gt_depth)
        # sat_img = transforms.ToPILImage()(sat_img_proj[0])
        # sat_img.save('sat_proj.png')
        # grd = transforms.ToPILImage()(grd_img_left[0])
        # grd.save('grd.png')
        # sat = transforms.ToPILImage()(sat_map[0])
        # sat.save('sat.png')

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)

        grd_feat_list, grd_conf_list = self.GrdFeatureNet(grd_img_left)

        shift_u = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        shift_v = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)
        heading = torch.zeros([B, 1], dtype=torch.float32, requires_grad=True, device=sat_map.device)

        gt_uv_dict = {}
        gt_feat_dict = {}
        pred_uv_dict = {}
        pred_feat_dict = {}
        shift_us_all = []
        shift_vs_all = []
        headings_all = []
        for level in range(len(sat_feat_list)):

            shift_us = []
            shift_vs = []
            headings = []
            for iter in range(self.N_iters):
                sat_feat = sat_feat_list[level]
                sat_conf = sat_conf_list[level]
                grd_feat = grd_feat_list[level]
                grd_conf = grd_conf_list[level]

                grd_H, grd_W = grd_feat.shape[-2:]
                sat_feat_proj, sat_conf_proj, dfeat_dpose, sat_uv, mask = self.project_map_to_grd(
                    sat_feat, sat_conf, shift_u, shift_v, heading, level, gt_depth=gt_depth)
                # [B, C, H, W], [B, 1, H, W], [3, B, C, H, W], [B, H, W, 2]

                grd_feat = grd_feat * mask[:, None, :, :]
                grd_conf = grd_conf * mask[:, None, :, :]

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
                    shift_u_new, shift_v_new, heading_new = self.LM_update(shift_u, shift_v, heading,
                                                            sat_feat_new,
                                                            sat_conf_new,
                                                            grd_feat_new,
                                                            grd_conf_new,
                                                            dfeat_dpose_new)  # only need to compare bottom half
                elif self.args.Optimizer == 'SGD':
                    # r = sat_feat_proj[:, :, grd_H // 2:, :] - grd_feat[:, :, grd_H // 2:, :]
                    # p = torch.mean(torch.abs(r), dim=[1, 2, 3])  # *100 #* 256 * 256 * 3
                    # dp_dshiftu = torch.autograd.grad(p, shift_u, retain_graph=True, create_graph=True,
                    #                              only_inputs=True)[0]
                    # dp_dshiftv = torch.autograd.grad(p, shift_v, retain_graph=True, create_graph=True,
                    #                                  only_inputs=True)[0]
                    # dp_dheading = torch.autograd.grad(p, heading, retain_graph=True, create_graph=True,
                    #                                 only_inputs=True)[0]
                    # print(dp_dshiftu)
                    # print(dp_dshiftv)
                    # print(dp_dheading)

                    shift_u_new, shift_v_new, heading_new = self.SGD_update(shift_u, shift_v, heading,
                                                                           sat_feat_new,
                                                                           sat_conf_new,
                                                                           grd_feat_new,
                                                                           grd_conf_new,
                                                                           dfeat_dpose_new)

                elif self.args.Optimizer == 'NN':
                    shift_u_new, shift_v_new, heading_new = self.NN_update(shift_u, shift_v, heading,
                                                                         sat_feat_new,
                                                                         sat_conf_new,
                                                                         grd_feat_new,
                                                                         grd_conf_new,
                                                                         dfeat_dpose_new)
                elif self.args.Optimizer == 'ADAM':
                    t = iter * self.args.level + level
                    if t==0:
                        m = 0
                        v = 0
                    shift_u_new, shift_v_new, heading_new, m, v = self.ADAM_update(shift_u, shift_v, heading,
                                                                         sat_feat_new,
                                                                         sat_conf_new,
                                                                         grd_feat_new,
                                                                         grd_conf_new,
                                                                         dfeat_dpose_new,
                                                                         m, v, t)


                shift_us.append(shift_u_new[:, 0])  # [B]
                shift_vs.append(shift_v_new[:, 0])  # [B]
                headings.append(heading_new[:, 0])  # [B]

                shift_u = shift_u_new.clone()
                shift_v = shift_v_new.clone()
                heading = heading_new.clone()

                if level not in pred_feat_dict.keys():
                    pred_feat_dict[level] = [sat_feat_proj]
                    pred_uv_dict[level] = [sat_uv / torch.tensor([sat_feat.shape[-1], sat_feat.shape[-2]], dtype=torch.float32).reshape(1, 1, 1, 2).to(sat_feat.device)]
                else:
                    pred_feat_dict[level].append(sat_feat_proj)
                    pred_uv_dict[level].append(sat_uv / torch.tensor([sat_feat.shape[-1], sat_feat.shape[-2]], dtype=torch.float32).reshape(1, 1, 1, 2).to(sat_feat.device))

                if level not in gt_uv_dict.keys() and mode == 'train':
                    gt_sat_feat_proj, _, _, gt_uv, _ = self.project_map_to_grd(
                        sat_feat, None, gt_shiftu, gt_shiftv, gt_heading, level, require_jac=False, gt_depth=gt_depth)
                    # [B, C, H, W], [B, H, W, 2]
                    gt_feat_dict[level] = gt_sat_feat_proj # [B, C, H, W]
                    gt_uv_dict[level] = gt_uv / torch.tensor([sat_feat.shape[-1], sat_feat.shape[-2]], dtype=torch.float32).reshape(1, 1, 1, 2).to(sat_feat.device)
                    # [B, H, W, 2]

            shift_us_all.append(torch.stack(shift_us, dim=1))  # [B, N_iters]
            shift_vs_all.append(torch.stack(shift_vs, dim=1))  # [B, N_iters]
            headings_all.append(torch.stack(headings, dim=1))  # [B, N_iters]

        shift_lats = torch.stack(shift_vs_all, dim=2)  # [B, N_iters, Level]
        shift_lons = torch.stack(shift_us_all, dim=2)  # [B, N_iters, Level]
        thetas = torch.stack(headings_all, dim=2)  # [B, N_iters, Level]

        if self.args.visualize:
            from visualize_utils import features_to_RGB, RGB_iterative_pose
            features_to_RGB(sat_feat_list, grd_feat_list, pred_feat_dict, gt_feat_dict, loop,
                            save_dir='./visualize/')
            RGB_iterative_pose(sat_map, grd_img_left, shift_lats, shift_lons, thetas, gt_shiftu, gt_shiftv, gt_heading,
                               self.meters_per_pixel[-1], self.args, loop, save_dir='./visualize/')


        if mode == 'train':

            if self.args.rotation_range == 0:
                coe_heading = 0
            else:
                coe_heading = self.args.coe_heading

            loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
            shift_lat_last, shift_lon_last, theta_last, \
            L1_loss, L2_loss, L3_loss, L4_loss \
                = loss_func(self.args.loss_method, grd_feat_list, pred_feat_dict, gt_feat_dict,
                            shift_lats, shift_lons, thetas, gt_shiftv[:, 0], gt_shiftu[:, 0], gt_heading[:, 0],
                            pred_uv_dict, gt_uv_dict,
                            self.args.coe_shift_lat, self.args.coe_shift_lon, coe_heading,
                            self.args.coe_L1, self.args.coe_L2, self.args.coe_L3, self.args.coe_L4)

            return loss, loss_decrease, shift_lat_decrease, shift_lon_decrease, thetas_decrease, loss_last, \
                       shift_lat_last, shift_lon_last, theta_last, \
                       L1_loss, L2_loss, L3_loss, L4_loss, grd_conf_list
        else:
            return shift_lats[:, -1, -1], shift_lons[:, -1, -1], thetas[:, -1, -1]

    def polar_transform(self, sat_feat, level):
        meters_per_pixel = self.meters_per_pixel[level]

        B, C, A, _ = sat_feat.shape

        grd_H = A // 2
        grd_W = A * 2

        v, u = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32),
                              torch.arange(0, 4 * grd_W, dtype=torch.float32))
        v = v.to(sat_feat.device)
        u = u.to(sat_feat.device)
        theta = u / grd_W * np.pi * 2
        radius = (1 - v / grd_H) * 40 / meters_per_pixel  # set radius as 40 meters

        us = A / 2 + radius * torch.cos(np.pi / 4 - theta)
        vs = A / 2 - radius * torch.sin(np.pi / 4 - theta)

        grids = torch.stack([us, vs], dim=-1).unsqueeze(dim=0).repeat(B, 1, 1, 1)  # [B, grd_H, grd_W, 2]

        polar_sat, _ = grid_sample(sat_feat, grids)

        return polar_sat

    def polar_coordinates(self, level):
        meters_per_pixel = self.meters_per_pixel[level]

        # B, C, A, _ = sat_feat.shape
        A = 512 / 2**(3-level)

        grd_H = A // 2
        grd_W = A * 2

        v, u = torch.meshgrid(torch.arange(0, grd_H, dtype=torch.float32),
                              torch.arange(0, 4 * grd_W, dtype=torch.float32))
        # v = v.to(sat_feat.device)
        # u = u.to(sat_feat.device)
        theta = u / grd_W * np.pi * 2
        radius = (1 - v / grd_H) * 40 / meters_per_pixel  # set radius as 40 meters

        us = A / 2 + radius * torch.cos(np.pi / 4 - theta)
        vs = A / 2 - radius * torch.sin(np.pi / 4 - theta)

        grids = torch.stack([us, vs], dim=-1).unsqueeze(dim=0)# .repeat(B, 1, 1, 1)  # [1, grd_H, grd_W, 2]

        # polar_sat, _ = grid_sample(sat_feat, grids)

        return grids

    def orien_corr(self, sat_map, grd_img_left, gt_shiftu=None, gt_shiftv=None, gt_heading=None, mode='train',
                file_name=None, gt_depth=None):
        '''
        :param sat_map: [B, C, A, A] A--> sidelength
        :param grd_img_left: [B, C, H, W]
        :return:
        '''

        B, _, ori_grdH, ori_grdW = grd_img_left.shape

        # A = sat_map.shape[-1]
        # sat_img_proj, _, _, _, _ = self.project_map_to_grd(
        #     grd_img_left, None, gt_shiftu, gt_shiftv, gt_heading, level=3, require_jac=True, gt_depth=gt_depth)
        # sat_img = transforms.ToPILImage()(sat_img_proj[0])
        # sat_img.save('sat_proj.png')
        # grd = transforms.ToPILImage()(grd_img_left[0])
        # grd.save('grd.png')
        # sat = transforms.ToPILImage()(sat_map[0])
        # sat.save('sat.png')

        sat_feat_list, sat_conf_list = self.SatFeatureNet(sat_map)

        grd_feat_list, grd_conf_list = self.GrdFeatureNet(grd_img_left)

        corr_list = []
        for level in range(len(sat_feat_list)):
            sat_feat = sat_feat_list[level]
            grd_feat = grd_feat_list[level]  # [B, C, H, W]
            B, C, H, W = grd_feat.shape
            grd_feat = F.normalize(grd_feat.reshape(B, -1)).reshape(B, -1, H, W)

            grids = self.polar_grids[level].detach().to(sat_feat.device).repeat(B, 1, 1, 1)  # [B, H, 4W, 2]
            polar_sat, _ = grid_sample(sat_feat, grids)
            # polar_sat = self.polar_transform(sat_feat, level)
            # [B, C, H, 4W]

            degree_per_pixel = 90 / W
            n = int(np.ceil(self.args.rotation_range / degree_per_pixel))
            sat_W = polar_sat.shape[-1]
            if sat_W - W < n:
                polar_sat1 = torch.cat([polar_sat[:, :, :, -n:], polar_sat, polar_sat[:, :, :, : (n - sat_W + W)]], dim=-1)
            else:
                polar_sat1 = torch.cat([polar_sat[:, :, :, -n:], polar_sat[:, :, :, : (W + n)]], dim=-1)

            # polar_sat1 = torch.cat([polar_sat, polar_sat[:, :, :, : (W-1)]], dim=-1)
            polar_sat2 = polar_sat1.reshape(1, B*C, H, -1)
            corr = F.conv2d(polar_sat2, grd_feat, groups=B)[0, :, 0, :]  # [B, 4W]

            denominator = F.avg_pool2d(polar_sat1.pow(2), (H, W), stride=1, divisor_override=1)[:, :, 0, :]  # [B, 4W]
            denominator = torch.sum(denominator, dim=1)  # [B, 4W]
            denominator = torch.maximum(torch.sqrt(denominator), torch.ones_like(denominator) * 1e-6)
            corr = 2 - 2 * corr / denominator

            orien = torch.argmin(corr, dim=-1)  # / (4 * W) * 360  # [B]
            orien = (orien - n) * degree_per_pixel

            corr_list.append((corr, degree_per_pixel))

        if mode == 'train':

            return self.triplet_loss(corr_list, gt_heading)
        else:
            return orien

    def triplet_loss(self, corr_list, gt_heading):
        gt = gt_heading * self.args.rotation_range #/ 360

        losses = []
        for level in range(len(corr_list)):
            corr = corr_list[level][0]
            degree_per_pixel = corr_list[level][1]
            B, W = corr.shape
            gt_idx = ((W - 1)/2 + torch.round(gt[:, 0]/degree_per_pixel)).long()

            # gt_idx = (torch.round(gt[:, 0] * (W-1)) % (W-1)).long()

            pos = corr[range(B), gt_idx]  # [B]
            pos_neg = pos[:, None] - corr  # [B, W]
            loss = torch.sum(torch.log(1 + torch.exp(pos_neg * 10))) / (B * (W - 1))
            losses.append(loss)

        return torch.sum(torch.stack(losses, dim=0))



