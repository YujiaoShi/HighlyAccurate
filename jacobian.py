import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision


class Conv2d_cir(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), n=1):
        super(Conv2d_cir, self).__init__()
        self.n = n
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
    def forward(self, x):
        out = torch.cat([x[:, :, :, -self.n:], x, x[:, :, :, :self.n]], dim=-1)
        out = F.pad(out, (0, 0, 1, 1), mode='constant') # pad last dim by (0, 0) and 2nd to last by (1, 1)
        out = self.conv(out)
        return out


def conv2d_jac(conv2d, x, jac):
    '''
    x: (N, C_in, H_in, W_in)
    jac: (q_dim, N, C_in, H_in, W_in)
    returns:
    y: (N, C_out, H_out, W_out)
    jac: (q_dim, N, C_out, H_out, W_out)
    '''
    q_dim, N, C_in, H_in, W_in = jac.shape
    jac = jac.reshape(-1, C_in, H_in, W_in)
    if conv2d.padding_mode != 'zeros':
        jac= F.conv2d(F.pad(jac, tuple(x for x in reversed(x.padding) for _ in range(2)), mode=conv2d.padding_mode),
                        conv2d.weight, None, conv2d.stride, (0,0), conv2d.dilation, conv2d.groups)
    else:
        jac = F.conv2d(jac, conv2d.weight, None, conv2d.stride, conv2d.padding, conv2d.dilation, conv2d.groups)
    _, C_out, H_out, W_out = jac.shape
    return conv2d(x), jac.reshape(q_dim, N, -1, H_out, W_out)


def maxpool_jac(maxpool, x, jac):
    # specifically designed for maxpool_2d, with kernel_size = 2, stride = 2
    y, indices = maxpool(x)

    q_dim, N, C_in, H_in, W_in = jac.shape
    indices = indices.unsqueeze(dim=0).repeat(q_dim, 1, 1, 1, 1).reshape(q_dim*N, C_in, H_in//2, W_in//2)

    jac = jac.reshape(-1, C_in, H_in, W_in)
    flattened_tensor = jac.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)

    return y, output.reshape(q_dim, N, C_in, H_in//2, W_in//2)


def activation_jac(activation, x, jac):
    '''
    actication function must be element-wsie
    x: (...)
    jac: (q_dim, ...)
    returns:
    y: (...)
    jac: (q_dim, ...)
    '''
    y = activation(x)
    # jac = jac * torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    # if relu:
    jac = jac * (x>0).float()
    return y, jac


def interpolate_jac(image, new_shape, jac):
    # jac.shape = [#pose, N, C, IH, IW]
    B, C, IH, IW = image.shape
    H, W = new_shape

    u0 = torch.arange(W, dtype=torch.float32, device=image.device) / (W - 1) * (IW - 1)
    v0 = torch.arange(H, dtype=torch.float32, device=image.device) / (H - 1) * (IH - 1)

    iy, ix = torch.meshgrid(v0, u0)

    with torch.no_grad():
        ix_nw = torch.floor(ix)  # north-west  upper-left-x
        iy_nw = torch.floor(iy)  # north-west  upper-left-y
        ix_ne = ix_nw + 1        # north-east  upper-right-x
        iy_ne = iy_nw            # north-east  upper-right-y
        ix_sw = ix_nw            # south-west  lower-left-x
        iy_sw = iy_nw + 1        # south-west  lower-left-y
        ix_se = ix_nw + 1        # south-east  lower-right-x
        iy_se = iy_nw + 1        # south-east  lower-right-y

        torch.clamp(ix_nw, 0, IW -1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH -1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW -1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH -1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW -1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH -1, out=iy_sw)

        torch.clamp(ix_se, 0, IW -1, out=ix_se)
        torch.clamp(iy_se, 0, IH -1, out=iy_se)

    nw = (ix_se - ix) * (iy_se - iy)  #[H, W]
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    nw = nw.unsqueeze(dim=0).unsqueeze(dim=0) # [1, 1, H, W]
    ne = ne.unsqueeze(dim=0).unsqueeze(dim=0)
    sw = sw.unsqueeze(dim=0).unsqueeze(dim=0)
    se = se.unsqueeze(dim=0).unsqueeze(dim=0)

    image = image.view(B, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(1, 1, H * W).repeat(B, C, 1)).view(B, C, H, W)
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(1, 1, H * W).repeat(B, C, 1)).view(B, C, H, W)
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(1, 1, H * W).repeat(B, C, 1)).view(B, C, H, W)
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(1, 1, H * W).repeat(B, C, 1)).view(B, C, H, W)

    out_val = (nw_val * nw + ne_val * ne + sw_val * sw + se_val * se)

    if jac is not None:
        M = jac.shape[0]
        jac1 = jac.permute(1, 0, 2, 3, 4).reshape(B, M*C, IH, IW)

        nw_jac = torch.gather(jac1, 2, (iy_nw * IW + ix_nw).long().view(1, 1, H * W).repeat(B, M*C, 1)).view(B, M*C, H, W)
        ne_jac = torch.gather(jac1, 2, (iy_ne * IW + ix_ne).long().view(1, 1, H * W).repeat(B, M*C, 1)).view(B, M*C, H, W)
        sw_jac = torch.gather(jac1, 2, (iy_sw * IW + ix_sw).long().view(1, 1, H * W).repeat(B, M*C, 1)).view(B, M*C, H, W)
        se_jac = torch.gather(jac1, 2, (iy_se * IW + ix_se).long().view(1, 1, H * W).repeat(B, M*C, 1)).view(B, M*C, H, W)

        jac_new = (nw_jac * nw + ne_jac * ne + sw_jac * sw + se_jac * se)

        jac_new = jac_new.reshape(B, M, C, IH, IW).permute(1, 0, 2, 3, 4)  # [M, B, C, IH, IW]

        return out_val, jac_new
    else:
        return out_val


def grid_sample(image, optical, jac=None):
    # values in optical within range of [0, H], and [0, W]
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0].view(N, 1, H, W)
    iy = optical[..., 1].view(N, 1, H, W)

    with torch.no_grad():
        ix_nw = torch.floor(ix)  # north-west  upper-left-x
        iy_nw = torch.floor(iy)  # north-west  upper-left-y
        ix_ne = ix_nw + 1        # north-east  upper-right-x
        iy_ne = iy_nw            # north-east  upper-right-y
        ix_sw = ix_nw            # south-west  lower-left-x
        iy_sw = iy_nw + 1        # south-west  lower-left-y
        ix_se = ix_nw + 1        # south-east  lower-right-x
        iy_se = iy_nw + 1        # south-east  lower-right-y

        torch.clamp(ix_nw, 0, IW -1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH -1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW -1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH -1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW -1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH -1, out=iy_sw)

        torch.clamp(ix_se, 0, IW -1, out=ix_se)
        torch.clamp(iy_se, 0, IH -1, out=iy_se)

    mask_x = (ix >= 0) & (ix <= IW - 1)
    mask_y = (iy >= 0) & (iy <= IH - 1)
    mask = mask_x * mask_y

    assert torch.sum(mask) > 0

    nw = (ix_se - ix) * (iy_se - iy) * mask
    ne = (ix - ix_sw) * (iy_sw - iy) * mask
    sw = (ix_ne - ix) * (iy - iy_ne) * mask
    se = (ix - ix_nw) * (iy - iy_nw) * mask

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1)).view(N, C, H, W)

    out_val = (nw_val * nw + ne_val * ne + sw_val * sw + se_val * se)

    if jac is not None:

        dout_dpx = (nw_val * (-(iy_se - iy) * mask) + ne_val * (iy_sw - iy) * mask +
                    sw_val * (-(iy - iy_ne) * mask) + se_val * (iy - iy_nw) * mask)
        dout_dpy = (nw_val * (-(ix_se - ix) * mask) + ne_val * (-(ix - ix_sw) * mask) +
                    sw_val * (ix_ne - ix) * mask + se_val * (ix - ix_nw) * mask)
        dout_dpxy = torch.stack([dout_dpx, dout_dpy], dim=-1)  # [N, C, H, W, 2]

        # assert jac.shape[1:] == [N, H, W, 2]
        jac_new = dout_dpxy[None, :, :, :, :, :] * jac[:, :, None, :, :, :]
        jac_new1 = torch.sum(jac_new, dim=-1)

        if torch.any(torch.isnan(jac)) or torch.any(torch.isnan(dout_dpxy)):
            print('Nan occurs')

        return out_val, jac_new1 #jac_new1 #jac_new.permute(4, 0, 1, 2, 3)
    else:
        return out_val, None


    # out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
    #            ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
    #            sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
    #            se_val.view(N, C, H, W) * se.view(N, 1, H, W))
    #
    # return out_val


# import numpy as np
# import PIL.Image as Image
# x = torch.from_numpy(np.random.rand(1, 3, 32, 32).astype(np.float32)).cuda()
# x.requires_grad = True
#
# grids = torch.from_numpy(np.random.uniform(-1, 1, size=[1, 32, 32, 2]).astype(np.float32)).cuda()
#
# img0 = F.grid_sample(x, grids, align_corners=True)
# img1 = grid_sample(x, (grids + 1)/2 * 31)
# print(torch.sum(torch.abs(img0 - img1)))



# y, dy_dgrids = grid_sample(x, grids)
#
# jac = torch.autograd.functional.jacobian(grid_sample, (x, grids))
#
# torch.sum(jac[0][1])
# torch.sum(dy_dgrids)
# temp = jac[0][1][0, :, :,:, 0, :, :, : ].reshape([3, 32*32, 32*32, 2])
# temp_diag = torch.diagonal(temp, dim1=1, dim2=2)
# torch.sum(temp_diag) - torch.sum(temp)
#
# a = 1
