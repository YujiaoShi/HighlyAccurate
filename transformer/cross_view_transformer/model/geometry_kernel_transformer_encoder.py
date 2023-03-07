import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List
from IPython import embed

import sympy
from scipy.optimize import fsolve

import time

def ResNetBottleNeck(c): return Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid(
        (xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1),
                    value=1)                   # 3 h w
    # 1 3 h w
    indices = indices[None]

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(
            mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(
            std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        decoder_blocks: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))

        # bev coordinates
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width,
                            h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()  # 3 3
        # 3 (h w)
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')
        grid = rearrange(grid, 'd (h w) -> d h w', h=h,
                         w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer(
            'grid', grid, persistent=False)                    # 3 h w
        self.learned_features = nn.Parameter(
            sigma * torch.randn(dim, h, w))    # d h w

    def get_prior(self):
        return self.learned_features


class KernelAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(
            dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(
            dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(
            dim, heads * dim_head, bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None, mask=None):
        """
        q: (b n d H W)
        k: (b n k g d)
        v: (b n k g d)
        mask: (b n k 1)
        """
        _, _, _, H, W = q.shape
        num_points = k.shape[-2]
        # Move feature dim to last for multi-head proj
        # (b, n, k, d)
        q = rearrange(q, 'b n d H W -> b n (H W) d')

        # Project with multiple heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        # Group the head dim with batch dim
        q = rearrange(q, 'b n q (m d) -> (b m) n q 1 d',
                      m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b n q g (m d) -> (b m) n q g d',
                      m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b n q g (m d) -> (b m) q (n g) d',
                      m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * \
            torch.einsum('b n Q c d, b n Q K d -> b n Q c K', q, k)
        dot = rearrange(dot, 'b n Q c K -> b Q (n c K)')
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1, num_points)
            mask = rearrange(mask, 'b h n Q g -> (b h) Q (n g)')
            dot[~mask] = -10**9
        att = dot.to(q).softmax(dim=-1)
        a = torch.einsum('b Q K, b Q K d -> b Q d', att, v)

        a = rearrange(a, '(b m) Q d -> b Q (m d)',
                      m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z


@torch.no_grad()
def bev2image_sampling(points, I, E, height, width):
    """
    bev points to images: each bev point -> image points
    Args:
        points: (k, 3), (x,y,z)
        I: (b, n, 3, 3)
        E: (b, n, 4, 4)
    Return:
        sampled points: (k, 6, 2)
    """
    # (k, 3) -> (k, 4)
    k = points.shape[0]
    b, n = I.shape[:2]
    points = torch.cat([points, torch.ones_like(points[..., :1])], -1)
    intrin_mat = F.pad(I, (0, 1, 0, 1), value=0)
    intrin_mat[..., -1, -1] = 1.0
    # (k, 3) -> (b, n, k, 4, 1)
    points = points.view(1, 1, k, 4).repeat(b, n, 1, 1).unsqueeze(-1)
    # (b, n, 4, 4) * (k, 4)^T
    point2image = (intrin_mat @ E).view(b, n, 1, 4, 4).repeat(1, 1, k, 1, 1)
    sample_points = (point2image @ points).squeeze(-1)  # (b, n, k, 4)

    # filter points
    eps = 1e-5
    # mask: (b, n, k, 4)
    mask = (sample_points[..., 2:3] > eps)
    sample_points = sample_points[..., 0:2] / \
        sample_points[..., 2:3].clamp(min=eps)

    sample_points[..., 0] /= width
    sample_points[..., 1] /= height

    # sample points in the image
    mask = (mask & (sample_points[..., 0:1] > 0.0)
            & (sample_points[..., 0:1] < 1.0)
            & (sample_points[..., 1:2] > 0.0)
            & (sample_points[..., 1:2] < 1.0))
    mask = torch.nan_to_num(mask)

    return sample_points, mask


class IndexBEVProjector(nn.Module):
    """GridBEVProjector, based on Grid Sampling (nearest)
    """

    def __init__(self, image_size, grid_size=(3, 3), height=-1.0):
        super().__init__()
        self.image_size = image_size
        self.grid_size = grid_size
        grid_h, grid_w = grid_size
        y = torch.arange(grid_h) - grid_h // 2
        x = torch.arange(grid_w) - grid_w // 2
        offsets = torch.stack(torch.meshgrid(
            x, y, indexing="xy")).permute(1, 2, 0)
        self.register_buffer("grid_offsets", offsets, persistent=False)
        self.bev_height = height

    def forward(self, bev_grids, images, I, E):
        """
        bev_grids: (3, H, W)
        images: (b, n, c, h, w), features
        I: (b, n, 3, 3)
        E: (b, n, 4, 4)
        im_size: (height, width)
        """
        b, n = I.shape[:2]
        # unfold feature maps
        bn, c, h, w = images.shape

        # bev_grids -> image_coords
        # (3, H, W) -> (H*W, 3), k=H*W
        bev_points = bev_grids.reshape(3, -1).transpose(0, 1)
        bev_points[:, -1] = self.bev_height

        # (b, n, k, 2), (b, n, k, 1)
        sample_points, sample_mask = bev2image_sampling(
            bev_points, I, E, self.image_size[0], self.image_size[1])
        num_grid_points = self.grid_size[0] * self.grid_size[1]
        sample_points[..., 0] *= w
        sample_points[..., 1] *= h
        sample_points = sample_points.round().long()
        grid_offsets = self.grid_offsets.view(1, 1, 1, num_grid_points, 2)

        # [b, n, k, 9, 2]
        sample_points = sample_points.unsqueeze(-2) + grid_offsets
        # restrict sample_points between 0~H-1
        sample_points[..., 0].clamp_(min=0, max=w-1)
        sample_points[..., 1].clamp_(min=0, max=h-1)
        # [b, n, k, 9]
        k = sample_points.shape[2]
        sample_points_inds = sample_points[..., 0] + sample_points[..., 1] * w
        # [b*n, k*9]
        sample_points_inds = sample_points_inds.view(
            b * n, k * num_grid_points)
        # [b*n*h*w, c]
        images = rearrange(images, "b c h w -> (b h w) c")
        ind_offsets = (torch.arange(b * n, device=images.device)
                       * (h * w)).view(b * n, 1)
        # b*n*k*9, 1
        sample_points_inds = (sample_points_inds + ind_offsets).view(-1)
        # [b*n*k*9, c]
        sample_feats = images[sample_points_inds].reshape(
            b, n, k, num_grid_points, c)
        # embed()
        return sample_feats, sample_mask.detach()


class UnfoldBEVProjector(nn.Module):

    def __init__(self, image_size, grid_size=(3, 3), height=-1.0):
        super().__init__()
        self.image_size = image_size
        self.grid_size = grid_size
        self.pad_size = (grid_size[0] // 2, grid_size[1] // 2)
        self.unfold = nn.Unfold(
            kernel_size=self.grid_size,
            padding=self.pad_size
        )
        self.bev_height = height

    def forward(self, bev_grids, images, I, E):
        """
        bev_grids: (3, H, W)
        images: (b*n, c, h, w), features
        I: (b, n, 3, 3)
        E: (b, n, 4, 4)
        im_size: (height, width)
        """
        # bev_grids -> image_coords
        # (3, H, W) -> (H*W, 3), k=H*W
        bev_points = bev_grids.reshape(
            3, -1).transpose(0, 1).requires_grad_(False)
        # z: bev height
        bev_points[:, -1] = self.bev_height

        # (b, n, k, 2), (b, n, k, 1)
        sample_points, sample_mask = bev2image_sampling(
            bev_points, I, E, self.image_size[0], self.image_size[1])
        sample_points = sample_points * 2.0 - 1.0

        # embed()

        b, n = I.shape[:2]
        # unfold feature maps
        bn, c, h, w = images.shape
        # (b*n, c*p, h, w)
        unfold_images = self.unfold(images).view(bn, -1, h, w)
        # (b, n, k, 2) -> (b * n, k, 1, 2)
        k = sample_points.shape[2]
        sample_points = sample_points.reshape(b * n, k, 1, 2)

        # grid-sample -> (b*n, c, k, 1)
        # reshape -> (b, n, c', num, k)
        num_grid_points = self.grid_size[0] * self.grid_size[1]
        sample_feats = F.grid_sample(
            unfold_images, sample_points, mode='nearest').reshape(b, n, c, num_grid_points, k)
        # permute -> (b, n, k, grid_points, C)
        sample_feats = sample_feats.permute(0, 1, 4, 3, 2)
        return sample_feats, sample_mask.detach()


class GeometryKernelAttention(nn.Module):

    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        bev_z: int,
        kernel_h: int,
        kernel_w: int,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,
        sampling_type: str = "index",
        use_kernel_conv: bool = True,
        kernel_conv_h: int = 1,
        kernel_conv_w: int = 7
    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        if sampling_type == "unfold":
            self.sampling = UnfoldBEVProjector(
                (image_height, image_width), grid_size=(kernel_h, kernel_w), height=bev_z)
        elif sampling_type == "index":
            self.sampling = IndexBEVProjector(
                (image_height, image_width), grid_size=(kernel_h, kernel_w), height=bev_z)
        else:
            raise NotImplementedError()

        self.feature_linear = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, dim, bias=False)
        )

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, dim, bias=False)
            )
        if use_kernel_conv:
            self.conv = nn.Conv2d(
                feat_dim, feat_dim, (kernel_conv_h, kernel_conv_w),
                padding=(kernel_conv_h // 2, kernel_conv_w // 2))
        else:
            self.conv = lambda x: x

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Linear(4, dim, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attn = KernelAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(
        self,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
        I_: torch.FloatTensor,
        E_: torch.FloatTensor,
        intrinsics_dict: dict
    ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """

        # print(f'Check dimensions: ')
        # print(f'x:       {x.shape}: ')
        # print(f'feature:{feature.shape} ')
        # print(f'I_inv:  {I_inv.shape} ')     
        # print(f'E_inv:  {E_inv.shape} ')
        # print(f'I_:     {I_.shape} ')     
        # print(f'E_:     {E_.shape} ')
        # print(f'I_inv {I_inv.shape} ')     
        # 
        '''
            Check dimensions: 
            x:       torch.Size([1, 128, 25, 25]): 
            feature:torch.Size([1, 4, 32, 64, 256]) 
            I_inv:  torch.Size([1, 4, 3, 3]) # b, n, 3, 3
            E_inv:  torch.Size([1, 4, 4, 4]) 
            I_:     torch.Size([4, 3, 3]) 
            E_:     torch.Size([1, 4, 4, 4])            
        '''                
        start = time.time()


        b, n, _, _, _ = feature.shape

        # b n 3 h w
        pixel = self.image_plane
        _, _, _, h, w = pixel.shape

        # b n 4 1
        c = E_inv[..., -1:]
        # (b n) 4 1 1
        c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]
        # (b n) d 1 1
        c_embed = self.cam_embed(c_flat)

        # 1 1 3 (h w)
        pixel_flat = rearrange(pixel, '... h w -> ... (h w)')
        # b n 3 (h w)
        cam = I_inv @ pixel_flat

        # Distortion and mirror for fisheye
        # given (x', y') -> get (x, y)
        # x' = x * (1 + k1 * (x ** 2 + y ** 2) + k2 * (x ** 2 + y ** 2) ** 2)
        # y' = y * (1 + k1 * (x ** 2 + y ** 2) + k2 * (x ** 2 + y ** 2) ** 2)

        fcam2_k1 = intrinsics_dict["fcam2"]["distortion_parameters"]["k1"].item()
        fcam2_k2 = intrinsics_dict["fcam2"]["distortion_parameters"]["k2"].item()
        fcam2_xi = intrinsics_dict["fcam2"]["mirror_parameters"]["xi"].item()

        fcam3_k1 = intrinsics_dict["fcam3"]["distortion_parameters"]["k1"].item()
        fcam3_k2 = intrinsics_dict["fcam3"]["distortion_parameters"]["k2"].item()
        fcam3_xi = intrinsics_dict["fcam3"]["mirror_parameters"]["xi"].item()

        # To handle the inverse of undistortion, it seems unavoidable to call the sympy function in a non-vectorized way
        for idx in range(h * w):
            # Try scipy
            # fcam2
            x_prime = cam[:, 2, 0, idx].item()
            y_prime = cam[:, 2, 1, idx].item()
            def func(x):
                r2 = x[0] ** 2 + x[1] ** 2
                return [x[0] * (1 + fcam2_k1 * r2 + fcam2_k2 * r2 ** 2) - x_prime, x[1] * (1 + fcam2_k1 * r2 + fcam2_k2 * r2 ** 2) - y_prime]
            
            root = fsolve(func, [x_prime, y_prime])
            # print("prime = ", x_prime, " ", y_prime)
            # print("root = ", root)
            cam[:, 2, 0, idx] = root[0]
            cam[:, 2, 1, idx] = root[1]

            # fcam3
            x_prime = cam[:, 3, 0, idx].item()
            y_prime = cam[:, 3, 1, idx].item()
            def func(x):
                r2 = x[0] ** 2 + x[1] ** 2
                return [x[0] * (1 + fcam2_k1 * r2 + fcam2_k2 * r2 ** 2) - x_prime, x[1] * (1 + fcam2_k1 * r2 + fcam2_k2 * r2 ** 2) - y_prime]
            
            root = fsolve(func, [x_prime, y_prime])
            # print("prime = ", x_prime, " ", y_prime)
            # print("root = ", root)
            cam[:, 3, 0, idx] = root[0]
            cam[:, 3, 1, idx] = root[1]

            # x of fcam2 in idx's pixel
            # x, y = sympy.symbols('x, y')
            # r2 = x ** 2 + y ** 2
            # y2 = y ** 2
            # x_prime = cam[:, 2, 0, idx].item()
            # y_prime = cam[:, 2, 1, idx].item()
            # eq1 = sympy.Eq(x * (1 + fcam2_k1 * r2 + fcam2_k2 * r2 ** 2), x_prime)
            # eq2 = sympy.Eq(y * (1 + fcam2_k1 * r2 + fcam2_k2 * r2 ** 2), y_prime)
            # eq3 = sympy.Eq(x + y2, 2)
            # eq4 = sympy.Eq(x - y, 1)
            # print("x = ", x)
            # print("y = ", y)
            # print("r2 = ", r2)
            # print("x_prime = ", x_prime)
            # print("y_prime = ", y_prime)
            # print("fcam2_k1 = ", fcam2_k1)
            # print("fcam2_k2 = ", fcam2_k2)
            # print("solving... ", sympy.solve((eq3, eq4), (x, y)))
            # print("solving... ", sympy.nonlinsolve([eq1, eq2], [x, y]))
            # # print("solving... ", sympy.solve((eq1, eq2), (x, y)))
            # ans = sympy.solve((eq1, eq2), (x, y))
            # print("ans = ", ans)

        print("1: ", cam[0, 2, :, 0])
        cam[:, 2:, 0: 2, :] *= cam[:, 2:, 2, :] + fcam2_xi
        # cam[:, 2:, 1, :] *= cam[:, 2:, 2, :] + fcam3_xi
        print("2: ", cam[0, 2, :, 0])
        print("shape1 = ", cam[:, 2:, :, :].shape)
        print("shape2 = ", cam[:, 2:, 2:, :].shape)
        cam[:, 2:, :, :] /= cam[:, 2:, 2:, :]

        print("3: ", cam[0, 2, :, 0])

        

        cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)
        # b n 4 (h w)
        d = E_inv @ cam
        # (b n) 4 h w
        d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)

        # 2 H W
        world = bev.grid[:2]
        # 1 d H W
        w_embed = self.bev_embed(world[None])
        # (b n) d H W
        bev_embed = w_embed - c_embed
        # (b n) d H W
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)
        # b n d H W
        query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)

        # (b n) d h w
        feature_flat = rearrange(feature, 'b n ... -> (b n) ...')

        feature_flat = self.conv(feature_flat)
        # project local patches using sampling
        # concat feature and embeddings for sampling
        d_feature = feature_flat.shape[1]     

        feature_embed = torch.cat([feature_flat, d_flat], dim=1)
        feature_embed, mask = self.sampling(
            bev.grid.detach().clone(), feature_embed, I_, E_)

        # b, n, q, num_points, c
        feature_flat = feature_embed[..., :d_feature]
        d_flat = feature_embed[..., d_feature:]

        # (b n) q, num_points, 4
        d_embed = self.img_embed(d_flat)

        # d_embed: b, n, q, num_points, d
        # c_embed: (b, n), d, 1, 1
        img_embed = d_embed - c_embed.view(b, n, 1, 1, d_embed.shape[-1])
        img_embed = img_embed / (img_embed.norm(dim=-1, keepdim=True) + 1e-7)

        # g: num_grid_points
        # b, n, q, g, c
        if self.feature_proj is not None:
            key_flat = img_embed + self.feature_proj(feature_flat)
        else:
            # (b, n) d, h, w
            key_flat = img_embed

        # (b, n) d, h, w
        val_flat = self.feature_linear(feature_flat)

        # Expand + refine the BEV embedding
        # b, n, d, H, W
        query = query_pos + x[:, None]

        end = time.time()
        print("time elapse for one attention forward = ", end - start)

        return self.cross_attn(query, key_flat, val_flat, mask=mask, skip=x if self.skip else None)

import numpy as np

def process_fisheye_image(image, fish_cam_dict):
    ''' camera coordinate to image plane '''
    '''

    return : an image (tensor of shape (C, H, W)
    '''
    # points = points.T
    # norm = np.linalg.norm(points, axis=1)

    # x = points[:,0] / norm
    # y = points[:,1] / norm
    # z = points[:,2] / norm

    # x /= z+fish_cam_dict['mirror_parameters']['xi']
    # y /= z+fish_cam_dict['mirror_parameters']['xi']

    # k1 = fish_cam_dict['distortion_parameters']['k1']
    # k2 = fish_cam_dict['distortion_parameters']['k2']
    # gamma1 = fish_cam_dict['projection_parameters']['gamma1']
    # gamma2 = fish_cam_dict['projection_parameters']['gamma2']
    # u0 = fish_cam_dict['projection_parameters']['u0']
    # v0 = fish_cam_dict['projection_parameters']['v0']

    # ro2 = x*x + y*y
    # x *= 1 + k1*ro2 + k2*ro2*ro2
    # y *= 1 + k1*ro2 + k2*ro2*ro2

    # x = gamma1*x + u0
    # y = gamma2*y + v0

    # return x, y, norm * points[:,2] / np.abs(points[:,2])

    return image
    
def preprocess_images(images, intrinsics_dict):

    """
    Preprocess images[2] and images[3]
    so that we can multiply them with 3x3 intrinsic form

    return: images (tensor of shape (4,C,H,W))    
    
    """

    images_processed = torch.zeros_like(images) 
    images_processed[0:1, :, :] = images[0:1, :, :]
    images_processed[2, :, :] = process_fisheye_image(images[2,:,:], intrinsics_dict['fcam2'])
    images_processed[3, :, :] = process_fisheye_image(images[3,:,:], intrinsics_dict['fcam3'])
    return images_processed

def get_fisheye_intrinsics(fish_cam_dict):

    g1 =  float(fish_cam_dict['projection_parameters']['gamma1'])
    g2 =  float(fish_cam_dict['projection_parameters']['gamma2'])
    fx = g1
    fy = g2
    cx =  float(fish_cam_dict['projection_parameters']['u0'])
    cy =  float(fish_cam_dict['projection_parameters']['v0'])
    print(f"        fx: {fx}  fy: {fy}  cx: {cx}  cy: {cy}")
    return torch.tensor([[fx, 0,  cx], [0, fy, cy], [0, 0, 1]])
 
    return intrinsic 
def setup_intrinsics(intrinsics_dict):

    """
    Setup fisheye cameras intrincis (in 3x3 form)
    Take in intrinsics['fcam2'] and intrinsics['fcam3'] as input
    Input argument: a dictionary
    return: Intrinsic matrices I (tensor of shape (4, 3, 3)) (4 cameras, 2 perspective 2 fisheye)
    
    """

    I = torch.zeros(1, len(intrinsics_dict), 3, 3)
    I[0, 0,:,:] = intrinsics_dict['pcam0']
    I[0, 1,:,:] = intrinsics_dict['pcam1']
    I[0, 2,:,:] = get_fisheye_intrinsics(intrinsics_dict['fcam2'])
    I[0, 3,:,:] = get_fisheye_intrinsics(intrinsics_dict['fcam3'])
    return I

class GeometryKernelEncoder(nn.Module):

    def __init__(
            self,
            backbone,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone


        # F.interpolate: Down/up samples the input to either the given size or the given scale_factor
        if scale < 1.0:
            self.down = lambda x: F.interpolate(
                x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = list()
        layers = list()

        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(
                torch.zeros(feat_shape)).shape

            cva = GeometryKernelAttention(
                feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim)
                                  for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch):

        """
        Input batch: dict
        {image: [B,C,A,A],
         intrinsics_dict: a dictionary of 4 cameras intrinsics information
            pcam0/pcam1: 3x3 tensors
            fcam2/fcam3: dictionary of fisheye camera information
         extrinsicx: B, N, 4, 4}

        Input image 'batch':
        :param left_camera_k: [B, 3, 3]
        :param grd_img_left: [B, C, H, W]       
        """
        # b, n, _, _, _ = batch['image'].shape
        # B, C, H, W = batch['image'].shape # (1, 3, 256, 1024)
        # print(f'Shape of input ground-img: {B},{C},{H},{W}') # 1, 3, 512, 512
        # print("Representin B, C, H, W")

        b, n, _, _, _= batch['image'].shape
        # b n c h w
        images = batch['image'].flatten(0, 1)

        print("GKT Encoder: ")
        print(f'    Grd Img shape: {batch["image"].shape}' )
        print(f'    Intrinsics_dict len: {len(batch["intrinsics_dict"])}' )
        print(f'    Extrinsics shape: {batch["extrinsics"].shape}' )
        """
        => TODO: Make shape be:
        Img: b, n, c, h, w => 1, 4, 3, 256, 1024
        Intrinsics_dict: a dictionary
        Extrinsics: 1, 4, 4, 4
        Then, (b,n) = 4 = b, n = 1, 4

        # Process batch['intrinsic'] here:
        # batch['intrincis']: a dict of 4 items
        # batch['intrincis']['pcam0']: tensor of shape (3x3)
        # batch['intrincis']['pcam1']: tensor of shape (3x3)
        # batch['intrincis']['fcam2']: dictionary
        # batch['intrincis']['fcam3']: dictionary
        """

        # Preprocess points (images[2] and images[3[])

        # images = preprocess_images(images, batch['intrinsics_dict'])
        # I should be (4, 4, 3, 3)
        I = setup_intrinsics(batch['intrinsics_dict'])
        I = I.to(batch['extrinsics'].device)
        # print(f'Intrinsics.shape : {I.shape}' )
        # b n 3 3
        I_inv = I.inverse()
        # I_inv = batch['intrinsics'].inverse()
        # b n 4 4
        E_inv = batch['extrinsics'].inverse()     


        

        features = [self.down(y) for y in self.backbone(self.norm(images))]


        # print(f'len(features): {len(features)}')            # 2
        # print(f'features[0].shape {features[0].shape}')    features[0].shape torch.Size([4, 32, 64, 256])
        # print(f'features[1].shape {features[1].shape}')    features[1].shape torch.Size([4, 112, 16, 64])

        # d H W
        x = self.bev_embedding.get_prior()
        # b d H W
        x = repeat(x, '... -> b ...', b=b)


        # 02/23 TODO: Error here!
        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
            x = cross_view(x, self.bev_embedding, feature, I_inv,
                           E_inv, I, batch['extrinsics'], batch['intrinsics_dict'])
            x = layer(x)

        return x        
        # for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
        #     feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)
        #     x = cross_view(x, self.bev_embedding, feature, I_inv,
        #                    E_inv, batch['intrinsics'], batch['extrinsics'])
        #     x = layer(x)

        # return x
