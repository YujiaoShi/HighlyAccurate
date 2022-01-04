import os.path

import numpy as np
from PIL import Image
from torchvision import transforms

def features_to_RGB(sat_feat_list, grd_feat_list, pred_feat_dict, gt_sat_feat_proj, loop=0, save_dir='./visualize/'):
    """Project a list of d-dimensional feature maps to RGB colors using PCA."""
    from sklearn.decomposition import PCA

    def reshape_normalize(x):
        '''
        Args:
            x: [B, C, H, W]

        Returns:

        '''
        B, C, H, W = x.shape
        x = x.transpose([0, 2, 3, 1]).reshape([-1, C])

        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator==0, 1, denominator)
        return x / denominator

    def normalize(x):
        denominator = np.linalg.norm(x, axis=-1, keepdims=True)
        denominator = np.where(denominator == 0, 1, denominator)
        return x / denominator

    # sat_shape = []
    # grd_shape = []
    for level in range(len(sat_feat_list)):
    # for level in [len(sat_feat_list)-1]:
        flatten = []

        sat_feat = sat_feat_list[level].data.cpu().numpy()  # [B, C, H, W]
        grd_feat = grd_feat_list[level].data.cpu().numpy()  # [B, C, H, W]
        s2g_feat = [feat.data.cpu().numpy() for feat in pred_feat_dict[level]]
        # a list with length iters, each item has shape [B, C, H, W]
        gt_a2g = gt_sat_feat_proj[level].data.cpu().numpy()   # [B, C, H, W]

        B, C, A, _ = sat_feat.shape
        B, C, H, W = grd_feat.shape
        # sat_shape.append([B, C, A, A])
        # grd_shape.append([B, C, H, W])

        flatten.append(reshape_normalize(sat_feat))
        flatten.append(reshape_normalize(grd_feat))
        flatten.append(reshape_normalize(gt_a2g[:, :, H//2:, :]))

        for feat in s2g_feat:
            flatten.append(reshape_normalize(feat[:, :, H//2:, :]))

        flatten = np.concatenate(flatten[:1], axis=0)

        # if level == 0:
        pca = PCA(n_components=3)
        pca.fit(reshape_normalize(sat_feat))

        pca_grd = PCA(n_components=3)
        pca_grd.fit(reshape_normalize(grd_feat))

    # for level in range(len(sat_feat_list)):
        sat_feat = sat_feat_list[level].data.cpu().numpy()  # [B, C, H, W]
        grd_feat = grd_feat_list[level].data.cpu().numpy()  # [B, C, H, W]
        s2g_feat = [feat.data.cpu().numpy() for feat in pred_feat_dict[level]]
        # a list with length iters, each item has shape [B, C, H, W]
        gt_s2g = gt_sat_feat_proj[level].data.cpu().numpy()   # [B, C, H, W]

        B, C, A, _ = sat_feat.shape
        B, C, H, W = grd_feat.shape
        sat_feat_new = ((normalize(pca.transform(reshape_normalize(sat_feat[..., :]))) + 1 )/ 2).reshape(B, A, A, 3)
        grd_feat_new = ((normalize(pca_grd.transform(reshape_normalize(grd_feat[:, :, H//2:, :]))) + 1) / 2).reshape(B, H//2, W, 3)
        gt_s2g_new = ((normalize(pca.transform(reshape_normalize(gt_s2g[:, :, H//2:, :]))) + 1) / 2).reshape(B, H//2, W, 3)

        for idx in range(B):
            sat = Image.fromarray((sat_feat_new[idx] * 255).astype(np.uint8))
            sat = sat.resize((512, 512))
            sat.save(os.path.join(save_dir, 'sat_feat_' + str(loop * B + idx) + '_level_' + str(level) + '.png'))

            grd = Image.fromarray((grd_feat_new[idx] * 255).astype(np.uint8))
            grd = grd.resize((1024, 128))
            grd.save(os.path.join(save_dir, 'grd_feat_' + str(loop * B + idx) + '_level_' + str(level) + '.png'))

            s2g = Image.fromarray((gt_s2g_new[idx] * 255).astype(np.uint8))
            s2g = s2g.resize((1024, 128))
            s2g.save(os.path.join(save_dir, 's2g_gt_feat_' + str(loop * B + idx) + '_level_' + str(level) + '.png'))

        # for iter in range(len(s2g_feat)):
        for iter in [len(s2g_feat)-1]:
            feat = s2g_feat[iter]
            feat_new = ((normalize(pca.transform(reshape_normalize(feat[:, :, H//2:, :]))) + 1) / 2).reshape(B, H//2, W, 3)

            for idx in range(B):
                img = Image.fromarray((feat_new[idx] * 255).astype(np.uint8))
                img = img.resize((1024, 128))
                img.save(os.path.join(save_dir, 's2g_feat_' + str(loop * B + idx) + '_level_' + str(level)
                                      + '_iter_' + str(iter) + '.png'))

    return


def RGB_iterative_pose(sat_img, grd_img, shift_lats, shift_lons, thetas, gt_shift_u, gt_shift_v, gt_theta,
                       meter_per_pixel, args, loop=0, save_dir='./visualize/'):
    '''
    This function is for KITTI dataset
    Args:
        sat_img: [B, C, H, W]
        shift_lats: [B, Niters, Level]
        shift_lons: [B, Niters, Level]
        thetas: [B, Niters, Level]
        meter_per_pixel: scalar

    Returns:

    '''

    import matplotlib.pyplot as plt

    B, _, A, _ = sat_img.shape

    # A = 512 - 128

    shift_lats = (A/2 - shift_lats.data.cpu().numpy() * args.shift_range_lat / meter_per_pixel).reshape([B, -1])
    shift_lons = (A/2 + shift_lons.data.cpu().numpy() * args.shift_range_lon / meter_per_pixel).reshape([B, -1])
    thetas = (- thetas.data.cpu().numpy() * args.rotation_range).reshape([B, -1])

    gt_u = (A/2 + gt_shift_u.data.cpu().numpy() * args.shift_range_lon / meter_per_pixel)
    gt_v = (A/2 - gt_shift_v.data.cpu().numpy() * args.shift_range_lat / meter_per_pixel)
    gt_theta = - gt_theta.cpu().numpy() * args.rotation_range

    for idx in range(B):
        img = np.array(transforms.functional.to_pil_image(sat_img[idx], mode='RGB'))
        # img = img[64:-64, 64:-64]
        # A = img.shape[0]

        fig, ax = plt.subplots()
        ax.imshow(img)
        init = ax.scatter(A/2, A/2, color='r', s=20, zorder=2)
        update = ax.scatter(shift_lons[idx, :-1], shift_lats[idx, :-1], color='m', s=15, zorder=2)
        pred = ax.scatter(shift_lons[idx, -1], shift_lats[idx, -1], color='g', s=20, zorder=2)
        gt = ax.scatter(gt_u[idx], gt_v[idx], color='b', s=20, zorder=2)
        # ax.legend((init, update, pred, gt), ('Init', 'Intermediate', 'Pred', 'GT'),
        #           frameon=False, fontsize=14, labelcolor='r', loc=2)
        # loc=1: upper right
        # loc=3: lower left

        # if args.rotation_range>0:
        init = ax.quiver(A/2, A/2, 1, 1, angles=0, color='r', zorder=2)
        # update = ax.quiver(shift_lons[idx, :], shift_lats[idx, :], 1, 1, angles=thetas[idx, :], color='r')
        pred = ax.quiver(shift_lons[idx, -1], shift_lats[idx, -1], 1, 1, angles=thetas[idx, -1], color='g', zorder=2)
        gt = ax.quiver(gt_u[idx], gt_v[idx], 1, 1, angles=gt_theta[idx], color='b', zorder=2)
        # ax.legend((init, pred, gt), ('pred', 'Updates', 'GT'), frameon=False, fontsize=16, labelcolor='r')
        #
        # # for i in range(shift_lats.shape[1]-1):
        # #     ax.quiver(shift_lons[idx, i], shift_lats[idx, i], shift_lons[idx, i+1], shift_lats[idx, i+1], angles='xy',
        # #               color='r')
        #
        ax.axis('off')

        plt.savefig(os.path.join(save_dir, 'points_' + str(loop * B + idx) + '.png'),
                    transparent=True, dpi=A, bbox_inches='tight')
        plt.close()

        grd = transforms.functional.to_pil_image(grd_img[idx], mode='RGB')
        grd.save(os.path.join(save_dir, 'grd_' + str(loop * B + idx) + '.png'))

        sat = transforms.functional.to_pil_image(sat_img[idx], mode='RGB')
        sat.save(os.path.join(save_dir, 'sat_' + str(loop * B + idx) + '.png'))


def RGB_iterative_pose_ford(sat_img, grd_img, shift_lats, shift_lons, thetas, gt_shift_u, gt_shift_v, gt_theta,
                       meter_per_pixel, args, loop=0, save_dir='./visualize/'):
    '''
    This function is for KITTI dataset
    Args:
        sat_img: [B, C, H, W]
        shift_lats: [B, Niters, Level]
        shift_lons: [B, Niters, Level]
        thetas: [B, Niters, Level]
        meter_per_pixel: scalar

    Returns:

    '''

    import matplotlib.pyplot as plt

    B, _, A, _ = sat_img.shape

    # A = 512 - 128

    shift_lats = (A/2 - shift_lats.data.cpu().numpy() * args.shift_range_lat / meter_per_pixel).reshape([B, -1])
    shift_lons = (A/2 - shift_lons.data.cpu().numpy() * args.shift_range_lon / meter_per_pixel).reshape([B, -1])
    thetas = (- thetas.data.cpu().numpy() * args.rotation_range).reshape([B, -1])

    gt_u = (A/2 - gt_shift_u.data.cpu().numpy() * args.shift_range_lat / meter_per_pixel)
    gt_v = (A/2 - gt_shift_v.data.cpu().numpy() * args.shift_range_lon / meter_per_pixel)
    gt_theta = - gt_theta.cpu().numpy() * args.rotation_range

    for idx in range(B):
        img = np.array(transforms.functional.to_pil_image(sat_img[idx], mode='RGB'))
        # img = img[64:-64, 64:-64]
        # A = img.shape[0]

        fig, ax = plt.subplots()
        ax.imshow(img)
        init = ax.scatter(A/2, A/2, color='r', s=20, zorder=2)
        update = ax.scatter(shift_lats[idx, :-1], shift_lons[idx, :-1], color='m', s=15, zorder=2)
        pred = ax.scatter(shift_lats[idx, -1], shift_lons[idx, -1], color='g', s=20, zorder=2)
        gt = ax.scatter(gt_u[idx], gt_v[idx], color='b', s=20, zorder=2)
        # ax.legend((init, update, pred, gt), ('Init', 'Intermediate', 'Pred', 'GT'),
        #           frameon=False, fontsize=14, labelcolor='r', loc=2)
        # loc=1: upper right
        # loc=3: lower left

        # if args.rotation_range>0:
        init = ax.quiver(A/2, A/2, 1, 1, angles=90, color='r', zorder=2)
        # update = ax.quiver(shift_lons[idx, :], shift_lats[idx, :], 1, 1, angles=thetas[idx, :], color='r')
        pred = ax.quiver(shift_lats[idx, -1], shift_lons[idx, -1], 1, 1, angles=thetas[idx, -1] + 90, color='g', zorder=2)
        gt = ax.quiver(gt_u[idx], gt_v[idx], 1, 1, angles=gt_theta[idx] + 90, color='b', zorder=2)
        # ax.legend((init, pred, gt), ('pred', 'Updates', 'GT'), frameon=False, fontsize=16, labelcolor='r')
        #
        # # for i in range(shift_lats.shape[1]-1):
        # #     ax.quiver(shift_lons[idx, i], shift_lats[idx, i], shift_lons[idx, i+1], shift_lats[idx, i+1], angles='xy',
        # #               color='r')
        #
        ax.axis('off')

        plt.savefig(os.path.join(save_dir, 'points_' + str(loop * B + idx) + '.png'),
                    transparent=True, dpi=A, bbox_inches='tight')
        plt.close()

        grd = transforms.functional.to_pil_image(grd_img[idx], mode='RGB')
        grd.save(os.path.join(save_dir, 'grd_' + str(loop * B + idx) + '.png'))

        sat = transforms.functional.to_pil_image(sat_img[idx], mode='RGB')
        sat.save(os.path.join(save_dir, 'sat_' + str(loop * B + idx) + '.png'))
