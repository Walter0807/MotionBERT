# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn

from lib.utils.utils_mesh import batch_rodrigues

class MeshLoss(nn.Module):
    def __init__(
            self,
            loss_3d=30.,
            loss_pose=1.,
            loss_shape=0.001,
            loss_type='MSE',
            device='cuda',
    ):
        super(MeshLoss, self).__init__()
        self.loss_3d = loss_3d
        self.loss_pose = loss_pose
        self.loss_shape = loss_shape

        self.device = device
        self.criterion_shape = nn.L1Loss().to(self.device)
        if loss_type == 'MSE': 
            self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
            self.criterion_regr = nn.MSELoss().to(self.device)
        elif loss_type == 'L1': 
            self.criterion_keypoints = nn.L1Loss(reduction='none').to(self.device)
            self.criterion_regr = nn.L1Loss().to(self.device)

    def forward(
            self,
            smpl_output,
            data_gt,
    ):
        # to reduce time dimension
        reduce = lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])
        # flatten for weight vectors
        flatten = lambda x: x.reshape(-1)
        # accumulate all predicted thetas from IEF
        accumulate_thetas = lambda x: torch.cat([output['theta'] for output in x],0)

        real_3d = reduce(data_gt['kp_3d'])
        data_3d_theta = reduce(data_gt['theta'])


        total_predict_thetas = accumulate_thetas(smpl_output)

        preds = smpl_output[-1]

        pred_j3d = preds['kp_3d']
        pred_theta = preds['theta']

        theta_size = pred_theta.shape[:2]

        pred_theta = reduce(pred_theta)
        pred_j3d = reduce(pred_j3d)

        loss_kp_3d = self.keypoint_3d_loss(pred_j3d, real_3d)
        loss_kp_3d = loss_kp_3d * self.loss_3d

        real_shape, pred_shape = data_3d_theta[:, 72:], pred_theta[:, 72:]
        real_pose, pred_pose = data_3d_theta[:, :72], pred_theta[:, :72]

        loss_dict = {
            'loss_kp_3d': loss_kp_3d,
        }
        if pred_theta.shape[0] > 0:
            loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape, real_pose, real_shape)
            loss_shape = loss_shape * self.loss_shape
            loss_pose = loss_pose * self.loss_pose
            loss_dict['loss_shape'] = loss_shape
            loss_dict['loss_pose'] = loss_pose

        gen_loss = torch.stack(list(loss_dict.values())).sum()

        return gen_loss, loss_dict


    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """

        # conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        # gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        # gt_keypoints_3d = gt_keypoints_3d
        # conf = conf
        pred_keypoints_3d = pred_keypoints_3d
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            # print(conf.shape, pred_keypoints_3d.shape, gt_keypoints_3d.shape)
            # return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
            return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas):
        pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(-1,3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1,3)).reshape(-1, 24, 3, 3)
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas
