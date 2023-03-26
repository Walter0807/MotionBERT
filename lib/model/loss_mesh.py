import torch
import torch.nn as nn
import ipdb
from lib.utils.utils_mesh import batch_rodrigues
from lib.model.loss import *

class MeshLoss(nn.Module):
    def __init__(
            self,
            loss_type='MSE',
            device='cuda',
    ):
        super(MeshLoss, self).__init__()
        self.device = device
        self.loss_type = loss_type
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
        data_3d_theta = reduce(data_gt['theta'])

        preds = smpl_output[-1]
        pred_theta = preds['theta']
        theta_size = pred_theta.shape[:2]
        pred_theta = reduce(pred_theta)
        preds_local = preds['kp_3d'] - preds['kp_3d'][:, :, 0:1,:]  # (N, T, 17, 3)
        gt_local = data_gt['kp_3d'] - data_gt['kp_3d'][:, :, 0:1,:]
        real_shape, pred_shape = data_3d_theta[:, 72:], pred_theta[:, 72:]
        real_pose, pred_pose = data_3d_theta[:, :72], pred_theta[:, :72]
        loss_dict = {}
        loss_dict['loss_3d_pos'] = loss_mpjpe(preds_local, gt_local)
        loss_dict['loss_3d_scale'] = n_mpjpe(preds_local, gt_local)
        loss_dict['loss_3d_velocity'] = loss_velocity(preds_local, gt_local)
        loss_dict['loss_lv'] = loss_limb_var(preds_local)
        loss_dict['loss_lg'] = loss_limb_gt(preds_local, gt_local)
        loss_dict['loss_a'] = loss_angle(preds_local, gt_local)
        loss_dict['loss_av'] = loss_angle_velocity(preds_local, gt_local)
        
        if pred_theta.shape[0] > 0:
            loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape, real_pose, real_shape)
            loss_norm = torch.norm(pred_theta, dim=-1).mean()
            loss_dict['loss_shape'] = loss_shape 
            loss_dict['loss_pose'] = loss_pose 
            loss_dict['loss_norm'] = loss_norm 
        return loss_dict
        
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
