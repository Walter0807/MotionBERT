import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.utils.utils_smpl import SMPL
from lib.utils.utils_mesh import rotation_matrix_to_angle_axis, rot6d_to_rotmat, smpl_aa_to_ortho6d

class SMPLRegressor(nn.Module):
    def __init__(self, args, dim_rep=512, num_joints=17, hidden_dim=2048, dropout_ratio=0.):
        super(SMPLRegressor, self).__init__()
        param_pose_dim = 24 * 6
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(num_joints*dim_rep, hidden_dim)
        self.fc2 = nn.Conv1d(param_pose_dim, hidden_dim, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.head_pose = nn.Conv1d(hidden_dim*2, param_pose_dim, kernel_size=5, stride=1, padding=2)

        self.smpl = SMPL(
            args.data_root,
            batch_size=64,
            create_transl=False,
        )
        self.J_regressor = self.smpl.J_regressor_h36m

    def forward(self, feat, init_pose=None, init_shape=None):
        N, T, J, C = feat.shape
        NT = N * T
        feat = feat.reshape(N, T, -1)
        feat = self.dropout(feat)
        feat_pose = feat.reshape(NT, -1)     # (N*T, J*C)
        init_pose = init_pose.reshape(N, T, -1).permute(0, 2, 1)   # (N, T, C) -> (N, C, T)

        feat_pose = self.fc1(feat_pose)
        feat_pose = self.bn1(feat_pose)
        feat_pose = self.relu1(feat_pose)        # (NT, C)

        feat_pred = self.fc2(init_pose)
        feat_pred = self.bn2(feat_pred)
        feat_pred = self.relu2(feat_pred)                            # (N, C, T)
        
        feat_pose = feat_pose.reshape(N, T, -1).permute(0, 2, 1)     # (NT, C) -> (N, C, T)
        feat_pose = torch.cat((feat_pose, feat_pred), 1)
        pred_pose = self.head_pose(feat_pose) + init_pose            # (N, C, T)
        
        pred_pose = pred_pose.permute(0, 2, 1).reshape(NT, -1)
        pred_shape = init_shape.reshape(NT, -1)
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(-1, 24, 3, 3)
        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )
        pred_vertices = pred_output.vertices*1000.0
        assert self.J_regressor is not None
        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
        pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
        output = [{
            'theta'  : torch.cat([pose, pred_shape], dim=1),    # (N*T, 72+10)
            'verts'  : pred_vertices,                           # (N*T, 6890, 3)
            'kp_3d'  : pred_joints,                             # (N*T, 17, 3)
        }]
        return output
    
class MeshRegressor(nn.Module):
    def __init__(self, args, backbone, dim_rep=512, num_joints=17, hidden_dim=2048, dropout_ratio=0.5):
        super(MeshRegressor, self).__init__()
        self.backbone = backbone
        self.head = SMPLRegressor(args, dim_rep, num_joints, hidden_dim, dropout_ratio)

    def forward(self, x, init_pose=None, init_shape=None):
        '''
            Input: (N x T x 17 x 3) 
        '''
        N, T, J, C = x.shape  
        feat = self.backbone.get_representation(x)
        init_pose = smpl_aa_to_ortho6d(init_pose)
        smpl_output = self.head(feat, init_pose, init_shape)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(N, T, -1)
            s['verts'] = s['verts'].reshape(N, T, -1, 3)
            s['kp_3d'] = s['kp_3d'].reshape(N, T, -1, 3)
        return smpl_output