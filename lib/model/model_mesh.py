import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.utils.utils_smpl import SMPL
from lib.utils.utils_mesh import rotation_matrix_to_angle_axis, rot6d_to_rotmat

class SMPLRegressor(nn.Module):
    def __init__(self, args, dim_rep=512, num_joints=17, hidden_dim=2048, dropout_ratio=0.):
        super(SMPLRegressor, self).__init__()
        param_pose_dim = 24 * 6
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(num_joints*dim_rep, hidden_dim)
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.fc2 = nn.Linear(num_joints*dim_rep, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.head_pose = nn.Linear(hidden_dim, param_pose_dim)
        self.head_shape = nn.Linear(hidden_dim, 10)
        nn.init.xavier_uniform_(self.head_pose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.head_shape.weight, gain=0.01)
        self.smpl = SMPL(
            args.data_root,
            batch_size=64,
            create_transl=False,
        )
        mean_params = np.load(self.smpl.smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.J_regressor = self.smpl.J_regressor_h36m

    def forward(self, feat, init_pose=None, init_shape=None):
        N, T, J, C = feat.shape
        NT = N * T
        feat = feat.reshape(N, T, -1)

        feat_pose = feat.reshape(NT, -1)     # (N*T, J*C)

        feat_pose = self.dropout(feat_pose)
        feat_pose = self.fc1(feat_pose)
        feat_pose = self.bn1(feat_pose)
        feat_pose = self.relu1(feat_pose)    # (NT, C)

        feat_shape = feat.permute(0,2,1)     # (N, T, J*C) -> (N, J*C, T)
        feat_shape = self.pool2(feat_shape).reshape(N, -1)          # (N, J*C)

        feat_shape = self.dropout(feat_shape)
        feat_shape = self.fc2(feat_shape)
        feat_shape = self.bn2(feat_shape)
        feat_shape = self.relu2(feat_shape)     # (N, C)

        pred_pose = self.init_pose.expand(NT, -1)   # (NT, C)
        pred_shape = self.init_shape.expand(N, -1)  # (N, C)

        pred_pose = self.head_pose(feat_pose) + pred_pose
        pred_shape = self.head_shape(feat_shape) + pred_shape
        pred_shape = pred_shape.expand(T, N, -1).permute(1, 0, 2).reshape(NT, -1)
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
        self.feat_J = num_joints
        self.head = SMPLRegressor(args, dim_rep, num_joints, hidden_dim, dropout_ratio)
        
    def forward(self, x, init_pose=None, init_shape=None, n_iter=3):
        '''
            Input: (N x T x 17 x 3) 
        '''
        N, T, J, C = x.shape  
        feat = self.backbone.get_representation(x)
        feat = feat.reshape([N, T, self.feat_J, -1])      # (N, T, J, C)
        smpl_output = self.head(feat)
        for s in smpl_output:
            s['theta'] = s['theta'].reshape(N, T, -1)
            s['verts'] = s['verts'].reshape(N, T, -1, 3)
            s['kp_3d'] = s['kp_3d'].reshape(N, T, -1, 3)
        return smpl_output