import torch
import numpy as np
import glob
import os
import io
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from lib.data.augmentation import Augmenter3D
from lib.utils.tools import read_pkl
from lib.utils.utils_data import flip_data, crop_scale
from lib.utils.utils_mesh import flip_thetas
from lib.utils.utils_smpl import SMPL
from torch.utils.data import Dataset, DataLoader
from lib.data.datareader_h36m import DataReaderH36M  
from lib.data.datareader_mesh import DataReaderMesh  
from lib.data.dataset_action import random_move  

class SMPLDataset(Dataset):
    def __init__(self, args, data_split, dataset): # data_split: train/test; dataset: h36m, coco, pw3d
        random.seed(0)
        np.random.seed(0)
        self.clip_len = args.clip_len
        self.data_split = data_split
        if dataset=="h36m":
            datareader = DataReaderH36M(n_frames=self.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=self.clip_len, dt_root=args.data_root, dt_file=args.dt_file_h36m)
        elif dataset=="coco":
            datareader = DataReaderMesh(n_frames=1, sample_stride=args.sample_stride, data_stride_train=1, data_stride_test=1, dt_root=args.data_root, dt_file=args.dt_file_coco, res=[640, 640])
        elif dataset=="pw3d":
            datareader = DataReaderMesh(n_frames=self.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=self.clip_len, dt_root=args.data_root, dt_file=args.dt_file_pw3d, res=[1920, 1920])
        else:
            raise Exception("Mesh dataset undefined.")

        split_id_train, split_id_test = datareader.get_split_id()                        # Index of clips
        train_data, test_data = datareader.read_2d()
        train_data, test_data = train_data[split_id_train], test_data[split_id_test]     # Input: (N, T, 17, 3)
        self.motion_2d = {'train': train_data, 'test': test_data}[data_split]

        dt = datareader.dt_dataset
        smpl_pose_train = dt['train']['smpl_pose'][split_id_train]                       # (N, T, 72)
        smpl_shape_train = dt['train']['smpl_shape'][split_id_train]                     # (N, T, 10)
        smpl_pose_test = dt['test']['smpl_pose'][split_id_test]                          # (N, T, 72)
        smpl_shape_test = dt['test']['smpl_shape'][split_id_test]                        # (N, T, 10)
        
        self.motion_smpl_3d = {'train': {'pose': smpl_pose_train, 'shape': smpl_shape_train}, 'test': {'pose': smpl_pose_test, 'shape': smpl_shape_test}}[data_split]
        self.smpl = SMPL(
            args.data_root,
            batch_size=1,
        )

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motion_2d)

    def __getitem__(self, index):
        raise NotImplementedError 

class MotionSMPL(SMPLDataset):
    def __init__(self, args, data_split, dataset):
        super(MotionSMPL, self).__init__(args, data_split, dataset)
        self.flip = args.flip
        
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        motion_2d = self.motion_2d[index]                                            # motion_2d: (T,17,3)     
        motion_2d[:,:,2] = np.clip(motion_2d[:,:,2], 0, 1)
        motion_smpl_pose = self.motion_smpl_3d['pose'][index].reshape(-1, 24, 3)     # motion_smpl_3d: (T, 24, 3)    
        motion_smpl_shape = self.motion_smpl_3d['shape'][index]                      # motion_smpl_3d: (T,10)    
        
        if self.data_split=="train":
            if self.flip and random.random() > 0.5:                                  # Training augmentation - random flipping
                motion_2d = flip_data(motion_2d)
                motion_smpl_pose = flip_thetas(motion_smpl_pose)                

            
        motion_smpl_pose = torch.from_numpy(motion_smpl_pose).reshape(-1, 72).float()
        motion_smpl_shape = torch.from_numpy(motion_smpl_shape).reshape(-1, 10).float()
        motion_smpl = self.smpl(
            betas=motion_smpl_shape,
            body_pose=motion_smpl_pose[:, 3:],
            global_orient=motion_smpl_pose[:, :3],
            pose2rot=True
        )
        motion_verts = motion_smpl.vertices.detach()*1000.0
        J_regressor = self.smpl.J_regressor_h36m
        J_regressor_batch = J_regressor[None, :].expand(motion_verts.shape[0], -1, -1).to(motion_verts.device)
        motion_3d_reg = torch.matmul(J_regressor_batch, motion_verts)                 # motion_3d: (T,17,3)  
        motion_verts = motion_verts - motion_3d_reg[:, :1, :]
        motion_3d_reg = motion_3d_reg - motion_3d_reg[:, :1, :]                       # motion_3d: (T,17,3)    
        motion_theta = torch.cat((motion_smpl_pose, motion_smpl_shape), -1)
        motion_smpl_3d = {
            'theta': motion_theta,       # smpl pose and shape
            'kp_3d': motion_3d_reg,      # 3D keypoints
            'verts': motion_verts,       # 3D mesh vertices
        }
        return motion_2d, motion_smpl_3d