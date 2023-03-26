import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import copy
import json
from collections import defaultdict
from lib.utils.utils_data import crop_scale, flip_data, resample, split_clips

def posetrack2h36m(x):
    '''
        Input: x (T x V x C)

        PoseTrack keypoints = [ 'nose',
                                'head_bottom',
                                'head_top',
                                'left_ear',
                                'right_ear',
                                'left_shoulder',
                                'right_shoulder',
                                'left_elbow',
                                'right_elbow',
                                'left_wrist',
                                'right_wrist',
                                'left_hip',
                                'right_hip',
                                'left_knee',
                                'right_knee',
                                'left_ankle',
                                'right_ankle']
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,0,:] = (x[:,11,:] + x[:,12,:]) * 0.5
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,8,:] = x[:,1,:]
    y[:,7,:] = (y[:,0,:] + y[:,8,:]) * 0.5
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,2,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    y[:,0,2] = np.minimum(x[:,11,2], x[:,12,2])
    y[:,7,2] = np.minimum(y[:,0,2], y[:,8,2])
    return y


class PoseTrackDataset2D(Dataset):
    def __init__(self, flip=True, scale_range=[0.25, 1]):
        super(PoseTrackDataset2D, self).__init__()
        self.flip = flip
        data_root = "data/motion2d/posetrack18_annotations/train/"
        file_list = sorted(os.listdir(data_root))
        all_motions = []
        all_motions_filtered = []
        self.scale_range = scale_range
        for filename in file_list:
            with open(os.path.join(data_root, filename), 'r') as file:
                json_dict = json.load(file)
                annots = json_dict['annotations']
                imgs = json_dict['images']
                motions = defaultdict(list)
                for annot in annots:
                    tid = annot['track_id']
                    pose2d = np.array(annot['keypoints']).reshape(-1,3)
                    motions[tid].append(pose2d)
            all_motions += list(motions.values())
        for motion in all_motions:
            if len(motion)<30:
                continue
            motion = np.array(motion[:30])
            if np.sum(motion[:,:,2]) <= 306:  # Valid joint num threshold
                continue
            motion = crop_scale(motion, self.scale_range) 
            motion = posetrack2h36m(motion)
            motion[motion[:,:,2]==0] = 0
            if np.sum(motion[:,0,2]) < 30:
                continue                      # Root all visible (needed for framewise rootrel)
            all_motions_filtered.append(motion)
        all_motions_filtered = np.array(all_motions_filtered)
        self.motions_2d = all_motions_filtered
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions_2d)

    def __getitem__(self, index):
        'Generates one sample of data'
        motion_2d = torch.FloatTensor(self.motions_2d[index])
        if self.flip and random.random()>0.5:                       
            motion_2d = flip_data(motion_2d)
        return motion_2d, motion_2d
    
class InstaVDataset2D(Dataset):
    def __init__(self, n_frames=81, data_stride=27, flip=True, valid_threshold=0.0, scale_range=[0.25, 1]):
        super(InstaVDataset2D, self).__init__()
        self.flip = flip
        self.scale_range = scale_range
        motion_all = np.load('data/motion2d/InstaVariety/motion_all.npy')
        id_all = np.load('data/motion2d/InstaVariety/id_all.npy')
        split_id = split_clips(id_all, n_frames, data_stride)  
        motions_2d = motion_all[split_id]                        # [N, T, 17, 3]
        valid_idx = (motions_2d[:,0,0,2] > valid_threshold)
        self.motions_2d = motions_2d[valid_idx]
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions_2d)

    def __getitem__(self, index):
        'Generates one sample of data'
        motion_2d = self.motions_2d[index]
        motion_2d = crop_scale(motion_2d, self.scale_range) 
        motion_2d[motion_2d[:,:,2]==0] = 0
        if self.flip and random.random()>0.5:                       
            motion_2d = flip_data(motion_2d)
        motion_2d = torch.FloatTensor(motion_2d)
        return motion_2d, motion_2d
        