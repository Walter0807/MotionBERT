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
    
class MotionDataset(Dataset):
    def __init__(self, args, subset_list, data_split): # data_split: train/test
        np.random.seed(0)
        self.data_root = args.data_root
        self.subset_list = subset_list
        self.data_split = data_split
        file_list_all = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list_all.append(os.path.join(data_path, i))
        self.file_list = file_list_all
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError 

class MotionDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(MotionDataset3D, self).__init__(args, subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d
    
    def __coco_to_h36m(self, kps_coco, n_dim=2):
        """
        kps_coco: np.array of shape (N, 17, 2) in COCO format
        returns:  np.array of shape (N, 17, 2) in H36M format
        """
        # this is the same function from `motionbert/motionbert_convert.ipynb` in the DL_project repo
        N = kps_coco.shape[0]
        h36m = np.zeros((N, 17, n_dim), dtype=np.float32)

        # Synthesize H36M joints not present in COCO
        # H36M[0] = Hip root = average of left and right hips
        h36m[:, 0] = (kps_coco[:, 11] + kps_coco[:, 12]) / 2  # (LHip + RHip) / 2

        # H36M[7] = Spine mid = average of hip root and neck
        neck = (kps_coco[:, 5] + kps_coco[:, 6]) / 2  # avg of shoulders as proxy
        h36m[:, 7] = (h36m[:, 0] + neck) / 2

        # H36M[8] = Thorax/Neck = average of shoulders
        h36m[:, 8] = neck

        # H36M[10] = Head top = average of ears (or use nose as fallback)
        h36m[:, 10] = (kps_coco[:, 3] + kps_coco[:, 4]) / 2  # avg of ears

        # Direct mappings (COCO index -> H36M index)
        h36m[:, 1]  = kps_coco[:, 12]  # RHip
        h36m[:, 2]  = kps_coco[:, 14]  # RKnee
        h36m[:, 3]  = kps_coco[:, 16]  # RAnkle
        h36m[:, 4]  = kps_coco[:, 11]  # LHip
        h36m[:, 5]  = kps_coco[:, 13]  # LKnee
        h36m[:, 6]  = kps_coco[:, 15]  # LAnkle
        h36m[:, 9]  = kps_coco[:, 0]   # Nose
        h36m[:, 11] = kps_coco[:, 5]   # LShoulder
        h36m[:, 12] = kps_coco[:, 7]   # LElbow
        h36m[:, 13] = kps_coco[:, 9]   # LWrist
        h36m[:, 14] = kps_coco[:, 6]   # RShoulder
        h36m[:, 15] = kps_coco[:, 8]   # RElbow
        h36m[:, 16] = kps_coco[:, 10]  # RWrist

        return h36m

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_3d = self.__coco_to_h36m(motion_file["data_label"], 3)  
        if self.data_split=="train":
            if self.synthetic or self.gt_2d:
                motion_3d = self.aug.augment3D(motion_3d)
                motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1                        # No 2D detection, use GT xy and c=1.
            elif motion_file["data_input"] is not None:     # Have 2D detection 
                motion_2d = self.__coco_to_h36m(motion_file["data_input"], 3)
                motion_2d = crop_scale(motion_2d) 
                if self.flip and random.random() > 0.5:                        # Training augmentation - random flipping
                    motion_2d = flip_data(motion_2d)
                    motion_3d = flip_data(motion_3d)
            else:
                raise ValueError('Training illegal.') 
        elif self.data_split=="test":                                           
            motion_2d = self.__coco_to_h36m(motion_file["data_input"], 3)
            motion_2d = crop_scale(motion_2d) 
            if self.gt_2d:
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1
        else:
            raise ValueError('Data split unknown.')   
        return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)