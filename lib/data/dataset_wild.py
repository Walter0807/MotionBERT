import torch
import numpy as np
import ipdb
import glob
import os
import io
import math
import random
import json
import pickle
import math
from torch.utils.data import Dataset, DataLoader
from lib.data.operations import crop_scale

def halpe2h36m(x):
    '''
        Input: x (T x V x C)  
       //Halpe 26 body keypoints
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"},
    '''
    T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:] = x[:,19,:]
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
    y[:,8,:] = x[:,18,:]
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,17,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    return y
    
def read_input(json_path='alphapose-results.json'):
    with open(json_path, "r") as read_file:
        results = json.load(read_file)
    kpts_all = []
    for item in results:
        kpts = np.array(item['keypoints']).reshape([-1,3])
        kpts_all.append(kpts)
    kpts_all = np.array(kpts_all)
    kpts_all = halpe2h36m(kpts_all)
    motion = crop_scale(kpts_all, [1.3, 1.3]) 
    return motion.astype(np.float32)

class WildDetDataset(Dataset):
    def __init__(self, json_path, clip_len=243):
        self.json_path = json_path
        self.clip_len = clip_len
        self.vid_all = read_input(json_path)
        
    def __len__(self):
        'Denotes the total number of samples'
        return math.ceil(len(self.vid_all) / self.clip_len)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        st = index*self.clip_len
        end = min((index+1)*self.clip_len, len(self.vid_all))
        return self.vid_all[st:end]