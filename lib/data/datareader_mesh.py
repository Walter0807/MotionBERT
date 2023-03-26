import numpy as np
import os, sys
import copy
from lib.utils.tools import read_pkl
from lib.utils.utils_data import split_clips

class DataReaderMesh(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, dt_root = 'data/mesh', dt_file = 'pw3d_det.pkl', res=[1920, 1920]):
        self.split_id_train = None
        self.split_id_test = None
        self.dt_dataset = read_pkl('%s/%s' % (dt_root, dt_file))
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence
        self.res = res
        
    def read_2d(self):
        if self.res is not None:
            res_w, res_h = self.res
            offset = [1, res_h / res_w]
        else:
            res = np.array(self.dt_dataset['train']['img_hw'])[::self.sample_stride].astype(np.float32)
            res_w, res_h = res.max(1)[:, None, None], res.max(1)[:, None, None]
            offset = 1
        trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)    # [N, 17, 2] 
        # res_w, res_h = self.res
        trainset = trainset / res_w * 2 - offset
        testset = testset / res_w * 2 - offset
        if self.read_confidence:
            train_confidence = self.dt_dataset['train']['confidence'][::self.sample_stride].astype(np.float32)  
            test_confidence = self.dt_dataset['test']['confidence'][::self.sample_stride].astype(np.float32)  
            if len(train_confidence.shape)==2: 
                train_confidence = train_confidence[:,:,None]
                test_confidence = test_confidence[:,:,None]
            trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]
        return trainset, testset
    
    def get_split_id(self):
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]                          
        vid_list_test = self.dt_dataset['test']['source'][::self.sample_stride]                          
        self.split_id_train = split_clips(vid_list_train, self.n_frames, self.data_stride_train)  
        self.split_id_test = split_clips(vid_list_test, self.n_frames, self.data_stride_test)  
        return self.split_id_train, self.split_id_test
    
    def get_sliced_data(self):
        train_data, test_data = self.read_2d()     
        train_labels, test_labels = self.read_3d() 
        split_id_train, split_id_test = self.get_split_id()
        train_data, test_data = train_data[split_id_train], test_data[split_id_test]                     # (N, 27, 17, 3)
        train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]             # (N, 27, 17, 3)
        return train_data, test_data, train_labels, test_labels

    