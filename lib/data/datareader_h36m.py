# Adapted from Optimizing Network Structure for 3D Human Pose Estimation (ICCV 2019) (https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/data.py)

import numpy as np
import os, sys
import random
import copy
from lib.utils.tools import read_pkl
from lib.utils.utils_data import split_clips
random.seed(0)
    
class DataReaderH36M(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, dt_root = 'data/motion3d', dt_file = 'h36m_cpn_cam_source.pkl'):
        self.gt_trainset = None
        self.gt_testset = None
        self.split_id_train = None
        self.split_id_test = None
        self.test_hw = None
        self.dt_dataset = read_pkl('%s/%s' % (dt_root, dt_file))
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence
        
    def read_2d(self):
        trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.dt_dataset['train']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            trainset[idx, :, :] = trainset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            testset[idx, :, :] = testset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        if self.read_confidence:
            if 'confidence' in self.dt_dataset['train'].keys():
                train_confidence = self.dt_dataset['train']['confidence'][::self.sample_stride].astype(np.float32)  
                test_confidence = self.dt_dataset['test']['confidence'][::self.sample_stride].astype(np.float32)  
                if len(train_confidence.shape)==2: # (1559752, 17)
                    train_confidence = train_confidence[:,:,None]
                    test_confidence = test_confidence[:,:,None]
            else:
                # No conf provided, fill with 1.
                train_confidence = np.ones(trainset.shape)[:,:,0:1]
                test_confidence = np.ones(testset.shape)[:,:,0:1]
            trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]
        return trainset, testset

    def read_3d(self):
        train_labels = self.dt_dataset['train']['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32)  # [N, 17, 3]
        test_labels = self.dt_dataset['test']['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32)    # [N, 17, 3]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.dt_dataset['train']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            train_labels[idx, :, :2] = train_labels[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
            train_labels[idx, :, 2:] = train_labels[idx, :, 2:] / res_w * 2
            
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            test_labels[idx, :, :2] = test_labels[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
            test_labels[idx, :, 2:] = test_labels[idx, :, 2:] / res_w * 2
            
        return train_labels, test_labels
    def read_hw(self):
        if self.test_hw is not None:
            return self.test_hw
        test_hw = np.zeros((len(self.dt_dataset['test']['camera_name']), 2))
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            test_hw[idx] = res_w, res_h
        self.test_hw = test_hw
        return test_hw
    
    def get_split_id(self):
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]                          # (1559752,)
        vid_list_test = self.dt_dataset['test']['source'][::self.sample_stride]                           # (566920,)
        self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train) 
        self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)
        return self.split_id_train, self.split_id_test
    
    def get_hw(self):
#       Only Testset HW is needed for denormalization
        test_hw = self.read_hw()                                     # train_data (1559752, 2) test_data (566920, 2)
        split_id_train, split_id_test = self.get_split_id()
        test_hw = test_hw[split_id_test][:,0,:]                      # (N, 2)
        return test_hw
    
    def get_sliced_data(self):
        train_data, test_data = self.read_2d()     # train_data (1559752, 17, 3) test_data (566920, 17, 3)
        train_labels, test_labels = self.read_3d() # train_labels (1559752, 17, 3) test_labels (566920, 17, 3)
        split_id_train, split_id_test = self.get_split_id()
        train_data, test_data = train_data[split_id_train], test_data[split_id_test]                # (N, 27, 17, 3)
        train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]        # (N, 27, 17, 3)
        # ipdb.set_trace()
        return train_data, test_data, train_labels, test_labels
    
    def denormalize(self, test_data):
#       data: (N, n_frames, 51) or data: (N, n_frames, 17, 3)        
        n_clips = test_data.shape[0]
        test_hw = self.get_hw()
        data = test_data.reshape([n_clips, -1, 17, 3])
        assert len(data) == len(test_hw)
        # denormalize (x,y,z) coordiantes for results
        for idx, item in enumerate(data):
            res_w, res_h = test_hw[idx]
            data[idx, :, :, :2] = (data[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
            data[idx, :, :, 2:] = data[idx, :, :, 2:] * res_w / 2
        return data # [n_clips, -1, 17, 3]
