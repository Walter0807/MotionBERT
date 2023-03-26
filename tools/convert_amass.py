import os
import sys
import copy
import pickle
import ipdb
import torch
import numpy as np
sys.path.insert(0, os.getcwd())
from lib.utils.utils_data import split_clips
from tqdm import tqdm

fileName = open('data/AMASS/amass_joints_h36m_60.pkl','rb')
joints_all = pickle.load(fileName)

joints_cam = []
vid_list = []
vid_len_list = []
scale_factor = 0.298

for i, item in enumerate(joints_all): # (17,N,3):
    item = item.astype(np.float32)
    vid_len = item.shape[1]
    vid_len_list.append(vid_len)
    for _ in range(vid_len):
        vid_list.append(i)
    real2cam = np.array([[1,0,0], 
                        [0,0,1], 
                        [0,-1,0]], dtype=np.float32)
    item = np.transpose(item, (1,0,2)) # (17,N,3) -> (N,17,3)
    motion_cam = item @ real2cam
    motion_cam *= scale_factor
    # motion_cam = motion_cam - motion_cam[0,0,:]
    joints_cam.append(motion_cam)

joints_cam_all = np.vstack(joints_cam)
split_id = datareader.split_clips(vid_list, n_frames=243, data_stride=81)
print(joints_cam_all.shape)   # (N,17,3)

max_x, minx_x = np.max(joints_cam_all[:,:,0]), np.min(joints_cam_all[:,:,0])
max_y, minx_y = np.max(joints_cam_all[:,:,1]), np.min(joints_cam_all[:,:,1])
max_z, minx_z = np.max(joints_cam_all[:,:,2]), np.min(joints_cam_all[:,:,2])
print(max_x, minx_x)
print(max_y, minx_y)
print(max_z, minx_z)

joints_cam_clip = joints_cam_all[split_id]
print(joints_cam_clip.shape)   # (N,27,17,3)

# np.save('doodle/joints_cam_clip_amass_60.npy', joints_cam_clip)

root_path = "data/motion3d/MB3D_f243s81/AMASS"
subset_name = "train"
save_path = os.path.join(root_path, subset_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

num_clips = len(joints_cam_clip)
for i in tqdm(range(num_clips)):
    motion = joints_cam_clip[i]
    data_dict = {
            "data_input": None,
            "data_label": motion
        }
    with open(os.path.join(save_path, "%08d.pkl" % i), "wb") as myprofile:  
        pickle.dump(data_dict, myprofile)


