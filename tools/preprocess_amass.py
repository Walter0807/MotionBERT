import torch
import numpy as np
import os
from os import path as osp
from human_body_prior.body_model.body_model import BodyModel
import copy
import pickle
import ipdb
import pandas as pd

df = pd.read_csv('./data/AMASS/fps.csv', sep=',',header=None)
fname_list = list(df[0][1:])

processed_dir = './data/AMASS/amass_fps60/'
J_reg_dir = 'data/AMASS/J_regressor_h36m_correct.npy'
all_motions = 'data/AMASS/all_motions_fps60.pkl'

file = open(all_motions, 'rb')
motion_data = pickle.load(file)
J_reg = np.load(J_reg_dir)
all_joints = []

max_len = 2916
with open('data/AMASS/clip_list.csv', 'w') as f:
    print('clip_id, fname, clip_len', file=f)
    for i, bdata in enumerate(motion_data):
        if i%200==0:
            print(i, 'seqs done.')
        comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        subject_gender = bdata['gender']
        if (str(subject_gender) != 'female') and (str(subject_gender) != 'male'):
            subject_gender = 'female'

        bm_fname = osp.join('data/AMASS/body_models/smplh/{}/model.npz'.format(subject_gender))
        dmpl_fname = osp.join('data/AMASS/body_models/dmpls/{}/model.npz'.format(subject_gender))

        # number of body parameters
        num_betas = 16
        # number of DMPL parameters
        num_dmpls = 8

        bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(comp_device)
        time_length = len(bdata['trans'])
        num_slice = time_length // max_len

        for sid in range(num_slice+1):
            start = sid*max_len
            end = min((sid+1)*max_len, time_length)
            body_parms = {
                'root_orient': torch.Tensor(bdata['poses'][start:end, :3]).to(comp_device), # controls the global root orientation
                'pose_body': torch.Tensor(bdata['poses'][start:end, 3:66]).to(comp_device), # controls the body
                'pose_hand': torch.Tensor(bdata['poses'][start:end, 66:]).to(comp_device), # controls the finger articulation
                'trans': torch.Tensor(bdata['trans'][start:end]).to(comp_device), # controls the global body position
                'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=(end-start), axis=0)).to(comp_device), # controls the body shape. Body shape is static
                'dmpls': torch.Tensor(bdata['dmpls'][start:end, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
            }
            body_trans_root = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand', 'dmpls', 'trans', 'root_orient']})
            mesh = body_trans_root.v.cpu().numpy()
            kpts = np.dot(J_reg, mesh)    # (17,T,3)
            all_joints.append(kpts)
            print(len(all_joints)-1, ',', fname_list[i], ',', end-start, file=f)
    fileName = open('data/AMASS/amass_joints_h36m_60.pkl','wb')
    pickle.dump(all_joints, fileName)
    print(len(all_joints))