import os
import os.path as osp
import numpy as np
import argparse
import pickle
from tqdm import tqdm
import time
import random
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.utils.utils_mesh import flip_thetas_batch
from lib.data.dataset_wild import WildDetDataset
# from lib.model.loss import *
from lib.model.model_action import ActionNet
from lib.utils.vismo import render_and_save, motion2video_mesh
from lib.utils.utils_smpl import *
from scipy.optimize import least_squares

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/action/MB_ft_NTU60_xsub.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/action/MB_ft_NTU60_xsub/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    opts = parser.parse_args()
    return opts

def err(p, x, y):
    return np.linalg.norm(p[0] * x + np.array([p[1], p[2], p[3]]) - y, axis=-1).mean()

def solve_scale(x, y):
    print('Estimating camera transformation.')
    best_res = 100000
    best_scale = None
    for init_scale in tqdm(range(0,2000,5)):
        p0 = [init_scale, 0.0, 0.0, 0.0]
        est = least_squares(err, p0, args = (x.reshape(-1,3), y.reshape(-1,3)))
        if est['fun'] < best_res:
            best_res = est['fun']
            best_scale = est['x'][0]
    print('Pose matching error = %.2f mm.' % best_res)
    return best_scale

opts = parse_args()
args = get_config(opts.config)

# root_rel
# args.rootrel = True

model_backbone = load_backbone(args)
model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, hidden_dim=args.hidden_dim)

if torch.cuda.is_available():
    print("--use cuda--")
    model = nn.DataParallel(model)
    model = model.cuda()

chk_filename = opts.evaluate if opts.evaluate else opts.resume
print('Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
original_state_dict = checkpoint['model']
new_state_dict = {k[len("module."):]: v for k, v in original_state_dict.items()}
model.load_state_dict(new_state_dict, strict=True)
model.eval()

testloader_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True,
        'drop_last': False
}

vid = imageio.get_reader(opts.vid_path,  'ffmpeg')
fps_in = vid.get_meta_data()['fps']
vid_size = vid.get_meta_data()['size']
os.makedirs(opts.out_path, exist_ok=True)

if opts.pixel:
    # Keep relative scale with pixel coornidates
    wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
else:
    # Scale to [-1,1]
    wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)

test_loader = DataLoader(wild_dataset, **testloader_params)

with torch.no_grad():
    for batch_input in tqdm(test_loader):
        print(batch_input.shape)
        N, T = batch_input.shape[:2]
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
        output = model(batch_input)   

print(softmax(output))