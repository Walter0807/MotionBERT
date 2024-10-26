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
from lib.data.dataset_wild_walking import AlphaPoseWildDataset
# from lib.model.loss import *
from lib.model.model_walking import WalkingNet
from lib.utils.vismo import render_and_save, motion2video_mesh
from lib.utils.utils_smpl import *
from scipy.optimize import least_squares

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/walking/MB_ft_walking.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='checkpoint/walking/FT_MB_release_MB_ft_walking/all/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-j', '--json_path', type=str, help='alphapose detection result json path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    opts = parser.parse_args()
    return opts

opts = parse_args()
args = get_config(opts.config)

model_backbone = load_backbone(args)
model = WalkingNet(backbone=model_backbone, dim_rep=args.dim_rep, hidden_dim=args.hidden_dim)

if torch.cuda.is_available():
    print("--use cuda--")
    model = nn.DataParallel(model)
    model = model.cuda()


chk_filename = opts.evaluate
print('Loading checkpoint', chk_filename)
checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
original_state_dict = checkpoint['model']
# new_state_dict = {k[len("module."):]: v for k, v in original_state_dict.items()}
model.load_state_dict(original_state_dict, strict=True)
model.eval()

testloader_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 2,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True,
        'drop_last': False
}

# os.makedirs(opts.out_path, exist_ok=True)

wild_dataset = AlphaPoseWildDataset(opts.json_path, n_frames=opts.clip_len)
test_loader = DataLoader(wild_dataset, **testloader_params)

with torch.no_grad():
    for batch_input in tqdm(test_loader):
        N, T = batch_input.shape[:2]
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
        output = model(batch_input)
        prob = nn.Softmax(dim=1)(output)
        print(f"Probability of Normal Person: {prob[0][0]:.3f}")
        print(f"Probability of Model Person: {prob[0][1]:.3f}")
        exit(0)
