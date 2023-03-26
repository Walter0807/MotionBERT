from __future__ import print_function
import os
import sys
import random
import copy
import argparse
import math
import pickle
import json
import glob
import numpy as np
sys.path.insert(0, os.getcwd())
from lib.utils.utils_data import crop_scale


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_action', type=str)
    args = parser.parse_args()
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))
    return args
    
def json2pose(json_dict):
    pose_h36m = np.zeros([17,3])
    idx2key = ['Hip',
               'R Hip',
               'R Knee',
               'R Ankle',
               'L Hip',
               'L Knee',
               'L Ankle',
               'Belly', 
               'Neck',
               'Nose',
               'Head', 
               'L Shoulder',
               'L Elbow',
               'L Wrist',
               'R Shoulder',
               'R Elbow',
               'R Wrist',
              ]
    for i in range(17):
        if idx2key[i]=='Belly' or idx2key[i]=='Head':
            pose_h36m[i] = 0, 0, 0
        else:
            item = json_dict[idx2key[i]]
            pose_h36m[i] = item['x'], item['y'], item['logits']
    return pose_h36m

def load_motion(json_path):
    json_dict = json.load(open(json_path, 'r'))
    pose_h36m = json2pose(json_dict)
    return pose_h36m

    
args = parse_args()
dataset_root = 'data/Motion2d/InstaVariety/InstaVariety_tracks/'
action_motions = []
dir_action = os.path.join(dataset_root, args.name_action)
for name_vid in sorted(os.listdir(dir_action)):
    dir_vid = os.path.join(dir_action, name_vid)
    for name_clip in sorted(os.listdir(dir_vid)):
        motion_path = os.path.join(dir_vid, name_clip)
        motion_list = sorted(glob.glob(motion_path+'/*.json'))
        if len(motion_list)==0:
            continue
        motion = [load_motion(i) for i in motion_list]
        motion = np.array(motion)
        motion = crop_scale(motion)
        motion[:,:,:2] = motion[:,:,:2] - motion[0:1,0:1,:2]
        motion[motion[:,:,2]==0] = 0
        action_motions.append(motion)
    print("%s Done, %d vids processed" % (name_vid, len(action_motions)))
print("%s Done, %d vids processed" % (args.name_action, len(action_motions)))
with open(os.path.join(dir_action, '%s.pkl' % args.name_action), 'wb') as f:
    pickle.dump(action_motions, f)
