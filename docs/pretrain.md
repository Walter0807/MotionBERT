# Pretrain

## Data

### AMASS

1. Please download data from the [official website](https://amass.is.tue.mpg.de/download.php) (SMPL+H).
2. We provide the preprocessing scripts as follows. Minor modifications might be necessary.
   - [tools/compress_amass.py](../tools/compress_amass.py): downsample the frame rate
   - [tools/preprocess_amass.py](../tools/preprocess_amass.py): render the mocap data and extract the 3D keypoints
   - [tools/convert_amass.py](../tools/convert_amass.py): slice them to motion clips


### Human 3.6M

Please refer to [pose3d.md](pose3d.md#data).

### InstaVariety

1. Please download data from [human_dynamics](https://github.com/akanazawa/human_dynamics/blob/master/doc/insta_variety.md#generating-tfrecords) to `data/motion2d`.
1. Use [tools/convert_insta.py](../tools/convert_insta.py) to preprocess the 2D keypoints (need to specify `name_action` ).

### PoseTrack

Please download PoseTrack18 from [MMPose](https://mmpose.readthedocs.io/en/latest/tasks/2d_body_keypoint.html#posetrack18) and unzip to `data/motion2d`.



The processed directory tree should look like this:

```
.
└── data/
    ├── motion3d/
    │   └── MB3D_f243s81/
    │       ├── AMASS
    │       └── H36M-SH
    ├── motion2d/
    │   ├── InstaVariety/
    │   │   ├── motion_all.npy
    │   │   └── id_all.npy
    │   └── posetrack18_annotations/
    │       ├── train
    │       └── ...
    └── ...
```



## Train

```bash
python train.py \
--config configs/pretrain/MB_pretrain.yaml \
-c checkpoint/pretrain/MB_pretrain
```



