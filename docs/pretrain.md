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



### PoseTrack

Please download PoseTrack18 from [MMPose](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html#posetrack18) (annotation files) and unzip to `data/motion2d`.



### InstaVariety

1. Please download data from [human_dynamics](https://github.com/akanazawa/human_dynamics/blob/master/doc/insta_variety.md#generating-tfrecords) to `data/motion2d`.
1. Use [tools/convert_insta.py](../tools/convert_insta.py) to preprocess the 2D keypoints (need to specify `name_action` ).
1. Load all the processed  `.pkl` files from step 2, concatenate them to `motion_list`, then run

```python
import numpy as np
ids = []
for i, x in enumerate(motion_list):
    ids.append(np.ones(len(x))*i)
motion_all = np.concatenate(motion_list)
id_all = np.concatenate(ids)
np.save('data/motion2d/InstaVariety/motion_all.npy', motion_all)
np.save('data/motion2d/InstaVariety/id_all.npy', id_all)

```

You can also download the preprocessed 2D keypoints from [here](https://1drv.ms/u/s!AvAdh0LSjEOlgVElzkVkWoFcJ1MR?e=TU2CeI) and unzip it to `data/motion2d/`.





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



