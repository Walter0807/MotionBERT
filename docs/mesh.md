# Human Mesh Recovery

## Data

1. Download the datasets [here](https://1drv.ms/f/s!AvAdh0LSjEOlfy-hqlHxdVMZxWM) and put them to  `data/mesh/`. We use Human3.6M, COCO, and PW3D for training and testing. Descriptions of the joint regressors could be found in [SPIN](https://github.com/nkolot/SPIN/tree/master/data).
2. Download the SMPL model(`basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`) from [SMPLify](https://smplify.is.tue.mpg.de/), put it  to `data/mesh/`, and rename it as `SMPL_NEUTRAL.pkl`


## Running

**Train from scratch:**

```bash
# with 3DPW
python train_mesh.py \
--config configs/mesh/MB_train_pw3d.yaml \
--checkpoint checkpoint/mesh/MB_train_pw3d

# H36M
python train_mesh.py \
--config configs/mesh/MB_train_h36m.yaml \
--checkpoint checkpoint/mesh/MB_train_h36m
```

**Finetune from a pretrained model:**

```bash
# with 3DPW
python train_mesh.py \
--config configs/mesh/MB_ft_pw3d.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/mesh/FT_MB_release_MB_ft_pw3d

# H36M
python train_mesh.py \
--config configs/mesh/MB_ft_h36m.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/mesh/FT_MB_release_MB_ft_h36m

```

**Evaluate:**

```bash
# with 3DPW
python train_mesh.py \
--config configs/mesh/MB_train_pw3d.yaml \
--evaluate checkpoint/mesh/MB_train_pw3d/best_epoch.bin 

# H36M
python train_mesh.py \
--config configs/mesh/MB_train_h36m.yaml \
--evaluate checkpoint/mesh/MB_train_h36m/best_epoch.bin 
```







