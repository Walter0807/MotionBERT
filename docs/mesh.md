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

### Refiner

1. Prepare the predictions of an image-based estimator. We provide an example: [mesh_hybrik.zip](https://1drv.ms/u/s!AvAdh0LSjEOlgUlqm-WkmP_PtcTP?e=GZcIHU) (please unzip the `.pkl` files to `data/mesh/`).
2. Use the pretrained motion representations to regress the residual in joint rotations:

  ```bash
  python train_mesh_refine.py \
  --config configs/mesh/MB_ft_pw3d_refine_hybrik.yaml \
  --pretrained checkpoint/pretrain/MB_release \
  --checkpoint checkpoint/mesh/FT_MB_release_MB_ft_pw3d_refine_hybrik
  ```

3. Evaluate with (download our checkpoint for HybrIK [here](https://1drv.ms/f/s!AvAdh0LSjEOlgUwk4DI_JvPYgt75?e=upm9fG)):

  ```bash
  python train_mesh_refine.py \
  --config configs/mesh/MB_ft_pw3d_refine_hybrik.yaml \
  --evaluate checkpoint/mesh/FT_MB_release_MB_ft_pw3d_refine_hybrik/best_epoch.bin
  ```







