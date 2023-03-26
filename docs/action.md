# Skeleton-based Action Recognition

## Data

The NTURGB+D 2D detection results are provided by [pyskl](https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md) using HRNet.

1. Download [`ntu60_hrnet.pkl`](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl) and  [`ntu120_hrnet.pkl`](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_hrnet.pkl) to  `data/action/`. 
2. Download the 1-shot split [here](https://1drv.ms/f/s!AvAdh0LSjEOlfi-hqlHxdVMZxWM) and put it to  `data/action/`. 

## Running

### NTURGB+D

**Train from scratch:**

```shell
# Cross-subject
python train_action.py \
--config configs/action/MB_train_NTU60_xsub.yaml \
--checkpoint checkpoint/action/MB_train_NTU60_xsub

# Cross-view
python train_action.py \
--config configs/action/MB_train_NTU60_xview.yaml \
--checkpoint checkpoint/action/MB_train_NTU60_xview
```

**Finetune from pretrained MotionBERT:**

```shell
# Cross-subject
python train_action.py \
--config configs/action/MB_ft_NTU60_xsub.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/action/FT_MB_release_MB_ft_NTU60_xsub

# Cross-view
python train_action.py \
--config configs/action/MB_ft_NTU60_xview.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/action/FT_MB_release_MB_ft_NTU60_xview
```

**Evaluate:**

```bash
# Cross-subject
python train_action.py \
--config configs/action/MB_train_NTU60_xsub.yaml \
--evaluate checkpoint/action/MB_train_NTU60_xsub/best_epoch.bin 

# Cross-view
python train_action.py \
--config configs/action/MB_train_NTU60_xview.yaml \
--evaluate checkpoint/action/MB_train_NTU60_xview/best_epoch.bin 
```

### NTURGB+D-120 (1-shot)

**Train from scratch:**

```bash
python train_action_1shot.py \
--config configs/action/MB_train_NTU120_oneshot.yaml \
--checkpoint checkpoint/action/MB_train_NTU120_oneshot
```

**Finetune from a pretrained model:**

```bash
python train_action_1shot.py \
--config configs/action/MB_ft_NTU120_oneshot.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/action/FT_MB_release_MB_ft_NTU120_oneshot
```

**Evaluate:**

```bash
python train_action_1shot.py \
--config configs/action/MB_train_NTU120_oneshot.yaml \
--evaluate checkpoint/action/MB_train_NTU120_oneshot/best_epoch.bin 
```



