# 3D Human Pose Estimation

## Data

1. Download the finetuned Stacked Hourglass detections and our preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to `data/motion3d`.

  > Note that the preprocessed data is only intended for reproducing our results more easily. If you want to use the dataset, please register to the [Human3.6m website](http://vision.imar.ro/human3.6m/) and download the dataset in its original format. Please refer to [LCN](https://github.com/CHUNYUWANG/lcn-pose#data) for how we prepare the H3.6M data.

2. Slice the motion clips (len=243, stride=81)

   ```bash
   python tools/convert_h36m.py
   ```

## Running

**Train from scratch:**

```bash
python train.py \
--config configs/pose3d/MB_train_h36m.yaml \
--checkpoint checkpoint/pose3d/MB_train_h36m
```

**Finetune from pretrained MotionBERT:**

```bash
python train.py \
--config configs/pose3d/MB_ft_h36m.yaml \
--pretrained checkpoint/pretrain/MB_release \
--checkpoint checkpoint/pose3d/FT_MB_release_MB_ft_h36m
```

**Evaluate:**

```bash
python train.py \
--config configs/pose3d/MB_train_h36m.yaml \
--evaluate checkpoint/pose3d/MB_train_h36m/best_epoch.bin         
```











