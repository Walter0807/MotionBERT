# MotionBERT: Unified Pretraining for Human Motion Analysis

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motionbert-unified-pretraining-for-human/monocular-3d-human-pose-estimation-on-human3)](https://paperswithcode.com/sota/monocular-3d-human-pose-estimation-on-human3?p=motionbert-unified-pretraining-for-human)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motionbert-unified-pretraining-for-human/one-shot-3d-action-recognition-on-ntu-rgbd)](https://paperswithcode.com/sota/one-shot-3d-action-recognition-on-ntu-rgbd?p=motionbert-unified-pretraining-for-human)

### [Project Page](https://motionbert.github.io/) | [Paper](https://arxiv.org/pdf/2210.06551.pdf)

This is the official PyTorch implementation of the paper "MotionBERT: Unified Pretraining for Human Motion Analysis".

<img src="https://motionbert.github.io/assets/teaser.gif" alt="" style="zoom: 60%;" />

## Installation

```bash
conda create -n motionbert python=3.7 anaconda
conda activate motionbert
# Please install PyTorch according to your CUDA version.
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Usage

| Task                              | Document                                                     |
| --------------------------------- | ------------------------------------------------------------ |
| Pretrain                          | TBD                                                          |
| 3D human pose estimation          | [docs/pose3d.md](docs/pose3d.md) |
| Skeleton-based action recognition | [docs/action.md](docs/action.md) |
| Mesh recovery                     | [docs/mesh.md](docs/mesh.md) |

## Using MotionBERT for new tasks

```python
'''	    
  x: 2D skeletons 
    type = <class 'torch.Tensor'>
    shape = [batch size * frames * joints(17) * channels(3)]
    
  MotionBERT: pretrained MotionBERT
    type = <class 'lib.model.DSTformer.DSTformer'>
    
  E: encoded motion representation
    type = <class 'torch.Tensor'>
    shape = [batch size * frames * joints(17) * channels(512)]
'''
E = MotionBERT.get_representation(x)
```

**Hints**

1. The model could handle different input lengths (no more than 243 frames). No need to explicitly specify the input length elsewhere.
2. The model uses 17 body keypoints ([H36M format](https://github.com/JimmySuen/integral-human-pose/blob/master/pytorch_projects/common_pytorch/dataset/hm36.py#L32)). If you are using other formats, please convert them before feeding to MotionBERT.
3. Please refer to [model_action.py](lib/model/model_action.py) and [model_mesh.py](lib/model/model_mesh.py) for examples of (easily) adapting MotionBERT to different downstream tasks.

## Model Zoo

| Model                                                  | Download Link                                                | Performance      |
| ------------------------------------------------------ | ------------------------------------------------------------ | ---------------- |
| MotionBERT (pretrained motion encoder weights)         | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgQhYxzswPARRLtZ5) | -                |
| 3D Pose (H36M-SH, scratch)                             | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgQ4voapR8XVTGcVj) | 39.1mm (MPJPE)   |
| 3D Pose (H36M-SH, finetuned)                           | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgQ8voapR8XVTGcVj) | 37.4mm (MPJPE)   |
| Action Recognition (NTU-RGB+D x-sub, finetuned)        | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgQovoapR8XVTGcVj) | 97.3% (Top1 Acc) |
| Action Recognition (NTU-RGB+D x-view, finetuned)       | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgQsvoapR8XVTGcVj) | 92.8% (Top1 Acc) |
| Action Recognition (NTU-RGB+D-120 one-shot, finetuned) | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgQwvoapR8XVTGcVj) | 67.4% (Top1 Acc) |
| Mesh Recovery (with 3DPW, finetuned)                   | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgRAvoapR8XVTGcVj) | 94.2mm (MPVE)    |

## TODO

- [ ] Scripts and docs for pretraining

- [ ] Demo for custom videos


## BibTeX

If you find our work useful for your research, please consider citing the paper:

```bibtex
@article{motionbert2022,
  title   =   {MotionBERT: Unified Pretraining for Human Motion Analysis}, 
  author  =   {Zhu, Wentao and Ma, Xiaoxuan and Liu, Zhaoyang and Liu, Libin and Wu, Wayne and Wang, Yizhou},
  year    =   {2022},
  journal =   {arXiv preprint arXiv:2210.06551},
}
```
