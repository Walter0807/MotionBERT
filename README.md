# MotionBERT

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> [![arXiv](https://img.shields.io/badge/arXiv-2210.06551-b31b1b.svg)](https://arxiv.org/abs/2210.06551) <a href="https://motionbert.github.io/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a> <a href="https://youtu.be/slSPQ9hNLjM"><img alt="Demo" src="https://img.shields.io/badge/-Demo-ea3323?logo=youtube"></a> [![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffab41)](https://huggingface.co/walterzhu/MotionBERT)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motionbert-unified-pretraining-for-human/monocular-3d-human-pose-estimation-on-human3)](https://paperswithcode.com/sota/monocular-3d-human-pose-estimation-on-human3?p=motionbert-unified-pretraining-for-human)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motionbert-unified-pretraining-for-human/one-shot-3d-action-recognition-on-ntu-rgbd)](https://paperswithcode.com/sota/one-shot-3d-action-recognition-on-ntu-rgbd?p=motionbert-unified-pretraining-for-human)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/motionbert-unified-pretraining-for-human/3d-human-pose-estimation-on-3dpw)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=motionbert-unified-pretraining-for-human)

This is the official PyTorch implementation of the paper *"[Learning Human Motion Representations: A Unified Perspective](https://arxiv.org/pdf/2210.06551.pdf)"*.

<img src="https://motionbert.github.io/assets/teaser.gif" alt="" style="zoom: 60%;" />

## Installation

```bash
conda create -n motionbert python=3.7 anaconda
conda activate motionbert
# Please install PyTorch according to your CUDA version.
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```



## Getting Started

| Task                              | Document                                                     |
| --------------------------------- | ------------------------------------------------------------ |
| Pretrain                          | [docs/pretrain.md](docs/pretrain.md)                                                          |
| 3D human pose estimation          | [docs/pose3d.md](docs/pose3d.md) |
| Skeleton-based action recognition | [docs/action.md](docs/action.md) |
| Mesh recovery                     | [docs/mesh.md](docs/mesh.md) |



## Applications

### In-the-wild inference (for custom videos)

Please refer to [docs/inference.md](docs/inference.md).

### Using MotionBERT for *human-centric* video representations

```python
'''	    
  x: 2D skeletons 
    type = <class 'torch.Tensor'>
    shape = [batch size * frames * joints(17) * channels(3)]
    
  MotionBERT: pretrained human motion encoder
    type = <class 'lib.model.DSTformer.DSTformer'>
    
  E: encoded motion representation
    type = <class 'torch.Tensor'>
    shape = [batch size * frames * joints(17) * channels(512)]
'''
E = MotionBERT.get_representation(x)
```



> **Hints**
>
> 1. The model could handle different input lengths (no more than 243 frames). No need to explicitly specify the input length elsewhere.
> 2. The model uses 17 body keypoints ([H36M format](https://github.com/JimmySuen/integral-human-pose/blob/master/pytorch_projects/common_pytorch/dataset/hm36.py#L32)). If you are using other formats, please convert them before feeding to MotionBERT. 
> 3. Please refer to [model_action.py](lib/model/model_action.py) and [model_mesh.py](lib/model/model_mesh.py) for examples of (easily) adapting MotionBERT to different downstream tasks.
> 4. For RGB videos, you need to extract 2D poses ([inference.md](docs/inference.md)), convert the keypoint format ([dataset_wild.py](lib/data/dataset_wild.py)), and then feed to MotionBERT ([infer_wild.py](infer_wild.py)).
>



## Model Zoo

<img src="https://motionbert.github.io/assets/demo.gif" alt="" style="zoom: 50%;" />

| Model                           | Download Link                                                | Config                                                       | Performance      |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------- |
| MotionBERT (162MB)              | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgS425shtVi9e5reN?e=6UeBa2) | [pretrain/MB_pretrain.yaml](configs/pretrain/MB_pretrain.yaml) | -                |
| MotionBERT-Lite (61MB)          | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgS27Ydcbpxlkl0ng?e=rq2Btn) | [pretrain/MB_lite.yaml](configs/pretrain/MB_lite.yaml)       | -                |
| 3D Pose (H36M-SH, scratch)      | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgSvNejMQ0OHxMGZC?e=KcwBk1) | [pose3d/MB_train_h36m.yaml](configs/pose3d/MB_train_h36m.yaml) | 39.2mm (MPJPE)   |
| 3D Pose (H36M-SH, ft)           | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgSoTqtyR5Zsgi8_Z?e=rn4VJf) | [pose3d/MB_ft_h36m.yaml](configs/pose3d/MB_ft_h36m.yaml)     | 37.2mm (MPJPE)   |
| Action Recognition (x-sub, ft)  | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgTX23yT_NO7RiZz-?e=nX6w2j) | [action/MB_ft_NTU60_xsub.yaml](configs/action/MB_ft_NTU60_xsub.yaml) | 97.2% (Top1 Acc) |
| Action Recognition (x-view, ft) | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgTaNiXw2Nal-g37M?e=lSkE4T) | [action/MB_ft_NTU60_xview.yaml](configs/action/MB_ft_NTU60_xview.yaml) | 93.0% (Top1 Acc) |
| Mesh (with 3DPW, ft)            | [OneDrive](https://1drv.ms/f/s!AvAdh0LSjEOlgTmgYNslCDWMNQi9?e=WjcB1F) | [mesh/MB_ft_pw3d.yaml](configs/mesh/MB_ft_pw3d.yaml)              | 88.1mm (MPVE)    |

In most use cases (especially with finetuning), `MotionBERT-Lite` gives a similar performance with lower computation overhead. 



## TODO

- [x] Scripts and docs for pretraining

- [x] Demo for custom videos



## Citation

If you find our work useful for your project, please consider citing the paper:

```bibtex
@article{motionbert2022,
  title   =   {Learning Human Motion Representations: A Unified Perspective}, 
  author  =   {Zhu, Wentao and Ma, Xiaoxuan and Liu, Zhaoyang and Liu, Libin and Wu, Wayne and Wang, Yizhou},
  year    =   {2022},
  journal =   {arXiv preprint arXiv:2210.06551},
}
```

