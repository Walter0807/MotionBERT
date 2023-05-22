# In-the-wild Inference

## 2D Pose

Please use [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose#quick-start) to extract the 2D keypoints for your video first. We use the *Fast Pose* model trained on *Halpe* dataset ([Link](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md#halpe-dataset-26-keypoints)). [Here](https://github.com/Walter0807/MotionBERT/files/11526727/sample.zip) we provide an example json file.

Note: Currently we only support single person. If your video contains multiple person, you may need to use the [Pose Tracking Module for AlphaPose](https://github.com/MVIG-SJTU/AlphaPose/tree/master/trackers) and set `--focus` to specify the target person id.



## 3D Pose

| ![pose_1](https://github.com/motionbert/motionbert.github.io/blob/main/assets/pose_1.gif?raw=true) | ![pose_2](https://raw.githubusercontent.com/motionbert/motionbert.github.io/main/assets/pose_2.gif) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |


1. Please download the checkpoint [here](https://1drv.ms/f/s!AvAdh0LSjEOlgT67igq_cIoYvO2y?e=bfEc73) and put it to `checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/`.
1. Run the following command to infer from the extracted 2D poses:
```bash
python infer_wild.py \
--vid_path <your_video.mp4> \
--json_path <alphapose-results.json> \
--out_path <output_path>
```



## Mesh

| ![mesh_1](https://raw.githubusercontent.com/motionbert/motionbert.github.io/main/assets/mesh_1.gif) | ![mesh_2](https://github.com/motionbert/motionbert.github.io/blob/main/assets/mesh_2.gif?raw=true) |
| ------------------------------------------------------------ | ----------- |

1. Please download the checkpoint [here](https://1drv.ms/f/s!AvAdh0LSjEOlgTmgYNslCDWMNQi9?e=WjcB1F) and put it to `checkpoint/mesh/FT_MB_release_MB_ft_pw3d/`
2. Run the following command to infer from the extracted 2D poses:
```bash
python infer_wild_mesh.py \
--vid_path <your_video.mp4> \
--json_path <alphapose-results.json> \
--out_path <output_path> \
--ref_3d_motion_path <3d-pose-results.npy> # Optional, use the estimated 3D motion for root trajectory.
```







