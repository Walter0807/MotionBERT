import logging
import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

## For MotionBERT
import torch
import torch.nn as nn
from lib.model.DSTformer import DSTformer
from lib.utils.tools import get_config
from lib.utils.learning import load_backbone
import open3d as o3d

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def halpe2h36m(joints, scores, vid_size):

    if joints.shape[0] != 1:
        joints = joints[:1, :, :]
        scores = scores[:1, :]

    x = np.concatenate([joints, scores[..., None]], axis=2)
    kpts = np.zeros([1, 17, 3])
    kpts[:, 0, :] = x[:, 19, :]
    kpts[:, 1, :] = x[:, 12, :]
    kpts[:, 2, :] = x[:, 14, :]
    kpts[:, 3, :] = x[:, 16, :]
    kpts[:, 4, :] = x[:, 11, :]
    kpts[:, 5, :] = x[:, 13, :]
    kpts[:, 6, :] = x[:, 15, :]
    kpts[:, 7, :] = (x[:, 18, :] + x[:, 19, :]) * 0.5
    kpts[:, 8, :] = x[:, 18, :]
    kpts[:, 9, :] = x[:, 0, :]
    kpts[:, 10, :] = x[:, 17, :]
    kpts[:, 11, :] = x[:, 5, :]
    kpts[:, 12, :] = x[:, 7, :]
    kpts[:, 13, :] = x[:, 9, :]
    kpts[:, 14, :] = x[:, 6, :]
    kpts[:, 15, :] = x[:, 8, :]
    kpts[:, 16, :] = x[:, 10, :]

    h, w = vid_size
    scale = min(w, h) / 2.0
    kpts[:, :, :2] = kpts[:, :, :2] - np.array([w, h]) / 2.0
    kpts[:, :, :2] = kpts[:, :, :2] / scale
    motion = kpts

    return torch.Tensor(motion[None, ...])


def process_one_image(
    args,
    img,
    detector,
    pose2d_estimator,
    pose3d_estimator,
    visualizer2D=None,
    visualizer3D=None,
    points3D=None,
    lineSet=None,
    show_interval=0,
):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
    )
    bboxes = bboxes[
        np.logical_and(
            pred_instance.labels == args.det_cat_id,
            pred_instance.scores > args.bbox_thr,
        )
    ]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose2d_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    pose_prediction = data_samples.get("pred_instances", None)

    coord_with_score = halpe2h36m(
        pose_prediction["keypoints"], pose_prediction["keypoint_scores"], img.shape[:2]
    )

    predicted_3d_pos = pose3d_estimator(coord_with_score)
    predicted_3d_pos[:, 0, 0, 2] = 0

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order="rgb")
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer2D is not None:
        visualizer2D.add_datasample(
            "result",
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr,
        )

    if visualizer3D is not None and points3D is not None and lineSet is not None:
        points3D.points = o3d.utility.Vector3dVector(
            predicted_3d_pos[0, 0].cpu().detach().numpy()
        )
        lineSet.points = points3D.points

        lineSet.lines = o3d.utility.Vector2iVector(
            np.asarray(
                [
                    [0, 1],
                    [1, 2],
                    [2, 3],
                    [0, 4],
                    [4, 5],
                    [5, 6],
                    [0, 7],
                    [7, 8],
                    [8, 10],
                    [9, 10],
                    [11, 8],
                    [11, 12],
                    [12, 13],
                    [8, 14],
                    [14, 15],
                    [15, 16],
                ]
            )
        )

        vis_ctr = visualizer3D.get_view_control()
        vis_ctr.set_up((0, -1, 0))

        visualizer3D.update_geometry(points3D)
        visualizer3D.update_geometry(lineSet)
        visualizer3D.poll_events()
        visualizer3D.update_renderer()


def main():

    parser = ArgumentParser()
    parser.add_argument("det_config", help="Config file for detection")
    parser.add_argument("det_checkpoint", help="Checkpoint file for detection")
    parser.add_argument("pose_config", help="Config file for pose")
    parser.add_argument("pose_checkpoint", help="Checkpoint file for pose")
    parser.add_argument(
        "--show", action="store_true", default=False, help="whether to show img"
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--det-cat-id",
        type=int,
        default=0,
        help="Category id for bounding box detection model",
    )
    parser.add_argument(
        "--bbox-thr", type=float, default=0.3, help="Bounding box score threshold"
    )
    parser.add_argument(
        "--nms-thr", type=float, default=0.3, help="IoU threshold for bounding box NMS"
    )
    parser.add_argument(
        "--kpt-thr", type=float, default=0.3, help="Visualizing keypoint thresholds"
    )
    parser.add_argument(
        "--draw-heatmap",
        action="store_true",
        default=False,
        help="Draw heatmap predicted by the model",
    )
    parser.add_argument(
        "--show-kpt-idx",
        action="store_true",
        default=False,
        help="Whether to show the index of keypoints",
    )
    parser.add_argument(
        "--skeleton-style",
        default="mmpose",
        type=str,
        choices=["mmpose", "openpose"],
        help="Skeleton style selection",
    )
    parser.add_argument(
        "--radius", type=int, default=3, help="Keypoint radius for visualization"
    )
    parser.add_argument(
        "--thickness", type=int, default=1, help="Link thickness for visualization"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.8, help="The transparency of bboxes"
    )
    parser.add_argument(
        "--draw-bbox", action="store_true", help="Draw bboxes of instances"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pose3d/MB_ft_h36m_global_lite.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        default="checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin",
        type=str,
        metavar="FILENAME",
        help="checkpoint to evaluate (file name)",
    )
    parser.add_argument(
        "--camera-id", type=int, default=0, help="for specifying the webcam ID"
    )

    args = parser.parse_args()
    mb_model_args = get_config(args.config)

    assert has_mmdet, "Please install mmdet to run the demo."
    assert args.show
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    model_backbone = load_backbone(mb_model_args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()
    elif torch.has_mps:
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cpu()
        
    checkpoint = torch.load(args.evaluate, map_location=lambda storage, loc: storage)
    model_backbone.load_state_dict(checkpoint["model_pos"], strict=True)
    pose3d_estimator = model_backbone
    pose3d_estimator.eval()

    # build detector
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose2d_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        device=args.device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))),
    )

    # build visualizer
    pose2d_estimator.cfg.visualizer.radius = args.radius
    pose2d_estimator.cfg.visualizer.alpha = args.alpha
    pose2d_estimator.cfg.visualizer.line_width = args.thickness
    visualizer2D = VISUALIZERS.build(pose2d_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer2D.set_dataset_meta(
        pose2d_estimator.dataset_meta, skeleton_style=args.skeleton_style
    )

    ## 3D visualization with Open3D
    visualizer3D = o3d.visualization.Visualizer()
    visualizer3D.create_window()

    points3D = o3d.geometry.PointCloud()
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[1, 0, 0]
    )
    lineSet = o3d.geometry.LineSet()
    visualizer3D.add_geometry(points3D)
    visualizer3D.add_geometry(coordinate_frame)
    visualizer3D.add_geometry(lineSet)

    cap = cv2.VideoCapture(args.camera_id)

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.resize(frame, (640,360))

        # topdown pose estimation
        process_one_image(
            args,
            frame,
            detector,
            pose2d_estimator,
            pose3d_estimator,
            visualizer2D,
            visualizer3D,
            points3D,
            lineSet,
            0.001,
        )

        # press ESC to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    visualizer3D.destroy_window()


if __name__ == "__main__":
    main()
