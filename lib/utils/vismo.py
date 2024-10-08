import numpy as np
import os
import cv2
import math
import copy
import imageio
import io
from tqdm import tqdm
from PIL import Image
from lib.utils.tools import ensure_dir
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib.utils.utils_smpl import *
import ipdb


from vispy import app, scene, io
from vispy.color import Color


def render_and_save(motion_input, save_path, keep_imgs=False, fps=25, color="#F96706#FB8D43#FDB381", with_conf=False, draw_face=False, metrics=None):
    ensure_dir(os.path.dirname(save_path))
    motion = copy.deepcopy(motion_input)
    if motion.shape[-1]==2 or motion.shape[-1]==3:
        motion = np.transpose(motion, (1,2,0))   #(T,17,D) -> (17,D,T) 
    if motion.shape[1]==2 or with_conf:
        colors = hex2rgb(color)
        if not with_conf:
            J, D, T = motion.shape
            motion_full = np.ones([J,3,T])
            motion_full[:,:2,:] = motion
        else:
            motion_full = motion
        motion_full[:,:2,:] = pixel2world_vis_motion(motion_full[:,:2,:])
        motion2video(motion_full, save_path=save_path, colors=colors, fps=fps)
    elif motion.shape[0]==6890:
        # motion_world = pixel2world_vis_motion(motion, dim=3)
        motion2video_mesh(motion, save_path=save_path, keep_imgs=keep_imgs, fps=fps, draw_face=draw_face)
    else:
        motion_world = pixel2world_vis_motion(motion, dim=3)
        motion2video_3d_vispy(motion_world, save_path=save_path, fps=fps, metrics=metrics)
        
def pixel2world_vis(pose):
#     pose: (17,2)
    return (pose + [1, 1]) * 512 / 2

def pixel2world_vis_motion(motion, dim=2, is_tensor=False):
#     pose: (17,2,N)
    N = motion.shape[-1]
    if dim==2:
        offset = np.ones([2,N]).astype(np.float32)
    else:
        offset = np.ones([3,N]).astype(np.float32)
        offset[2,:] = 0
    if is_tensor:
        offset = torch.tensor(offset)
    return (motion + offset) * 512 / 2

def vis_data_batch(data_input, data_label, n_render=10, save_path='doodle/vis_train_data/'):
    '''
        data_input: [N,T,17,2/3]
        data_label: [N,T,17,3]
    '''
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 
    for i in range(min(len(data_input), n_render)):
        render_and_save(data_input[i][:,:,:2], '%s/input_%d.mp4' % (save_path, i))
        render_and_save(data_label[i], '%s/gt_%d.mp4' % (save_path, i))

def get_img_from_fig(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return img

def rgb2rgba(color):
    return (color[0], color[1], color[2], 255)

def hex2rgb(hex, number_of_colors=3):
    h = hex
    rgb = []
    for i in range(number_of_colors):
        h = h.lstrip('#')
        hex_color = h[0:6]
        rgb_color = [int(hex_color[i:i+2], 16) for i in (0, 2 ,4)]
        rgb.append(rgb_color)
        h = h[6:]
    return rgb

def joints2image(joints_position, colors, transparency=False, H=1000, W=1000, nr_joints=49, imtype=np.uint8, grayscale=False, bg_color=(255, 255, 255)):
#     joints_position: [17*2]
    nr_joints = joints_position.shape[0]

    if nr_joints == 49: # full joints(49): basic(15) + eyes(2) + toes(2) + hands(30)
        limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], \
                   [8, 9], [8, 13], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15], [15, 16],
                   ]#[0, 17], [0, 18]] #ignore eyes

        L = rgb2rgba(colors[0]) if transparency else colors[0]
        M = rgb2rgba(colors[1]) if transparency else colors[1]
        R = rgb2rgba(colors[2]) if transparency else colors[2]

        colors_joints = [M, M, L, L, L, R, R,
                  R, M, L, L, L, L, R, R, R,
                  R, R, L] + [L] * 15 + [R] * 15

        colors_limbs = [M, L, R, M, L, L, R,
                  R, L, R, L, L, L, R, R, R,
                  R, R]
    elif nr_joints == 15: # basic joints(15) + (eyes(2))
        limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7],
                   [8, 9], [8, 12], [9, 10], [10, 11], [12, 13], [13, 14]]
                    # [0, 15], [0, 16] two eyes are not drawn

        L = rgb2rgba(colors[0]) if transparency else colors[0]
        M = rgb2rgba(colors[1]) if transparency else colors[1]
        R = rgb2rgba(colors[2]) if transparency else colors[2]

        colors_joints = [M, M, L, L, L, R, R,
                         R, M, L, L, L, R, R, R]

        colors_limbs = [M, L, R, M, L, L, R,
                        R, L, R, L, L, R, R]
    elif nr_joints == 17: # H36M, 0: 'root',
    #                             1: 'rhip',
    #                             2: 'rkne',
    #                             3: 'rank',
    #                             4: 'lhip',
    #                             5: 'lkne',
    #                             6: 'lank',
    #                             7: 'belly',
    #                             8: 'neck',
    #                             9: 'nose',
    #                             10: 'head',
    #                             11: 'lsho',
    #                             12: 'lelb',
    #                             13: 'lwri',
    #                             14: 'rsho',
    #                             15: 'relb',
    #                             16: 'rwri'
        limbSeq = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]

        L = rgb2rgba(colors[0]) if transparency else colors[0]
        M = rgb2rgba(colors[1]) if transparency else colors[1]
        R = rgb2rgba(colors[2]) if transparency else colors[2]

        colors_joints = [M, R, R, R, L, L, L, M, M, M, M, L, L, L, R, R, R]
        colors_limbs = [R, R, R, L, L, L, M, M, M, L, R, M, L, L, R, R]
        
    else:
        raise ValueError("Only support number of joints be 49 or 17 or 15")

    if transparency:
        canvas = np.zeros(shape=(H, W, 4))
    else:
        canvas = np.ones(shape=(H, W, 3)) * np.array(bg_color).reshape([1, 1, 3])
    hips = joints_position[0]
    neck = joints_position[8]
    torso_length = ((hips[1] - neck[1]) ** 2 + (hips[0] - neck[0]) ** 2) ** 0.5
    head_radius = int(torso_length/4.5)
    end_effectors_radius = int(torso_length/15)
    end_effectors_radius = 7
    joints_radius = 7
    for i in range(0, len(colors_joints)):
        if i in (17, 18):
            continue
        elif i > 18:
            radius = 2
        else:
            radius = joints_radius
        if len(joints_position[i])==3:                 # If there is confidence, weigh by confidence
            weight = joints_position[i][2]
            if weight==0:
                continue
        cv2.circle(canvas, (int(joints_position[i][0]),int(joints_position[i][1])), radius, colors_joints[i], thickness=-1)
        
    stickwidth = 2
    for i in range(len(limbSeq)):
        limb = limbSeq[i]
        cur_canvas = canvas.copy()
        point1_index = limb[0]
        point2_index = limb[1]
        point1 = joints_position[point1_index]
        point2 = joints_position[point2_index]
        if len(point1)==3:                             # If there is confidence, weigh by confidence
            limb_weight = min(point1[2], point2[2])
            if limb_weight==0:
                bb = bounding_box(canvas)
                canvas_cropped = canvas[:,bb[2]:bb[3], :]
                continue
        X = [point1[1], point2[1]]
        Y = [point1[0], point2[0]]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        alpha = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(alpha), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors_limbs[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        bb = bounding_box(canvas)
        canvas_cropped = canvas[:,bb[2]:bb[3], :]
    canvas = canvas.astype(imtype)
    canvas_cropped = canvas_cropped.astype(imtype)
    if grayscale:
        if transparency:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGBA2GRAY)
            canvas_cropped = cv2.cvtColor(canvas_cropped, cv2.COLOR_RGBA2GRAY)
        else:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
            canvas_cropped = cv2.cvtColor(canvas_cropped, cv2.COLOR_RGB2GRAY)
    return [canvas, canvas_cropped]


def motion2video(motion, save_path, colors, h=512, w=512, bg_color=(255, 255, 255), transparency=False, motion_tgt=None, fps=25, save_frame=False, grayscale=False, show_progress=True, as_array=False):
    nr_joints = motion.shape[0]
#     as_array = save_path.endswith(".npy")
    vlen = motion.shape[-1]

    out_array = np.zeros([vlen, h, w, 3]) if as_array else None
    videowriter = None if as_array else imageio.get_writer(save_path, fps=fps)

    if save_frame:
        frames_dir = save_path[:-4] + '-frames'
        ensure_dir(frames_dir)

    iterator = range(vlen)
    if show_progress: iterator = tqdm(iterator)
    for i in iterator:
        [img, img_cropped] = joints2image(motion[:, :, i], colors, transparency=transparency, bg_color=bg_color, H=h, W=w, nr_joints=nr_joints, grayscale=grayscale)
        if motion_tgt is not None:
            [img_tgt, img_tgt_cropped] = joints2image(motion_tgt[:, :, i], colors, transparency=transparency, bg_color=bg_color, H=h, W=w, nr_joints=nr_joints, grayscale=grayscale)
            img_ori = img.copy()
            img = cv2.addWeighted(img_tgt, 0.3, img_ori, 0.7, 0)
            img_cropped = cv2.addWeighted(img_tgt, 0.3, img_ori, 0.7, 0)
            bb = bounding_box(img_cropped)
            img_cropped = img_cropped[:, bb[2]:bb[3], :]
        if save_frame:
            save_image(img_cropped, os.path.join(frames_dir, "%04d.png" % i))
        if as_array: out_array[i] = img
        else: videowriter.append_data(img)

    if not as_array:
        videowriter.close()

    return out_array


def plot_body_metric(metrics): 
    """
        metrics : dict
    """
    n_plots = sum([len(metrics[key]) for key in metrics])
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    fig.tight_layout()

    plot_idx = 0

    for body_part in metrics: 
        for metric_name in metrics[body_part]: 

            plot_title = f"{body_part} - {metric_name}"
            x = np.arange(0, len(metrics[body_part][metric_name]))
            y = metrics[body_part][metric_name]
            axes[plot_idx].plot(x, y)
            axes[plot_idx].set_title(plot_title)
            plot_idx += 1

    return fig, axes




def motion2video_3d_vispy(motion, save_path, fps=25, metrics=None):
    canvas = scene.SceneCanvas(show=False)
    view = canvas.central_widget.add_view()

    camera = scene.cameras.TurntableCamera(elevation=2, azimuth=0, roll=0, distance=4)
    view.camera = camera

    # Define joint pairs
    joint_pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], 
                            [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], 
                            [11, 12], [12, 13], [14, 15], [15, 16]])
    joint_pairs_left = np.array([[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]])
    joint_pairs_right = np.array([[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]])

    # Create line and scatter plots for joints and limbs
    lines = scene.Line(parent=view.scene, color='gray', method='gl', width=20)
    scatter = scene.Markers(parent=view.scene, edge_color='green')

    # Set up colors
    color_mid = Color("#00457E").rgb
    color_left = Color("#02315E").rgb
    color_right = Color("#2F70AF").rgb

    colors = np.full((len(joint_pairs), 3), color_mid)
    colors[np.isin(joint_pairs, joint_pairs_left).all(axis=1)] = color_left
    colors[np.isin(joint_pairs, joint_pairs_right).all(axis=1)] = color_right
    lines.set_data(color=np.repeat(colors, 2, axis=0))

    motion = motion - np.reshape(motion[3, :, 0], (1, -1, 1))
    print(motion.shape)
    j3d_adjusted = motion[:, [0, 2, 1], ] # (x, z, y)
    j3d_adjusted = - j3d_adjusted 

    axis_length = 1.0
    axis_points = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ])
    axis_connects = np.array([[0, 1], [0, 2], [0, 3]])

    axis_lines = scene.Line(pos=axis_points, connect=axis_connects, 
                            method='gl', parent=view.scene, width=10)

    # Add axis labels
    axis_labels = ['X', 'Y', 'Z']
    label_offset = 0.1  # Offset to position labels slightly away from axis ends
    for i, label in enumerate(axis_labels):
        pos = axis_points[i+1] + label_offset
        text = scene.Text(label, pos=pos, color='white',
                          font_size=24, parent=view.scene)


    writer = imageio.get_writer(save_path, fps=fps)
    text = scene.Text(f'Frame : 0', color='white', pos=[2, 2, 0], font_size=80)
    view.add(text)

    textankle = scene.Text(f'', color='white', pos=[2, 2, 0], font_size=80)
    view.add(textankle)

    for f in tqdm(range(j3d_adjusted.shape[2])):
        j3d = j3d_adjusted[:, :, f] * 0.004 # (17, 3)
        
        # Update joint positions    
        
        scatter.set_data(j3d, edge_color='black', face_color='white', size=10)

        # Update limb positions
        connects = np.c_[j3d[joint_pairs[:, 0]], j3d[joint_pairs[:, 1]]].reshape(-1, 3)
        lines.set_data(pos=connects, connect='segments')
        
        text.text = f'Frame: {f}'
        highest_point = np.max(j3d[:, 2])
    
        # Position the text above the highest point
        text_pos = [0, 0, highest_point]
        text.pos = text_pos
        textankle.text = f"{j3d[6, 2]:2f}"
        textankle.pos = j3d[6, :]

        # Set view limits
        view.camera.set_range()

        # Render the scene and save the frame
        canvas.update()
        frame = canvas.render()
        writer.append_data(frame)

    writer.close()

    if metrics: 
        fig, axes = plot_body_metric(metrics)
        file_name = save_path.replace(".mp4", "_metrics.png")
        plt.savefig(file_name)


def motion2video_3d(motion, save_path, fps=25, keep_imgs = False):
#     motion: (17,3,N)
    videowriter = imageio.get_writer(save_path, fps=fps)
    vlen = motion.shape[-1]
    save_name = save_path.split('.')[0]
    frames = []
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    
    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"
    for f in tqdm(range(vlen)):
        j3d = motion[:,:,f]
        fig = plt.figure(0, figsize=(10, 10))
        ax = plt.axes(projection="3d")
        ax.set_xlim(-512, 0)
        ax.set_ylim(-256, 256)
        ax.set_zlim(-512, 0)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        ax.view_init(elev=12., azim=80)
        plt.tick_params(left = False, right = False , labelleft = False ,
                        labelbottom = False, bottom = False)
        for i in range(len(joint_pairs)):
            limb = joint_pairs[i]
            xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
            if joint_pairs[i] in joint_pairs_left:
                ax.plot(-xs, -zs, -ys, color=color_left, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            elif joint_pairs[i] in joint_pairs_right:
                ax.plot(-xs, -zs, -ys, color=color_right, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            else:
                ax.plot(-xs, -zs, -ys, color=color_mid, lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
            
        frame_vis = get_img_from_fig(fig)
        videowriter.append_data(frame_vis)
        plt.close()
    videowriter.close()

def motion2video_mesh(motion, save_path, fps=25, keep_imgs = False, draw_face=True):
    videowriter = imageio.get_writer(save_path, fps=fps)
    vlen = motion.shape[-1]
    draw_skele = (motion.shape[0]==17)
    save_name = save_path.split('.')[0]
    smpl_faces = get_smpl_faces()
    frames = []
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]

    
    X, Y, Z = motion[:, 0], motion[:, 1], motion[:, 2]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    
    for f in tqdm(range(vlen)):
        j3d = motion[:,:,f]
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fig = plt.figure(0, figsize=(8, 8))
        ax = plt.axes(projection="3d", proj_type = 'ortho')
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.view_init(elev=-90, azim=-90)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        
        # plt.savefig("filename.png", transparent=True, bbox_inches="tight", pad_inches=0)
        
        if draw_skele:
            for i in range(len(joint_pairs)):
                limb = joint_pairs[i]
                xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
                ax.plot(-xs, -zs, -ys, c=[0,0,0], lw=3, marker='o', markerfacecolor='w', markersize=3, markeredgewidth=2) # axis transformation for visualization
        elif draw_face:
            ax.plot_trisurf(j3d[:, 0], j3d[:, 1], triangles=smpl_faces, Z=j3d[:, 2], color=(166/255.0,188/255.0,218/255.0,0.9))
        else:
            ax.scatter(j3d[:, 0], j3d[:, 1], j3d[:, 2], s=3, c='w', edgecolors='grey')
        frame_vis = get_img_from_fig(fig, dpi=128)
        plt.cla()
        videowriter.append_data(frame_vis)
        plt.close()
    videowriter.close()

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def bounding_box(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox
