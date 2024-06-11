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

#max_angle_knee_right = 180
#max_angle_knee_left = 180




def remove_coordinates(coordinate, *args):
    #Coordinate
    #X = 0
    #Y = 1
    #Z = 2
    for x in args:
        #Removes Z-Coordinate (3rd Element of Vector)
        x[coordinate] = 0
       # print(x)
    return args

def calculate_angle(A, B, C, max_angle_name, coordinate):
    
    # X,Y oder Z Koordinate entfernen, falls gewünscht
    if coordinate in (0,1,2):
        A,B,C = remove_coordinates(coordinate,A,B,C)

    #Vektoren AB und BC erstellen
    AB = B - A
    BC = C - B

    # Skalarprodukt und Normen berechnen
    dot_product = np.dot(AB, BC)
    norm_AB = np.linalg.norm(AB)
    norm_BC = np.linalg.norm(BC)

    # Winkel in Radiant berechnen
    angle = np.arccos(dot_product / (norm_AB * norm_BC))
    # Winkel in Grad umwandeln
    angle_deg = np.degrees(angle)

    # Winkel von 180° abziehen, um eigentlichen Winkel zu erhalten
    result = 180-angle_deg

    # Falls berechneter Winkel größer als bisheriger, maximaler Winkel, aktualisiere Winkel mit der maximalen Abweichung von 180 Grad
    if abs(result-180) > abs(max_angle_name-180): 
        max_angle_name = result

    return result, max_angle_name

# da Motionbert bei der Y-Achse von oben zu zählen beginnt (d.h. ganz oben ist Wert 0 und nach unten hin wird der Wert immer größer), müssen wir den maximalen y-Wert berechnen, um den tiefsten Punkt für die Bodenkontaktzeit ermitteln zu können
def updateMaxYValue(maxYValue, newYValue):
    if newYValue > maxYValue:
        maxYValue = newYValue
    return maxYValue

def updateMinYValue(minYValue, newYValue):
    if newYValue < minYValue:
        minYValue = newYValue
    return minYValue
 

def midpoint(p1, p2):
    return np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, 0]) #[[(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, 0]] #


def calc_hip_neck_yaxis_angle(hip_r_kp, hip_l_kp, neck_r_kp, neck_l_kp):
    hip_r_kp[2] = 0
    hip_l_kp[2] = 0
    neck_r_kp[2] = 0
    neck_l_kp[2] = 0
    # Vektor von Hüfte zu Nacken
    vector_a = midpoint(neck_r_kp, neck_l_kp) - midpoint(hip_r_kp, hip_l_kp)
    # Y-Achse
    y_axis = [0,1,0]
    # Berechne Winkel zwischen Hüfte-Nacken und Y-Achse
    angle_hip_neck_yaxis = np.arccos(np.dot(vector_a, y_axis) / (np.linalg.norm(vector_a) * np.linalg.norm(y_axis)))
    angle_deg = np.degrees(angle_hip_neck_yaxis)
    return 180-angle_deg

def process_joints(joints):
    hip = joints[4]
    knee = joints[5]
    foot = joints[6]
    coordinate = 2 #Berechnet VALUGUS Winkel durch 0-llen der Z-Koordinate
    return calculate_angle(hip, knee, foot, max_angle_knee_right, coordinate)
    

def render_and_save(motion_input, save_path, video_basename, keep_imgs=False, fps=25, color="#F96706#FB8D43#FDB381", with_conf=False, draw_face=False):
    print(">>>>>>> RENDER and SAVE")
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

        motion2video_3d(motion_world, save_path=save_path, video_basename=video_basename, keep_imgs=keep_imgs, fps=fps, counter = 0, counter_dez = 0, counter_boden = 0)
        
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
    print(">>> joints2image wird aufgerufen")
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
    #######
    # FIXME: delete after testing or implement
    
    r_knee_angle = process_joints(joints_position)
    
    print("knee",r_knee_angle)
    
    cv2.putText(canvas, f"Knee: {r_knee_angle:.2f}Grad",
            (int(joints_position[5][0]),int(joints_position[5][1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 2)
    
    ########
    
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
        
        # FIXME: delete
        cv2.putText(canvas, f"Test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 20, (0, 255, 0), 2)
        
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        bb = bounding_box(canvas)
        canvas_cropped = canvas[:,bb[2]:bb[3], :]
    canvas = canvas.astype(imtype)
    canvas_cropped = canvas_cropped.astype(imtype)
    # FIXME: delete
    cv2.putText(canvas_cropped, f"Test", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 20, (0, 255, 0), 2)
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

def motion2video_3d(motion, save_path, video_basename, fps=25, keep_imgs = False, counter = 0, counter_dez=0, counter_boden = 0):
#     motion: (17,3,N)
    print(">>>>>>>>> Motion To Vid")
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

    max_angle_knee_right = 180
    max_angle_knee_left = 180
    #max_angle_elb_right = 180
    max_hip_neck_angle = 0
    reakt_idx_right = 0
    reakt_idx_left = 0

    maxYValueFootRight = 0
    maxYValueFootLeft = 0
    minYValueHead = 100000 

    bodenkontaktzeit_right = 0
    bodenkontaktzeit_left = 0
    flag_l = True
    flag_r = True
    
    anzahl_frames = len(tqdm(range(vlen)))-1

    for idx, f in enumerate(tqdm(range(vlen))):
        print("idx: " , idx)

    #for f in tqdm(range(vlen)):
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


######################
        #Calculates the angles and writes angles to Video frames

        rhip = j3d[1, :]
        rkne = j3d[2, :]
        rank = j3d[3, :]
        lhip = j3d[4, :]
        lkne = j3d[5, :]
        lank = j3d[6, :]
      
        #rwri = j3d[16, :]
        #relb = j3d[15, :]
        rsho = j3d[14, :]

        rhip = j3d[1, :]
        lhip = j3d[4, :]
        lsho = j3d[11, :]

        # Keypoints der Fußknöchel im aktuellen Frame
        r_ankle_y = float(j3d[3, 1])
        l_ankle_y = float(j3d[6, 1])

        r_knee_x, r_knee_y = int(j3d[2, 0]), int(j3d[2, 1])
        #l_knee_x, l_knee_y = int(j3d[5, 0]), int(j3d[5, 1])

        # bestimme, an welcher Position im Video/Bild die Ausgaben erscheinen
        knee_textposition = (r_knee_x, r_knee_y)
        knee_max_textposition = (r_knee_x, r_knee_y + 50)
        #l_text_position = (r_knee_x, r_knee_y + 50)
        #l_text_max_position = (r_knee_x, r_knee_y + 150)
        #r_text_elb_position = (r_knee_x, r_knee_y + 200)
        #r_text_max_elb_position = (r_knee_x, r_knee_y + 250)
        hip_neck_angle_textposition = (r_knee_x, r_knee_y + 50)
        sprunghöhe_textposition = (r_knee_x, r_knee_y + 100)
        bodenkontaktzeit_textposition = (r_knee_x, r_knee_y + 150)
        reaktiv_index_textposition = (r_knee_x, r_knee_y + 200)
        
        # bestimme die Koordinate, die genullt werden soll für Winkelberechnung
        coordinate = 2

        # Berechne Ellenbogen Winkel
        # r_elb_angle , max_angle_elb_right = calculate_angle(rsho, relb, rwri, max_angle_elb_right, 3)    
        # cv2.putText(frame_vis, f"Elbow right: {r_elb_angle:.2f} Degree",r_text_elb_position,cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1)
        # cv2.putText(frame_vis, f"Elbow max right: {max_angle_elb_right:.2f} Degree",r_text_max_elb_position,cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1)


################ seitliche Rumpfbewegung ###############

        hip_neck_angle = calc_hip_neck_yaxis_angle(rhip, lhip, rsho, lsho)

        # aktualisiere den Wert der größten seitlichen Rumpfbewegung
        if (hip_neck_angle > max_hip_neck_angle):
            max_hip_neck_angle = hip_neck_angle

        if (hip_neck_angle > 5):
            # färbe rot, wenn Körperhaltung zu schief
            cv2.putText(frame_vis, f"Hip Neck y-Axis: {hip_neck_angle:.2f} Degree",hip_neck_angle_textposition,cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 1)
        else:
            cv2.putText(frame_vis, f"Hip Neck y-Axis: {hip_neck_angle:.2f} Degree",hip_neck_angle_textposition,cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1)
        
        
############### Sprunghöhe ############################
        # Da wir die Sprunghöhe nur als Vergleichswert zwischen Sprung mit rechtem und mit linkem Fuß benötigen, wird hier die max. Diff. des y-Werts des Kopfs als Sprunghöhe bezeichnet

        # Sprunghöhe Teil 1: Messe den y-Wert des Kopfs im ersten Frame und speichere den Wert  
        head_y = float(j3d[10, 1])
        if( idx == 0):
            # speichere den y-Wert des Kopfs im ersten Videoframe
            head_first_y_val = head_y

        # Sprunghöhe Teil 2: Aktualisiere den maximalen y-Wert vom Kopf, falls er kleiner als zuvor ist
        minYValueHead = updateMinYValue(minYValueHead, head_y)
        # Diff. am Ende des Videos sollte positiv sein
        head_diff = head_first_y_val - minYValueHead 
        print(f"Differenz von head y: {head_diff}")
        print(f"First y val head: {head_first_y_val}")
        print(f"Minimaler y val head: {minYValueHead}")

        cv2.putText(frame_vis, f"Sprunghoehe: {head_diff:.2f}", sprunghöhe_textposition, cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1)

############### Kennzahlen beim Sprung mit rechtem Fuß ############################

        # separiere Videos, bei denen mit rechtem Bein gesprungen wird von denen, bei denen mit linkem Bein gesprungen wird, anhand des Videonamens
        if "rechts" in video_basename:

            # Berechne Valgus/Varus anhand des rechten Kniewinkels
            r_knee_angle, max_angle_knee_right = calculate_angle(rhip, rkne, rank, max_angle_knee_right, coordinate)
            # gib Winkel auf Video aus
            if (r_knee_angle  < 170):
                # färbe Ausgabe rot wenn Valgus/Varus Kniewinkel zu groß
                cv2.putText(frame_vis, f"Right knee angle: {r_knee_angle:.2f} Degree",knee_textposition,cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 1)
            else:
                cv2.putText(frame_vis, f"Right knee angle: {r_knee_angle:.2f} Degree",knee_textposition,cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1)
            
            #if (max_angle_knee_right  < 170):
                # färbe Ausgabe rot wenn Valgus/Varus Kniewinkel zu groß
            #    cv2.putText(frame_vis, f"Right knee max angle: {max_angle_knee_right:.2f} Degree",knee_max_textposition,cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 1) 
            #else:
            #    cv2.putText(frame_vis, f"Right knee max angle: {max_angle_knee_right:.2f} Degree",knee_max_textposition,cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1)  

            if( idx == 0):
            # gib den y-Wert des rechten Fußes im ersten Frame aus 
                print("Y Wert rechter Knöchel im 1. Frame: " , r_ankle_y)
            
            # Bodenkontaktzeit rechter Fuß
            # maxYValueFootRight gibt den derzeit höchsten (eigentlich tiefsten) y-Wert des rechten Fußes an
            # prüfe, ob der Wert im aktuellen Frame tiefer ist und aktualisiere den Wert
            maxYValueFootRightNew = updateMaxYValue(maxYValueFootRight, r_ankle_y)
            # Bodenkontaktzeit messen: 
            # Wenn sich der größte (eigentlich tiefste) Wert nicht mehr ändert und die Differenz vom aktuellen y-Wert und 
            # dem tiefsten y-Wert eine bestimmte Toleranz (hier 2) nicht übersteigt und 
            # der y-Wert des linken Fußes tiefer ist als der des rechten Fußes (d.h. der linke Fuß ist noch in der Luft)
            # dann ist der Fuß gerade vermutlich am Boden
            # zähle dann die Bodenkontaktzeit weiter
            if(flag_r == True): # and counter_boden > 10):
                if (maxYValueFootRightNew == maxYValueFootRight and (abs(maxYValueFootRight-r_ankle_y)<2) and (l_ankle_y < r_ankle_y) and r_ankle_y>490):
                    # counter_dez gibt an, in welcher Sekunde wir gerade sind
                    # Wir berechnen die Bodenkontaktzeit in Anzahl an Frames, da unsere Videos sehr kurz sind 
                    bodenkontaktzeit_right += 1
                else:
                    # sobald die bodenkontaktzeit einmal hochgezählt wurde und im nächsten Frame nicht mehr, ist die Bodenkontaktzeit abgeschlossen
                    # damit am Ende des Videos, wenn person wieder steht, nicht hochgezählt wird
                    if (bodenkontaktzeit_right >0):
                        flag_r = False 
            # überschreibe den max. y-value in jedem Frame, falls größer
            maxYValueFootRight = maxYValueFootRightNew

            print(f"Bodenkontaktzeit rechts in Anzahl Frames: {bodenkontaktzeit_right}")
            print(f"Bodenkontaktzeit rechts in Sekunden: {bodenkontaktzeit_right/25}")

            print("Bisher max. Y Wert rechter Knöchel: " , maxYValueFootRight)

            # Berechne Reaktivitätsindex = Sprunghöhe / Bodenkontaktzeit
            # Reaktivitätsindex ist nur im letzten Frame aussagekräftig, daher nur Ausgabe am Ende des Videos betrachten!
            if (bodenkontaktzeit_right != 0): 
                cv2.putText(frame_vis, f"Bodenkontaktzeit rechtes: {bodenkontaktzeit_right:.2f}", bodenkontaktzeit_textposition, cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1)
                reakt_idx_right = head_diff / (bodenkontaktzeit_right/25) # Angabe in Sekunden
                cv2.putText(frame_vis, f"Reaktiv.-index rechts: {reakt_idx_right:.2f}", reaktiv_index_textposition, cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1)
            

            ############# Speichere Winkel f. rechte Seite in Textdatei #################

            # Datei mit Verlauf der Kennzahlen im Zeitverlauf
            with open(r'./outputs/angle_output_verlauf_rechts.txt', 'a') as f:
                ## increase counter for frames
                counter +=1 
                #counter_boden +=1
                if counter >= 25:
                    counter = 0
                    # Alle 25 Frames eine Sekunde hochzählen, da Video 24 Frames/Sekunde hat
                    counter_dez +=1
                    f.write(f"#####################"+ '\n')
                    f.write(f"Sekunde {counter_dez}"+ '\n')

                # Define the data to be written
                # Use a for loop to write each line of data to the file

                f.write(f"Winkel rechtes Knie von Front: {r_knee_angle:.2f} Degree" + '\n')
                f.write(f"Seitliche Rumpfbewegung / Hüfte Nacken y-Axe: {hip_neck_angle:.2f} Degree"+ '\n')
                f.write(f"Bodenkontaktzeit rechter Fuß in Sekunden: {bodenkontaktzeit_right/25:.2f}"+ '\n')
                f.write(f"Sprunghöhe: {head_diff}" + '\n')


            # Speichere Endergebnisse im letzten Frame
            if (idx == anzahl_frames): 
                # erstelle eine Datei für die Endergebnisse der rechten Seite
                filename_results_right = video_basename + '_results.txt'  # Create a unique filename
                output_path_results_right = os.path.join('./outputs', filename_results_right)  # Path for the results file
                print("output_path_results_right: " , output_path_results_right)

                with open(output_path_results_right, 'w') as f:
                    f.write(f"Winkel rechtes Knie mit max. Abweichung von 180 Grad von Front: {max_angle_knee_right} Grad" + '\n')
                    f.write(f"Maximale seitliche Rumpfbewegung: {max_hip_neck_angle} Grad" + '\n')
                    f.write(f"Sprunghöhe: {head_diff}" + '\n')
                    f.write(f"Bodenkontaktzeit rechter Fuß in Sekunden: {bodenkontaktzeit_right/25:.2f}"+ '\n')
                    if (reakt_idx_right != None):
                        f.write(f"Reaktiv.-index rechts: {reakt_idx_right:.2f}" + '\n') 

############### Kennzahlen beim Sprung mit linkem Fuß ############################

        if "links" in video_basename:

            # Berechne Valgus/Varus anhand des linken Kniewinkels
            l_knee_angle , max_angle_knee_left = calculate_angle(lhip, lkne, lank, max_angle_knee_left, coordinate)

            # gib Winkel auf Video aus
            if (l_knee_angle  < 170):
                # färbe Ausgabe rot wenn Valgus/Varus Kniewinkel zu groß
                cv2.putText(frame_vis, f"Left knee angle: {l_knee_angle:.2f} Degree",knee_textposition,cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 1)
            else:
                cv2.putText(frame_vis, f"Left knee angle: {l_knee_angle:.2f} Degree",knee_textposition,cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1)

            #if (max_angle_knee_left  < 170):
                # färbe Ausgabe rot wenn Valgus/Varus Kniewinkel zu groß
            #    cv2.putText(frame_vis, f"Left knee max angle: {max_angle_knee_left:.2f} Degree",knee_max_textposition,cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 1) 
            #else:
            #    cv2.putText(frame_vis, f"Left knee max angle: {max_angle_knee_left:.2f} Degree",knee_max_textposition,cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1)  

            if( idx == 0):
                # gib den y-Wert des rechten Fußes im ersten Frame aus 
                print("Y Wert linker Knöchel im 1. Frame: " , l_ankle_y)

            print("Berechnung Bodenkontaktzeit linker Fuß")
            # Bodenkontaktzeit linker Fuß
            maxYValueFootLeftNew = updateMaxYValue(maxYValueFootLeft, l_ankle_y)

            # Bodenkontaktzeit darf nur gezählt werden, wenn der linke Fuß tiefer ist als der rechte Fuß (is beim Aufkommen am Boden immer der Fall)
            # und der linke Fuß sich nicht mehr als 2 Pixel bewegt
            if(flag_l == True): #and counter_boden > 10):
                if (maxYValueFootLeftNew == maxYValueFootLeft and (abs(maxYValueFootLeft-l_ankle_y)<2) and (l_ankle_y > r_ankle_y) and l_ankle_y>490):
                    # counter_dez gibt an, in welcher Sekunde wir gerade sind
                    # Wir geben die Bodenkontaktzeit in Anzahl an Frames aus, da unsere Videos sehr kurz sind 
                    bodenkontaktzeit_left += 1
                else:
                    if (bodenkontaktzeit_left >0):
                        flag_l = False 
            # überschreibe den max. y-value in jedem Frame, falls größer 
            maxYValueFootLeft = maxYValueFootLeftNew

            print(f"bodenkontaktzeit links in Anzahl Frames: {bodenkontaktzeit_left}")
            print(f"bodenkontaktzeit links in Sekunden: {bodenkontaktzeit_left/25}")

            print("Bisher max. Y Wert linker Knöchel: " , maxYValueFootLeft)

            # Berechne Reaktivitätsindex = Sprunghöhe / Bodenkontaktzeit
            # Reaktivitätsindex ist nur im letzten Frame aussagekräftig, daher nur Ausgabe am Ende des Videos betrachten!
            if (bodenkontaktzeit_left != 0):
                cv2.putText(frame_vis, f"Bodenkontaktzeit links: {bodenkontaktzeit_left:.2f}", bodenkontaktzeit_textposition, cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1)
                reakt_idx_left = head_diff / (bodenkontaktzeit_left/25)
                cv2.putText(frame_vis, f"Reaktiv.-index links: {reakt_idx_left:.2f}", reaktiv_index_textposition, cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 0), 1)
        
        
            ############# Speichere Winkel für linke Seite in Textdatei #################

            # Datei mit Verlauf der Kennzahlen im Zeitverlauf
            with open(r'./outputs/angle_output_verlauf_links.txt', 'a') as f:
                ## create counter for frames
                counter +=1 
                counter_boden +=1
                if counter >= 25:
                    counter = 0
                    # Alle 25 Frames eine Sekunde hochzählen, da Video 24 Frames pro Sekunde hat
                    counter_dez +=1
                    f.write(f"#####################"+ '\n')
                    f.write(f"Sekunde {counter_dez}"+ '\n')

                # Speichere aktuelle Werte
                f.write(f"Winkel linkes Knie von Front: {l_knee_angle:.2f} Grad" + '\n') 
                f.write(f"Seitliche Rumpfbewegung / Hüfte Nacken y-Axe: {hip_neck_angle:.2f} Grad"+ '\n')
                f.write(f"Bodenkontaktzeit linker Fuß in Sekunden: {bodenkontaktzeit_left/25:.2f}"+ '\n')
                f.write(f"Sprunghöhe: {head_diff}" + '\n')

            # Speichere Endergebnisse im letzten Frame
            if (idx == anzahl_frames): 
                # erstelle eine Datei für die Endergebnisse der linken Seite
                filename_results_left = video_basename + '_results.txt'  # Create a unique filename
                output_path_results_left = os.path.join('./outputs', filename_results_left)  # Path for the results file
                print("output_path_results_left: " , output_path_results_left)

                with open(output_path_results_left, 'w') as f:
                    f.write(f"Winkel linkes Knie mit max. Abweichung von 180 Grad von Front: {max_angle_knee_left} Grad" + '\n')
                    f.write(f"Maximale seitliche Rumpfbewegung: {max_hip_neck_angle} Grad" + '\n')
                    f.write(f"Sprunghöhe: {head_diff}" + '\n')
                    f.write(f"Bodenkontaktzeit linker Fuß in Sekunden: {bodenkontaktzeit_left/25:.2f}"+ '\n')
                    if (reakt_idx_left != None):
                        f.write(f"Reaktiv.-index links: {reakt_idx_left:.2f}" + '\n') 

######################

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
