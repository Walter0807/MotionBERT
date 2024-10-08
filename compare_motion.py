import os
import numpy as np
import argparse
from tqdm import tqdm
import imageio
import copy

from lib.utils.utils_skeleton import translate_to_origin

from vispy import app, scene, io
from vispy.color import Color



JOINT_PAIRS = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], 
                            [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], 
                            [11, 12], [12, 13], [14, 15], [15, 16]])
JOINT_PAIRS_LEFT = np.array([[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]])
JOINT_PAIRS_RIGHT = np.array([[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]])

COLOR_MID = Color("#00457E").rgb
COLOR_LEFT = Color("#02315E").rgb
COLOR_RIGHT = Color("#2F70AF").rgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_motion", type=str, required=True, help=".npy file of the student motion")
    parser.add_argument('--teacher_motion', type=str, required=True, help=".npy file of the teacher motion")
    opts = parser.parse_args()
    return opts

def load_skeleton(path): 
    skeleton_data = np.load(path) 
    return skeleton_data


def setup_world(): 
    canvas = scene.SceneCanvas(show=True)
    view = canvas.central_widget.add_view()
    camera = scene.cameras.TurntableCamera(elevation=2, azimuth=0, roll=0, distance=4)
    view.camera = camera
    return canvas, camera




class Skeleton: 
    

    def __init__(self, motion, view): 

        assert motion.shape[1] == 17 and motion.shape[2] == 3
        

        self.parent = view.scene 
        lines = scene.Line(parent=view.scene, color='gray', method='gl', width=20)
        scatter = scene.Markers(parent=view.scene, edge_color='green')
        
        colors = np.full((len(JOINT_PAIRS), 3), COLOR_MID)
        colors[np.isin(JOINT_PAIRS, JOINT_PAIRS_LEFT).all(axis=1)] = COLOR_LEFT
        colors[np.isin(JOINT_PAIRS, JOINT_PAIRS_RIGHT).all(axis=1)] = COLOR_RIGHT
        lines.set_data(color=np.repeat(colors, 2, axis=0))

        self.frame_text = scene.Text(f'Frame: 0', color='white', font_size=80, parent=view.scene)
        
        self.lines = lines
        self.scatter = scatter

        self.frame_idx = 0 
        # translate to root 
        preprocessed_motion = self._preprocess_motion(motion)
        self.motion = self.translate_to_origin(preprocessed_motion, joint_idx=0) # (N, 17, 3)
        

    def _preprocess_motion(self, motion): 
        _motion = copy.deepcopy(motion)
        _motion[:, :, 0] = - motion[:, :, 0]
        _motion[:, :, 1] = - motion[:, :, 2]
        _motion[:, :, 2] = - motion[:, :, 1]
        return _motion

    def translate_to_origin(self, motion, joint_idx): 
        translated_motion = motion - np.reshape(motion[0, joint_idx, :], (1, 1, -1))
        return translated_motion
    
    def update(self):
        j3d = self.motion[self.frame_idx, :, :] # (17, 3)
        # Update joint positions
        self.scatter.set_data(j3d, edge_color='black', face_color='white', size=10)
        
        # Update limb positions
        connects = np.c_[j3d[JOINT_PAIRS[:, 0]], j3d[JOINT_PAIRS[:, 1]]].reshape(-1, 3)
        self.lines.set_data(pos=connects, connect='segments')

        # update text 
        self.frame_text.text = f'Frame: {self.frame_idx}'
        highest_point = np.max(j3d[:, 2])
        self.frame_text.pos = [j3d[0, 0], j3d[0, 1], highest_point + 0.1]

        # update frame index 
        self.frame_idx += 1

    def __len__(self): 
        return len(self.motion)


        




class ComparisonWorld: 

    def __init__(self, teacher_motion, student_motion, fps=30, y_vertical=True): 
        self.fps = fps 
        self.y_vertical = y_vertical

        self._setup_world(teacher_motion, student_motion)


    def _setup_world(self, teacher_motion, student_motion): 
        canvas = scene.SceneCanvas(show=True)
        view = canvas.central_widget.add_view()
        camera = scene.cameras.TurntableCamera(elevation=0, azimuth=0)
        view.camera = camera
        self.view = view 
        self.camera = camera

        tch_skeleton = Skeleton(teacher_motion, view)
        self.tch_skeleton = tch_skeleton
        std_skeleton = Skeleton(student_motion, view)
        self.std_skeleton = std_skeleton

        self.curr_frame = 0 
        self.max_frame = max(len(tch_skeleton), len(std_skeleton))

        self.timer = app.Timer(interval=1/self.fps, connect=self._on_timer, start=True)

    def _on_timer(self, event):
        if self.curr_frame >= self.max_frame:
            self.curr_frame = 0

        self.tch_skeleton.update()
        # self.std_skeleton.update()

        self.curr_frame += 1


    def _display_axis(self): 
        axis_length = 4.0
        axis_points = np.array([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length]
        ])

        axis_connects = np.array([[0, 1], [0, 2], [0, 3]])

        axis_colors = np.array([
            [1, 1, 1, 1],  # Red,
            [1, 0, 0, 1],  # Red
            [0, 1, 0, 1],  # Green
            [0, 0, 1, 1]   # Blue
        ])

        axis_lines = scene.Line(pos=axis_points, connect=axis_connects, color=axis_colors,
                            method='gl', parent=self.view.scene, width=5)
        
        # Add axis labels
        axis_labels = ['X', 'Y', 'Z']
    
        label_offset = 0.1  # Offset to position labels slightly away from axis ends
        for i, label in enumerate(axis_labels):
            pos = axis_points[i+1] + label_offset
            text = scene.Text(label, pos=pos, color=axis_colors[i+1],
                            font_size=60, parent=self.view.scene)


    def translate_to_origin(self, motion, joint_idx): 
        assert motion.shape[1] == 17 and motion.shape[2] == 3
        translated_motion = motion - np.reshape(motion[0, joint_idx, :], (1, 1, -1))
        return translated_motion

    def display(self): 
        self._display_axis()
        app.run()



if __name__ == "__main__": 

    opts = parse_args() 

    student_motion = load_skeleton(opts.student_motion)
    teacher_motion = load_skeleton(opts.teacher_motion)



    # translate to origin 

    world = ComparisonWorld(student_motion, teacher_motion) 

    world.display()

    




