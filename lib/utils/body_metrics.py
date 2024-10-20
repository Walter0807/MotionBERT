import numpy as np 
from .vismo import pixel2world_vis_motion



JOINT_NAME_TO_INDEX = {
    'root': 0,
    'rhip': 1,
    'rkne': 2,
    'rank': 3,
    'lhip': 4,
    'lkne': 5,
    'lank': 6,
    'belly': 7,
    'neck': 8,
    'nose': 9,
    'head': 10,
    'lsho': 11,
    'lelb': 12,
    'lwri': 13,
    'rsho': 14,
    'relb': 15,
    'rwri': 16
}



def compute_velocity(positions, dt):
    # Compute velocity as the first derivative of positions
    velocity = np.diff(positions, axis=0) / dt
    return velocity

def compute_acceleration(velocity, dt):
    # Compute acceleration as the first derivative of velocity
    acceleration = np.diff(velocity, axis=0) / dt
    return acceleration

def compute_jerk(acceleration, dt): 
    jerk = np.diff(acceleration, axis=0) / dt
    return jerk

def compute_center_of_mass(keypoints):
    center_of_mass = np.mean(keypoints, axis=1)
    return center_of_mass


def compute_x_axis_rotation_angle(skeleton, joints):
    """
    compute x axis rotation agle for given joints. 
    For example, computing the shoulders rotation w.r.t x axis. 
    The function needs two joints coordinates and will extract the relevant 
    information from the skeleton data. 

    Arg
    ---- 
        skeleton : (N, 17, 3) array 
        joints : list of joint indices
    """    

    if len(joints) != 2: 
        raise ValueError("You should provide two joints in order to compute the x axis rotation") 

    # extract joints 
    joint1 = skeleton[:, joints[0], [0, 2]] 
    joint2 = skeleton[:, joints[1], [0, 2]]

    # Compute direction vector of shoulder segment
    segment_direction = joint2 - joint1 
    magnitude_direction = np.linalg.norm(segment_direction, ord=2, axis=1)
    
    quadrant = np.zeros((len(segment_direction,)))
    first_q = ((segment_direction[:, 0] >= 0) & (segment_direction[:, 1] >= 0))
    second_q = ((segment_direction[:, 0] < 0) & (segment_direction[:, 1] > 0))
    third_q = ((segment_direction[:, 0] < 0) & (segment_direction[:, 1] < 0))
    fourth_q = ((segment_direction[:, 0] > 0) & (segment_direction[:, 1] < 0))
    quadrant[first_q] = 1
    quadrant[second_q] = 2
    quadrant[third_q] = 3
    quadrant[fourth_q] = 4

    # compute angle based on quandrant 
    rotation_angle_rad = np.zeros_like(quadrant) 
    rotation_angle_rad[quadrant == 1] = np.arccos(segment_direction[quadrant == 1][:, 0] / magnitude_direction[quadrant == 1])
    rotation_angle_rad[quadrant == 2] = np.pi - np.arcsin(segment_direction[quadrant == 2][:, 1] / magnitude_direction[quadrant == 2])
    rotation_angle_rad[quadrant == 3] = 2*np.pi + np.arctan2(segment_direction[quadrant == 3][:, 1], segment_direction[quadrant == 3][:, 0])
    rotation_angle_rad[quadrant == 4] = 2*np.pi + np.arcsin(segment_direction[quadrant == 4][:, 1] / magnitude_direction[quadrant == 4])

    # print(quadrant)
    
    # Convert rotation angle to degrees
    rotation_angle_deg = np.degrees(rotation_angle_rad)
    # print(rotation_angle_deg)    
    return rotation_angle_deg


def compute_angle(skeleton, joints):
    """
    Given a set of joints (2 segments), compute the angle based on the centered joint.
    For example, computing the right leg angle 
    The function needs a list of joints coordinates and will extract the relevant 
    segment from the skeleton data. 

    Arg
    ---- 
        skeleton : (N, 17, 3) array 
        joints : [[seg_start, seg_end], [seg_start, seg_end]]
    """    

    if len(joints) != 2: 
        raise ValueError("You should provide two joints in order to compute the x axis rotation") 

    # extract joints 
    segment1 = skeleton[:, joints[0][0], :] - skeleton[:, joints[0][1], :] # (N, 3)
    segment2 = skeleton[:, joints[1][0], :] - skeleton[:, joints[1][1], :] # (N, 3)
    
    # Calculate dot product of direction vector and unit x-axis vector
    dot_product = np.sum(np.multiply(segment1, segment2), axis=1) # (N, 3) -> (N,)

    # Calculate magnitude of direction vector
    magnitude_segment1 = np.linalg.norm(segment1, ord=2, axis=1)
    magnitude_segment2 = np.linalg.norm(segment2, ord=2, axis=1)
    # Calculate rotation angle (in radians)
    rotation_angle_rad = np.arccos(dot_product / (magnitude_segment1 * magnitude_segment2)) # (N, )
    
    # Convert rotation angle to degrees
    rotation_angle_deg = np.degrees(rotation_angle_rad)
    
    return rotation_angle_deg


def get_all_metrics(skeleton, y_last=False): 
    skeleton = - skeleton
    
    body_segments = {
        "shoulders": [14, 11], 
        "hips" : [1, 5], 
    } 

    body_angles = {
        "right_leg": [[2, 1], [2, 3]], 
        "left_leg" : [[5, 4], [5, 6]], 
        "right_arm" : [[15, 14], [15, 16]], 
        "left_arm" : [[12, 11], [12, 13]], 
    }

    
    y_index = 2 if y_last else 1
    x_index = 0
    z_index = 1 if y_last else 2

    shoulder_x_axis_rotation = compute_x_axis_rotation_angle(skeleton, body_segments["shoulders"])
    hips_x_axis_rotation = compute_x_axis_rotation_angle(skeleton, body_segments["hips"])

    shoulder_rot_velocity = compute_velocity(shoulder_x_axis_rotation, 1/30)
    shoulder_rot_accel = compute_acceleration(shoulder_rot_velocity, 1/30)
    shoulder_rot_jerk = compute_jerk(shoulder_rot_accel, 1/30)

    hips_rot_vel = compute_velocity(hips_x_axis_rotation, 1/30) 
    hips_rot_accel = compute_acceleration(hips_rot_vel, 1/30) 
    hips_rot_jerk = compute_jerk(hips_rot_accel, 1/30)
    
    right_leg_angle = compute_angle(skeleton, body_angles["right_leg"])
    left_leg_angle = compute_angle(skeleton, body_angles["left_leg"])
    right_arm_angle = compute_angle(skeleton, body_angles["right_arm"])
    left_arm_angle = compute_angle(skeleton, body_angles["left_arm"])
    left_leg_angle_velocity = compute_velocity(left_leg_angle, 1/30)
    left_leg_angle_accel = compute_acceleration(left_leg_angle_velocity, 1/30)

    right_ankle_y = skeleton[:, JOINT_NAME_TO_INDEX["rank"], y_index]
    right_ankle_z = skeleton[:, JOINT_NAME_TO_INDEX["rank"], z_index]
    left_ankle_y = skeleton[:, JOINT_NAME_TO_INDEX["lank"], y_index]
    left_ankle_z = skeleton[:, JOINT_NAME_TO_INDEX["lank"], z_index]
    left_ankle_x = skeleton[:, JOINT_NAME_TO_INDEX["lank"], x_index]
    
    output = {
        "shoulders" : {
            "rotation" : shoulder_x_axis_rotation, 
            "rotation_velocity" : shoulder_rot_velocity, 
            "rotation_acceleration" : shoulder_rot_accel,
            "rotation_jerk" : shoulder_rot_jerk
        }, 
        "hips" : {
            "rotation" : hips_x_axis_rotation, 
            "rotation_velocity" : hips_rot_vel, 
            "rotation_acceleration" : hips_rot_accel, 
            "rotation_jerk" : hips_rot_jerk
        }, 
        "left_leg" : {
            "angle" : left_leg_angle, 
            "angle_velocity": left_leg_angle_velocity, 
            "angle_acceleration": left_leg_angle_accel
        },
        "right_leg" : {
            "angle" : right_leg_angle
        },
        "left_arm" : {
            "left_arm_angle" : left_arm_angle
        },
        "right_arm" : {
            "right_arm_angle" : right_arm_angle
        }, 
        "right_ankle" : {
            "y_coord": right_ankle_y, 
            "z_coord": right_ankle_z
        }, 
        "left_ankle" : {
            "y_corod": left_ankle_y, 
            "z_coord": left_ankle_z, 
            "x_coord": left_ankle_x
        }
    }

    return output