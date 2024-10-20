import numpy as np 






def flip_h36m_motion(motion_data):
    """
    Flip H36M motion data using NumPy.
    
    :param motion_data: NumPy array of shape (frames, joints, 3) or (batch, frames, joints, 3)
    :return: Flipped motion data with the same shape as input
    """
    # Define the joint pairs to be swapped
    swap_pairs = [
        (1, 4),  # right hip <-> left hip
        (2, 5),  # right knee <-> left knee
        (3, 6),  # right ankle <-> left ankle
        (11, 14),  # left shoulder <-> right shoulder
        (12, 15),  # left elbow <-> right elbow
        (13, 16),  # left wrist <-> right wrist
    ]
    
    # Create a copy of the input data
    flipped_data = motion_data.copy()
    
    # Swap the paired joints
    for pair in swap_pairs:
        flipped_data[..., pair[0], :], flipped_data[..., pair[1], :] = \
            flipped_data[..., pair[1], :].copy(), flipped_data[..., pair[0], :].copy()
    
    # Negate the x-coordinates to mirror the motion
    flipped_data[..., 0] = -flipped_data[..., 0]
    
    return flipped_data