import numpy as np 





def translate_to_origin(skeleton, joint_idx): 
    """
    Translate the skeleton based on a specific joint to the origin (0, 0, 0) 

    Argument: 
    
    skeleton : np.array (N, 17, 3) 
    joint_idx : int
    coords : np.array (3, )

    Output: 
    
    translated_skeleton : np.array (N, 17, 3)
    """

    assert skeleton.shape[1] == 17 and skeleton.shape[2] == 3
    translated_skeleton = skeleton - np.reshape(skeleton[0, joint_idx, :], (1, 1, -1))
    return translated_skeleton
    