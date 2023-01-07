import numpy as np

def depth(flow, confidence, ep, K, thres=10):
    """
    params:
        @flow: np.array(h, w, 2)
        @confidence: np.array(h, w, 2)
        @K: np.array(3, 3)
        @ep: np.array(3,) the epipole you found epipole.py note it is uncalibrated and you need to calibrate it in this function!
    return value:
        depth_map: np.array(h, w)
    """
    depth_map = np.zeros_like(confidence)

    """
    STUDENT CODE BEGINS
    """
    S = np.matrix([[1118,0,357],[0,1121,268],[0,0,1]])
    K_inv = np.linalg.inv(K)
    EP = np.matmul(K_inv,ep)
    for i in range(confidence.shape[0]):
        for j in range(confidence.shape[1]):
            if confidence[i,j] > thres:
                c = flow[i,j,0]/357
                d = flow[i,j,1]/268
                X = [j, i, 1]
                X_ = np.matmul(K_inv, X)
                a = (X_[0] - EP[0])
                b = (X_[1] - EP[1])
                depth_map[i,j] = np.sqrt((a**2 + b**2)/(c**2 + d**2))

    """
    STUDENT CODE ENDS
    """

    truncated_depth_map = np.maximum(depth_map, 0)
    valid_depths = truncated_depth_map[truncated_depth_map > 0]
    # You can change the depth bound for better visualization if your depth is in different scale
    depth_bound = valid_depths.mean() + 10 * np.std(valid_depths)
    # print(f'depth bound: {depth_bound}')

    truncated_depth_map[truncated_depth_map > depth_bound] = 0
    truncated_depth_map = truncated_depth_map / truncated_depth_map.max()
    

    return truncated_depth_map
