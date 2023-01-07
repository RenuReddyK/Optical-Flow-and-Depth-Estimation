import numpy as np

def compute_planar_params(flow_x, flow_y, K,
                                up=[256, 0], down=[512, 256]):
    """
    params:
        @flow_x: np.array(h, w)
        @flow_y: np.array(h, w)
        @K: np.array(3, 3)
        @up: upper left index [i,j] of image region to consider.
        @down: lower right index [i,j] of image region to consider.
    return value:
        sol: np.array(8,)
    """
    """
    STUDENT CODE BEGINS
    """
    
    u = flow_x/K[0,0]
    v = flow_y/K[1,1]
    A =[]
    B = []
    K_inv = np.linalg.inv(K)
    for i in range(up[0],down[0]):
        for j in range(up[1],down[1]):
            X = [j, i, 1]
            x = np.matmul(K_inv,X)[0]
            y = np.matmul(K_inv,X)[1]
            A.append([x*x, x*y, x, y, 1, 0, 0, 0])
            A.append([x*y, y*y, 0, 0, 0, y, x, 1])
            B.append([u[i,j]])
            B.append([v[i,j]])

    A = np.array(A)
    B = np.array(B)

    q = np.linalg.lstsq(A,B,rcond = None)[0]
    sol = q.flatten()

    """
    STUDENT CODE ENDS
    """
    return sol
    
