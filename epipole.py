import numpy as np
def epipole(u,v,smin,thresh,num_iterations = 1000):
    ''' Takes flow (u,v) with confidence smin and finds the epipole using only the points with confidence above the threshold thresh 
    (for both sampling and finding inliers)
    params:
        @u: np.array(h,w)
        @v: np.array(h,w)
        @smin: np.array(h,w)
    return value:
        @best_ep: np.array(3,)
        @inliers: np.array(n,) 
    
    u, v and smin are (h,w), thresh is a scalar
    output should be best_ep and inliers, which have shapes, respectively (3,) and (n,) 
    '''
    
    x =[]
    y =[]
    print("check1")
    for i in range(-256, 256):
        for j in range(-256, 256):
            x.append(j)
            y.append(i)
    
    x = np.array(x).flatten() #.reshape((512,512))
    y = np.array(y).flatten() #.reshape((512,512))
    u_correct = u[smin>thresh]
    u_correct.reshape([-1,1])
    v_correct = v[smin>thresh]
    v_correct.reshape([-1,1])

    Flatted_smin = smin.flatten()
    Original_indices = np.where(Flatted_smin>thresh)
    Original_indices = Original_indices[0]
    z = np.zeros((u_correct.shape[0]))
    U = np.hstack((u_correct,v_correct,z))
    x_updated = x[Flatted_smin > thresh]
    y_updated = y[Flatted_smin > thresh]
    o = np.ones((x_updated.shape[0])).T
    Xp = np.vstack((x_updated,y_updated,o))
    U = U.reshape((Xp.shape[0],Xp.shape[1]))
    Xp_x_U = np.cross(Xp.T,U.T)

    sample_size = 2

    eps = 10**-2

    best_num_inliers = -1
    best_inliers = None
    best_ep = None

    for i in range(num_iterations): #Make sure to vectorize your code or it will be slow! Try not to introduce a nested loop inside this one
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(0,np.sum((smin>thresh))))
        sample_indices = permuted_indices[:sample_size] #indices for thresholded arrays you find above
        test_indices = permuted_indices[sample_size:] #indices for thresholded arrays you find above

        e=Xp_x_U[sample_indices]
        U_,S,Vt_ = np.linalg.svd(e)
        ep = Vt_.T[:,-1]

        E = np.dot(Xp_x_U[test_indices], ep)
        distance = np.abs(E)
        
        test_inliers = test_indices[np.where(distance <eps)[0]]
        inliers = np.append(sample_indices, test_inliers)
        inliers = Original_indices[inliers]

        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_ep = ep
            best_inliers = inliers

    return best_ep, best_inliers


