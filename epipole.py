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

    """
    You can do the thresholding on smin using thresh outside the RANSAC loop here. 
    Make sure to keep some way of going from the indices of the arrays you get below back to the indices of a flattened u/v/smin
    STUDENT CODE BEGINS
    """
    
    print("v",v.shape)
    print("u",u.shape)
    x =[]
    y =[]
    print("check1")
    for i in range(-256, 256):
        for j in range(-256, 256):
            x.append(j)
            y.append(i)
    
    x = np.array(x).flatten() #.reshape((512,512))
    y = np.array(y).flatten() #.reshape((512,512))
    print("check2")
    u_correct = u[smin>thresh]
    u_correct.reshape([-1,1])
    v_correct = v[smin>thresh]
    v_correct.reshape([-1,1])
    print("check3")

    Flatted_smin = smin.flatten()
    Original_indices = np.where(Flatted_smin>thresh)
    Original_indices = Original_indices[0]
    print("check4")
    z = np.zeros((u_correct.shape[0]))
    U = np.hstack((u_correct,v_correct,z))
    print("checkcheck")
    x_updated = x[Flatted_smin > thresh]
    y_updated = y[Flatted_smin > thresh]
    print("checkcheckcheck")
    o = np.ones((x_updated.shape[0])).T
    print("check5")
    Xp = np.vstack((x_updated,y_updated,o))
    #Xp = Xp.flatten()
    print(Xp.shape)
    U = U.reshape((Xp.shape[0],Xp.shape[1]))
    print(U.shape)
    Xp_x_U = np.cross(Xp.T,U.T)
    print("check6")
    # xp = np.array(xp)
    # yp = np.array(yp)
    # 
    
    # x_p = np.vstack((xp,yp,o))


    # z = np.zeros((u.shape[0],1)).T
    # U = np.vstack(u,v,z)
    """ 
    STUDENT CODE ENDS
    """

    sample_size = 2

    eps = 10**-2

    best_num_inliers = -1
    best_inliers = None
    best_ep = None

    for i in range(num_iterations): #Make sure to vectorize your code or it will be slow! Try not to introduce a nested loop inside this one
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(0,np.sum((smin>thresh))))
        sample_indices = permuted_indices[:sample_size] #indices for thresholded arrays you find above
        test_indices = permuted_indices[sample_size:] #indices for thresholded arrays you find above

        """
        STUDENT CODE BEGINS
        """
        e=Xp_x_U[sample_indices]
        U_,S,Vt_ = np.linalg.svd(e)
        ep = Vt_.T[:,-1]

        E = np.dot(Xp_x_U[test_indices], ep)
        distance = np.abs(E)
        #print(distance)
        
        test_inliers = test_indices[np.where(distance <eps)[0]]
        inliers = np.append(sample_indices, test_inliers)
        inliers = Original_indices[inliers]

        """
        STUDENT CODE ENDS
        """

        #NOTE: inliers need to be indices in flattened original input (unthresholded), 
        #sample indices need to be before the test indices for the autograder
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_ep = ep
            best_inliers = inliers

    print(best_ep.shape)
    print(best_inliers.shape)

    return best_ep, best_inliers


# if __name__ == "__main__":
#     q=[]
#     w =[]
#     for i in range(-256,256,1):
#         for j in range(-256,256,1):
#             q.append(i)
#             w.append(j)
#     print(len(q))
#     print(len(w))
#     q = np.array(q).reshape((512,512))
#     w = np.array(w).reshape((512,512))
#     print("q",q)
#     print("w",w)
#     print(np.vstack((q,w)))
#     yy = np.arange(0,512*512)//512 -256
#     xx = np.arange(0,512*512)%512 -256
#     print(yy)
#     print(xx)
#     z = np.zeros((q.shape[0],1))
#     print(z.shape)
#     
