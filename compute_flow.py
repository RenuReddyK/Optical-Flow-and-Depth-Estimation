import numpy as np
import pdb

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
        @x: int
        @y: int
    return value:
        flow: np.array(2,)
        conf: np.array(1,)
    """
    """
    STUDENT CODE BEGINS
    """
    C_I_x = []
    C_I_y = []
    It_x =[]
    #It_y = []
    #print(Ix.shape)
    for i in [x+2, x+1, x, x-1, x-2]:
        for j in [y+2, y+1, y, y-1, y-2]:
            if 0 <= i < Ix.shape[0] and 0 <= j < Iy.shape[1]:
                C_I_x_1 = Ix[j,i]
                C_I_x.append(C_I_x_1)
                It_x.append(It[j,i])
                C_I_y_1 = Iy[j,i]
                C_I_y.append(C_I_y_1)
                #It_y.append(It[i,j])
    #print(c)
    C_I_x = np.array(C_I_x)
    C_I_y = np.array(C_I_y)
    new_It = (np.array(It_x)).T
    new_It = new_It.reshape((new_It.shape[0],1))
    A = np.vstack((C_I_x,C_I_y)).T
    #print("A",A.shape)
    
    B = -new_It
    #print("B",B.shape)
    X = np.linalg.lstsq(A,B,rcond = None)[0]
    flow = X
    flow = flow.reshape((flow.shape[0],))
    #print("Flow",flow)
    U, D, Vt = np.linalg.svd(A)
    conf = np.min(D)
    #print("conf",conf)
    """
    STUDENT CODE ENDS
    """
    return flow, conf


def flow_lk(Ix, Iy, It, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
    return value:
        flow: np.array(h, w, 2)
        conf: np.array(h, w)
    """
    image_flow = np.zeros([Ix.shape[0], Ix.shape[1], 2])
    confidence = np.zeros([Ix.shape[0], Ix.shape[1]])
    for x in range(Ix.shape[1]):
        for y in range(Ix.shape[0]):
            flow, conf = flow_lk_patch(Ix, Iy, It, x, y)
            image_flow[y, x, :] = flow
            confidence[y, x] = conf
    return image_flow, confidence

# if __name__ == "__main__":
#     Ix = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]).reshape((5,5))
#     Iy = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]).reshape((5,5))
#     print("Ix",Ix)
#     print("Iy",Iy)
#     C_I_x =[]
#     C_I_y = []
#     for j in [4,3,2,1,0]:
#         for i in [0,1,2,3,4]:
#             C_I_x_1 = Ix[i,j]
#             C_I_y_1 = Iy[i,j]
#             C_I_x.append(C_I_x_1)
#             C_I_y.append(C_I_y_1)
#     C_I_x = np.array(C_I_x)
#     print(C_I_x)
#     C_I_y = np.array(C_I_y)
#     A = np.vstack((C_I_x,C_I_y))
#     print(A)
#     # B = np.array([1,2,3,4,5]).T
#     # X = np.linalg.lstsq(A,B)[0]
#     # flow = X
#     # U, D, Vt = np.linalg.svd(A)
#     # print(flow)
#     # conf = np.min(np.diag(D))
    

