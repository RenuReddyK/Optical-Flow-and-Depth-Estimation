import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_flow(image, flow_image, confidence, threshmin=10):
    """
    params:
        @img: np.array(h, w)
        @flow_image: np.array(h, w, 2)
        @confidence: np.array(h, w)
        @threshmin: confidence must be greater than threshmin to be kept
    return value:
        None
    """
    
    xx = []
    yy = []
    flow_xx = []
    flow_yy = []
    for i in range(confidence.shape[0]):
        for j in range(confidence.shape[1]):
            if confidence[i,j] > threshmin:
                flow_xx.append(flow_image[i,j][0])
                flow_yy.append(flow_image[i,j][1])
                xx.append(j)
                yy.append(i)
    x = np.array(xx)
    y = np.array(yy)
    flow_x = np.array(flow_xx)
    flow_y = np.array(flow_yy)
    
    plt.imshow(image, cmap='gray')
    plt.quiver(x, y, (flow_x*10).astype(int), (flow_y*10).astype(int), 
                    angles='xy', scale_units='xy', scale=1., color='red', width=0.001)
    
    return





    

