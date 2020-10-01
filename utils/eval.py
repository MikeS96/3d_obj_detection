# Code written by Miguel Saavedra & Gustavo Salazar, 2020.
# Compute 3D IoU and BeV IoU of for 3D object detection tasks

import numpy as np
from shapely.geometry import Polygon
import shapely.ops as so
import matplotlib.pyplot as plt

def volume_box3d(corners: np.array):
    ''' 
    Compute the volume of a 3D bounding box given their 8 points.
    :param corners: <np.float: 8, 3> point cloud mapped in the image frame.
    :return Volume of the current bounding box as a float.
    '''
    a = np.sqrt(np.sum((corners[0,:] - corners[3,:])**2)) # Height
    b = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2)) # Lenghth
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2)) # Width
    return a*b*c

def box3d_iou(corners1: np.array, 
              corners2: np.array, 
              vis_result: bool = False):
    ''' 
    Compute 3D bounding box IoU.
    :param corners1: <np.float: 8, 3> Bounding predicted box coordinates.
    :param corners2: <np.float: 8, 3> Bounding ground truth box coordinates.
    :param vis_result: Visualize union of bounding boxes in XY plane.
    :return iou: 3D IoU between ground truth and prediction.
    :return iou_2d: BEV IoU in XY plane.
    '''

    # Compute corners within xy plane for ground truth and bounding box
    rect1_xy = [[corners1[i,0], corners1[i,1]] for i in [0, 1, 5, 4]]
    rect2_xy = [[corners2[i,0], corners2[i,1]] for i in [0, 1, 5, 4]] 

    # Create the polygon out of the given points
    poly1 = Polygon(rect1_xy).convex_hull  
    poly2 = Polygon(rect2_xy).convex_hull

    # Area of each polygon
    area1 = poly1.area
    area2 = poly2.area
       
    if not poly1.intersects(poly2):
        inter_area = 0 
    else:
        inter_area = poly1.intersection(poly2).area 
        # Plot the union of the shapes if desired
        if vis_result:
            #cascaded union can work on a list of shapes
            new_shape = so.cascaded_union([poly1,poly2])
            # exterior coordinates split into two arrays, xs and ys
            xs, ys = new_shape.exterior.xy
            #plot it
            fig, axs = plt.subplots()
            axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')
            plt.show() #if not interactive

        
    # Computing IoU in 2D (XY plane)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    # Intersection in Z
    zmax = min(corners1[0,2], corners2[0,2])
    zmin = max(corners1[3,2], corners2[3,2])
    
    # Intersection volume
    inter_vol = inter_area * max(0.0, zmax - zmin)

    # Computing the volume of the 3D bounding boxes
    vol1 = volume_box3d(corners1)
    vol2 = volume_box3d(corners2)

    # 3D IoU volume computation
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d