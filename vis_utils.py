# Code written by Miguel Saavedra & Gustavo Salazar, 2020.

"""
Set 
Note: Util functions to plot the 3D bounding box in the image frame
      Describe longer Gus
"""

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import rcParams
from shapely.geometry import MultiPoint, box

from pyquaternion import Quaternion
import os.path as osp
from nuscenes import NuScenes

# Utils for Lidar and Radar
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.scripts.export_2d_annotations_as_json import get_2d_boxes, post_process_coords

from typing import Tuple, List, Dict, Union

def plot_3d_image_(nusc: NuScenes,
                   camera_token: str,
                   label: str,
                   sample_token: str,
                   bbox_3d,
                   pointsensor_channel: str = 'LIDAR_TOP'):    
    """
    Given a 3D box, this method plots the Bounding box in the image frame
    :param nusc: NuScenes instance.
    :param camera_token: Camera sample_data token.
    :param label: Class' label.
    :param sample_token: Sample data token belonging to a camera keyframe.
    :param bbox_3d: box object with the 3D bbox info.
    :param pointsensor_channel: Channel of the point cloud sensor.
    """
    
    # Sample record
    sample_record = nusc.get('sample', sample_token)
    # Sample cam sensor
    cam = nusc.get('sample_data', camera_token)
    # Sample point cloud
    pointsensor_token = sample_record['data'][pointsensor_channel]
    pointsensor = nusc.get('sample_data', pointsensor_token)

    # Obtain transformation matrices
    # From camera to ego
    cs_rec_cam = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    # From ego to world coordinate frame Cam
    pose_rec_cam = nusc.get('ego_pose', cam['ego_pose_token'])

    # From LiDAR to ego
    cs_rec_point = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    # Transformation metadata from ego to world coordinate frame
    pose_rec_point = nusc.get('ego_pose', pointsensor['ego_pose_token'])

    # Transform box to camera frame
    # From LiDAR point sensor to ego vehicle
    bbox_3d.rotate(Quaternion(cs_rec_point['rotation']))
    bbox_3d.translate(np.array(cs_rec_point['translation']))

    #  Move box to world coordinate frame
    bbox_3d.rotate(Quaternion(pose_rec_point['rotation']))
    bbox_3d.translate(np.array(pose_rec_point['translation']))

    # Move box to ego vehicle coord system.
    bbox_3d.translate(-np.array(pose_rec_cam['translation']))
    bbox_3d.rotate(Quaternion(pose_rec_cam['rotation']).inverse)

    #  Move box to sensor coord system (cam).
    bbox_3d.translate(-np.array(cs_rec_cam['translation']))
    bbox_3d.rotate(Quaternion(cs_rec_cam['rotation']).inverse)

    # Draw vertical lines of 3D bounding box with this method
    def draw_rect(selected_corners, color, axes):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axes.plot([prev[0], corner[0]], [prev[1], corner[1]], color = color, linewidth=2)
                prev = corner

    # Map corners to 2D image plane
    cs_record_calib = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    corners = view_points(bbox_3d.corners(), view = np.array(cs_record_calib['camera_intrinsic']), normalize = True)[:2, :]

    # Create axis and image
    fig, axes = plt.subplots(1, 1, figsize=(18, 9))
    # Open image of the interest camera
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))
    axes.imshow(im)

    # Draw the sides of the bounding box
    colors = ('b', 'r', 'k')

    for i in range(4):
        axes.plot([corners.T[i][0], corners.T[i + 4][0]],
                [corners.T[i][1], corners.T[i + 4][1]],
                color=colors[2], linewidth = 2)
        
    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0], axes)
    draw_rect(corners.T[4:], colors[1], axes)

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    axes.plot([center_bottom[0], center_bottom_forward[0]],
            [center_bottom[1], center_bottom_forward[1]],
            color=colors[0], linewidth=2)
        
    axes.set_title(nusc.get('sample_data', cam['token'])['channel'])
    axes.axis('off')
    axes.set_aspect('equal')
