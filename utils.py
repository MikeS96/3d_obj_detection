# Code written by Miguel Saavedra & Gustavo Salazar, 2020.

"""
Set 
Note: Util functions to sample 2d bounding boxes, 3d targets w.r.t to camera frame and point clouds of a specific instance.
      Describe longer Gus
"""

import numpy as np
import cv2
from PIL import Image
import json

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

def save_to_bin(dataroot: str ,
                data_bin):    
    """
    Given a dataroot and an array with the info pointclouds in the camera frame, this method 
    stores the data in a .bin file.
    :param dataroot: dataroot to store the file.
    :param data_bin: array with the info.
    """
    with open(dataroot, 'wb') as f:
        f.write(data_bin)

def load_file(dataroot: str):    
    """
    Given a dataroot this method load the data from a .txt file in JSON format.
    :param dataroot: dataroot to load the file.
    :return datan: dictionary with the info from file.
    """
    with open(dataroot) as json_file:
        data = json.load(json_file)
    return data

def save_in_file(dataroot: str ,
                 data_json: dict):    
    """
    Given a dataroot and a dictionary with the info, this method stores the data
    in a .txt file in JSON format.
    :param dataroot: dataroot to store the file.
    :param data_json: dictionary with the info.
    """
    with open(dataroot, 'w') as outfile:
        json.dump(data_json, outfile)
    

def bbox_3d_to_2d(nusc: NuScenes,
                  camera_token: str,
                  annotation_token: str,
                  visualize: bool = False) -> List:

    """
    Get the 2D annotation bounding box for a given `sample_data_token`. return None if no
    intersection (bounding box).
    :param nusc: NuScenes instance.
    :param camera_token: Camera sample_data token.
    :param annotation_token: Sample data token belonging to a camera keyframe.
    :param visualize: bool to plot the resulting bounding box.
    :return: List of 2D annotation record that belongs to the input `sample_data_token`
    """
    
    # Obtain camera sample_data
    cam_data = nusc.get('sample_data', camera_token)

    # Get the calibrated sensor and ego pose record to get the transformation matrices.
    
    # From camera to ego
    cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    # From ego to world coordinate frame
    pose_rec = nusc.get('ego_pose', cam_data['ego_pose_token'])
    # Camera intrinsic parameters
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])
    
    # Obtain the annotation from the token
    annotation_metadata =  nusc.get('sample_annotation', annotation_token)

    # Get the box in global coordinates from sample ann token 
    box = nusc.get_box(annotation_metadata['token'])

    # Mapping the box from world coordinate-frame to camera sensor
    
    # Move them to the ego-pose frame.
    box.translate(-np.array(pose_rec['translation']))
    box.rotate(Quaternion(pose_rec['rotation']).inverse)

    # Move them to the calibrated sensor frame.
    box.translate(-np.array(cs_rec['translation']))
    box.rotate(Quaternion(cs_rec['rotation']).inverse)

    # Filter out the corners that are not in front of the calibrated sensor.
    corners_3d = box.corners() # 8 corners of the 3d bounding box
    in_front = np.argwhere(corners_3d[2, :] > 0).flatten() # corners that are behind the sensor are removed
    corners_3d = corners_3d[:, in_front]

    # Project 3d box to 2d.
    corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()
    
    # Filter points that are outside the image
    final_coords = post_process_coords(corner_coords)

    if final_coords is None:
        return None

    min_x, min_y, max_x, max_y = [int(coord) for coord in final_coords]
    
    if visualize:
        # Load image from dataroot
        img_path = osp.join(nusc.dataroot, cam_data['filename'])
        img = cv2.imread(img_path, 1)

        # Draw rectangle on image with coords
        img_r = cv2.rectangle(img, (min_x,min_y),(max_x,max_y),(255, 165, 0) , 3)
        img_r = img_r[:, :, ::-1]

        plt.figure(figsize=(12, 4), dpi=100)
        plt.imshow(img_r)
        plt.show()
        
    return final_coords

def get_camera_data(nusc: NuScenes,
                    annotation_token: str,
                    box_vis_level: BoxVisibility = BoxVisibility.ANY):
    
    """
    Given an annotation token (3d detection in world coordinate frame) this method 
    returns the camera in which the annotation is located. If the box is splitted 
    between 2 cameras, it brings the first one found.
    :param nusc: NuScenes instance.
    :param annotation_token: Annotation token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :return camera channel.
    """
    #Get sample annotation
    ann_record = nusc.get('sample_annotation', annotation_token)

    sample_record = nusc.get('sample', ann_record['sample_token'])
  
    boxes, cam = [], []

    #Stores every camera
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]

    #Try with every camera a match for the annotation
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level,
                                                selected_anntokens=[annotation_token])
        if len(boxes) > 0:
            break  # Breaks if find an image that matches
    assert len(boxes) < 2, "Found multiple annotations. Something is wrong!"
    
    return cam

def target_to_cam(nusc: NuScenes,
                  camera_token: str,
                  annotation_token: str,
                  camera_channel: str = 'CAM_FRONT'):
    """
    Given an annotation token (3d detection in world coordinate frame) and camera sample_data token,
    transform the label from world-coordinate frame to camera.
    :param nusc: NuScenes instance.
    :param camera_token: Camera sample_data token.
    :param annotation_token: Camera sample_annotation token.
    :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
    :return box with the labels for the 3d detection task in the camera channel frame.
    """
    
    # Camera sample        
    cam_data = nusc.get('sample_data', camera_token) # Sample camera info
        
    # From camera to ego
    cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    # Transformation metadata from ego to world coordinate frame
    pose_rec = nusc.get('ego_pose', cam_data['ego_pose_token'])
    
    # Obtain the annotation from the token
    annotation_metadata =  nusc.get('sample_annotation', annotation_token)
    
    # Obtain box parameters
    box = nusc.get_box(annotation_metadata['token'])
                                           
    # Move them to the ego-pose frame.
    box.translate(-np.array(pose_rec['translation']))
    box.rotate(Quaternion(pose_rec['rotation']).inverse)

    # Move them to the calibrated sensor frame.
    box.translate(-np.array(cs_rec['translation']))
    box.rotate(Quaternion(cs_rec['rotation']).inverse)
    
    return box

def map_pointcloud_to_image_(nusc: NuScenes,
                             bbox,
                             pointsensor_token: str,
                             camera_token: str,
                             min_dist: float = 1.0,
                             visualize: bool = False) -> Tuple:
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
    plane.
    :param nusc: NuScenes instance.
    :param bbox: object coordinates in the current image.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, ori_points<np.float: 3, n)>, image <Image>).
    """
    cam = nusc.get('sample_data', camera_token) # Sample camera info
    pointsensor = nusc.get('sample_data', pointsensor_token) # Sample point cloud
    # pcl_path is the path from root to the pointCloud file
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename']) 
    # Open the pointCloud path using the Lidar or Radar class
    if pointsensor['sensor_modality'] == 'lidar':
        # Read point cloud with LidarPointCloud (4 x samples) --> X, Y, Z and intensity
        pc = LidarPointCloud.from_file(pcl_path)
        # To access the points pc.points
    else:
        # Read point cloud with LidarPointCloud (18 x samples) --> 
        # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L296
        pc = RadarPointCloud.from_file(pcl_path)
                    
    # Open image of the interest camera
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token']) # Transformation matrix of pointCloud
    # Transform the Quaternion into a rotation matrix and use method rotate in PointCloud class to rotate
    # Map from the laser sensor to ego_pose
    # The method is a dot product between cs_record['rotation'] (3 x 3) and points (3 x points)
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    # Add the traslation vector between ego vehicle and sensor
    # The method translate is an addition cs_record['translation'] (3,) and points (3 x points)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    # Same step as before, map from ego_pose to world coordinate frame
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    # Same step as before, map from world coordinate frame to ego vehicle frame for the timestamp of the image.
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    # Same step as before, map from ego at camera timestamp to camera
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    
    # Save original points (X, Y and Z) coordinates
    ori_points = pc.points[:3, :]
    
    # Retrieve the color from the depth.
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    # Normalization means to divide the X and Y coordinates by the depth
    # The output dim (3 x n_points) where the 3rd row are ones.
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    
    # bounding box coordinates
    min_x, min_y, max_x, max_y = [int(points_b) for points_b in bbox]

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    # Create a mask of bools the same size of depth points
    mask = np.ones(depths.shape[0], dtype=bool)
    # Keep points that are at least 1m in front of the camera 
    mask = np.logical_and(mask, depths > min_dist)
    # Keep points such as X coordinate is bigger than bounding box minimum coordinate
    mask = np.logical_and(mask, points[0, :] > min_x + 1)
    # remove points such as X coordinate is bigger than bounding box maximum coordinate
    mask = np.logical_and(mask, points[0, :] < max_x - 1)
    # Keep points such as Y coordinate is bigger than bounding box minimum coordinate
    mask = np.logical_and(mask, points[1, :] > min_y + 1)
    # remove points such as Y coordinate is bigger than bounding box maximum coordinate
    mask = np.logical_and(mask, points[1, :] < max_y - 1)
    # Keep only the interest points
    points = points[:, mask]
    coloring = coloring[mask]
    ori_points = ori_points[:, mask]
    
    if visualize:
        plt.figure(figsize=(9, 16))
        plt.imshow(im)
        plt.scatter(points[0, :], points[1, :], c = coloring, s = 5)
        plt.axis('off')
        
    return points, coloring, ori_points, im