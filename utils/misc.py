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
import open3d as o3d

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

def middle_func(axis_array):
    """
    Given an array for an axis calc the middle point.
    :param axis_array: array to calc middle.
    :return middle_result: operation result.
    """

    middle_result = ((np.max(axis_array)-np.min(axis_array))//2)+np.min(axis_array)

    return middle_result

def load_pcl_txt(dataroot: str,
                shape_info):    
    """
    Given a dataroot this method load the PointCloud from a .txt file and reshape to its original size.
    :param dataroot: dataroot to load the file.
    :return data_pcl: PointCloud from file.
    """
    data_pcl = np.loadtxt(dataroot).reshape(shape_info[0], shape_info[1])
    
    return data_pcl


def save_to_txt(dataroot: str ,
                data_txt):    
    """
    Given a dataroot and an array with the info pointclouds in the camera frame, this method 
    stores the data in a .txt file.
    :param dataroot: dataroot to store the file.
    :param data_txt: array with the info.
    """

    a_file = open(dataroot, "w")

    for row in data_txt:
        np.savetxt(a_file, row)

    a_file.close()

def load_file(dataroot: str):    
    """
    Given a dataroot this method load the data from a .txt file in JSON format.
    :param dataroot: dataroot to load the file.
    :return data: dictionary with the info from file.
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
                  pointsensor_token: str,
                  annotation_token: str,
                  pointsensor_channel: str = 'LIDAR_TOP'):
    """
    Given an annotation token (3d detection in world coordinate frame) and pointsensor sample_data token,
    transform the label from world-coordinate frame to LiDAR.
    :param nusc: NuScenes instance.
    :param camera_token: Lidar/radar sample_data token.
    :param annotation_token: Camera sample_annotation token.
    :param camera_channel: Laser channel name, e.g. 'LIDAR_TOP'.
    :return box with the labels for the 3d detection task in the LiDAR channel frame.
    """
    
    # Point LiDAR sample        
    point_data = nusc.get('sample_data', pointsensor_token) # Sample LiDAR info
        
    # From LiDAR to ego
    cs_rec = nusc.get('calibrated_sensor', point_data['calibrated_sensor_token'])
    # Transformation metadata from ego to world coordinate frame
    pose_rec = nusc.get('ego_pose', point_data['ego_pose_token'])
    
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
                             dist_thresh: float = 0.1,
                             visualize: bool = False) -> Tuple:
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
    plane.
    :param nusc: NuScenes instance.
    :param bbox: object coordinates in the current image.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :param dist_thresh: Threshold to consider points within floor plane.
    :return (points_ann <np.float: 2, n)>, coloring_ann <np.float: n>, ori_points_ann<np.float: 3, n)>, image <Image>).
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

    # Save original points (X, Y and Z) coordinates in LiDAR frame
    ori_points = pc.points[:3, :].copy()

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
    
    # Retrieve the color from the depth.
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    # Normalization means to divide the X and Y coordinates by the depth
    # The output dim (3 x n_points) where the 3rd row are ones.
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    
    # bounding box coordinates
    min_x, min_y, max_x, max_y = [int(points_b) for points_b in bbox]

    # Floor segmentation
    points_img, coloring_img, ori_points_img = segment_floor(points, coloring, ori_points,
                                                            (im.size[0], im.size[1]), dist_thresh, 1.0)

    # Filter the points within the annotaiton
    # Create a mask of bools the same size of depth points
    mask_ann = np.ones(coloring_img.shape[0], dtype=bool)
    # Keep points such as X coordinate is bigger than bounding box minimum coordinate
    mask_ann = np.logical_and(mask_ann, points_img[0, :] > min_x + 1)
    # remove points such as X coordinate is bigger than bounding box maximum coordinate
    mask_ann = np.logical_and(mask_ann, points_img[0, :] < max_x - 1)
    # Keep points such as Y coordinate is bigger than bounding box minimum coordinate
    mask_ann = np.logical_and(mask_ann, points_img[1, :] > min_y + 1)
    # remove points such as Y coordinate is bigger than bounding box maximum coordinate
    mask_ann = np.logical_and(mask_ann, points_img[1, :] < max_y - 1)
    # Keep only the interest points
    points_ann = points_img[:, mask_ann]
    coloring_ann = coloring_img[mask_ann]
    ori_points_ann = ori_points_img[:, mask_ann]
    
    if visualize:
        plt.figure(figsize=(9, 16))
        plt.imshow(im)
        plt.scatter(points_ann[0, :], points_ann[1, :], c = coloring_ann, s = 5)
        plt.axis('off')
        
    return points_ann, coloring_ann, ori_points_ann, im

def segment_floor(points: np.array,
                  coloring: np.array,
                  ori_points: np.array,
                  imsize: Tuple[float, float] = (1600, 900),
                  dist_thresh: float = 0.3,
                  min_dist: float = 1.0) -> Tuple:
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
    plane.
    :param points: <np.float: 2, n> point cloud mapped in the image frame
    :param coloring: <np.float: n> depth of the point cloud in the camera frame
    :param ori_points: <np.float: 3, n> point cloud in LiDAR coordinate frame
    :param imsize: Size of image to render. The larger the slower this will run.
    :param dist_thresh: Threshold to consider points within floor plane.
    :param min_dist: Distance from the camera below which points are discarded.
    :return (points_img <np.float: 2, n)>, coloring_img <np.float: n>, ori_points_img<np.float: 3, n)>).
    """

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    mask_img = np.ones(coloring.shape[0], dtype=bool)
    mask_img = np.logical_and(mask_img, coloring > min_dist)
    mask_img = np.logical_and(mask_img, points[0, :] > 1)
    mask_img = np.logical_and(mask_img, points[0, :] < imsize[0] - 1)
    mask_img = np.logical_and(mask_img, points[1, :] > 1)
    mask_img = np.logical_and(mask_img, points[1, :] < imsize[1] - 1)

    # Filter the points within the image with the generated mask
    points_img = points[:, mask_img]
    coloring_img = coloring[mask_img]
    ori_points_img = ori_points[:, mask_img]
    
    # Segmenting the point cloud's floor
    lidar_points = np.asarray(ori_points_img.T, np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points)

    # inliers are the indeces of the the inliers (plane points)
    plane_model, inliers = pcd.segment_plane(distance_threshold = dist_thresh,
                                            ransac_n = 20,
                                            num_iterations = 1000)
    # Obtaining the plane's equation
    [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # These lines plot the point cloud inliers and outliers
    # inlier_cloud = pcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([outlier_cloud])

    # Create a floor mask with the indeces inverted, points that are not in plane are of interest
    mask_floor = np.arange(points_img.shape[1])
    mask_floor = np.full(points_img.shape[1], True, dtype=bool)
    mask_floor[inliers] = False

    # Filter the points which are floor
    points_img = points_img[:, mask_floor]
    coloring_img = coloring_img[mask_floor]
    ori_points_img = ori_points_img[:, mask_floor]
        
    return points_img, coloring_img, ori_points_img

def parse_features_to_numpy(path: str):
    """
    Given a features' vector path, read the txt file and parse the info as a numpy array
    :param path: Path to the features' vector file.
    :return (descriptor_array <np.float: 1, 640)>.
    """

    # Reading the file
    file = open(path,"r")
    file = file.read()
    # Replace undesired elements
    file = file.replace('(', '').replace(')', '').replace(' ', '')
    # Split the string and create a numpy array with floats
    descriptor_array = np.array(file.split(',')).astype(float)
    
    return descriptor_array

