import numpy as np
import cv2
import matplotlib.pyplot as plt
from pyquaternion.quaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points
import os.path as osp
#from google.colab.patches import cv2_imshow
from nuscenes import NuScenes
#from google.colab import drive

#drive.mount('/content/drive')

nusc = NuScenes(version='v1.0-mini',dataroot='/content/drive/My Drive/AI_S/ML/NuScenes/v1.0-mini')

#select scene 9
my_scene = nusc.scene[9]

#gets first token from scene
first_sample_token = my_scene['first_sample_token']

#gets sample from token
my_sample = nusc.get('sample', first_sample_token)

#gets sample's token
my_sample_token = my_sample['token']

#which camera or sensor
sensor_camera= 'CAM_FRONT_RIGHT'

#gets sample data for sensor
cam_data = nusc.get('sample_data', my_sample['data'][sensor_camera])

#gets ann token
my_annotation_token = my_sample['anns'][15]

#gets sample ann from token
my_annotation_metadata =  nusc.get('sample_annotation', my_annotation_token)

from nuscenes.scripts.export_2d_annotations_as_json import get_2d_boxes
#get_2d_boxes(sample_data_token: str, visibilities: List[str]) -> List[OrderedDict]:

s_rec = nusc.get('sample', cam_data['sample_token'])

# Get the calibrated sensor and ego pose record to get the transformation matrices.
cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
pose_rec = nusc.get('ego_pose', cam_data['ego_pose_token'])
camera_intrinsic = np.array(cs_rec['camera_intrinsic'])


# Get the box in global coordinates from sample ann token  -------annotation --------------------
box = nusc.get_box(my_annotation_metadata['token'])

# Move them to the ego-pose frame.
box.translate(-np.array(pose_rec['translation']))
box.rotate(Quaternion(pose_rec['rotation']).inverse)

# Move them to the calibrated sensor frame.
box.translate(-np.array(cs_rec['translation']))
box.rotate(Quaternion(cs_rec['rotation']).inverse)

# Filter out the corners that are not in front of the calibrated sensor.
corners_3d = box.corners()
in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
corners_3d = corners_3d[:, in_front]

# Project 3d box to 2d.
corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

print("Box number of coords: ", np.array(corner_coords).shape)

corner = np.array(corner_coords)

#gets min and max corners
min_x = int(min(corner[:, 0]))
min_y = int(min(corner[:, 1]))
max_x = int(max(corner[:, 0]))
max_y = int(max(corner[:, 1]))

print("min_x: ",min_x)
print("min_y: ",min_y)
print("max_x: ",max_x)
print("max_y: ",max_y)

#load image from dataroot
img_path = osp.join(nusc.dataroot, cam_data['filename'])
img = cv2.imread(img_path,1)

#draw rectangle on image with coords
img_r = cv2.rectangle(img, (min_x,min_y),(max_x,max_y),(255, 165, 0) , 3)

plt.imshow(img_r)
