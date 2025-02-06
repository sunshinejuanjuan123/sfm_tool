import argparse
import os
from tqdm import tqdm
import json
import shutil
import numpy as np
import cv2
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R
from sfm_tools.feature_extract_match.model.read_write_model import Image, Camera, rotmat2qvec, write_model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="use meta json to creat colmap sparse init models"
    )
    parser.add_argument("--data_root", help="path to uniscene data")
    parser.add_argument("--output_path", help="path to 3dgs format results")
    args = parser.parse_args()

    data_root = args.data_root 
    output_path = args.output_path

    unisceneproto = os.path.join(data_root, "plannerGt/unisceneproto.json")
    uniscene = json.load(open(unisceneproto, "r"))
    
    cam_info_all = {'center_camera_fov120': {'colmap_id': 1},
                    'left_front_camera': {'colmap_id': 2},
                    'left_rear_camera': {'colmap_id': 3},
                    'right_front_camera': {'colmap_id': 4},
                    'right_rear_camera': {'colmap_id': 5},
                    'rear_camera': {'colmap_id': 6},
                    'center_camera_fov30': {'colmap_id': 7}}

    for cam_info in uniscene['cameras']:
        if cam_info['camera_name'] in cam_info_all.keys():

            cam_info_all[cam_info['camera_name']]['id'] = cam_info['camera_id']
            
            # intrinsic fx, fy, cx, cy
            fx, fy, cx, cy = cam_info['intrinsic']['fx'], cam_info['intrinsic']['fy'], cam_info['intrinsic']['cx'], cam_info['intrinsic']['cy']
            cam_info_all[cam_info['camera_name']]['intrinsic'] = np.array([fx, fy, cx, cy])
            
            # extrinsic 4*4
            quat = cam_info["extrinsic"]["quaternion"]
            trsl = cam_info["extrinsic"]["translation"]
            cam_info_all[cam_info['camera_name']]['extrinsic'] = np.eye(4)
            cam_info_all[cam_info['camera_name']]['extrinsic'][:3, :3] = R.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
            cam_info_all[cam_info['camera_name']]['extrinsic'][:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) @ cam_info_all[cam_info['camera_name']]['extrinsic'][:3, :3]
            cam_info_all[cam_info['camera_name']]['extrinsic'][:3, 3] = np.array([trsl["x"], trsl["y"], trsl["z"]]) 
    
    cam_id_2_name = {v['id']: k for k, v in cam_info_all.items()}
    
    pose_info = {}
    for ego_info in tqdm(uniscene['ego_status']):
        timestamp = int(round(ego_info['timestamp'], 3)*1000)
        quat = ego_info['ego_orientation']
        trsl = ego_info['ego_position']
        pose_info[timestamp] = np.eye(4)
        pose_info[timestamp][:3, :3] = R.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
        pose_info[timestamp][:3, 3] = np.array([trsl["x"], trsl["y"], trsl["z"]])   

    os.makedirs(output_path, exist_ok=True)
    cameras, images, points3D = {}, {}, {}
    image_id = 0
    
    # select all frames
    for sensor_info in tqdm(uniscene['sensor_frames']):
        timestamp = int(round(sensor_info['timestamp'], 3)*1000)
        lidar2enu = pose_info[timestamp]
        for camera_data in sensor_info['camera_data']:
            if camera_data['sensor_id'] in cam_id_2_name.keys():
                cam = cam_id_2_name[camera_data['sensor_id']]
                sensor2lidar = cam_info_all[cam]['extrinsic']
                dst_cam = os.path.join(output_path, "images", cam)
                if not os.path.exists(dst_cam):
                    os.makedirs(dst_cam, exist_ok=True)
                src_img_name = camera_data['file_path'].split("/")[-1]
                src_img_name, suffix = os.path.splitext(src_img_name)
                src_img_abs_path = os.path.join(data_root, camera_data['file_path'])
                rgb_img = cv2.imread(src_img_abs_path)
                h, w, _ = rgb_img.shape

                K = cam_info_all[cam]['intrinsic']
                fx, fy, cx, cy = K[0], K[1], K[2], K[3]

                if cam in ['center_camera_fov30', 'center_camera_fov120']:
                    h_new, w_new = 1280, 1920
                    rgb_img = cv2.resize(rgb_img, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
                    fx *= w_new / w
                    fy *= h_new / h
                    cx *= w_new / w
                    cy *= h_new / h 
                    cv2.imwrite(os.path.join(dst_cam, str(timestamp)+suffix), rgb_img)
                    h, w = h_new, w_new
                else:
                    shutil.copy(src_img_abs_path, os.path.join(dst_cam, str(timestamp)+suffix))

                sensor2enu = np.matmul(lidar2enu, sensor2lidar)
                enu2sensor = np.linalg.inv(sensor2enu)
                qvec = rotmat2qvec(enu2sensor[:3, :3])
                tvec = enu2sensor[:3, 3]

                image_id += 1
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=cam_info_all[cam]['colmap_id'],
                    name=cam+"/"+str(timestamp)+suffix,
                    xys=np.array([]),
                    point3D_ids=np.array([])
                )
                
                instrinsic = np.array([fx, fy, cx, cy])
                cameras[cam_info_all[cam]['colmap_id']] = Camera(
                    id=cam_info_all[cam]['colmap_id'],
                    model="PINHOLE",
                    width=w,
                    height=h,
                    params=instrinsic
                )

    cameras = OrderedDict(sorted(cameras.items(), key=lambda x: x[0]))
    
    # write colmap sparse init model
    output_model_path = os.path.join(output_path, "colmap/sparse_init")
    os.makedirs(output_model_path, exist_ok=True)
    
    write_model(
        cameras = cameras,
        images = images,
        points3D = points3D,
        path = output_model_path,
        ext = ".txt"
    )



