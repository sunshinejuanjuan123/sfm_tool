import argparse
import os
from tqdm import tqdm
import json
from datetime import datetime
import shutil
import numpy as np
import cv2
from sfm_tools.feature_extract_match.model.read_write_model import Image, Camera, rotmat2qvec, write_model
from mapxtoolkit.utils.transform import Transform
from scipy.spatial.transform import Rotation

def convert_timestamp(timestamp):
    timestamp_obj = datetime.strptime(timestamp, '%Y-%m-%d-%H-%M-%S-%f')
    unix_timestamp = int(timestamp_obj.timestamp()*10)
    return unix_timestamp

def convert_pose(lidar_pose, t):
    Translation = np.identity(4)
    Translation[:3, :3] = Rotation.from_euler("xyz", lidar_pose[3:6]).as_matrix()
    Translation[:3, 3] = t.LLH2ENU(lidar_pose[:3])
    return Translation

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="use meta json to creat colmap sparse init models"
    )
    parser.add_argument("--data_root", help="path to pvb data")
    parser.add_argument("--output_path", help="path to 3dgs format results")
    args = parser.parse_args()

    data_root = args.data_root 
    output_path = args.output_path

    meta_dir = os.path.join(data_root, "meta_json")
    
    os.makedirs(output_path, exist_ok=True)

    cameras, images, points3D = {}, {}, {}

    # select all frame idx 
    json_idxs = sorted(os.listdir(meta_dir))
    
    cameras_lists = ['center_camera_fov120',
                     'left_front_camera',
                     'left_rear_camera',
                     'right_front_camera',
                     'right_rear_camera',
                     'rear_camera',
                     'center_camera_fov30']
    
    image_id = 0
    
    for jdx, json_idx in enumerate(tqdm(json_idxs)):
        meta_info = json.load(open(os.path.join(meta_dir, json_idx), "r"))
        image_info = meta_info['meta_info']

        # pose
        if jdx == 0:
            t = Transform(*image_info["lidar_pose"][:3])
            lidar2enu = convert_pose(image_info["lidar_pose"], t)
        else:
            lidar2enu = convert_pose(image_info["lidar_pose"], t)
        
        lidar_abs_path = os.path.join(data_root, image_info['lidar_path']['car_center'])
        time_stamp = convert_timestamp(lidar_abs_path.split("/")[-1][:-4])
        
        for idx, cam in enumerate(cameras_lists):

            # extrinsic
            sensor2lidar = image_info['sensor2lidar'][cam]
            img_abs_path = os.path.join(data_root, image_info['cam_path'][cam])
            img_name = img_abs_path.split("/")[-1]

            rgb_img = cv2.imread(img_abs_path)
            h, w, _ = rgb_img.shape
            
            dst_data_img = os.path.join(output_path, 'images', cam)
            os.makedirs(dst_data_img, exist_ok=True)

            img_prefix, suffix = os.path.splitext(img_name)
            img_time_stamp = convert_timestamp(img_prefix)
            img_name = str(img_time_stamp) + suffix

            # intrinsic
            K = np.array(image_info['sensor2img'][cam])
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

            # resize center_camera image size 
            if cam in ['center_camera_fov30', 'center_camera_fov120']:
                h_new, w_new = 1280, 1920
                rgb_img = cv2.resize(rgb_img, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
                fx *= w_new / w
                fy *= h_new / h
                cx *= w_new / w
                cy *= h_new / h

                cv2.imwrite(os.path.join(dst_data_img, img_name), rgb_img)
                h, w = h_new, w_new
            else:
                shutil.copy(img_abs_path, os.path.join(dst_data_img, img_name))
            
            sensor2enu = np.matmul(lidar2enu, sensor2lidar)

            w2c = np.linalg.inv(sensor2enu)
            qvec = rotmat2qvec(w2c[:3, :3])
            tvec = w2c[:3, 3]

            image_id += 1
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id = idx+1,
                name = cam + "/" + img_name,
                xys=np.array([]),
                point3D_ids=np.array([])
            )
            instrinsic = np.array([fx, fy, cx, cy])
            cameras[idx+1] = Camera(
                id=idx+1,
                model="PINHOLE",
                width=w,
                height=h,
                params=instrinsic
            )
            
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



