import argparse
import os
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from sfm_tools.feature_extract_match.model.read_write_model import read_model, write_model, Image, Point3D, rotmat2qvec

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

    for jdx, sensor_info in enumerate(tqdm(uniscene['sensor_frames'])):
            timestamp = int(round(sensor_info['timestamp'], 3)*1000)
            lidar2enu = pose_info[timestamp]
            if jdx == 0:
                for camera_data in sensor_info['camera_data']:
                    if camera_data['sensor_id'] in cam_id_2_name.keys():
                        cam = cam_id_2_name[camera_data['sensor_id']]
                        sensor2lidar = cam_info_all[cam]['extrinsic']
                        if cam == "center_camera_fov120":
                            init_pose = np.matmul(lidar2enu, sensor2lidar) 
    
    sparse_sfm_dir = os.path.join(output_path, "colmap/sparse_sfm")
    cameras, images, points3D = read_model(sparse_sfm_dir, ext=".bin")

    new_cameras, new_images, new_points3D = {}, {}, {}
    for idx in tqdm(images.keys()):
        Rw2c = images[idx].qvec2rotmat()
        Tw2c = images[idx].tvec
        w2c = np.eye(4)
        w2c[:3, :3] = Rw2c
        w2c[:3, 3] = Tw2c
        w2c_enu = w2c @ np.linalg.inv(init_pose)
        new_images[images[idx].id] = Image(
            id=images[idx].id,
            qvec=rotmat2qvec(w2c_enu[:3, :3]),
            tvec=w2c_enu[:3, 3],
            camera_id=images[idx].camera_id,
            name=images[idx].name,
            xys=images[idx].xys,
            point3D_ids=images[idx].point3D_ids,
        )
    

    for jdx in tqdm(points3D.keys()):
        xyz=points3D[jdx].xyz
        xyz=np.hstack((xyz, 1))
        new_xyz = init_pose @ xyz
        new_xyz = new_xyz[:3]
        new_points3D[points3D[jdx].id] = Point3D(
            id=points3D[jdx].id,
            xyz=np.array(new_xyz),
            rgb=points3D[jdx].rgb,
            error=points3D[jdx].error,
            image_ids=points3D[jdx].image_ids,
            point2D_idxs=points3D[jdx].point2D_idxs,
        )
    
    new_cameras = cameras
    output_enu_model = os.path.join(output_path, "colmap/sparse_sfm_enu")
    os.makedirs(output_enu_model, exist_ok=True)

    write_model(new_cameras,
                new_images,
                new_points3D,
                output_enu_model,
                ext=".bin")
    
    sparse_init_dir = os.path.join(output_path, "colmap/sparse_init")
    cameras, images, points3D = read_model(sparse_init_dir, ext=".txt")

    new_cameras, new_images, new_points3D = {}, {}, {}
    for idx in tqdm(images.keys()):
        Rw2c = images[idx].qvec2rotmat()
        Tw2c = images[idx].tvec
        w2c = np.eye(4)
        w2c[:3, :3] = Rw2c
        w2c[:3, 3] = Tw2c
        w2c_enu = w2c @ np.linalg.inv(init_pose)
        new_images[images[idx].id] = Image(
            id=images[idx].id,
            qvec=rotmat2qvec(w2c_enu[:3, :3]),
            tvec=w2c_enu[:3, 3],
            camera_id=images[idx].camera_id,
            name=images[idx].name,
            xys=images[idx].xys,
            point3D_ids=images[idx].point3D_ids,
        )

    new_cameras, new_points3D = cameras, points3D
    output_enu_model = os.path.join(output_path, "colmap/sparse_init_enu")
    os.makedirs(output_enu_model, exist_ok=True)

    write_model(new_cameras,
                new_images,
                new_points3D,
                output_enu_model,
                ext=".txt")