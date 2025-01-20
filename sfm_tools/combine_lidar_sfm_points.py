import argparse 
import os
from tqdm import tqdm
import json
import open3d as o3d
import numpy as np
from sfm_tools.feature_extract_match.model.read_write_model import read_model, read_points3D_text, write_model
import cv2
from datetime import datetime
import random
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
        description="combine lidar points and sfm points together"
    )
    parser.add_argument("--data_root", help="path to pvb data")
    parser.add_argument("--gs_data_root", help="path to 3dgs format results")
    args = parser.parse_args()

    data_root = args.data_root
    meta_dir = os.path.join(data_root, "meta_json")
    json_idxs = sorted(os.listdir(meta_dir))

    gs_data_root = args.gs_data_root
    sparse_dir = os.path.join(gs_data_root, "colmap/sparse_sfm")
    cameras, images, points3D = read_model(sparse_dir, ext=".bin")

    project_lidar_camera_lists = ['center_camera_fov120', 'rear_camera']
    lidar_points_dir = os.path.join(sparse_dir, "../lidar_points")
    os.makedirs(lidar_points_dir, exist_ok=True)
    
    lidar_points_path = os.path.join(lidar_points_dir, "points3D.txt")

    with open(lidar_points_path, 'w') as j:
        i = 1

        for jdx, json_idx in enumerate(tqdm(json_idxs)):
            json_file = os.path.join(meta_dir, json_idx)
            meta_info = json.load(open(json_file, "r"))
            image_info = meta_info["meta_info"]

            # pose
            if jdx == 0:
                t = Transform(*image_info["lidar_pose"][:3])
                lidar2enu = convert_pose(image_info["lidar_pose"], t)
            else:
                lidar2enu = convert_pose(image_info["lidar_pose"], t)

            lidar_abs_pcd = os.path.join(data_root, image_info["lidar_path"]["car_center"])
            time_stamp = convert_timestamp(lidar_abs_pcd.split("/")[-1][:-4])
            pcd_data = o3d.io.read_point_cloud(lidar_abs_pcd)
            points = np.array(pcd_data.points)
            nan_rows = np.isnan(points).any(axis=1)

            points = points[~nan_rows]
            if points.shape[0] > 10000:
                indices = np.random.choice(points.shape[0], 10000, replace=False)
                points = points[indices]

            homogeneous_positions = np.hstack([points, np.ones((points.shape[0], 1))])
            transformed_positions = np.dot(lidar2enu, homogeneous_positions.T).T[:, :3]

            for cam in project_lidar_camera_lists:
                for ii in images.keys():
                    cam_ii, image_name = images[ii].name.split("/")
                    image_timestamp, _ = os.path.splitext(image_name)
                    if cam == cam_ii and image_timestamp == str(time_stamp):
                        ii_unique = ii
                
                K = cameras[images[ii_unique].camera_id].params
                fx, fy, cx, cy = K[0], K[1], K[2], K[3]
                intrinsic_matrix = np.array([[fx, 0, cx, 0],
                                             [0, fy, cy, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]])
                Rw2c = images[ii_unique].qvec2rotmat()
                Tw2c = images[ii_unique].tvec
                w2c = np.eye(4)
                w2c[:3, :3] = Rw2c
                w2c[:3, 3] = Tw2c 
                img_abs_path = os.path.join(gs_data_root, "images", images[ii_unique].name)
                rgb = cv2.imread(img_abs_path)
                h, w, _ = rgb.shape
                for m in transformed_positions:
                    if abs(m[0]) > 100000:
                        continue
                    m_l = np.array([m[0], m[1], m[2], 1])
                    uv_homogeneous = intrinsic_matrix @ w2c @ m_l
                    u, v = (uv_homogeneous[:2] / uv_homogeneous[2]).astype(int)

                    if 0 < u < w and 0 < v < h and uv_homogeneous[2] > 0:  
                        rgb_point = rgb[v, u]
                        error = random.uniform(0, 1)

                        j.write(f'{i} {m[0]:.3f} {m[1]:.3f} {m[2]:.3f} {rgb_point[2]} {rgb_point[1]} {rgb_point[0]} {error:.3f} 1 1 2 2 {random.randint(1,300)} {random.randint(1,2000)}\n')
                        i += 1

    lidar_points = read_points3D_text(lidar_points_path)
    sfm_points = points3D

    offset = max([i for i in lidar_points.keys()]) + 1
    for k, v in tqdm(sfm_points.items()):
        assert k + offset not in lidar_points
        lidar_points[k + offset] = v._replace(id=k + offset)
    
    combine_path = os.path.join(gs_data_root, "colmap/sparse/0")
    if not os.path.exists(combine_path):
        os.makedirs(combine_path, exist_ok=True)

    write_model(
        cameras, images, lidar_points, combine_path, ext=".bin"
    )