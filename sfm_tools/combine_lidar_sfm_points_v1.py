"""Combine LiDAR and SfM points; supports sparse_sfm_enu / sparse_sfm_enu_no_opt."""

import argparse
import os
import random

import cv2
import json
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from sfm_tools.feature_extract_match.model.read_write_model import (
    read_model,
    read_points3D_text,
    write_model,
)


def _resolve_sparse_enu_dir(gs_data_root):
    for name in ("sparse_sfm_enu", "sparse_sfm_enu_no_opt"):
        path = os.path.join(gs_data_root, "colmap", name)
        if os.path.isfile(os.path.join(path, "cameras.bin")):
            return path
    raise FileNotFoundError(
        f"No sparse SfM ENU model under {gs_data_root}/colmap "
        "(expected sparse_sfm_enu or sparse_sfm_enu_no_opt)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="combine lidar points and sfm points together"
    )
    parser.add_argument("--data_root", help="path to pvb data")
    parser.add_argument("--gs_data_root", help="path to 3dgs format results")
    args = parser.parse_args()

    data_root = args.data_root
    unisceneproto = os.path.join(data_root, "plannerGt/unisceneproto.json")
    with open(unisceneproto, "r", encoding="utf-8") as f:
        uniscene = json.load(f)

    gs_data_root = args.gs_data_root
    sparse_dir = _resolve_sparse_enu_dir(gs_data_root)
    cameras, images, points3D = read_model(sparse_dir, ext=".bin")

    project_lidar_camera_lists = ["center_camera_fov30", "rear_camera"]
    lidar_points_dir = os.path.join(sparse_dir, "../lidar_points")
    os.makedirs(lidar_points_dir, exist_ok=True)

    lidar_points_path = os.path.join(lidar_points_dir, "points3D.txt")

    with open(lidar_points_path, "w", encoding="utf-8") as j:
        i = 1

        pose_info = {}
        for ego_info in tqdm(uniscene["ego_status"]):
            timestamp = int(round(ego_info["timestamp"], 3) * 1000)
            quat = ego_info["ego_orientation"]
            trsl = ego_info["ego_position"]
            pose_info[timestamp] = np.eye(4)
            pose_info[timestamp][:3, :3] = R.from_quat(
                [quat["x"], quat["y"], quat["z"], quat["w"]]
            ).as_matrix()
            pose_info[timestamp][:3, 3] = np.array([trsl["x"], trsl["y"], trsl["z"]])

        for sensor_info in tqdm(uniscene["sensor_frames"]):
            timestamp = int(round(sensor_info["timestamp"], 3) * 1000)
            lidar2enu = pose_info[timestamp]
            lidar_abs_path = os.path.join(
                data_root, sensor_info["lidar_data"][0]["file_path"]
            )
            pcd_data = o3d.io.read_point_cloud(lidar_abs_path)
            points = np.array(pcd_data.points)
            nan_rows = np.isnan(points).any(axis=1)

            points = points[~nan_rows]
            if points.shape[0] > 10000:
                indices = np.random.choice(points.shape[0], 10000, replace=False)
                points = points[indices]

            homogeneous_positions = np.hstack(
                [points, np.ones((points.shape[0], 1))]
            )
            transformed_positions = np.dot(lidar2enu, homogeneous_positions.T).T[
                :, :3
            ]

            for cam in project_lidar_camera_lists:
                for ii in images.keys():
                    cam_ii, image_name = images[ii].name.split("/")
                    image_timestamp, _ = os.path.splitext(image_name)
                    if cam == cam_ii and image_timestamp == str(timestamp):
                        ii_unique = ii

                K = cameras[images[ii_unique].camera_id].params
                fx, fy, cx, cy = K[0], K[1], K[2], K[3]
                intrinsic_matrix = np.array(
                    [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                )
                Rw2c = images[ii_unique].qvec2rotmat()
                Twc2 = images[ii_unique].tvec
                w2c = np.eye(4)
                w2c[:3, :3] = Rw2c
                w2c[:3, 3] = Twc2
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

                        j.write(
                            f"{i} {m[0]:.3f} {m[1]:.3f} {m[2]:.3f} "
                            f"{rgb_point[2]} {rgb_point[1]} {rgb_point[0]} {error:.3f} "
                            f"1 1 2 2 {random.randint(1, 300)} {random.randint(1, 2000)}\n"
                        )
                        i += 1

    lidar_points = read_points3D_text(lidar_points_path)
    sfm_points = points3D

    offset = max(lidar_points.keys()) + 1
    for k, v in tqdm(sfm_points.items()):
        assert k + offset not in lidar_points
        lidar_points[k + offset] = v._replace(id=k + offset)

    combine_path = os.path.join(gs_data_root, "colmap/sparse/0")
    os.makedirs(combine_path, exist_ok=True)

    write_model(cameras, images, lidar_points, combine_path, ext=".bin")
