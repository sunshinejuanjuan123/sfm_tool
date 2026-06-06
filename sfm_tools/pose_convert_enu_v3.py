"""Convert sparse_init poses to ENU (before feature matching).

Only writes colmap/sparse_init_enu; does not require sparse_sfm.
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from sfm_tools.feature_extract_match.model.read_write_model import (
    Image,
    read_model,
    rotmat2qvec,
    write_model,
)


def _load_init_pose(data_root):
    unisceneproto = os.path.join(data_root, "plannerGt/unisceneproto.json")
    with open(unisceneproto, "r", encoding="utf-8") as f:
        uniscene = json.load(f)

    cam_info_all = {
        "center_camera_fov120": {},
        "left_front_camera": {},
        "left_rear_camera": {},
        "right_front_camera": {},
        "right_rear_camera": {},
        "rear_camera": {},
        "center_camera_fov30": {},
        "front_camera_fov195": {},
        "rear_camera_fov195": {},
        "right_camera_fov195": {},
    }

    for cam_info in uniscene["cameras"]:
        if cam_info["camera_name"] not in cam_info_all:
            continue
        cam_info_all[cam_info["camera_name"]]["id"] = cam_info["camera_id"]
        quat = cam_info["extrinsic"]["quaternion"]
        trsl = cam_info["extrinsic"]["translation"]
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R.from_quat(
            [quat["x"], quat["y"], quat["z"], quat["w"]]
        ).as_matrix()
        extrinsic[:3, :3] = (
            np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) @ extrinsic[:3, :3]
        )
        extrinsic[:3, 3] = np.array([trsl["x"], trsl["y"], trsl["z"]])
        cam_info_all[cam_info["camera_name"]]["extrinsic"] = extrinsic

    cam_id_2_name = {v["id"]: k for k, v in cam_info_all.items() if "id" in v}

    pose_info = {}
    for ego_info in uniscene["ego_status"]:
        timestamp = int(round(ego_info["timestamp"], 3) * 1000)
        quat = ego_info["ego_orientation"]
        trsl = ego_info["ego_position"]
        pose_info[timestamp] = np.eye(4)
        pose_info[timestamp][:3, :3] = R.from_quat(
            [quat["x"], quat["y"], quat["z"], quat["w"]]
        ).as_matrix()
        pose_info[timestamp][:3, 3] = np.array([trsl["x"], trsl["y"], trsl["z"]])

    init_pose = None
    for jdx, sensor_info in enumerate(uniscene["sensor_frames"]):
        if jdx != 0:
            break
        timestamp = int(round(sensor_info["timestamp"], 3) * 1000)
        lidar2enu = pose_info[timestamp]
        for camera_data in sensor_info["camera_data"]:
            if camera_data["sensor_id"] not in cam_id_2_name:
                continue
            cam = cam_id_2_name[camera_data["sensor_id"]]
            if cam == "center_camera_fov120":
                sensor2lidar = cam_info_all[cam]["extrinsic"]
                init_pose = np.matmul(lidar2enu, sensor2lidar)

    if init_pose is None:
        raise RuntimeError("Failed to compute init_pose from center_camera_fov120")
    return init_pose


def convert_sparse_init_to_enu(output_path, init_pose):
    sparse_init_dir = os.path.join(output_path, "colmap/sparse_init")
    cameras, images, points3D = read_model(sparse_init_dir, ext=".txt")

    new_images = {}
    for idx in tqdm(images.keys(), desc="sparse_init -> ENU"):
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

    output_enu_model = os.path.join(output_path, "colmap/sparse_init_enu")
    os.makedirs(output_enu_model, exist_ok=True)
    write_model(cameras, new_images, points3D, output_enu_model, ext=".txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert sparse_init camera poses to ENU (pre feature matching)"
    )
    parser.add_argument("--data_root", help="path to uniscene data")
    parser.add_argument("--output_path", help="path to 3dgs format results")
    args = parser.parse_args()

    init_pose = _load_init_pose(args.data_root)
    convert_sparse_init_to_enu(args.output_path, init_pose)
