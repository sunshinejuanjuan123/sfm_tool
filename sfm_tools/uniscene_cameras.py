"""Shared uniscene camera metadata helpers for 3DGS preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

PINHOLE_CAMERA_COLMAP = {
    "center_camera_fov120": 1,
    "left_front_camera": 2,
    "left_rear_camera": 3,
    "right_front_camera": 4,
    "right_rear_camera": 5,
    "rear_camera": 6,
    "center_camera_fov30": 7,
}

# Pinhole cameras used for feature matching / triangulation (no fisheye).
SFM_CAMERAS = frozenset(PINHOLE_CAMERA_COLMAP.keys())

FISHEYE_SLOT_BY_KEYWORD = (
    (("front",), 8),
    (("rear",), 9),
    (("right",), 10),
    (("left",), 11),
)


def is_fisheye_name(camera_name: str, camera_model: str | None = None) -> bool:
    if camera_model == "FISHEYE":
        return True
    name = camera_name.lower()
    return "fov200" in name or "fov195" in name


def fisheye_colmap_id(camera_name: str) -> int | None:
    name = camera_name.lower()
    if not is_fisheye_name(camera_name):
        return None
    for keywords, colmap_id in FISHEYE_SLOT_BY_KEYWORD:
        if any(k in name for k in keywords):
            return colmap_id
    return None


def colmap_id_for_camera(camera_name: str, camera_model: str | None = None) -> int | None:
    if camera_name in PINHOLE_CAMERA_COLMAP:
        return PINHOLE_CAMERA_COLMAP[camera_name]
    return fisheye_colmap_id(camera_name)


def discover_fisheye_dirs(data_root: Path) -> dict[str, Path]:
    """Scan camera/ and camera_distorted/ for fisheye image folders."""
    found: dict[str, Path] = {}
    for sub in ("camera_distorted", "camera"):
        root = data_root / sub
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            if not is_fisheye_name(entry.name):
                continue
            if entry.name not in found or sub == "camera_distorted":
                found[entry.name] = entry
    return found


def load_uniscene(data_root: str | Path) -> dict:
    proto_path = Path(data_root) / "plannerGt" / "unisceneproto.json"
    with open(proto_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_cam_info_all(uniscene: dict) -> tuple[dict, dict[int, str]]:
    cam_info_all: dict[str, dict] = {}
    for name, colmap_id in PINHOLE_CAMERA_COLMAP.items():
        cam_info_all[name] = {"colmap_id": colmap_id}

    for cam_info in uniscene.get("cameras", []):
        name = cam_info.get("camera_name")
        if not name:
            continue
        colmap_id = colmap_id_for_camera(name, cam_info.get("camera_model"))
        if colmap_id is None:
            continue
        if name not in cam_info_all:
            cam_info_all[name] = {"colmap_id": colmap_id}

        cam_info_all[name]["id"] = cam_info["camera_id"]
        fx = cam_info["intrinsic"]["fx"]
        fy = cam_info["intrinsic"]["fy"]
        cx = cam_info["intrinsic"]["cx"]
        cy = cam_info["intrinsic"]["cy"]
        cam_info_all[name]["intrinsic"] = np.array([fx, fy, cx, cy], dtype=np.float64)
        distortion = cam_info["intrinsic"].get("distortion", [0.0, 0.0, 0.0, 0.0])
        cam_info_all[name]["distortion"] = np.array(distortion[:4], dtype=np.float64)
        cam_info_all[name]["is_fisheye"] = is_fisheye_name(
            name, cam_info.get("camera_model")
        )

        quat = cam_info["extrinsic"]["quaternion"]
        trsl = cam_info["extrinsic"]["translation"]
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
        extrinsic[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) @ extrinsic[:3, :3]
        extrinsic[:3, 3] = np.array([trsl["x"], trsl["y"], trsl["z"]])
        cam_info_all[name]["extrinsic"] = extrinsic
        cam_info_all[name]["size"] = [cam_info["height"], cam_info["width"]]

    active_cams = {k: v for k, v in cam_info_all.items() if "id" in v and "size" in v}
    cam_id_2_name = {v["id"]: k for k, v in active_cams.items()}
    return active_cams, cam_id_2_name


def build_pose_info(uniscene: dict) -> dict[int, np.ndarray]:
    pose_info: dict[int, np.ndarray] = {}
    for ego_info in uniscene.get("ego_status", []):
        timestamp = int(round(ego_info["timestamp"], 3) * 1000)
        quat = ego_info["ego_orientation"]
        trsl = ego_info["ego_position"]
        pose = np.eye(4)
        pose[:3, :3] = R.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
        pose[:3, 3] = np.array([trsl["x"], trsl["y"], trsl["z"]])
        pose_info[timestamp] = pose
    return pose_info


def find_init_pose(
    uniscene: dict,
    active_cams: dict,
    cam_id_2_name: dict[int, str],
    pose_info: dict[int, np.ndarray],
) -> np.ndarray:
    for sensor_info in uniscene.get("sensor_frames", []):
        timestamp = int(round(sensor_info["timestamp"], 3) * 1000)
        lidar2enu = pose_info[timestamp]
        for camera_data in sensor_info.get("camera_data", []):
            cam_name = cam_id_2_name.get(camera_data.get("sensor_id"))
            if cam_name != "center_camera_fov120":
                continue
            sensor2lidar = active_cams[cam_name]["extrinsic"]
            return np.matmul(lidar2enu, sensor2lidar)
    raise RuntimeError("center_camera_fov120 not found in sensor frames")


def pipeline_camera_names(uniscene: dict) -> list[str]:
    return [cam["camera_name"] for cam in uniscene.get("cameras", []) if cam.get("camera_name")]
