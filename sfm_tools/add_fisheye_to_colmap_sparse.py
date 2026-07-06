#!/usr/bin/env python3
"""Add fov120 + 4 fisheye cameras to an existing 7v colmap/sparse/0 model.

Keeps existing SfM image poses unchanged. New views use static rig extrinsics
relative to the reference camera (center_camera_fov30) and inherit per-frame
poses from sparse/0.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict, defaultdict

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from sfm_tools.feature_extract_match.model.read_write_model import (
    Camera,
    Image,
    read_model,
    rotmat2qvec,
    write_model,
)

REF_CAMERA = "center_camera_fov30"
INIT_CAMERA = "center_camera_fov120"

FISHEYE_CAMERAS = {
    "front_camera_fov195": 8,
    "rear_camera_fov195": 9,
    "right_camera_fov195": 10,
    "left_camera_fov195": 11,
}

EXTRA_PINHOLE_CAMERAS = {
    "center_camera_fov120": 7,
}

ALL_EXTRA_CAMERAS = {**EXTRA_PINHOLE_CAMERAS, **FISHEYE_CAMERAS}
EXPECTED_11V_NAMES = {
    REF_CAMERA,
    "left_front_camera",
    "left_rear_camera",
    "right_front_camera",
    "right_rear_camera",
    "rear_camera",
    INIT_CAMERA,
    *FISHEYE_CAMERAS.keys(),
}

AXIS_FIX = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)


def _load_uniscene(data_root: str) -> dict:
    proto_path = os.path.join(data_root, "plannerGt/unisceneproto.json")
    if not os.path.isfile(proto_path):
        raise FileNotFoundError(f"unisceneproto.json not found: {proto_path}")
    with open(proto_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_gt_jsonl(data_root: str) -> None:
    gt_path = os.path.join(data_root, "pvbGt/gt.jsonl")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"gt.jsonl not found: {gt_path}")


def _load_pose_info(uniscene: dict) -> dict[int, np.ndarray]:
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
    return pose_info


def _scaled_intrinsics(cam_info: dict, target_w: int, target_h: int) -> tuple[float, float, float, float]:
    h_native, w_native = cam_info["size"]
    fx, fy, cx, cy = cam_info["intrinsic"]
    fx *= target_w / w_native
    fy *= target_h / h_native
    cx *= target_w / w_native
    cy *= target_h / h_native
    return fx, fy, cx, cy


def _image_w2c(image: Image) -> np.ndarray:
    w2c = np.eye(4)
    w2c[:3, :3] = image.qvec2rotmat()
    w2c[:3, 3] = image.tvec
    return w2c


def _index_ref_images(images: dict) -> dict[int, Image]:
    ref_by_ts: dict[int, Image] = {}
    for image in images.values():
        cam, fname = image.name.split("/", 1)
        if cam != REF_CAMERA:
            continue
        ts = int(os.path.splitext(fname)[0])
        ref_by_ts[ts] = image
    if not ref_by_ts:
        raise RuntimeError(f"sparse/0 has no reference camera images: {REF_CAMERA}")
    return ref_by_ts


def _get_init_pose(
    uniscene: dict,
    cam_info_all: dict,
    pose_info: dict[int, np.ndarray],
    cam_id_2_name: dict,
) -> np.ndarray:
    for jdx, sensor_info in enumerate(uniscene["sensor_frames"]):
        if jdx > 0:
            break
        timestamp = int(round(sensor_info["timestamp"], 3) * 1000)
        lidar2enu = pose_info[timestamp]
        for camera_data in sensor_info["camera_data"]:
            cam = cam_id_2_name.get(camera_data["sensor_id"])
            if cam == INIT_CAMERA:
                return lidar2enu @ cam_info_all[cam]["extrinsic"]
    raise RuntimeError(f"failed to resolve init_pose from first-frame {INIT_CAMERA}")


def _w2c_calib(
    cam: str,
    timestamp: int,
    cam_info_all: dict,
    pose_info: dict[int, np.ndarray],
    init_pose: np.ndarray,
) -> np.ndarray:
    sensor2enu = pose_info[timestamp] @ cam_info_all[cam]["extrinsic"]
    return np.linalg.inv(sensor2enu) @ init_pose


def _compute_cam_from_rig(
    ref_cam: str,
    new_cam: str,
    cam_info_all: dict,
    pose_info: dict[int, np.ndarray],
    init_pose: np.ndarray,
) -> np.ndarray:
    timestamp = next(iter(pose_info))
    w2c_ref = _w2c_calib(ref_cam, timestamp, cam_info_all, pose_info, init_pose)
    w2c_new = _w2c_calib(new_cam, timestamp, cam_info_all, pose_info, init_pose)
    return w2c_new @ np.linalg.inv(w2c_ref)


def _remove_extra_entries(cameras: dict, images: dict) -> tuple[dict, dict]:
    extra_ids = set(ALL_EXTRA_CAMERAS.values())
    kept_images = {
        iid: img for iid, img in images.items() if img.camera_id not in extra_ids
    }
    kept_cameras = {
        cid: cam for cid, cam in cameras.items() if cid not in extra_ids
    }
    return kept_cameras, kept_images


def _validate_11v(sparse_dir: str) -> None:
    cameras, images, points3D = read_model(sparse_dir, ext=".bin")
    names = defaultdict(set)
    for image in images.values():
        names[image.camera_id].add(image.name.split("/")[0])

    print("cameras:", len(cameras), "images:", len(images), "points:", len(points3D))
    for cid in sorted(names):
        count = sum(1 for image in images.values() if image.camera_id == cid)
        print(f"  id={cid}: {sorted(names[cid])} ({count} imgs)")

    found_names = set().union(*names.values())
    missing_names = sorted(EXPECTED_11V_NAMES - found_names)
    if missing_names:
        raise RuntimeError(f"11v check failed, missing camera views: {missing_names}")
    if len(cameras) != 11:
        raise RuntimeError(f"11v check failed, expected 11 cameras, got {len(cameras)}")
    if len(names) != 11:
        raise RuntimeError(f"11v check failed, expected 11 camera_id groups, got {len(names)}")

    for cid in sorted(cameras):
        cam = cameras[cid]
        print(
            f"  camera {cid}: model={cam.model} w={cam.width} h={cam.height} "
            f"params={np.array2string(cam.params, precision=3, separator=' ')}"
        )
    print("11v camera check OK")


def add_fisheye_to_colmap_sparse(
    data_root: str,
    gs_data_root: str,
    sparse_dir: str,
    overwrite: bool = False,
) -> None:
    _validate_gt_jsonl(data_root)
    uniscene = _load_uniscene(data_root)

    if not os.path.isfile(os.path.join(sparse_dir, "cameras.bin")):
        raise FileNotFoundError(f"sparse model not found: {sparse_dir}")

    cameras, images, points3D = read_model(sparse_dir, ext=".bin")
    n_points_before = len(points3D)
    n_images_before = len(images)
    base_images = {iid: images[iid] for iid in images}

    extra_ids = set(ALL_EXTRA_CAMERAS.values())
    existing_extra_ids = {img.camera_id for img in images.values()} & extra_ids
    if existing_extra_ids and not overwrite:
        raise RuntimeError(
            f"sparse/0 already contains extra camera_id {sorted(existing_extra_ids)}; "
            "pass --overwrite to replace them"
        )
    if existing_extra_ids and overwrite:
        cameras, images = _remove_extra_entries(cameras, images)

    ref_cam = cameras.get(1)
    if ref_cam is None:
        raise RuntimeError("sparse/0 missing reference camera_id=1 for target resolution")
    target_w, target_h = ref_cam.width, ref_cam.height

    needed_names = {REF_CAMERA, INIT_CAMERA, *ALL_EXTRA_CAMERAS.keys()}
    cam_info_all: dict[str, dict] = {name: {} for name in needed_names}
    cam_id_2_name: dict = {}
    for cam_info in uniscene["cameras"]:
        name = cam_info["camera_name"]
        if name not in cam_info_all:
            continue
        cam_info_all[name]["id"] = cam_info["camera_id"]
        cam_id_2_name[cam_info["camera_id"]] = name
        quat = cam_info["extrinsic"]["quaternion"]
        trsl = cam_info["extrinsic"]["translation"]
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R.from_quat(
            [quat["x"], quat["y"], quat["z"], quat["w"]]
        ).as_matrix()
        extrinsic[:3, :3] = AXIS_FIX @ extrinsic[:3, :3]
        extrinsic[:3, 3] = np.array([trsl["x"], trsl["y"], trsl["z"]])
        cam_info_all[name]["extrinsic"] = extrinsic

        fx, fy, cx, cy = (
            cam_info["intrinsic"]["fx"],
            cam_info["intrinsic"]["fy"],
            cam_info["intrinsic"]["cx"],
            cam_info["intrinsic"]["cy"],
        )
        cam_info_all[name]["intrinsic"] = np.array([fx, fy, cx, cy], dtype=np.float64)
        cam_info_all[name]["size"] = [cam_info["height"], cam_info["width"]]
        if name in FISHEYE_CAMERAS:
            distortion = cam_info["intrinsic"].get("distortion", [0.0, 0.0, 0.0, 0.0])
            cam_info_all[name]["distortion"] = np.array(distortion[:4], dtype=np.float64)

    pose_info = _load_pose_info(uniscene)
    init_pose = _get_init_pose(uniscene, cam_info_all, pose_info, cam_id_2_name)
    ref_by_ts = _index_ref_images(images)
    cam_from_rig = {
        cam: _compute_cam_from_rig(REF_CAMERA, cam, cam_info_all, pose_info, init_pose)
        for cam in ALL_EXTRA_CAMERAS
    }

    images_dir = os.path.join(gs_data_root, "images")
    next_image_id = max(images.keys(), default=0) + 1
    added_per_cam: dict[str, int] = defaultdict(int)

    for sensor_info in tqdm(uniscene["sensor_frames"], desc="add extra cameras"):
        timestamp = int(round(sensor_info["timestamp"], 3) * 1000)
        ref_image = ref_by_ts.get(timestamp)
        if ref_image is None:
            continue
        w2c_ref = _image_w2c(ref_image)

        for camera_data in sensor_info["camera_data"]:
            cam = cam_id_2_name.get(camera_data["sensor_id"])
            if cam not in ALL_EXTRA_CAMERAS:
                continue

            src_suffix = os.path.splitext(camera_data["file_path"].split("/")[-1])[1] or ".jpg"
            rel_name = f"{cam}/{timestamp}{src_suffix}"
            img_path = os.path.join(images_dir, rel_name)
            if not os.path.isfile(img_path):
                continue

            src_img_path = os.path.join(data_root, camera_data["file_path"])
            if not os.path.isfile(src_img_path):
                continue

            w2c_new = cam_from_rig[cam] @ w2c_ref
            colmap_id = ALL_EXTRA_CAMERAS[cam]
            images[next_image_id] = Image(
                id=next_image_id,
                qvec=rotmat2qvec(w2c_new[:3, :3]),
                tvec=w2c_new[:3, 3],
                camera_id=colmap_id,
                name=rel_name,
                xys=np.array([]),
                point3D_ids=np.array([]),
            )
            next_image_id += 1
            added_per_cam[cam] += 1

            if colmap_id not in cameras:
                fx, fy, cx, cy = _scaled_intrinsics(cam_info_all[cam], target_w, target_h)
                if cam in FISHEYE_CAMERAS:
                    k1, k2, k3, k4 = cam_info_all[cam]["distortion"]
                    cameras[colmap_id] = Camera(
                        id=colmap_id,
                        model="OPENCV_FISHEYE",
                        width=target_w,
                        height=target_h,
                        params=np.array([fx, fy, cx, cy, k1, k2, k3, k4], dtype=np.float64),
                    )
                else:
                    cameras[colmap_id] = Camera(
                        id=colmap_id,
                        model="PINHOLE",
                        width=target_w,
                        height=target_h,
                        params=np.array([fx, fy, cx, cy], dtype=np.float64),
                    )

    if not added_per_cam:
        raise RuntimeError(
            "No extra camera images added. Check images/ dirs and pvbGt paths in unisceneproto."
        )

    for iid, image in base_images.items():
        if images[iid] != image:
            raise RuntimeError(f"existing sparse/0 image id={iid} was modified unexpectedly")

    cameras = OrderedDict(sorted(cameras.items(), key=lambda x: x[0]))
    write_model(cameras, images, points3D, sparse_dir, ext=".bin")

    print(f"Wrote merged model to {sparse_dir}")
    print(f"  cameras: {len(cameras)} (added extra ids: {sorted(extra_ids & set(cameras))})")
    print(f"  images:  {len(images)} (+{len(images) - n_images_before})")
    print(f"  points3D: {len(points3D)} (unchanged={len(points3D) == n_points_before})")
    for cam in sorted(added_per_cam):
        print(f"  + {cam}: {added_per_cam[cam]} images (camera_id={ALL_EXTRA_CAMERAS[cam]})")

    print("\nValidation:")
    _validate_11v(sparse_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add fov120 + 4 fisheye cameras to existing 7v colmap/sparse/0"
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Scene root with plannerGt/unisceneproto.json and pvbGt/gt.jsonl",
    )
    parser.add_argument(
        "--gs_data_root",
        default=None,
        help="3dgs_format root (default: {data_root}/3dgs_format)",
    )
    parser.add_argument(
        "--sparse_dir",
        default=None,
        help="COLMAP sparse model dir (default: {gs_data_root}/colmap/sparse/0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing camera_id 7-11 entries in sparse/0",
    )
    args = parser.parse_args()

    gs_data_root = args.gs_data_root or os.path.join(args.data_root, "3dgs_format")
    sparse_dir = args.sparse_dir or os.path.join(gs_data_root, "colmap", "sparse", "0")

    add_fisheye_to_colmap_sparse(
        data_root=args.data_root,
        gs_data_root=gs_data_root,
        sparse_dir=sparse_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
