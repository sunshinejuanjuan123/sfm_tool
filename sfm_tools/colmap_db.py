import argparse
import json
import os

import numpy as np

from sfm_tools.feature_extract_match.model.read_write_model import read_model, rotmat2qvec
from sfm_tools.uniscene_cameras import SFM_CAMERAS, colmap_id_for_camera

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="postprocess colmap db & rigid_ba"
    )
    parser.add_argument("--gs_data_root", help="path to 3dgs format results")
    args = parser.parse_args()

    data_root = args.gs_data_root
    sparse_init = os.path.join(data_root, "colmap/sparse_init")
    cameras, images, points3D = read_model(sparse_init, ext=".txt")
    cam_rigid = dict()

    image_prefixes = sorted({img.name.split("/")[0] for img in images.values()})
    camera_name_2_id = {
        name: colmap_id_for_camera(name)
        for name in image_prefixes
        if colmap_id_for_camera(name) is not None
    }

    ref_camera_id = camera_name_2_id["center_camera_fov120"]
    cam_rigid["ref_camera_id"] = ref_camera_id
    rigid_cam_list = []

    ref_image_name = None
    for img in images.values():
        cam_name, image_name = img.name.split("/")
        if cam_name == "center_camera_fov120":
            ref_image_name = image_name
            ref_image_id = img.id
            break
    if ref_image_name is None:
        raise RuntimeError("center_camera_fov120 not found in sparse_init images")

    ref_extrinsic = np.eye(4)
    ref_extrinsic[:3, :3] = images[ref_image_id].qvec2rotmat()
    ref_extrinsic[:3, 3] = images[ref_image_id].tvec
    ref_extrinsic = np.linalg.inv(ref_extrinsic)

    for cam in sorted(camera_name_2_id.keys()):
        if cam not in SFM_CAMERAS:
            continue
        rigid_cam = dict()
        rigid_cam["camera_id"] = camera_name_2_id[cam]

        cur_image_id = None
        for img in images.values():
            cam_name, image_name = img.name.split("/")
            if cam_name == cam and image_name == ref_image_name:
                cur_image_id = img.id
                break
        if cur_image_id is None:
            continue

        cur_extrinsic = np.eye(4)
        cur_extrinsic[:3, :3] = images[cur_image_id].qvec2rotmat()
        cur_extrinsic[:3, 3] = images[cur_image_id].tvec
        cur_extrinsic = np.linalg.inv(cur_extrinsic)

        rel_extrinsic = np.linalg.inv(cur_extrinsic) @ ref_extrinsic

        qvec = rotmat2qvec(rel_extrinsic[:3, :3])
        tvec = rel_extrinsic[:3, 3]

        rigid_cam["image_prefix"] = cam
        rigid_cam["cam_from_rig_rotation"] = qvec.tolist()
        rigid_cam["cam_from_rig_translation"] = tvec.tolist()
        rigid_cam_list.append(rigid_cam)

    cam_rigid["cameras"] = rigid_cam_list
    if not rigid_cam_list:
        raise RuntimeError("no SFM cameras found for rig config")

    rigid_config_path = os.path.join(data_root, "colmap/cam_rigid_config.json")
    with open(rigid_config_path, "w+") as f:
        json.dump([cam_rigid], f, indent=4)
