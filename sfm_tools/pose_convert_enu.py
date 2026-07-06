import argparse
import os

import numpy as np
from tqdm import tqdm

from sfm_tools.feature_extract_match.model.read_write_model import (
    Image,
    Point3D,
    read_model,
    rotmat2qvec,
    write_model,
)
from sfm_tools.uniscene_cameras import (
    build_cam_info_all,
    build_pose_info,
    find_init_pose,
    load_uniscene,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="use meta json to creat colmap sparse init models"
    )
    parser.add_argument("--data_root", help="path to uniscene data")
    parser.add_argument("--output_path", help="path to 3dgs format results")
    args = parser.parse_args()

    data_root = args.data_root
    output_path = args.output_path

    uniscene = load_uniscene(data_root)
    active_cams, cam_id_2_name = build_cam_info_all(uniscene)
    pose_info = build_pose_info(uniscene)
    init_pose = find_init_pose(uniscene, active_cams, cam_id_2_name, pose_info)

    sparse_sfm_candidates = [
        ("sparse_sfm", "sparse_sfm_enu"),
        ("sparse_sfm_no_opt", "sparse_sfm_enu_no_opt"),
    ]
    for sparse_name, enu_name in sparse_sfm_candidates:
        sparse_sfm_dir = os.path.join(output_path, "colmap", sparse_name)
        cameras_bin = os.path.join(sparse_sfm_dir, "cameras.bin")
        if not os.path.isfile(cameras_bin):
            continue

        cameras, images, points3D = read_model(sparse_sfm_dir, ext=".bin")

        new_cameras, new_images, new_points3D = {}, {}, {}
        for idx in tqdm(images.keys(), desc=f"{sparse_name} -> ENU"):
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

        for jdx in tqdm(points3D.keys(), desc="points3D -> ENU"):
            xyz = points3D[jdx].xyz
            xyz = np.hstack((xyz, 1))
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
        output_enu_model = os.path.join(output_path, "colmap", enu_name)
        os.makedirs(output_enu_model, exist_ok=True)

        write_model(
            new_cameras,
            new_images,
            new_points3D,
            output_enu_model,
            ext=".bin",
        )

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

    write_model(
        new_cameras,
        new_images,
        new_points3D,
        output_enu_model,
        ext=".txt",
    )
