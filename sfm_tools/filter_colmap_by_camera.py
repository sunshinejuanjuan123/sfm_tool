#!/usr/bin/env python3
"""Filter a COLMAP sparse model to keep only selected camera_id entries."""

import argparse
import os
import shutil

import numpy as np
from tqdm import tqdm

from sfm_tools.feature_extract_match.model.read_write_model import read_model, write_model
from sfm_tools.uniscene_cameras import SFM_CAMERAS


def filter_sparse_init_sfm(sparse_init_dir, output_dir):
    """Keep only SFM pinhole images; fisheye (fov200) stay in sparse_init for pose/K only."""
    cameras, images, points3D = read_model(sparse_init_dir, ext=".txt")

    filtered_images = {}
    used_camera_ids = set()
    for image_id, image in images.items():
        cam_name = image.name.split("/")[0]
        if cam_name in SFM_CAMERAS:
            filtered_images[image_id] = image
            used_camera_ids.add(image.camera_id)

    filtered_cameras = {
        cid: cam for cid, cam in cameras.items() if cid in used_camera_ids
    }

    os.makedirs(output_dir, exist_ok=True)
    write_model(filtered_cameras, filtered_images, {}, output_dir, ext=".txt")
    print(
        f"sparse_init_sfm: cameras {len(cameras)} -> {len(filtered_cameras)}, "
        f"images {len(images)} -> {len(filtered_images)}"
    )
    return output_dir


def resolve_sparse_enu_dir(gs_data_root):
    for name in ("sparse_sfm_enu", "sparse_sfm_enu_no_opt"):
        path = os.path.join(gs_data_root, "colmap", name)
        if os.path.isfile(os.path.join(path, "cameras.bin")):
            return path
    raise FileNotFoundError(
        f"No sparse SfM ENU model under {gs_data_root}/colmap "
        "(expected sparse_sfm_enu or sparse_sfm_enu_no_opt)"
    )


def filter_colmap_by_camera_ids(input_dir, output_dir, camera_ids):
    allowed = set(camera_ids)
    cameras, images, points3D = read_model(input_dir, ext=".bin")

    kept_image_ids = set()
    filtered_images = {}
    for image_id, image in images.items():
        if image.camera_id in allowed:
            kept_image_ids.add(image_id)
            filtered_images[image_id] = image

    filtered_points = {}
    for point_id, point in tqdm(points3D.items(), desc="filter points3D"):
        keep_mask = np.isin(point.image_ids, list(kept_image_ids))
        if not np.any(keep_mask):
            continue
        image_ids = point.image_ids[keep_mask]
        point2D_idxs = point.point2D_idxs[keep_mask]
        filtered_points[point_id] = point._replace(
            image_ids=image_ids,
            point2D_idxs=point2D_idxs,
        )

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    write_model(cameras, filtered_images, filtered_points, output_dir, ext=".bin")

    print(
        f"Filtered model: images {len(images)} -> {len(filtered_images)}, "
        f"points3D {len(points3D)} -> {len(filtered_points)}"
    )
    kept_cams = sorted({images[i].camera_id for i in filtered_images})
    print(f"Kept camera_id: {kept_cams}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter COLMAP sparse model by camera_id")
    parser.add_argument("--gs_data_root", help="Path to 3dgs_format")
    parser.add_argument(
        "--sparse_init_dir",
        help="Input sparse_init dir; with --sparse_init_sfm_out writes SFM-only model",
    )
    parser.add_argument(
        "--sparse_init_sfm_out",
        help="Output dir for SFM-only sparse_init (7 pinhole, no fov200)",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Input sparse dir (default: auto-detect sparse_sfm_enu*)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output sparse dir (default: colmap/sparse_sfm_enu_7v)",
    )
    parser.add_argument(
        "--camera_ids",
        nargs="+",
        type=int,
        default=None,
        help="COLMAP camera_id values to keep",
    )
    args = parser.parse_args()

    if args.sparse_init_dir and args.sparse_init_sfm_out:
        filter_sparse_init_sfm(args.sparse_init_dir, args.sparse_init_sfm_out)
        raise SystemExit(0)

    if not args.gs_data_root:
        parser.error("--gs_data_root is required unless using --sparse_init_dir + --sparse_init_sfm_out")
    if not args.camera_ids:
        parser.error("--camera_ids is required for sparse model filtering")

    input_dir = args.input_dir or resolve_sparse_enu_dir(args.gs_data_root)
    output_dir = args.output_dir or os.path.join(
        args.gs_data_root, "colmap", "sparse_sfm_enu_7v"
    )
    filter_colmap_by_camera_ids(input_dir, output_dir, args.camera_ids)
