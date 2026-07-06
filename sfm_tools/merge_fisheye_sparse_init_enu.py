#!/usr/bin/env python3
"""Merge fisheye camera entries from sparse_init_enu into colmap/sparse/0."""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict

from sfm_tools.feature_extract_match.model.read_write_model import read_model, write_model
from sfm_tools.uniscene_cameras import is_fisheye_name


def merge_fisheye(gs_data_root: str, sparse_dir: str | None = None) -> None:
    gs_data_root = os.path.abspath(gs_data_root)
    sparse_dir = sparse_dir or os.path.join(gs_data_root, "colmap/sparse/0")
    init_enu = os.path.join(gs_data_root, "colmap/sparse_init_enu")

    cameras, images, points3D = read_model(sparse_dir, ext=".bin")
    init_cameras, init_images, _ = read_model(init_enu, ext=".txt")

    existing_ids = set(cameras.keys())
    next_image_id = max(images.keys(), default=0) + 1
    added = 0

    for img in init_images.values():
        cam_name = img.name.split("/")[0]
        if not is_fisheye_name(cam_name):
            continue
        if img.camera_id in existing_ids and any(
            i.camera_id == img.camera_id for i in images.values()
        ):
            continue
        if img.camera_id not in cameras:
            cameras[img.camera_id] = init_cameras[img.camera_id]
        images[next_image_id] = img._replace(id=next_image_id)
        next_image_id += 1
        added += 1

    if added == 0:
        print("No fisheye images added (already present?)")
        return

    cameras = OrderedDict(sorted(cameras.items()))
    write_model(cameras, images, points3D, sparse_dir, ext=".bin")
    print(f"Merged {added} fisheye images into {sparse_dir}")
    print(f"  cameras: {len(cameras)}, images: {len(images)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_data_root", required=True)
    parser.add_argument("--sparse_dir", default=None)
    args = parser.parse_args()
    merge_fisheye(args.gs_data_root, args.sparse_dir)
