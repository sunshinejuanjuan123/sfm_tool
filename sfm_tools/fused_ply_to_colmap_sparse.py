#!/usr/bin/env python3
"""Convert COLMAP stereo_fusion fused.ply into colmap/sparse/0 for Street Gaussians."""

import argparse
import os
import shutil

import numpy as np
import open3d as o3d
from tqdm import tqdm

from sfm_tools.feature_extract_match.model.read_write_model import (
    Point3D,
    read_model,
    write_model,
)


def ply_to_points3D(ply_path, voxel_size=0.05, max_points=0):
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise ValueError(f"No points in fused point cloud: {ply_path}")
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    points = np.asarray(pcd.points)
    if pcd.has_colors():
        colors = (np.asarray(pcd.colors) * 255.0).astype(np.uint8)
    else:
        colors = np.full((points.shape[0], 3), 128, dtype=np.uint8)

    if max_points > 0 and points.shape[0] > max_points:
        indices = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[indices]
        colors = colors[indices]

    points3D = {}
    ref_image_id = 1
    ref_point2d_idx = 0
    for idx, (xyz, rgb) in enumerate(
        tqdm(zip(points, colors), total=len(points), desc="ply -> points3D"), start=1
    ):
        points3D[idx] = Point3D(
            id=idx,
            xyz=xyz.astype(np.float64),
            rgb=rgb,
            error=0.01,
            image_ids=np.array([ref_image_id], dtype=np.int32),
            point2D_idxs=np.array([ref_point2d_idx], dtype=np.int32),
        )
    return points3D


def fused_ply_to_colmap_sparse(
    gs_data_root,
    fused_ply,
    sparse_src_dir,
    sparse_dst_dir,
    voxel_size=0.05,
    max_points=0,
):
    cameras, images, _ = read_model(sparse_src_dir, ext=".bin")
    points3D = ply_to_points3D(fused_ply, voxel_size=voxel_size, max_points=max_points)

    if os.path.isdir(sparse_dst_dir):
        shutil.rmtree(sparse_dst_dir)
    os.makedirs(sparse_dst_dir, exist_ok=True)
    write_model(cameras, images, points3D, sparse_dst_dir, ext=".bin")
    print(f"Wrote {len(points3D)} points to {sparse_dst_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert fused.ply to colmap/sparse/0")
    parser.add_argument("--gs_data_root", required=True, help="Path to 3dgs_format")
    parser.add_argument(
        "--fused_ply",
        default=None,
        help="Path to fused.ply (default: gs_data_root/colmap/dense/fused.ply)",
    )
    parser.add_argument(
        "--sparse_src_dir",
        default=None,
        help="Source cameras/images model (default: colmap/sparse_sfm_enu_7v)",
    )
    parser.add_argument(
        "--sparse_dst_dir",
        default=None,
        help="Output sparse model dir (default: colmap/sparse/0)",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="Voxel downsample size in meters (0 to disable)",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Optional hard cap on number of points after downsampling",
    )
    args = parser.parse_args()

    gs_data_root = args.gs_data_root
    fused_ply = args.fused_ply or os.path.join(gs_data_root, "colmap", "dense", "fused.ply")
    sparse_src_dir = args.sparse_src_dir or os.path.join(
        gs_data_root, "colmap", "sparse_sfm_enu_7v"
    )
    sparse_dst_dir = args.sparse_dst_dir or os.path.join(gs_data_root, "colmap", "sparse", "0")

    if not os.path.isfile(fused_ply):
        raise FileNotFoundError(f"fused.ply not found: {fused_ply}")
    if not os.path.isfile(os.path.join(sparse_src_dir, "cameras.bin")):
        raise FileNotFoundError(f"sparse source model not found: {sparse_src_dir}")

    fused_ply_to_colmap_sparse(
        gs_data_root,
        fused_ply,
        sparse_src_dir,
        sparse_dst_dir,
        voxel_size=args.voxel_size,
        max_points=args.max_points,
    )
