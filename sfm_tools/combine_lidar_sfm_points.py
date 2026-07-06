import argparse 
import os
from pathlib import Path
from tqdm import tqdm
import json
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from sfm_tools.feature_extract_match.model.read_write_model import read_model, read_points3D_text, write_model
import cv2
import random

def load_gt_jsonl(data_root):
    gt_path = os.path.join(data_root, "pvbGt/gt.jsonl")
    frames = []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            frames.append(json.loads(line))
    return frames


def lidar_extrinsic_matrix(lidar_info):
    ext = lidar_info.get("extrinsic")
    if isinstance(ext, list):
        return np.array(ext, dtype=np.float64)
    if isinstance(ext, dict):
        quat = ext.get("quaternion") or ext.get("rotation")
        trsl = ext.get("translation")
        mat = np.eye(4)
        if quat is not None and trsl is not None:
            if isinstance(quat, dict):
                q = [quat["x"], quat["y"], quat["z"], quat["w"]]
            else:
                q = quat
            mat[:3, :3] = R.from_quat(q).as_matrix()
            if isinstance(trsl, dict):
                mat[:3, 3] = [trsl["x"], trsl["y"], trsl["z"]]
            else:
                mat[:3, 3] = trsl
        return mat
    return np.eye(4)


def lidar_name_from_path(file_path):
    parts = Path(file_path).parts
    lidar_idx = parts.index("lidar") if "lidar" in parts else -1
    if lidar_idx >= 0 and lidar_idx + 1 < len(parts):
        return parts[lidar_idx + 1]
    return None


def _resolve_sparse_enu_dir(gs_data_root):
    for name in ("sparse_sfm_enu", "sparse_sfm_enu_no_opt"):
        path = os.path.join(gs_data_root, "colmap", name)
        if os.path.isfile(os.path.join(path, "cameras.bin")):
            return path
    raise FileNotFoundError(
        f"No sparse SfM ENU model under {gs_data_root}/colmap "
        "(expected sparse_sfm_enu or sparse_sfm_enu_no_opt)"
    )


def collect_frame_lidar_points(
    data_root,
    gt_frame,
    sensor_info,
    ego2enu,
    max_points_per_lidar=10000,
    lidar_names=None,
    lidar_source_dir=None,
    lidar_extrinsic_name=None,
):
    merged = []
    allowed = set(lidar_names) if lidar_names else None
    for lidar_entry in sensor_info.get("lidar_data", []):
        rel_path = lidar_entry["file_path"]
        lidar_name = lidar_name_from_path(rel_path)
        if lidar_name is None:
            continue
        if lidar_source_dir is not None:
            if lidar_extrinsic_name is not None and lidar_name != lidar_extrinsic_name:
                continue
        elif allowed is not None and lidar_name not in allowed:
            continue
        extrinsic_key = lidar_extrinsic_name or lidar_name
        lidar_info = gt_frame.get("sensors", {}).get("lidar", {}).get(extrinsic_key)
        if not lidar_info:
            continue
        if lidar_source_dir is not None:
            lidar_abs_path = os.path.join(
                data_root, lidar_source_dir, os.path.basename(rel_path)
            )
        else:
            lidar_abs_path = os.path.join(data_root, rel_path)
        if not os.path.isfile(lidar_abs_path):
            continue
        pcd_data = o3d.io.read_point_cloud(lidar_abs_path)
        points = np.array(pcd_data.points)
        nan_rows = np.isnan(points).any(axis=1)
        points = points[~nan_rows]
        if points.shape[0] == 0:
            continue
        # if points.shape[0] > max_points_per_lidar:
        #     indices = np.random.choice(points.shape[0], max_points_per_lidar, replace=False)
        #     points = points[indices]
        lidar2ego = lidar_extrinsic_matrix(lidar_info)
        lidar2enu = ego2enu @ lidar2ego
        homogeneous_positions = np.hstack([points, np.ones((points.shape[0], 1))])
        # 每一帧存储，进行debug
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(homogeneous_positions[:, :3])
        # # 按帧存储，文件名包含时间戳或lidar_entry中的关键信息方便debug
        # frame_id = sensor_info.get("frame_id", "unknown")
        # lidar_name_str = lidar_name if 'lidar_name' in locals() else "unknown_lidar"
        # save_dir = "/mnt/iag/yuanweizhong/datasets/L29_new/L29_new/3dgs_format_v1/lidar_points"
        # os.makedirs(save_dir, exist_ok=True)
        # ply_path = os.path.join(
        #     save_dir,
        #     f"lidar_points_{lidar_name_str}_frame{frame_id}.ply"
        # )
        # o3d.io.write_point_cloud(ply_path, pcd)
        # print(f"[DEBUG] Saved per-frame lidar points: {ply_path}")

        merged.append(np.dot(lidar2enu, homogeneous_positions.T).T[:, :3])
    if not merged:
        return np.empty((0, 3))
    return np.vstack(merged)


def write_points3d_ply(lidar_points_path, ply_path):
    lidar_points = read_points3D_text(lidar_points_path)
    if not lidar_points:
        raise ValueError(f"No points found in {lidar_points_path}")
    xyz = np.array([p.xyz for p in lidar_points.values()], dtype=np.float64)
    rgb = np.array([p.rgb for p in lidar_points.values()], dtype=np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)
    return len(lidar_points)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="combine lidar points and sfm points together"
    )
    parser.add_argument("--data_root", help="path to pvb data")
    parser.add_argument("--gs_data_root", help="path to 3dgs format results")
    parser.add_argument(
        "--lidar_names",
        nargs="+",
        default=None,
        help="only fuse these lidar sensors (e.g. top_center_lidar); default: all in sensor_frames",
    )
    parser.add_argument(
        "--lidar_source_dir",
        default=None,
        help="read pcd basename from uniscene but load files under this data_root-relative dir "
        "(e.g. lidar/top_center_lidar_compensation)",
    )
    parser.add_argument(
        "--lidar_extrinsic_name",
        default=None,
        help="gt.jsonl extrinsic key when lidar_source_dir differs from sensor path name "
        "(e.g. top_center_lidar for compensation clouds)",
    )
    parser.add_argument(
        "--export_ply",
        action="store_true",
        help="also write lidar_points/points3D.ply next to points3D.txt",
    )
    parser.add_argument(
        "--skip_sparse_merge",
        action="store_true",
        help="only write lidar_points/points3D.txt (and optional ply), skip colmap/sparse/0 merge",
    )
    parser.add_argument(
        "--no_lidar",
        action="store_true",
        help="no-lidar mode: skip lidar projection, use SfM points directly as final sparse/0 "
        "and copy dynamic-object template plys from --dynamic_template_dir",
    )
    parser.add_argument(
        "--dynamic_template_dir",
        default=None,
        help="directory with per-type template plys (e.g. car_1.ply, suv_3.ply, truck_5.ply, "
        "pedestrian_127.ply, non_motor_rider_16.ply) used to fabricate dynamic object models "
        "in no-lidar mode",
    )
    args = parser.parse_args()

    data_root = args.data_root
    unisceneproto = os.path.join(data_root, "plannerGt/unisceneproto.json")
    uniscene = json.load(open(unisceneproto, "r"))

    gs_data_root = args.gs_data_root
    sparse_dir = _resolve_sparse_enu_dir(gs_data_root)

    cameras, images, points3D = read_model(sparse_dir, ext=".bin")

    # ---- no-lidar mode: skip lidar, use SfM only + template dynamic objects ----
    if args.no_lidar:
        import shutil

        object_type_id_2_class = {
            0: "Other",
            1: "Car",
            2: "Pedestrian",
            3: "Cyclist",
            4: "Other",
            5: "Truck",
            6: "Other",
        }

        # gid class -> preferred case_2 template prefixes (longest first so
        # "non_motor_rider" wins over "motorcycle" over "car"/"suv"/"truck"/"bus")
        class_to_template_prefixes = {
            "Car": ["car", "suv"],
            "Truck": ["truck", "huge_vehicle"],
            "Pedestrian": ["pedestrian"],
            "Cyclist": ["non_motor_rider", "motorcycle"],
            "Other": ["suv", "car", "bus", "truck"],
        }
        known_prefixes = [
            "non_motor_rider", "huge_vehicle", "motorcycle",
            "pedestrian", "truck", "car", "suv", "bus",
        ]

        def detect_prefix(stem):
            for p in known_prefixes:
                if stem == p or stem.startswith(p + "_"):
                    return p
            return None

        # collect template plys per prefix
        templates_by_prefix = {}
        if args.dynamic_template_dir and os.path.isdir(args.dynamic_template_dir):
            for fname in sorted(os.listdir(args.dynamic_template_dir)):
                if not fname.endswith(".ply") or fname == "sample.ply":
                    continue
                prefix = detect_prefix(fname[:-4])
                if prefix is None:
                    continue
                templates_by_prefix.setdefault(prefix, []).append(
                    os.path.join(args.dynamic_template_dir, fname)
                )

        # gather gids from tracks (same logic as extract_dynamic_object.py)
        sdc_track_index = uniscene.get("sdc_track_index", 0)
        gids = []
        for track_idx, objects_info in enumerate(uniscene["tracks"]):
            if track_idx == sdc_track_index:
                continue
            cls = object_type_id_2_class[int(objects_info["object_type"])]
            gids.append(f"{cls}_{objects_info['object_id']}")

        dyna_out_dir = os.path.join(gs_data_root, "aggregate_lidar", "dynamic_objects")
        os.makedirs(dyna_out_dir, exist_ok=True)

        n_copied = 0
        n_missing = 0
        for gid in tqdm(sorted(set(gids)), desc="copy dynamic templates"):
            cls = gid.rsplit("_", 1)[0]
            prefixes = class_to_template_prefixes.get(cls, ["car", "suv"])
            src = None
            for p in prefixes:
                if templates_by_prefix.get(p):
                    src = templates_by_prefix[p][0]
                    break
            if src is None:
                print(f"[no_lidar] WARNING no template for class '{cls}' (gid={gid}); skipping")
                n_missing += 1
                continue
            dst = os.path.join(dyna_out_dir, f"{gid}.ply")
            shutil.copyfile(src, dst)
            n_copied += 1
        print(f"[no_lidar] copied {n_copied} dynamic-object template plys to {dyna_out_dir} "
              f"({n_missing} missing)")

        # write SfM model directly as final sparse/0 (no lidar merge)
        combine_path = os.path.join(gs_data_root, "colmap/sparse/0")
        os.makedirs(combine_path, exist_ok=True)
        write_model(cameras, images, points3D, combine_path, ext=".bin")
        print(f"[no_lidar] wrote SfM-only sparse/0: {len(points3D)} points to {combine_path}")
        raise SystemExit(0)

    # ---- default lidar + sfm path ----
    gt_frames = load_gt_jsonl(data_root)

    project_lidar_camera_lists = ['center_camera_fov30', 'rear_camera']
    lidar_points_dir = os.path.join(sparse_dir, "../lidar_points")
    os.makedirs(lidar_points_dir, exist_ok=True)
    
    lidar_points_path = os.path.join(lidar_points_dir, "points3D.txt")

    with open(lidar_points_path, 'w') as j:
        i = 1

        pose_info = {}
        for ego_info in tqdm(uniscene['ego_status']):
            timestamp = int(round(ego_info['timestamp'], 3)*1000)
            quat = ego_info['ego_orientation']
            trsl = ego_info['ego_position']
            pose_info[timestamp] = np.eye(4)
            pose_info[timestamp][:3, :3] = R.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
            pose_info[timestamp][:3, 3] = np.array([trsl["x"], trsl["y"], trsl["z"]])   
        
        for sensor_info in tqdm(uniscene['sensor_frames']):
            timestamp = int(round(sensor_info['timestamp'], 3)*1000)
            lidar2enu = pose_info[timestamp]
            lidar_abs_path = os.path.join(data_root, sensor_info['lidar_data'][0]['file_path'])
            pcd_data = o3d.io.read_point_cloud(lidar_abs_path)
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
                    if cam == cam_ii and  image_timestamp == str(timestamp):
                        ii_unique = ii 
                
                K = cameras[images[ii_unique].camera_id].params
                fx, fy, cx, cy = K[0], K[1], K[2], K[3]
                intrinsic_matrix = np.array([[fx, 0, cx, 0],
                                            [0, fy, cy, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])
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

                        j.write(f'{i} {m[0]:.3f} {m[1]:.3f} {m[2]:.3f} {rgb_point[2]} {rgb_point[1]} {rgb_point[0]} {error:.3f} 1 1 2 2 {random.randint(1,300)} {random.randint(1,2000)}\n')
                        i += 1

    lidar_points = read_points3D_text(lidar_points_path)
    print(f"Wrote {len(lidar_points)} lidar points to {lidar_points_path}")

    if args.export_ply:
        ply_path = os.path.join(lidar_points_dir, "points3D.ply")
        n_ply = write_points3d_ply(lidar_points_path, ply_path)
        print(f"Wrote {n_ply} lidar points to {ply_path}")

    if not args.skip_sparse_merge:
        sfm_points = points3D
        combine_path = os.path.join(gs_data_root, "colmap/sparse/0")
        if not os.path.exists(combine_path):
            os.makedirs(combine_path, exist_ok=True)

        if not lidar_points:
            print("Warning: no lidar points projected; writing SfM sparse model only to sparse/0")
            write_model(cameras, images, sfm_points, combine_path, ext=".bin")
        else:
            offset = max(lidar_points.keys()) + 1
            for k, v in tqdm(sfm_points.items()):
                assert k + offset not in lidar_points
                lidar_points[k + offset] = v._replace(id=k + offset)

            write_model(
                cameras, images, lidar_points, combine_path, ext=".bin"
            )
            print(f"Merged sparse/0: {len(lidar_points)} points (lidar + sfm)")