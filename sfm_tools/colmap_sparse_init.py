import argparse
import os
from collections import OrderedDict

import cv2
import numpy as np
from tqdm import tqdm

from sfm_tools.feature_extract_match.model.read_write_model import Camera, Image, rotmat2qvec, write_model
from sfm_tools.filter_colmap_by_camera import filter_sparse_init_sfm
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

    os.makedirs(output_path, exist_ok=True)
    cameras, images, points3D = {}, {}, {}
    image_id = 0

    h_new, w_new = active_cams["center_camera_fov120"]["size"]
    for cam in active_cams.keys():
        h, w = active_cams[cam]["size"]
        h_new = min(h_new, h)
        w_new = min(w_new, w)

    init_pose = find_init_pose(uniscene, active_cams, cam_id_2_name, pose_info)
    for sensor_info in tqdm(uniscene["sensor_frames"]):
        timestamp = int(round(sensor_info["timestamp"], 3) * 1000)
        lidar2enu = pose_info[timestamp]
        for camera_data in sensor_info["camera_data"]:
            if camera_data["sensor_id"] not in cam_id_2_name:
                continue
            cam = cam_id_2_name[camera_data["sensor_id"]]
            sensor2lidar = active_cams[cam]["extrinsic"]
            dst_cam = os.path.join(output_path, "images", cam)
            os.makedirs(dst_cam, exist_ok=True)
            src_img_name = camera_data["file_path"].split("/")[-1]
            src_img_name, suffix = os.path.splitext(src_img_name)
            src_img_abs_path = os.path.join(data_root, camera_data["file_path"])
            rgb_img = cv2.imread(src_img_abs_path)
            h, w, _ = rgb_img.shape

            K = active_cams[cam]["intrinsic"]
            fx, fy, cx, cy = K[0], K[1], K[2], K[3]

            rgb_img = cv2.resize(rgb_img, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
            fx *= w_new / w
            fy *= h_new / h
            cx *= w_new / w
            cy *= h_new / h
            cv2.imwrite(os.path.join(dst_cam, str(timestamp) + suffix), rgb_img)
            h, w = h_new, w_new

            sensor2enu = np.matmul(lidar2enu, sensor2lidar)
            sensor2enu = np.linalg.inv(init_pose) @ sensor2enu

            enu2sensor = np.linalg.inv(sensor2enu)
            qvec = rotmat2qvec(enu2sensor[:3, :3])
            tvec = enu2sensor[:3, 3]

            image_id += 1
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=active_cams[cam]["colmap_id"],
                name=cam + "/" + str(timestamp) + suffix,
                xys=np.array([]),
                point3D_ids=np.array([]),
            )

            if active_cams[cam].get("is_fisheye", False):
                k1, k2, k3, k4 = active_cams[cam]["distortion"]
                camera_params = np.array([fx, fy, cx, cy, k1, k2, k3, k4])
                camera_model = "OPENCV_FISHEYE"
            else:
                camera_params = np.array([fx, fy, cx, cy])
                camera_model = "PINHOLE"
            cameras[active_cams[cam]["colmap_id"]] = Camera(
                id=active_cams[cam]["colmap_id"],
                model=camera_model,
                width=w,
                height=h,
                params=camera_params,
            )

    cameras = OrderedDict(sorted(cameras.items(), key=lambda x: x[0]))

    output_model_path = os.path.join(output_path, "colmap/sparse_init")
    os.makedirs(output_model_path, exist_ok=True)

    write_model(
        cameras=cameras,
        images=images,
        points3D=points3D,
        path=output_model_path,
        ext=".txt",
    )

    filter_sparse_init_sfm(
        output_model_path,
        os.path.join(output_path, "colmap/sparse_init_sfm"),
    )
