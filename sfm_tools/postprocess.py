import argparse
import os
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import json

def get_cam_name(n, cam_info_all):
    for cam_name, info in cam_info_all.items():
        if info.get('colmap_id') == n:
            return cam_name
    return None 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="postprocess 3dgs render image to uniscene format"
    )
    parser.add_argument("--data-root", help="path to uniscene data")
    parser.add_argument("--gs-data-root", help="path to 3dgs format data")
    args = parser.parse_args()

    data_root = args.data_root
    gs_data_root = args.gs_data_root

    unisceneproto = os.path.join(data_root, "plannerGt/unisceneproto.json")
    uniscene = json.load(open(unisceneproto, "r"))

    reconstruction_dir = os.path.join(gs_data_root, "v1/street-gaussians-ns")
    subdirs = [sub for sub in os.listdir(reconstruction_dir) if os.path.isdir(os.path.join(reconstruction_dir, sub))]

    subdirs.sort() 

    # 最新训练时间戳
    render_dir = os.path.join(reconstruction_dir, subdirs[-1], "renders/all/rgb")

    cam_info_all = {'center_camera_fov120': {'colmap_id': 1},
                    'left_front_camera': {'colmap_id': 2},
                    'left_rear_camera': {'colmap_id': 3},
                    'right_front_camera': {'colmap_id': 4},
                    'right_rear_camera': {'colmap_id': 5},
                    'rear_camera': {'colmap_id': 6},
                    'center_camera_fov30': {'colmap_id': 7}}

    for cam_info in uniscene['cameras']:
        if cam_info['camera_name'] in cam_info_all.keys():
            cam_info_all[cam_info['camera_name']]['id'] = cam_info['camera_id']
            cam_info_all[cam_info['camera_name']]['size'] = [cam_info["height"], cam_info["width"]]

    cam_id_2_name = {v['id']: k for k, v in cam_info_all.items()}

    if "center_camera_fov120" in os.listdir(render_dir):
        # multicameras
        for cam in tqdm(cam_info_all.keys()):
            os.makedirs(os.path.join(gs_data_root, "postprocess", cam), exist_ok=True)
            for img_name in os.listdir(os.path.join(render_dir, cam)):
                render_img_rgb = cv2.imread(os.path.join(render_dir, cam, img_name))
                src_img_rgb = cv2.imread(os.path.join(gs_data_root, "images", cam, img_name))
                pil_image = Image.open(os.path.join(gs_data_root, "mask_combine", cam, img_name.replace(".jpg", ".png")))
                ego_car_mask = np.array(pil_image, dtype="int64")
                combine_rgb = src_img_rgb * (1-ego_car_mask[:, :, None]) + render_img_rgb * ego_car_mask[:, :, None]

                for sensor_info in uniscene['sensor_frames']:
                    timestamp = int(round(sensor_info['timestamp'], 3)*1000)
                    img_timestamp, _ = os.path.splitext(img_name)
                    if img_timestamp == str(timestamp):
                        for camera_data in sensor_info['camera_data']:
                            if camera_data['sensor_id'] in cam_id_2_name.keys():
                                cam_src = cam_id_2_name[camera_data['sensor_id']]
                                if cam == cam_src:
                                    src_img_name = camera_data['file_path'].split("/")[-1]
                                    combine_rgb = cv2.resize(combine_rgb, (cam_info_all[cam]['size'][1], cam_info_all[cam]['size'][0]), interpolation=cv2.INTER_NEAREST)
                                    cv2.imwrite(os.path.join(gs_data_root, "postprocess", cam, src_img_name), combine_rgb)
    else:
        # per camera
        cam = get_cam_name(int(os.path.basename(os.path.normpath(gs_data_root)).split('_')[-1]), cam_info_all)
        os.makedirs(os.path.join(gs_data_root, "postprocess", cam), exist_ok=True)
        for img_name in tqdm(os.listdir(render_dir)):
            render_img_rgb = cv2.imread(os.path.join(render_dir, img_name))
            src_img_rgb = cv2.imread(os.path.join(gs_data_root, "images", cam, img_name))
            pil_image = Image.open(os.path.join(gs_data_root, "mask_combine", cam, img_name.replace(".jpg", ".png")))
            ego_car_mask = np.array(pil_image, dtype="int64")
            combine_rgb = src_img_rgb * (1-ego_car_mask[:, :, None]) + render_img_rgb * ego_car_mask[:, :, None]

            for sensor_info in uniscene['sensor_frames']:
                timestamp = int(round(sensor_info['timestamp'], 3)*1000)
                img_timestamp, _ = os.path.splitext(img_name)
                if img_timestamp == str(timestamp):
                    for camera_data in sensor_info['camera_data']:
                        if camera_data['sensor_id'] in cam_id_2_name.keys():
                            cam_src = cam_id_2_name[camera_data['sensor_id']]
                            if cam == cam_src:
                                src_img_name = camera_data['file_path'].split("/")[-1]
                                combine_rgb = cv2.resize(combine_rgb, (cam_info_all[cam]['size'][1], cam_info_all[cam]['size'][0]), interpolation=cv2.INTER_NEAREST)
                                cv2.imwrite(os.path.join(gs_data_root, "postprocess", cam, src_img_name), combine_rgb)
