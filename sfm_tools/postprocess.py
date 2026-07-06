import argparse
import os
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import json

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def get_cam_name(n, cam_info_all):
    for cam_name, info in cam_info_all.items():
        if info.get('colmap_id') == n:
            return cam_name
    return None


def infer_single_cam_name(gs_data_root, cam_info_all):
    suffix = os.path.basename(os.path.normpath(gs_data_root)).split('_')[-1]
    if suffix.isdigit():
        cam = get_cam_name(int(suffix), cam_info_all)
        if cam is not None:
            return cam

    images_dir = os.path.join(gs_data_root, "images")
    if os.path.isdir(images_dir):
        cams = [
            cam for cam in cam_info_all.keys()
            if os.path.isdir(os.path.join(images_dir, cam))
        ]
        if len(cams) == 1:
            return cams[0]

    raise ValueError(
        "Cannot infer single camera name from gs_data_root. "
        "Use a per-camera path ending with the COLMAP camera id, e.g. *_1, "
        "or keep exactly one known camera directory under images/."
    )


def find_render_mp4(render_dir, cam):
    for sub in ("", "images_2", "images_ud_2"):
        mp4_path = (
            os.path.join(render_dir, f"{cam}.mp4")
            if not sub
            else os.path.join(render_dir, sub, f"{cam}.mp4")
        )
        if os.path.isfile(mp4_path):
            return mp4_path
    return None


def list_gs_image_names(gs_data_root, cam):
    img_dir = os.path.join(gs_data_root, "images", cam)
    return sorted(
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )


def iter_render_frames(render_dir, cam, gs_data_root):
    cam_dir = os.path.join(render_dir, cam)
    if os.path.isdir(cam_dir):
        names = sorted(
            f for f in os.listdir(cam_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        )
        if names:
            for name in names:
                img = cv2.imread(os.path.join(cam_dir, name))
                if img is not None:
                    yield name, img
            return

    flat_names = sorted(
        f for f in os.listdir(render_dir)
        if os.path.isfile(os.path.join(render_dir, f))
        and os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if flat_names:
        for name in flat_names:
            img = cv2.imread(os.path.join(render_dir, name))
            if img is not None:
                yield name, img
        return

    mp4_path = find_render_mp4(render_dir, cam)
    if mp4_path is None:
        return

    img_names = list_gs_image_names(gs_data_root, cam)
    cap = cv2.VideoCapture(mp4_path)
    frame_idx = 0
    try:
        while frame_idx < len(img_names):
            ok, frame = cap.read()
            if not ok:
                break
            yield img_names[frame_idx], frame
            frame_idx += 1
    finally:
        cap.release()


def has_render_output(render_dir, cam):
    cam_dir = os.path.join(render_dir, cam)
    if os.path.isdir(cam_dir):
        if any(
            os.path.splitext(f)[1].lower() in IMAGE_EXTS
            for f in os.listdir(cam_dir)
        ):
            return True
    return find_render_mp4(render_dir, cam) is not None


def process_camera(cam, render_dir, gs_data_root, uniscene, cam_info_all, cam_id_2_name):
    out_dir = os.path.join(gs_data_root, "postprocess", cam)
    os.makedirs(out_dir, exist_ok=True)
    written = 0

    for img_name, render_img_rgb in iter_render_frames(render_dir, cam, gs_data_root):
        src_img_rgb = cv2.imread(os.path.join(gs_data_root, "images", cam, img_name))
        if src_img_rgb is None or render_img_rgb is None:
            continue
        if render_img_rgb.shape[:2] != src_img_rgb.shape[:2]:
            render_img_rgb = cv2.resize(
                render_img_rgb,
                (src_img_rgb.shape[1], src_img_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        mask_path = os.path.join(
            gs_data_root, "mask_combine", cam,
            os.path.splitext(img_name)[0] + ".png",
        )
        pil_image = Image.open(mask_path)
        ego_car_mask = np.array(pil_image, dtype="int64")
        combine_rgb = (
            src_img_rgb * (1 - ego_car_mask[:, :, None])
            + render_img_rgb * ego_car_mask[:, :, None]
        )

        img_timestamp, _ = os.path.splitext(img_name)
        for sensor_info in uniscene['sensor_frames']:
            timestamp = int(round(sensor_info['timestamp'], 3) * 1000)
            if img_timestamp != str(timestamp):
                continue
            for camera_data in sensor_info['camera_data']:
                if camera_data['sensor_id'] not in cam_id_2_name:
                    continue
                cam_src = cam_id_2_name[camera_data['sensor_id']]
                if cam != cam_src:
                    continue
                src_img_name = camera_data['file_path'].split("/")[-1]
                resized = cv2.resize(
                    combine_rgb,
                    (cam_info_all[cam]['size'][1], cam_info_all[cam]['size'][0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                cv2.imwrite(os.path.join(out_dir, src_img_name), resized)
                written += 1

    return written


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="postprocess 3dgs render image to uniscene format"
    )
    parser.add_argument("--data-root", help="path to uniscene data")
    parser.add_argument("--gs-data-root", help="path to 3dgs format data")
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("EXPERIMENT_NAME", "v_new_anno"),
        help="experiment subdir under gs-data-root (default: v_new_anno or $EXPERIMENT_NAME)",
    )
    parser.add_argument(
        "--render-timestamp",
        default=os.environ.get("RENDER_TIMESTAMP", ""),
        help="render run timestamp subdir (default: latest or $RENDER_TIMESTAMP)",
    )
    parser.add_argument(
        "--camera-filter",
        default=os.environ.get("CAMERA_FILTER", ""),
        help="only process cameras whose name contains this substring",
    )
    args = parser.parse_args()

    data_root = args.data_root
    gs_data_root = args.gs_data_root

    unisceneproto = os.path.join(data_root, "plannerGt/unisceneproto.json")
    uniscene = json.load(open(unisceneproto, "r"))

    reconstruction_dir = os.path.join(gs_data_root, args.experiment_name, "street-gaussians-ns")
    subdirs = [sub for sub in os.listdir(reconstruction_dir) if os.path.isdir(os.path.join(reconstruction_dir, sub))]

    subdirs.sort()

    render_timestamp = args.render_timestamp or subdirs[-1]
    render_dir = os.path.join(reconstruction_dir, render_timestamp, "renders/all/rgb")
    print(f"postprocess: render_dir={render_dir}")

    cam_info_all = {'center_camera_fov120': {'colmap_id': 1},
                    'left_front_camera': {'colmap_id': 2},
                    'left_rear_camera': {'colmap_id': 3},
                    'right_front_camera': {'colmap_id': 4},
                    'right_rear_camera': {'colmap_id': 5},
                    'rear_camera': {'colmap_id': 6},
                    'center_camera_fov30': {'colmap_id': 7},
                    'front_camera_fov195': {'colmap_id': 8},
                    'rear_camera_fov195': {'colmap_id': 9},
                    'right_camera_fov195': {'colmap_id': 10}}

    for cam_info in uniscene['cameras']:
        if cam_info['camera_name'] in cam_info_all.keys():
            cam_info_all[cam_info['camera_name']]['id'] = cam_info['camera_id']
            cam_info_all[cam_info['camera_name']]['size'] = [cam_info["height"], cam_info["width"]]

    cam_id_2_name = {v['id']: k for k, v in cam_info_all.items()}

    render_cams = [
        cam for cam in cam_info_all.keys()
        if has_render_output(render_dir, cam)
        and (not args.camera_filter or args.camera_filter in cam)
    ]

    if not render_cams:
        cam = infer_single_cam_name(gs_data_root, cam_info_all)
        render_cams = [cam]

    total_written = 0
    for cam in tqdm(render_cams):
        written = process_camera(
            cam, render_dir, gs_data_root, uniscene, cam_info_all, cam_id_2_name,
        )
        total_written += written
        if written == 0:
            print(f"WARNING: no postprocess images written for {cam}")

    print(f"postprocess done: {total_written} images written for {len(render_cams)} camera(s)")
