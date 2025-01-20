import argparse
import os
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm 
import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="use segments result & sam to get ego car mask"
    )
    parser.add_argument("--root_path", help="path to 3dgs format results")
    parser.add_argument("--model_path", help="path to sam model path")
    args = parser.parse_args()
    root_path = args.root_path
    model_path = args.model_path

    device = "cuda"
    sam = sam_model_registry["vit_h"](checkpoint=model_path).to(device)
    predictor = SamPredictor(sam)

    images_dir = os.path.join(root_path, 'images')
    seg_dir = os.path.join(root_path, 'segs')
    mask_dir = os.path.join(root_path, 'mask_combine')

    os.makedirs(mask_dir, exist_ok=True)

    for cam in tqdm(os.listdir(images_dir)):
        mask_cam_dir = os.path.join(mask_dir, cam)
        os.makedirs(mask_cam_dir, exist_ok=True)

        src_cam_dir = os.path.join(images_dir, cam)
        pil_image = Image.open(os.path.join(images_dir, cam, os.listdir(os.path.join(images_dir, cam))[0]))
        img_seg = np.array(pil_image, dtype="int64")
        h, w, _ = img_seg.shape

        mask_combine = np.ones((h, w))
        if cam == "center_camera_fov120":
            for img_name in tqdm(os.listdir(src_cam_dir)):
                pil_image = Image.open(os.path.join(seg_dir, cam, img_name.replace(".jpg", ".png")))
                img_seg = np.array(pil_image, dtype="int64")
                masks = np.ones((h, w))
                masks[(img_seg == 64)] = 0
                mask_combine *= masks
            
            for img_name in tqdm(os.listdir(os.path.join(src_cam_dir))):
                cv2.imwrite(os.path.join(mask_cam_dir, img_name.replace(".jpg", ".png")), mask_combine)
        elif cam == "left_rear_camera":
            for img_name in tqdm(os.listdir(src_cam_dir)):
                rgb_img = cv2.imread(os.path.join(images_dir, cam, img_name))
                h, w, _ = rgb_img.shape

                predictor = SamPredictor(sam)
                predictor.set_image(rgb_img)

                input_point = np.array([[10, 10]])

                masks, scores, logit = predictor.predict(
                    point_coords = input_point,
                    point_labels = np.array([1]),
                    multimask_output = False
                )

                masks = masks.reshape(h, w, 1).astype(np.uint8)
                masks = (1 - masks).astype(np.uint8)
            
                masks = np.squeeze(masks)
                mask_combine *= masks

            for img_name in tqdm(os.listdir(os.path.join(src_cam_dir))):
                cv2.imwrite(os.path.join(mask_cam_dir, img_name.replace(".jpg", ".png")), mask_combine)
        elif cam == "right_rear_camera":
            for img_name in tqdm(os.listdir(src_cam_dir)):
                rgb_img = cv2.imread(os.path.join(images_dir, cam, img_name))
                h, w, _ = rgb_img.shape

                predictor = SamPredictor(sam)
                predictor.set_image(rgb_img)

                input_point = np.array([[1900, 10]])

                masks, scores, logit = predictor.predict(
                    point_coords = input_point,
                    point_labels = np.array([1]),
                    multimask_output = False
                )

                masks = masks.reshape(h, w, 1).astype(np.uint8)
                masks = (1 - masks).astype(np.uint8)
                masks = np.squeeze(masks)
                mask_combine *= masks

            for img_name in tqdm(os.listdir(os.path.join(src_cam_dir))):
                cv2.imwrite(os.path.join(mask_cam_dir, img_name.replace(".jpg", ".png")), mask_combine)
        else:
            for img_name in tqdm(os.listdir(os.path.join(src_cam_dir))):
                cv2.imwrite(os.path.join(mask_cam_dir, img_name.replace(".jpg", ".png")), mask_combine)