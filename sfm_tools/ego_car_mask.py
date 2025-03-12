import argparse
import os
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm 
import numpy as np

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

    src_img_dir = os.path.join(root_path, 'images')
    seg_dir = os.path.join(root_path, 'segs')
    dst_img_dir = os.path.join(root_path, 'mask_combine')

    os.makedirs(dst_img_dir, exist_ok=True)
 
    for cam in tqdm(os.listdir(src_img_dir)):
        os.makedirs(os.path.join(dst_img_dir, cam), exist_ok=True)
        pil_image = Image.open(os.path.join(src_img_dir, cam, os.listdir(os.path.join(src_img_dir, cam))[0]))
        img_seg = np.array(pil_image, dtype="int64")
        h, w, _ = img_seg.shape
        mask_combine = np.ones((h, w))
        if cam == "center_camera_fov120":
            ego_car_combine = np.ones((h, w))
            for img_name in os.listdir(os.path.join(src_img_dir, cam)):
                pil_image = Image.open(os.path.join(seg_dir, cam, img_name.replace(".jpg", ".png")))
                img_seg = np.array(pil_image, dtype="int64")
                visual_mask = np.ones_like(img_seg)
                visual_mask[(img_seg == 64)] = 0
                ego_car_combine *= visual_mask                 
            
            ego_car_combine[:700, 600:] = 1
            for img_name in os.listdir(os.path.join(src_img_dir, cam)):
                rgb_img = cv2.imread(os.path.join(src_img_dir, cam, img_name))
                
                # 创建黑色mask
                black_mask = np.all(rgb_img == 0, axis=-1) 
                visual_mask = (1-black_mask).astype(np.uint8)

                visual_mask *= ego_car_combine.astype(np.uint8)
                cv2.imwrite(os.path.join(dst_img_dir, cam, img_name.replace(".jpg", ".png")), visual_mask)
            
        elif cam in ["left_front_camera", "right_front_camera", "rear_camera", "center_camera_fov30"]:
            for img_name in os.listdir(os.path.join(src_img_dir, cam)):
                rgb_img = cv2.imread(os.path.join(src_img_dir, cam, img_name))
                
                # 创建黑色mask
                black_mask = np.all(rgb_img == 0, axis=-1) 
                visual_mask = (1-black_mask).astype(np.uint8)
                cv2.imwrite(os.path.join(dst_img_dir, cam, img_name.replace(".jpg", ".png")), visual_mask)
        elif cam in ["left_rear_camera"]:
            ego_car_combine = np.ones((h, w))
            for img_name in os.listdir(os.path.join(src_img_dir, cam)):
                pil_image = Image.open(os.path.join(seg_dir, cam, img_name.replace(".jpg", ".png")))
                img_seg = np.array(pil_image, dtype="int64")
                visual_mask = np.ones_like(img_seg)
                visual_mask[(img_seg == 55)] = 0
                ego_car_combine *= visual_mask               

            ego_car_combine[:, 230:] = 1
            for img_name in os.listdir(os.path.join(src_img_dir, cam)):
                rgb_img = cv2.imread(os.path.join(src_img_dir, cam, img_name))
                
                # 创建黑色mask
                black_mask = np.all(rgb_img == 0, axis=-1) 
                visual_mask = (1-black_mask).astype(np.uint8)

                visual_mask *= ego_car_combine.astype(np.uint8)
                cv2.imwrite(os.path.join(dst_img_dir, cam, img_name.replace(".jpg", ".png")), visual_mask)
        elif cam in ["right_rear_camera"]:
            ego_car_combine = np.ones((h, w))
            for img_name in os.listdir(os.path.join(src_img_dir, cam)):
                pil_image = Image.open(os.path.join(seg_dir, cam, img_name.replace(".jpg", ".png")))
                img_seg = np.array(pil_image, dtype="int64")
                visual_mask = np.ones_like(img_seg)
                visual_mask[(img_seg == 55)] = 0
                ego_car_combine *= visual_mask               

            ego_car_combine[:, :1750] = 1
            for img_name in os.listdir(os.path.join(src_img_dir, cam)):
                rgb_img = cv2.imread(os.path.join(src_img_dir, cam, img_name))
                
                # 创建黑色mask
                black_mask = np.all(rgb_img == 0, axis=-1) 
                visual_mask = (1-black_mask).astype(np.uint8)

                visual_mask *= ego_car_combine.astype(np.uint8)
                cv2.imwrite(os.path.join(dst_img_dir, cam, img_name.replace(".jpg", ".png")), visual_mask)