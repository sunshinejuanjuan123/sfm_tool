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
    parser.add_argument("--data-root", help="path to uniscene data root")
    parser.add_argument("--gs-data-root", help="path to 3dgs format results")
    parser.add_argument("--model-path", help="path to sam model path")
    parser.add_argument("--car-type", help="car type")
    args = parser.parse_args()

    data_root = args.data_root
    gs_data_root = args.gs_data_root
    model_path = args.model_path
    car_type = args.car_type

    device = "cuda"
    sam = sam_model_registry["vit_h"](checkpoint=model_path).to(device)
    predictor = SamPredictor(sam)

    src_img_dir = os.path.join(gs_data_root, 'images')
    seg_dir = os.path.join(gs_data_root, 'segs')
    dst_img_dir = os.path.join(gs_data_root, 'mask_combine')

    os.makedirs(dst_img_dir, exist_ok=True)

    def is_fisheye_cam(cam_name: str) -> bool:
        name = cam_name.lower()
        return "fov200" in name or "fov195" in name

    uniscene_ego_car_mask = os.path.join(data_root, "pvbGt/ego_car_masks")

    if os.path.exists(uniscene_ego_car_mask):
        print("use uniscene ego car mask")
        for cam in tqdm(os.listdir(src_img_dir)):
            os.makedirs(os.path.join(dst_img_dir, cam), exist_ok=True)
            pil_image = Image.open(os.path.join(src_img_dir, cam, os.listdir(os.path.join(src_img_dir, cam))[0]))
            img_seg = np.array(pil_image, dtype="int64")
            h, w, _ = img_seg.shape
            for img_name in os.listdir(os.path.join(uniscene_ego_car_mask, cam)):
                mask_img = cv2.imread(os.path.join(uniscene_ego_car_mask, cam, img_name))
                mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(dst_img_dir, cam, img_name), mask_img)
    else:
        if car_type == "zhiji":
            print("zhiji ego car mask")
            for cam in tqdm(os.listdir(src_img_dir)):
                os.makedirs(os.path.join(dst_img_dir, cam), exist_ok=True)
                pil_image = Image.open(os.path.join(src_img_dir, cam, os.listdir(os.path.join(src_img_dir, cam))[0]))
                img_seg = np.array(pil_image, dtype="int64")
                h, w, _ = img_seg.shape
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
                    
                elif cam in [
                    "left_front_camera",
                    "right_front_camera",
                    "rear_camera",
                    "center_camera_fov30",
                ] or is_fisheye_cam(cam):
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
        elif car_type == "pap":
            print("pap ego car mask")
            for cam in tqdm(os.listdir(src_img_dir)):
                mask_cam_dir = os.path.join(dst_img_dir, cam)
                os.makedirs(mask_cam_dir, exist_ok=True)

                src_cam_dir = os.path.join(src_img_dir, cam)
                pil_image = Image.open(os.path.join(src_img_dir, cam, os.listdir(os.path.join(src_img_dir, cam))[0]))
                img_seg = np.array(pil_image, dtype="int64")
                h, w, _ = img_seg.shape

                mask_combine = np.ones((h, w))
                if cam == "center_camera_fov120":
                    for img_name in os.listdir(src_cam_dir):
                        pil_image = Image.open(os.path.join(seg_dir, cam, img_name.replace(".jpg", ".png")))
                        img_seg = np.array(pil_image, dtype="int64")
                        masks = np.ones((h, w))
                        masks[(img_seg == 64)] = 0
                        mask_combine *= masks
                    
                    for img_name in os.listdir(os.path.join(src_cam_dir)):
                        cv2.imwrite(os.path.join(mask_cam_dir, img_name.replace(".jpg", ".png")), mask_combine)
                elif cam == "left_rear_camera":
                    for img_name in os.listdir(src_cam_dir):
                        rgb_img = cv2.imread(os.path.join(src_img_dir, cam, img_name))
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

                    for img_name in os.listdir(os.path.join(src_cam_dir)):
                        cv2.imwrite(os.path.join(mask_cam_dir, img_name.replace(".jpg", ".png")), mask_combine)
                elif cam == "right_rear_camera":
                    for img_name in os.listdir(src_cam_dir):
                        rgb_img = cv2.imread(os.path.join(src_img_dir, cam, img_name))
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

                    for img_name in os.listdir(os.path.join(src_cam_dir)):
                        cv2.imwrite(os.path.join(mask_cam_dir, img_name.replace(".jpg", ".png")), mask_combine)
                else:
                    for img_name in os.listdir(os.path.join(src_cam_dir)):
                        cv2.imwrite(os.path.join(mask_cam_dir, img_name.replace(".jpg", ".png")), mask_combine)
        else:
            print("not deal ego car mask")
            for cam in tqdm(os.listdir(src_img_dir)):
                mask_cam_dir = os.path.join(dst_img_dir, cam)
                os.makedirs(mask_cam_dir, exist_ok=True)

                src_cam_dir = os.path.join(src_img_dir, cam)
                pil_image = Image.open(os.path.join(src_img_dir, cam, os.listdir(os.path.join(src_img_dir, cam))[0]))
                img_seg = np.array(pil_image, dtype="int64")
                h, w, _ = img_seg.shape

                mask_combine = np.ones((h, w))
                for img_name in os.listdir(os.path.join(src_cam_dir)):
                    cv2.imwrite(os.path.join(mask_cam_dir, img_name.replace(".jpg", ".png")), mask_combine)