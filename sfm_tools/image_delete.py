from tqdm import tqdm
import numpy as np
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="pose center")
    parser.add_argument("--gs_data_root", help="path to 3dgs format results")
    parser.add_argument("--cam", help="cam list")
    args = parser.parse_args()

    data_root = args.gs_data_root
    cameras = args.cam

    image_delete_file = open(os.path.join(data_root, "1.txt"), "w")
    
    for cam in os.listdir(os.path.join(data_root, "images")):
        if cam == cameras:
            continue
        for img_name in os.listdir(os.path.join(data_root, "images", cam)):
            image_delete_file.write("{}\n".format(os.path.join(cam, img_name)))
    
    image_delete_file.close()
