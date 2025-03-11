import numpy as np
import os
import argparse
from sfm_tools.feature_extract_match.model.read_write_model import read_model, rotmat2qvec
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="postprocess colmap db & rigid_ba"
    )
    parser.add_argument("--gs_data_root", help="path to 3dgs format results")
    args = parser.parse_args()

    data_root = args.gs_data_root
    sparse_init = os.path.join(data_root, "colmap/sparse_init")
    cameras, images, points3D = read_model(sparse_init, ext=".txt")
    cam_rigid = dict()

    camera_name_2_id = {'center_camera_fov120': 1,
                    'left_front_camera': 2,
                    'left_rear_camera': 3, 
                    'right_front_camera': 4,  
                    'right_rear_camera': 5,
                    'rear_camera': 6,
                    'center_camera_fov30': 7}
    
    ref_camera_id = camera_name_2_id['center_camera_fov120']
    cam_rigid["ref_camera_id"] = ref_camera_id
    rigid_cam_list = []

    for cam in camera_name_2_id.keys():
        rigid_cam = dict()
        rigid_cam["camera_id"] = camera_name_2_id[cam]
        
        for ii in images.keys():
            cam_ii, image_name_ii = images[ii].name.split("/")
            if cam_ii == "center_camera_fov120":
                ii_unique = ii
                break
        for jj in images.keys():
            cam_jj, image_name_jj = images[jj].name.split("/")
            if cam_jj == cam and image_name_jj == image_name_ii:
                jj_unique = jj
                break

        ref_extrinsic = np.eye(4)
        ref_extrinsic[:3, :3] = images[ii_unique].qvec2rotmat()
        ref_extrinsic[:3, 3] = images[ii_unique].tvec
        ref_extrinsic = np.linalg.inv(ref_extrinsic)
        
        cur_extrinsic = np.eye(4)
        cur_extrinsic[:3, :3] = images[jj_unique].qvec2rotmat()
        cur_extrinsic[:3, 3] = images[jj_unique].tvec
        cur_extrinsic = np.linalg.inv(cur_extrinsic)

        rel_extrinsic = np.linalg.inv(cur_extrinsic) @ ref_extrinsic

        qvec = rotmat2qvec(rel_extrinsic[:3, :3])
        tvec = rel_extrinsic[:3, 3]

        rigid_cam["image_prefix"] = cam
        rigid_cam['cam_from_rig_rotation'] = qvec.tolist()
        rigid_cam['cam_from_rig_translation'] = tvec.tolist()
        rigid_cam_list.append(rigid_cam)
    
    cam_rigid["cameras"] = rigid_cam_list

    rigid_config_path = os.path.join(data_root, "colmap/cam_rigid_config.json")
    with open(rigid_config_path, "w+") as f:
        json.dump([cam_rigid], f, indent=4)   