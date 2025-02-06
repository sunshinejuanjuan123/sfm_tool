import sqlite3
import numpy as np
import os
import argparse
from sfm_tools.feature_extract_match.model.read_write_model import read_model, rotmat2qvec
import json

# 解码 pair_id 为图像对
def decode_pair_id(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    return image_id1, image_id2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="postprocess colmap db & rigid_ba"
    )
    parser.add_argument("--gs_data_root", help="path to 3dgs format results")
    args = parser.parse_args()

    data_root = args.gs_data_root
    
    db_path = os.path.join(data_root, "colmap/feature_sp_sg/colmap.db")
    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()

    cursor.execute("SELECT pair_id, rows FROM two_view_geometries")
    pairs = cursor.fetchall()

    cursor.execute("SELECT image_id, name FROM images")
    images = cursor.fetchall()
    image_dict = {image_id: name for image_id, name in images}

    for pair in pairs:
        image_id1, image_id2 = decode_pair_id(pair[0])
        name1, name2 = image_dict[image_id1], image_dict[image_id2]
        cam1, cam2 = name1.split("/")[0], name2.split("/")[0]
        if cam1 == "center_camera_fov30" and cam2 not in ["center_camera_fov30"]: 
            cursor.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair[0],))
        if cam2 == "center_camera_fov30" and cam1 not in ["center_camera_fov30"]: 
            cursor.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair[0],))
        
        if cam1 == "center_camera_fov120" and cam2 not in ["center_camera_fov120", "left_front_camera", "right_front_camera"]: 
            cursor.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair[0],))
        if cam2 == "center_camera_fov120" and cam1 not in ["center_camera_fov120", "left_front_camera", "right_front_camera"]: 
            cursor.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair[0],))

        if cam1 == "left_front_camera" and cam2 not in ["left_front_camera", "center_camera_fov120", "left_rear_camera"]:
            cursor.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair[0],))
        if cam2 == "left_front_camera" and cam1 not in ["left_front_camera", "center_camera_fov120", "left_rear_camera"]:
            cursor.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair[0],))
        
        if cam1 == "left_rear_camera" and cam2 not in ["left_rear_camera", "left_front_camera", "rear_camera"]:
            cursor.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair[0],))
        if cam2 == "left_rear_camera" and cam1 not in ["left_rear_camera", "left_front_camera", "rear_camera"]:
            cursor.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair[0],))
        
        if cam1 == "rear_camera" and cam2 not in ["rear_camera", "left_rear_camera", "right_rear_camera"]:
            cursor.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair[0],))
        if cam2 == "rear_camera" and cam1 not in ["rear_camera", "left_rear_camera", "right_rear_camera"]:
            cursor.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair[0],))
        
        if cam1 == "right_rear_camera" and cam2 not in ["right_rear_camera", "rear_camera", "right_front_camera"]:
            cursor.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair[0],))
        if cam2 == "right_rear_camera" and cam1 not in ["right_rear_camera", "rear_camera", "right_front_camera"]:
            cursor.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair[0],))
        
        if cam1 == "right_front_camera" and cam2 not in ["right_front_camera", "right_rear_camera", "center_camera_fov120"]:
            cursor.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair[0],))
        if cam2 == "right_front_camera" and cam1 not in ["right_front_camera", "right_rear_camera", "center_camera_fov120"]:
            cursor.execute("DELETE FROM two_view_geometries WHERE pair_id = ?", (pair[0],))

    conn.commit()
    conn.close()

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

