from sfm_tools.feature_extract_match.model.read_write_model import read_model, write_model, rotmat2qvec, Image, Point3D
from tqdm import tqdm
import numpy as np
import os
import json
from scipy.spatial.transform import Rotation as R
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="pose center")
    parser.add_argument("--data_root", help="path to uniscene data root")
    parser.add_argument("--gs_data_root", help="path to 3dgs format results")
    args = parser.parse_args()

    data_root = args.data_root
    gs_data_root = args.gs_data_root

    ori_sfm_dir = os.path.join(gs_data_root, "colmap/sparse/0")

    cameras, images, points3D = read_model(ori_sfm_dir, ext=".bin")

    output_dir = os.path.join(gs_data_root, "colmap/sparse/1")
    os.makedirs(output_dir, exist_ok=True)

    pose_cameras_center = []
    for idx in tqdm(images.keys()):
        Rw2c = images[idx].qvec2rotmat()
        Tw2c = images[idx].tvec
        w2c = np.eye(4)
        w2c[:3, :3] = Rw2c
        w2c[:3, 3] = Tw2c
        c2w = np.linalg.inv(w2c)
        pose_camera = c2w[:3, 3]
        pose_cameras_center.append(pose_camera)
    
    pose_cameras_center = np.array(pose_cameras_center)
    pose_center = np.mean(pose_cameras_center, axis=0)
    output_pose_center = 100 * pose_center

    new_cameras = cameras
    new_images = {}
    new_points3d = {}

    for idx in tqdm(images.keys()):
        Rw2c = images[idx].qvec2rotmat()
        Tw2c = images[idx].tvec
        w2c = np.eye(4)
        w2c[:3, :3] = Rw2c
        w2c[:3, 3] = Tw2c
        c2w = np.linalg.inv(w2c)
        Rc2w = c2w[:3, :3]
        Tc2w = c2w[:3, 3]
        new_Tc2w = Tc2w - pose_center
        new_c2w = np.eye(4)
        new_c2w[:3, :3] = Rc2w
        new_c2w[:3, 3] = new_Tc2w
        new_w2c = np.linalg.inv(new_c2w)
        new_qvec = rotmat2qvec(new_w2c[:3, :3])
        new_tvec = new_w2c[:3, 3]
        new_images[idx] = Image(
            id=idx,
            qvec=new_qvec,
            tvec=new_tvec,
            camera_id=images[idx].camera_id,
            name=images[idx].name,
            xys=images[idx].xys,
            point3D_ids=images[idx].point3D_ids,
        )
    
    for jdx in tqdm(points3D.keys()):
        xyz = points3D[jdx].xyz
        new_xyz = xyz - pose_center
        new_points3d[points3D[jdx].id] = Point3D(
            id=points3D[jdx].id,
            xyz=np.array(new_xyz),
            rgb=points3D[jdx].rgb,
            error=points3D[jdx].error,
            image_ids=points3D[jdx].image_ids,
            point2D_idxs=points3D[jdx].point2D_idxs,
        )

    write_model(new_cameras,
                new_images,
                new_points3d,
                output_dir,
                ext=".bin")
    
    unisceneproto = os.path.join(data_root, "plannerGt/unisceneproto.json")
    uniscene = json.load(open(unisceneproto, "r"))
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
            
            # intrinsic fx, fy, cx, cy
            fx, fy, cx, cy = cam_info['intrinsic']['fx'], cam_info['intrinsic']['fy'], cam_info['intrinsic']['cx'], cam_info['intrinsic']['cy']
            cam_info_all[cam_info['camera_name']]['intrinsic'] = np.array([fx, fy, cx, cy])
            
            # extrinsic 4*4
            quat = cam_info["extrinsic"]["quaternion"]
            trsl = cam_info["extrinsic"]["translation"]
            cam_info_all[cam_info['camera_name']]['extrinsic'] = np.eye(4)
            cam_info_all[cam_info['camera_name']]['extrinsic'][:3, :3] = R.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
            cam_info_all[cam_info['camera_name']]['extrinsic'][:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]) @ cam_info_all[cam_info['camera_name']]['extrinsic'][:3, :3]
            cam_info_all[cam_info['camera_name']]['extrinsic'][:3, 3] = np.array([trsl["x"], trsl["y"], trsl["z"]]) 

    ue_info = {}
    ue_info["position"] = [output_pose_center[0], -output_pose_center[1], output_pose_center[2]]
    ue_info["extrinsic"] = {}
    for cam in cam_info_all.keys():
        ue_info["extrinsic"][cam] = cam_info_all[cam]['extrinsic'].tolist()
    
    with open(os.path.join(output_dir, "position.json"), "w") as fout:
        json.dump(ue_info, fout, indent=4) 