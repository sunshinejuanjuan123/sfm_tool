from sfm_tools.feature_extract_match.model.read_write_model import read_model, write_model, rotmat2qvec, Image, Point3D
from tqdm import tqdm
import numpy as np
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="pose center")
    parser.add_argument("--gs_data_root", help="path to 3dgs format results")
    args = parser.parse_args()

    data_root = args.gs_data_root

    ori_sfm_dir = os.path.join(data_root, "colmap/sparse/0")

    cameras, images, points3D = read_model(ori_sfm_dir, ext=".bin")

    output_dir = os.path.join(data_root, "colmap/sparse/1")
    os.makedirs(output_dir, exist_ok=True)

    scale_txt = open(os.path.join(output_dir,  "scale.txt"), "w")
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
    scale_txt.write("(X={},Y={},Z={})".format(output_pose_center[0], -output_pose_center[1], output_pose_center[2]))
    scale_txt.close()
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
    
    d_jx = 0
    for jdx in tqdm(points3D.keys()):
        d_jx += 1
        xyz = points3D[jdx].xyz
        new_xyz = xyz - pose_center
        new_points3d[d_jx] = Point3D(
            id=d_jx,
            xyz=np.array(new_xyz),
            rgb=points3D[jdx].rgb,
            error=points3D[jdx].error,
            image_ids=points3D[idx].image_ids,
            point2D_idxs=points3D[idx].point2D_idxs,
        )

    write_model(new_cameras,
                new_images,
                new_points3d,
                output_dir,
                ext=".bin")