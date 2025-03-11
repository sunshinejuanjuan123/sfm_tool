import argparse
import os
from tqdm import tqdm
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import cv2
import random
from sfm_tools.feature_extract_match.model.read_write_model import read_model

def get_box_corners(center, dimensions, orientation):

    cx, cy, cz = center
    length, width, height = dimensions
    q = orientation

    dx  = length / 2.0
    dy = width / 2.0
    dz = height / 2.0

    corners = np.array(
        [
            [dx, dy, dz],
            [-dx, dy, dz],
            [-dx, -dy, dz],
            [dx, -dy, dz],
            [dx, dy, -dz],
            [-dx, dy, -dz],
            [-dx, -dy, -dz],
            [dx, -dy, -dz],
        ]
    )

    rotation = R.from_quat([q[1], q[2], q[3], q[0]])  
    rotated_corners = rotation.apply(corners)

    world_corners = rotated_corners + center

    return world_corners

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="extract dynamic objects"
    )
    parser.add_argument("--data_root", help="path to pvb data")
    parser.add_argument("--gs_data_root", help="path to 3dgs format results")
    parser.add_argument("--cam", help="project camera")
    args = parser.parse_args()

    data_root = args.data_root
    unisceneproto = os.path.join(data_root, "plannerGt/unisceneproto.json")
    uniscene = json.load(open(unisceneproto, "r"))

    gs_data_root = args.gs_data_root
    sparse_dir = os.path.join(gs_data_root, "colmap/sparse_sfm_enu")
    cameras1, images1, points3D1 = read_model(sparse_dir, ext=".bin")

    sparse_dir = os.path.join(gs_data_root, "colmap/sparse_init_enu")
    cameras2, images2, points3D2 = read_model(sparse_dir, ext=".txt")

    project_camera = args.cam
    track_info = {}
    pose_info = {}

    for ego_info in tqdm(uniscene['ego_status']):
        timestamp = int(round(ego_info['timestamp'], 3)*1000)
        quat = ego_info['ego_orientation']
        trsl = ego_info['ego_position']
        pose_info[timestamp] = np.eye(4)
        pose_info[timestamp][:3, :3] = R.from_quat([quat["x"], quat["y"], quat["z"], quat["w"]]).as_matrix()
        pose_info[timestamp][:3, 3] = np.array([trsl["x"], trsl["y"], trsl["z"]])   

    for sensor_info in tqdm(uniscene['sensor_frames']):
        timestamp = int(round(sensor_info['timestamp'], 3)*1000)
        track_info[timestamp] = []

    object_type_id_2_class = {
        0: "Other",
        1: "Car",
        2: "Pedestrian",
        3: "Cyclist",
        4: "Other",
        5: "Truck",
        6: "Other"}

    for objects_info in uniscene['tracks']:
        for track_frame_info in objects_info['object_states']:
            object_timestamp = int(round(track_frame_info['timestamp'], 3)*1000)

            skip_timestamp = False
            for ii in images1.keys():
                cam, image_name = images1[ii].name.split("/")
                if cam == project_camera and image_name == (str(object_timestamp)+".jpg"):
                    K = cameras1[images1[ii].camera_id].params
                    fx, fy, cx, cy = K[0], K[1], K[2], K[3]
                    intrinsic_matrix = np.array([[fx, 0, cx, 0],
                                                    [0, fy, cy, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]])
                    Rw2c = images1[ii].qvec2rotmat()
                    Twc2 = images1[ii].tvec
                    w2c = np.eye(4)
                    w2c[:3, :3] = Rw2c
                    w2c[:3, 3] = Twc2
                    rgb_img = cv2.imread(os.path.join(gs_data_root, 'images', images1[ii].name))
                    h, w, _ = rgb_img.shape
                    ii_unique = ii 
                    skip_timestamp = True
            
            if not skip_timestamp:
                continue

            for jj in images2.keys():
                cam, image_name = images2[jj].name.split("/")
                if cam == project_camera and image_name == (str(object_timestamp)+".jpg"):
                    jj_unique = jj 

            Rw2c = images2[jj_unique].qvec2rotmat()
            Twc2 = images2[jj_unique].tvec
            w2c_opt = np.eye(4)
            w2c_opt[:3, :3] = Rw2c
            w2c_opt[:3, 3] = Twc2

            opt = np.linalg.inv(w2c) @ w2c_opt

            if object_timestamp in track_info.keys():
                center_x, center_y, center_z = track_frame_info['position_from_ego']['x'], track_frame_info['position_from_ego']['y'], track_frame_info['position_from_ego']['z']
                box_length, box_width, box_height = track_frame_info['length'], track_frame_info['width'], track_frame_info['height']
                yaw_info = track_frame_info['heading_from_ego']

                center_vcs = np.array([center_x, center_y, center_z, 1])
                center_wcs = pose_info[object_timestamp] @ center_vcs
                
                R_opt = opt[:3, :3]
                t_opt = opt[:3, 3]
                
                rotation_vcs = R.from_euler("xyz", [0, 0, yaw_info], degrees=False).as_matrix()

                rotation_wcs = pose_info[object_timestamp][:3, :3] @ rotation_vcs
                rotation_wcs = R_opt @ rotation_wcs
                rotation_wcs = R.from_matrix(rotation_wcs).as_quat()
                translation = center_wcs[:3]
                translation = R_opt @ translation + t_opt

                speed_x, speed_y = track_frame_info['velocity_x'], track_frame_info['velocity_y']
                speed = np.sqrt(speed_x**2 + speed_y**2)

                track_info[object_timestamp].append(
                    {
                        "type": object_type_id_2_class[int(objects_info['object_type'])],
                        "gid": object_type_id_2_class[int(objects_info['object_type'])] + "_" + str(objects_info['object_id']),
                        "translation": translation.tolist(),
                        "size": [box_length, box_width, box_height],
                        "rotation": [rotation_wcs[3], rotation_wcs[0], rotation_wcs[1], rotation_wcs[2]],
                        "is_moving": bool(speed > 0.2)
                    }
                )
    annotations = []
    for timestamp in track_info.keys():
        annotations.append({"timestamp": timestamp, "objects": track_info[timestamp]})

    with open(os.path.join(gs_data_root, "annotation.json"), "w") as fout:
            json.dump({"frames": annotations}, fout, indent=4)
    
    sparse_dir = os.path.join(gs_data_root, "colmap/sparse_sfm_enu")
    cameras, images, points3D = read_model(sparse_dir, ext=".bin")

    annotation_path = os.path.join(gs_data_root, "annotation.json")
    with open(annotation_path, "r") as f:
         annotation_data = json.load(f)

    lidar_project_camera_list = [project_camera]
    print("lidar_project_camera_list:{}".format(lidar_project_camera_list))
    obj_pcd = {}
    annotation_frames = annotation_data['frames']

    i = 1
    
    for sensor_info in tqdm(uniscene['sensor_frames']):
        timestamp = int(round(sensor_info['timestamp'], 3)*1000)
        lidar2enu = pose_info[timestamp]

        obbs = []
        for idx in range(len(annotation_frames)):
            if annotation_frames[idx]['timestamp'] == timestamp:
                for object in annotation_frames[idx]["objects"]:
                    if object["is_moving"]:
                        if object["gid"] not in obj_pcd:
                                obj_pcd[object["gid"]] = {
                                    'xyz': [],
                                    'rgb': [],
                                }
                        q = object["rotation"]
                        rotation_matrix = R.from_quat([q[1], q[2], q[3], q[0]])
                        obj = {
                            'gid': object["gid"],
                            'translation': object["translation"],
                            'size': object["size"],
                            'rotation': rotation_matrix.as_matrix(),
                        }
                        translation = object["translation"]
                        lwh = object["size"]
                        rotation = object["rotation"]
                        world_corners = get_box_corners(translation, lwh, rotation)
                        obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(world_corners))
                        scale_x = 1.0
                        scale_y = 1.0
                        scale_z = 1.0
                        extents = np.array(obb.extent) * np.array([scale_x, scale_y, scale_z]) # 更新边界长度
                        obb = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, extents)
                        obj['obb'] = obb
                        obbs.append(obj)
            else:
                continue
        
        lidar_abs_path = os.path.join(data_root, sensor_info['lidar_data'][0]['file_path'])
        pcd_data = o3d.io.read_point_cloud(lidar_abs_path)
        points = np.array(pcd_data.points)
        nan_rows = np.isnan(points).any(axis=1)
        points = points[~nan_rows]
        
        homogeneous_positions = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed_positions = np.dot(lidar2enu, homogeneous_positions.T).T[:, :3]

        obj_pcds = {}
        if len(obbs) > 0:
            pcds = point_cloud = o3d.geometry.PointCloud()
            pcds.points = o3d.utility.Vector3dVector(transformed_positions)
            for obj in obbs:
                obb = obj['obb']
                inliers_indices = obb.get_point_indices_within_bounding_box(pcds.points)
                inliers_pcd =  pcds.select_by_index(inliers_indices, invert=False)
                obj_pcds[obj['gid']] = np.array(inliers_pcd.points)
        
        obj_ixd = 0

        for idx in images.keys():
            if images[idx].name.split("/")[0] not in lidar_project_camera_list:
                continue
            if images[idx].name.split("/")[-1][:-4] == str(timestamp):
                Rw2c = np.array(images[idx].qvec2rotmat())
                Tw2c = np.array(images[idx].tvec)
                T = np.eye(4)
                T[:3, :3] = Rw2c 
                T[:3, 3] = Tw2c   
                w2c = T
                K = cameras[images[idx].camera_id].params
                fx, fy, cx, cy = K[0], K[1], K[2], K[3]

                img_abs_path = os.path.join(gs_data_root, "images", images[idx].name)
                rgb = cv2.imread(img_abs_path)
                h, w, _ = rgb.shape

                intrinsic_matrix = np.array([[fx, 0, cx, 0],
                                            [0, fy, cy, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])
                for gid, pts in obj_pcds.items():
                    assert obbs[obj_ixd]['gid'] == gid
                    obj = obbs[obj_ixd]
                    t = obj['translation']
                    rot = obj['rotation']
                    o2w = np.eye(4)
                    o2w[:3,:3] = rot
                    o2w[:3,3] = t
                    w2o = np.linalg.inv(o2w)
                    for pt in pts:
                        if abs(pt[0]) >100000:
                            continue
                        m_1= np.array([pt[0],pt[1],pt[2],1])
                        uv_homogeneous = intrinsic_matrix @ w2c @ m_1
                        u, v = (uv_homogeneous[:2] / uv_homogeneous[2]).astype(int)
                        if 0 <= u < w and 0 <= v < h and uv_homogeneous[2]>0:
                            rgb_point = rgb[v, u] / 255.
                            error = random.uniform(0,1)
                            pt_obj = w2o@m_1
                            pt_obj = pt_obj[:3] / pt_obj[3]
                            obj_pcd[gid]['xyz'].append(pt_obj)
                            obj_pcd[gid]['rgb'].append(rgb_point)
                            i += 1
                    obj_ixd += 1

    save_path = gs_data_root + f"/aggregate_lidar/dynamic_objects/"
    os.makedirs(gs_data_root + f"/aggregate_lidar/", exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    for gid, pcd in obj_pcd.items():
        if np.array(pcd['xyz']).shape[0] > 0:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array(pcd['xyz']).astype(np.float32))
            point_cloud.colors = o3d.utility.Vector3dVector(np.array(pcd['rgb']).astype(np.float32))
            o3d.io.write_point_cloud(str(save_path + f"{gid}.ply"), point_cloud)