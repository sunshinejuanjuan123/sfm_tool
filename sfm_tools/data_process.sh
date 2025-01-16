#!/bin/bash

data_root=$1
gs_data_root=$2

data_root=$(realpath "$data_root")
gs_data_root=$(realpath "$gs_data_root")

log_step() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting: $1"
}

# colmap sparse init camera intrinsics, extrinsics & pose
log_step "Initialize camera intrinsics, extrinsics, and pose"
python sfm_tools/colmap_sparse_init.py \
        --data_root $data_root \
        --output_path $gs_data_root \

# images segmentation
log_step "Segment images"
python street-gaussians/dependencies/Mask2Former/segs_generate.py \
        --root_path $gs_data_root \
        --config-file street-gaussians/dependencies/Mask2Former/configs/mapillary-vistas/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml \
        --opts MODEL.WEIGHTS street-gaussians/dependencies/Mask2Former/models/model_final_90ee2d.pkl

# mask ego car
log_step "Ego car mask"
python sfm_tools/ego_car_mask.py --root_path $gs_data_root

# feature extraction & matching
log_step "Feature extraction & Matching"
python sfm_tools/feature_extract_match/sp_sg_colmap_db.py --root_path $gs_data_root

# colmap point triangulator & bundle adjuestment
log_step "Point triangulator & BA"
mkdir -p $gs_data_root/colmap/sparse_sfm
colmap point_triangulator --image_path $gs_data_root/images \
    --database_path $gs_data_root/colmap/feature_sp_sg/colmap.db \
    --input_path $gs_data_root/colmap/sparse_init/ \
    --output_path $gs_data_root/colmap/sparse_sfm/ \

# sfm & lidar points combine
log_step "Combine sfm & lidar points"
python sfm_tools/combine_lidar_sfm_points.py \
    --data_root $data_root \
    --gs_data_root $gs_data_root \

# dynamic objects annotation & pcd
log_step "Dynamic objects info"
python sfm_tools/extract_dynamic_object.py \
    --data_root $data_root \
    --gs_data_root $gs_data_root