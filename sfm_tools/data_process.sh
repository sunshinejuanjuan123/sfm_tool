data_root=$1
gs_data_root=$2

data_root=$(realpath "$data_root")
gs_data_root=$(realpath "$gs_data_root")

log_step() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting: $1"
}

# colmap sparse init camera intrinsics, extrinsics & pose
log_step "Initialize camera intrinsics, extrinsics, and pose"
python -m sfm_tools.colmap_sparse_init \
        --data_root $data_root \
        --output_path $gs_data_root \

# images segmentation
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
log_step "Segment images"
python $SCRIPT_DIR/mask2former/segs_generate.py \
        --root_path $gs_data_root \
        --config-file $SCRIPT_DIR/../config/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml \
        --opts MODEL.WEIGHTS $SCRIPT_DIR/../model/Mask2Former/model_final_90ee2d.pkl

# mask ego car
log_step "Ego car mask"
python -m sfm_tools.ego_car_mask \
    --root_path $gs_data_root \
    --model_path $SCRIPT_DIR/../model/Mask2Former/sam_vit_h_4b8939.pth

# feature extraction & matching
log_step "Feature extraction & Matching"
python -m sfm_tools.feature_extract_match.sp_sg_colmap_db --root_path $gs_data_root
python -m sfm_tools.colmap_db --gs_data_root $gs_data_root

# colmap point triangulator & bundle adjuestment
log_step "Point triangulator & BA"
mkdir -p $gs_data_root/colmap/sparse_sfm
colmap point_triangulator \
    --image_path $gs_data_root/images \
    --database_path $gs_data_root/colmap/feature_sp_sg/colmap.db \
    --input_path $gs_data_root/colmap/sparse_init/ \
    --output_path $gs_data_root/colmap/sparse_sfm/ \
    --refine_extrinsics 1 \
    
colmap rig_bundle_adjuster \
    --input_path $gs_data_root/colmap/sparse_sfm \
    --output_path $gs_data_root/colmap/sparse_sfm \
    --rig_config_path $gs_data_root/colmap/cam_rigid_config.json \
    --estimate_rig_relative_poses 0 \
    --RigBundleAdjustment.refine_relative_poses 1 \
    --BundleAdjustment.max_num_iterations 50 \
    --BundleAdjustment.refine_focal_length 0 \
    --BundleAdjustment.refine_principal_point 0 \
    --BundleAdjustment.refine_extra_params 0 \
    --BundleAdjustment.refine_extrinsics 1 \

# pose enu
log_step "pose enu"
python -m sfm_tools.pose_convert_enu \
    --data_root $data_root \
    --output_path $gs_data_root \

# sfm & lidar points combine
log_step "Combine sfm & lidar points"
python -m sfm_tools.combine_lidar_sfm_points \
    --data_root $data_root \
    --gs_data_root $gs_data_root \

# dynamic objects annotation & pcd
log_step "Dynamic objects info"
python -m sfm_tools.extract_dynamic_object \
    --data_root $data_root \
    --gs_data_root $gs_data_root

# pose center
log_step "pose center"
python -m sfm_tools.pose_center \
    --gs_data_root $gs_data_root
