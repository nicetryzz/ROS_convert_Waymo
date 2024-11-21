#!/bin/bash

# 设置默认参数
BASE_DIR="/home/hqlab/workspace/dataset/parkinglot"
BAG_SUBDIR="raw_data/cql_circle_2024-10-26-01-28-40.bag"
OUTPUT_SUBDIR="data/10_26"
INPUT_SIZE=518
ENCODER="vitl"

# echo "Running RosbagReader..."
# python src/rosbag_reader.py \
#     --base_dir ${BASE_DIR} \
#     --bag_subdir ${BAG_SUBDIR} \
#     --output_subdir ${OUTPUT_SUBDIR}

# echo "Running LidarTransform..."
# source activate etc
# python src/lidar_transform.py \
#     --base_dir ${BASE_DIR} \
#     --ply_dir "${OUTPUT_SUBDIR}/lidar" \
#     --pose_file "${OUTPUT_SUBDIR}/lidar_poses.txt" \
#     --output_dir "${OUTPUT_SUBDIR}/lidar_world"

# echo "Running DepthEstimator..."
# source activate depthanything
# export PYTHONPATH="/home/hqlab/workspace/depth_estimation/Depth-Anything-V2:${PYTHONPATH}"
# python src/depth_estimator.py \
#     --base_dir ${BASE_DIR} \
#     --input_subdir "${OUTPUT_SUBDIR}/images" \
#     --output_subdir "${OUTPUT_SUBDIR}/depths" \
#     --input_size ${INPUT_SIZE} \
#     --encoder ${ENCODER} \
#     --pred-only

# echo "Running NormalEstimator..."
# source activate stable_normal
# python src/normal_estimator.py \
#     --base_dir ${BASE_DIR} \
#     --input_subdir "${OUTPUT_SUBDIR}/images" \
#     --output_subdir "${OUTPUT_SUBDIR}/normals"


# echo "Running SemanticEstimator..."
# source activate dinov2
# export PYTHONPATH="/home/hqlab/workspace/base_model/dinov2:${PYTHONPATH}"
# python src/semantic_estimator.py \
#     --base_dir ${BASE_DIR} \
#     --input_subdir "${OUTPUT_SUBDIR}/images" \
#     --output_subdir "${OUTPUT_SUBDIR}/masks" 
#     # --save_vis


# echo "Running PlaneDetector..."
# source activate etc
# python src/plane_detector.py \
#     --base_dir ${BASE_DIR} \
#     --input_subdir ${OUTPUT_SUBDIR} \
#     --output_subdir "${OUTPUT_SUBDIR}/planes" \
#     --save_vis

echo "Running WaymoConvertor..."
source activate dinov2
python src/waymo_convertor.py \
    --base_dir ${BASE_DIR} \
    --subdir ${OUTPUT_SUBDIR}

echo "数据处理完成!"
