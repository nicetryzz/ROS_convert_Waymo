#!/bin/bash

# 初始化conda
source ~/anaconda3/etc/profile.d/conda.sh
# 设置默认参数
BASE_DIR="/home/hqlab/workspace/dataset/parkinglot"
BAG_SUBDIR="raw_data/2025-07-22-01-08-14.bag"
OUTPUT_SUBDIR="data/20000"
INPUT_SIZE=518
ENCODER="vitl"

echo "Running RosbagReader..."
python src/rosbag_reader.py \
    --base_dir ${BASE_DIR} \
    --bag_subdir ${BAG_SUBDIR} \
    --output_subdir ${OUTPUT_SUBDIR}

echo "Running RosbagSampler..."
python src/rosbag_sampler.py \
    --source_dir ${BASE_DIR}/${OUTPUT_SUBDIR} \
    --sampling_rate 5 \
    --start_frame 0 \
    --end_frame 335

echo "Running PoseAdjuster..."
conda activate etc
python src/pose_adjusting.py \
    --base_dir ${BASE_DIR} \
    --subdir ${OUTPUT_SUBDIR}

echo "Running LidarTransform..."
conda activate etc
python src/lidar_transform.py \
    --base_dir ${BASE_DIR} \
    --ply_dir "${OUTPUT_SUBDIR}/lidar" \
    --pose_file "${OUTPUT_SUBDIR}/poses_adjusted.txt" \
    --output_dir "${OUTPUT_SUBDIR}/lidar_world_fix" \
    --npz_dir "${OUTPUT_SUBDIR}/lidar_npz"

echo "Running DepthEstimator..."
conda  activate depthanything
export PYTHONPATH="/home/hqlab/workspace/depth_estimation/Depth-Anything-V2:${PYTHONPATH}"
python src/depth_estimator.py \
    --base_dir ${BASE_DIR} \
    --input_subdir "${OUTPUT_SUBDIR}/images" \
    --output_subdir "${OUTPUT_SUBDIR}/depths" \
    --input_size ${INPUT_SIZE} \
    --encoder ${ENCODER} \
    --pred-only

echo "Running NormalEstimator..."
conda activate stable_normal
python src/normal_estimator.py \
    --base_dir ${BASE_DIR} \
    --input_subdir "${OUTPUT_SUBDIR}/images" \
    --output_subdir "${OUTPUT_SUBDIR}/normals"


echo "Running SemanticEstimator..."
conda  activate dinov2
export PYTHONPATH="/home/hqlab/workspace/base_model/dinov2:${PYTHONPATH}"
python src/semantic_estimator.py \
    --base_dir ${BASE_DIR} \
    --input_subdir "${OUTPUT_SUBDIR}/images" \
    --output_subdir "${OUTPUT_SUBDIR}/masks" \
    --save_vis


echo "Running Sequence Depth Estimator"
conda activate depthanything-vedio
python src/image_sequence_depth_estimator.py \
    --base_input_dir ${BASE_DIR}/${OUTPUT_SUBDIR}/"images" \
    --base_output_dir ${BASE_DIR}/${OUTPUT_SUBDIR}/"video_depth" \
    --save_npz


echo "Running PlaneDetector..."
conda activate etc
python src/plane_detector.py \
    --base_dir ${BASE_DIR} \
    --input_subdir ${OUTPUT_SUBDIR} \
    --output_subdir "${OUTPUT_SUBDIR}/planes" \
    --use_multiprocess \
    --fine_curvature_threshold 0 \
    --angle_threshold 15

echo "Running WaymoConvertor..."
conda  activate etc
python src/waymo_convertor.py \
    --base_dir ${BASE_DIR} \
    --subdir ${OUTPUT_SUBDIR}

# echo "Running DTUConvertor..."
# conda deactivate && source activate etc
# python src/DTU_convertor.py \
#     --base_dir ${BASE_DIR} \
#     --subdir ${OUTPUT_SUBDIR}

echo "数据处理完成!"
