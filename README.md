# ROS数据预处理工具

本仓库提供了一套完整的ROS数据预处理工具，用于处理和转换自动驾驶数据集。主要功能包括数据格式转换、单目深度估计、法向量估计以及平面检测。

## 一键运行（推荐）

仓库内提供了 `run_task.sh`，串联执行：rosbag解析→采样→位姿修正→点云坐标转换/导出npz→深度/法向/语义→平面→Waymo打包。

```bash
bash run_task.sh
```

注意：脚本里包含多个 conda 环境切换（例如 `etc`、`depthanything` 等），请先按你的机器环境准备好对应环境与依赖。

## RosbagReader

`rosbag_reader.py` 模块用于从ROS包中提取和同步多相机图像数据与点云数据。支持6个环视相机的图像提取，并将点云数据转换为PLY格式保存。

### 输入

- **rosbag文件**：包含多相机图像数据和点云数据的ROS包文件。
- **输出目录**：用于存储提取后的图像和点云数据的目录路径。

### 输出

- **图像数据**：提取的多相机图像，按相机名称存储为PNG格式。输出目录结构如下：

- **点云数据**：提取的点云数据，存储为PLY格式。

## RosbagSampler

`rosbag_sampler.py` 用于对 `RosbagReader` 的输出进行抽帧/采样，生成更稀疏的序列（例如每 5 帧取 1 帧），减少后续模型推理开销。

`run_task.sh` 中示例参数：

```bash
python src/rosbag_sampler.py \
  --source_dir ${BASE_DIR}/${OUTPUT_SUBDIR} \
  --sampling_rate 5 \
  --start_frame 0 \
  --end_frame 335
```

## PoseAdjuster

`pose_adjusting.py` 用于对位姿进行修正/平滑，并在输出目录下生成 `poses_adjusted.txt`。

`run_task.sh` 里后续模块（如 `lidar_transform.py`、`waymo_convertor.py`）默认使用该文件。

## LidarTransform

`lidar_transform.py` 模块用于：

1) 将雷达点云从雷达坐标系转换到世界坐标系并保存为 PLY（用于检查/可视化/下游处理）。

2) 将每帧点云导出为 **局部坐标系** 下的 rays 格式（NPZ），用于以“光线 + 深度”的形式使用点云。

### 输入

- **点云数据**：原始雷达点云数据（PLY格式）。
- **位姿数据**：雷达在世界坐标系下的位姿信息（3*4的RT矩阵，TXT格式）。

### 输出

- **转换后的点云**：世界坐标系下的点云数据（PLY格式）。

- **ray格式点云（NPZ, 局部坐标系）**：每个 `.ply` 对应一个 `.npz`，包含：
  - `rays_o`: `[N, 3]`，每条光线的起点。在当前实现中为全 0（局部雷达坐标系原点）。
  - `rays_d`: `[N, 3]`，每条光线的方向（局部坐标系下单位向量）。
  - `ranges`: `[N]`，每条光线的距离/量测深度（与 `points` 的模长一致）。

> 说明：现在 `ply2npz` 导出的是**局部坐标系** rays，因此**不依赖位姿文件**。如果你需要把 rays 变到世界系，请在使用阶段再用对应帧位姿做转换：
> - `rays_o_world = t`（位姿平移）
> - `rays_d_world = R @ rays_d_local`（位姿旋转）

## DepthEstimator

`depth_estimator.py` 模块使用Depth-Anything-V2模型对提取的图像进行单目深度估计。支持处理6个环视相机的图像，并输出深度图和可视化结果。

### 输入

- **图像数据**：从ROS包中提取的多相机图像（PNG格式）。

### 输出

- **深度图数据**：估计的深度图，按相机名称存储为NPZ格式。

## NormalEstimator

`normal_estimator.py` 模块使用StableNormal模型对提取的图像进行法向量估计。支持处理6个环视相机的图像，并输出法向量图和可视化结果。

### 输入

- **图像数据**：从ROS包中提取的多相机图像（PNG格式）。

### 输出

- **法向量数据**：估计的法向量图，按相机名称存储为PNG格式。

## SemanticEstimator

`semantic_estimator.py` 模块使用DINOv2模型对图像进行语义分割。支持处理6个环视相机的图像，并输出分割结果和可视化图像。

### 输入

- **图像数据**：从ROS包中提取的多相机图像（PNG格式）。
- **ego_masks**：手动标注的自车mask。

### 输出

- **分割数据**：语义分割结果，存储为NPZ格式。
- **可视化结果**：（可选）分割结果的可视化图像。

注：当前 `run_task.sh` 默认把语义分割输出目录设为 `masks/`（不是 `semantic/`）。

## SequenceDepthEstimator

`image_sequence_depth_estimator.py` 用于生成视频序列深度（时序一致性更好），`run_task.sh` 默认输出到 `video_depth/`，并可保存 npz。

## PlaneDetector

`plane_detector.py` 模块用于检测和生成场景中的平面。该模块结合深度图、法向量和语义分割信息，通过区域生长的方式检测平面区域。

### 输入

- **深度图**：从Depth-Anything-V2生成的深度图（NPZ格式）
- **法向量**：从StableNormal生成的法向量图（NPZ格式）
- **语义分割**：从DINOv2生成的语义分割结果（NPZ格式）

### 输出

- **平面检测结果**：检测到的平面参数和分割结果，存储为NPZ格式，包含：
  - `planes`：平面参数数组 [N, 4]，每行为[nx, ny, nz, d]表示平面方程 nx*x + ny*y + nz*z + d = 0
  - `segments`：平面分割结果 [H, W]，像素值表示对应的平面ID
- **可视化结果**：（可选）平面检测的可视化图像，不同颜色表示不同平面


## 目前已完成输出

- [x] images/            # 多相机图像
- [x] lidar/             # 原始点云（PLY）
- [x] lidar_world_fix/   # 世界坐标系点云（PLY）
- [x] lidar_npz/         # rays格式点云（NPZ，局部坐标系）
- [x] depths/            # 单目深度（Depth-Anything-V2）
- [x] normals/           # 法向（StableNormal）
- [x] masks/             # 语义分割（DINOv2，脚本中目录名为 masks）
- [x] video_depth/       # 序列深度（image_sequence_depth_estimator）
- [x] planes/            # 平面检测输出
- [x] scenario.pt        # WaymoConvertor 输出