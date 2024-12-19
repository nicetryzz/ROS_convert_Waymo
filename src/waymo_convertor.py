import numpy as np
from numpy.linalg import inv
import os
from pathlib import Path
import pickle
import argparse
import imageio.v2 as imageio 
import json

class WaymoConverter:
    def __init__(self, base_dir: str, subdir: str):
        """初始化转换器
        
        Args:
            base_dir: 数据集根目录
            subdir: 子目录，包含数据和输出目录
        """
        # 设置路径
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / subdir
        
        # 相机列表
        self.cameras = ['camera_FRONT', 'camera_FRONT_RIGHT', 'camera_BACK_RIGHT',
                       'camera_BACK', 'camera_BACK_LEFT', 'camera_FRONT_LEFT']
        
        # 定义坐标系转换矩阵（从雷达坐标系到统一坐标系）
        # 从 (y_back, x_left, z_up) 到 (x_front, y_left, z_up)
        self.pandar_to_waymo = np.array([
            [0, -1, 0, 0],   # y_back -> x_front
            [1, 0, 0, 0],  # x_left -> y_left
            [0, 0, 1, 0],   # -z_down -> -z_down
            [0, 0, 0, 1]
        ], dtype=np.float32)

    def load_poses(self, path):
        """加载位姿数据
        
        Args:
            path: 位姿文件路径，例如 'data/10_26/lidar_poses.txt'
        
        Returns:
            np.ndarray: 形状为[N, 4, 4]的位姿矩阵数组，其中：
                - N 是帧数
                - 每个4x4矩阵表示一个变换矩阵 [R|t]
                    - R: 3x3旋转矩阵
                    - t: 3x1平移向量
        
        文件格式：
            - 每行包含12个数字（3x4矩阵展开）
            - 格式为：[r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3]
            - 函数会自动添加第4行 [0 0 0 1] 使其成为完整的4x4变换矩阵
        """
        # 加载数据
        data = np.loadtxt(path)  # [N, 12]
        
        # 重塑为[N, 3, 4]矩阵
        data = data.reshape(-1, 3, 4)  # [N, 3, 4]
        
        # 添加最后一行 [0, 0, 0, 1]
        new_rows = np.array([[0, 0, 0, 1]] * data.shape[0]).reshape(-1, 1, 4)  # [N, 1, 4]
        data = np.concatenate((data, new_rows), axis=1)  # [N, 4, 4]
        
        return data

    def load_camera_params(self, camera_name: str) :
        """加载相机参数（内参和外参）
        
        Args:
            camera_name: 相机名称，如 'camera_FRONT'
            
        Returns:
            tuple[np.ndarray, np.ndarray]: 
                - extrinsic: [4, 4] 相机外参矩阵
                - intrinsic: [3, 3] 相机内参矩阵
        """
        param_path = self.base_dir / "extrinsic" / "camera" / f"{camera_name}.json"
        
        # 读取JSON文件
        with open(param_path, 'r') as f:
            params = json.load(f)
        
        # 解析外参矩阵
        extrinsic = np.array(params["extrinsic"]).reshape(4, 4)
        
        # 解析内参矩阵
        intrinsic = np.array(params["intrinsic"]).reshape(3, 3)
        
        # 解析畸变系数
        distortion = np.array(params["distortion"])
        
        return extrinsic, intrinsic, distortion

    def process_camera(self, camera_name):
        """处理单个相机数据"""
        img = imageio.imread(self.data_dir / "images" / f"{camera_name}" / "000000.png")
        
        extrinsic, intrinsic, distortion = self.load_camera_params(camera_name)

        # 计算相机到世界的变换
        c2w = self.lidar_pose @ self.pandar_to_waymo @ extrinsic
        
        # 加载相机时间戳并转换为NumPy数组
        with open(self.data_dir / "time" / f"{camera_name}.json", 'r') as f:
            timestamp = json.load(f)
        timestamp = np.array([float(t)/1e9 for t in timestamp], dtype=np.float64)  # 转换为NumPy数组
        
        return {
            "hw": np.tile(np.array([img.shape[0],img.shape[1]]), (self.frame_num, 1)),
            "c2v": np.tile(self.pandar_to_waymo @ extrinsic, (self.frame_num, 1, 1)).astype(np.float32),
            "sensor_v2w": self.lidar_pose,
            "c2w": c2w.astype(np.float32),
            "global_frame_ind": np.arange(self.frame_num),
            "intr": np.repeat(intrinsic[:3,:3].reshape(1,3,3), self.frame_num,  axis=0),
            "distortion": np.repeat(distortion.reshape(1,5), self.frame_num,  axis=0),
            "timestamp": timestamp  # NumPy数组格式的时间戳
        }

    def process_lidar(self):
        """处理激光雷达数据"""
        # 加载激光雷达时间戳并转换为NumPy数组
        with open(self.data_dir / "time" / "pointcloud.json", 'r') as f:
            timestamp = json.load(f)
        timestamp = np.array([float(t)/1e9 for t in timestamp], dtype=np.float64)  # 转换为NumPy数组
        
        return {
            "l2v": np.tile(np.eye(4), (self.frame_num, 1, 1)).astype(np.float32),
            "l2w": self.lidar_pose,  # [N, 4, 4] 激光雷达位姿
            "global_frame_ind": np.arange(self.frame_num),
            "timestamp": timestamp  # NumPy数组格式的时间戳
        }

    def process_egocar(self):
        """处理自车数据"""   
        # 加载激光雷达时间戳作为自车时间戳,并转换为NumPy数组
        with open(self.data_dir / "time" / "pointcloud.json", 'r') as f:
            timestamp = json.load(f)
        timestamp = np.array([float(t)/1e9 for t in timestamp], dtype=np.float64)  # 转换为NumPy数组
        
        return {
            "v2w": self.lidar_pose,  # [N, 4, 4] 自车位姿
            "global_frame_ind": np.arange(self.frame_num),
            "timestamp": timestamp  # NumPy数组格式的时间戳
        }

    def convert(self):
        """转换数据到Waymo格式"""
        # 加载位姿和标定数据
        self.lidar_pose = self.load_poses(self.data_dir / "lidar_poses.txt")
        self.frame_num = self.lidar_pose.shape[0]
        
        # 构建数据字典
        data = {
            "scene_id": self.data_dir.name,
            "metas": {
                "n_frames": self.frame_num
            },
            "observers": {},
            "objects": {}
        }
        
        # 添加自车
        data["observers"]["ego_car"] = {
            "class_name": "EgoVehicle",
            "n_frames": self.frame_num,
            "data": self.process_egocar(),
        }
        
        # 处理每个相机
        for cam_i, camera_name in enumerate(self.cameras):
            data["observers"][camera_name] = {
                "class_name": "Camera",
                "n_frames": self.frame_num,
                "data": self.process_camera(camera_name),
            }
        
        # 添加激光雷达
        data["observers"]["lidar_TOP"] = {
            "class_name": "RaysLidar",
            "n_frames": self.frame_num,
            "data": self.process_lidar(),
        }
        
        # 保存为.pt文件
        output_path = self.data_dir / "scenario.pt"
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"数据已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert dataset to Waymo format')
    parser.add_argument('--base_dir', type=str, 
                    default="/home/hqlab/workspace/dataset/parkinglot",
                    help='Base directory')
    parser.add_argument('--subdir', type=str, 
                    default="data/10_26",
                    help='Subdirectory containing data')
    args = parser.parse_args()
    
    # 创建转换器并执行转换
    converter = WaymoConverter(
        base_dir=args.base_dir,
        subdir=args.subdir
    )
    converter.convert()

if __name__ == "__main__":
    main()