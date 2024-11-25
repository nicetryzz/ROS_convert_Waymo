import numpy as np
from numpy.linalg import inv
import json
import os
from pathlib import Path
import imageio.v2 as imageio
from typing import Dict, List

class DTUConverter:
    def __init__(self, base_dir: str, subdir: str):
        self.base_dir = Path(base_dir)
        self.subdir = Path(subdir)
        self.data_dir = self.base_dir / self.subdir
        
        # 相机列表
        self.cameras = ['camera_FRONT', 'camera_FRONT_RIGHT', 'camera_BACK_RIGHT',
                       'camera_BACK', 'camera_BACK_LEFT', 'camera_FRONT_LEFT']
                       
        # 坐标转换矩阵
        self.pandar_to_waymo = np.array([
            [0, -1, 0, 0],   # y_back -> x_front
            [1, 0, 0, 0],    # x_left -> y_left
            [0, 0, 1, 0],    # -z_down -> -z_down
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 加载雷达位姿
        self.lidar_pose_pandar = self.load_poses(self.data_dir / "lidar_poses.txt")

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

    def load_camera_params(self, camera_name: str):
        """加载相机参数（内参和外参）
        
        Args:
            camera_name: 相机名称
            
        Returns:
            tuple: (extrinsic, intrinsic) 都是4x4矩阵
        """
        param_path = self.base_dir / "extrinsic" / "camera" / f"{camera_name}.json"
        
        with open(param_path, 'r') as f:
            params = json.load(f)
        
        # 解析外参矩阵
        extrinsic = np.array(params["extrinsic"]).reshape(4, 4)
        
        # 解析内参矩阵并扩展为4x4
        intrinsic = np.array(params["intrinsic"]).reshape(3, 3)
        intrinsic_4x4 = np.eye(4)
        intrinsic_4x4[:3, :3] = intrinsic
        
        return extrinsic, intrinsic_4x4

    def calculate_scene_box(self, front_distance=30, side_distance=15, back_distance=5, up_distance=2):
        """计算场景的AABB包围盒
        
        Args:
            front_distance: 前方距离 (米)
            side_distance: 左右距离 (米)
            back_distance: 后方距离 (米)
            up_distance: 上方距离 (米)
        
        Returns:
            tuple: (min_point, max_point) 包围盒的最小和最大点
        """
        all_points = []
        
        for camera_name in self.cameras:
            # 加载相机参数
            extrinsic, _ = self.load_camera_params(camera_name)
            
            # 定义视锥体的8个角点（相机坐标系下）
            corners = np.array([
                [-side_distance, -up_distance, -back_distance, 1],
                [-side_distance, -up_distance, front_distance, 1],
                [-side_distance, up_distance, -back_distance, 1],
                [-side_distance, up_distance, front_distance, 1],
                [side_distance, -up_distance, -back_distance, 1],
                [side_distance, -up_distance, front_distance, 1],
                [side_distance, up_distance, -back_distance, 1],
                [side_distance, up_distance, front_distance, 1],
            ])
            
            # 将角点转换到世界坐标系
            for idx, lidar_pose in enumerate(self.lidar_pose_pandar):
                # 计算相机到世界的变换
                c2w = lidar_pose @ self.pandar_to_waymo @ inv(extrinsic)
                
                # 转换角点
                world_points = corners @ c2w.T
                all_points.append(world_points[:, :3])  # 只保留xyz坐标
        
        # 将所有点转换为numpy数组
        all_points = np.concatenate(all_points, axis=0)
        
        # 计算AABB
        bmin = all_points.min(axis=0)
        bmax = all_points.max(axis=0)
        
        return bmin.tolist(), bmax.tolist()

    def generate_meta_data(self):
        """生成meta_data.json文件"""
        # 读取第一张图片获取尺寸
        first_img = imageio.imread(str(self.data_dir / "images" / "camera_FRONT" / "000000.png"))
        height, width = first_img.shape[:2]
        
        # 计算场景包围盒
        aabb_min, aabb_max = self.calculate_scene_box(
            front_distance=10,  # 前方30米
            side_distance=10,   # 左右各15米
            back_distance=10,    # 后方5米
            up_distance=3       # 上方2米
        )
        
        frames = []
        
        for camera_name in self.cameras:
            # 加载相机参数
            extrinsic, intrinsic = self.load_camera_params(camera_name)
            
            # 计算相机到世界的变换
            c2w = self.lidar_pose_pandar @ self.pandar_to_waymo @ inv(extrinsic)
            
            # 获取图像列表
            img_dir = self.data_dir / "images" / camera_name
            img_files = sorted(img_dir.glob("*.png"))
            
            for idx, img_file in enumerate(img_files):
                frame = {
                    'rgb_path': str(Path("images") / camera_name / f"{img_file.stem}.png"),
                    'camtoworld': c2w[idx].tolist(),  # 使用计算后的相机到世界的变换
                    'intrinsics': intrinsic.tolist(),  # 使用4x4的内参矩阵
                    'mono_depth_path': str(Path("depths") / camera_name / f"{img_file.stem}.npy"),
                    'mono_normal_path': str(Path("normals") / camera_name / f"{img_file.stem}.npy")
                }
                frames.append(frame)
        
        meta_data = {
            'camera_model': 'OPENCV',
            'height': height,
            'width': width,
            'has_mono_prior': True,
            'worldtogt': [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ],
            'scene_box': {
                'aabb': [aabb_min, aabb_max],  # 使用计算得到的AABB
                'near': 0.25,
                'far': max(30, np.linalg.norm(np.array(aabb_max) - np.array(aabb_min))),  # 根据AABB大小调整far平面
                'radius': np.linalg.norm(np.array(aabb_max) - np.array(aabb_min)) / 2,  # 根据AABB大小计算半径
                'collider_type': 'box'
            },
            'frames': frames
        }
        
        # 保存meta_data.json
        with open(self.data_dir / "meta_data.json", 'w') as f:
            json.dump(meta_data, f, indent=2)

    def convert(self):
        """执行转换"""
        print("开始生成meta_data.json...")
        self.generate_meta_data()
        print(f"生成完成，文件保存在: {self.data_dir}/meta_data.json")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate meta_data.json for DTU format')
    parser.add_argument('--base_dir', type=str, 
                    default="/home/hqlab/workspace/dataset/parkinglot",
                    help='Base directory')
    parser.add_argument('--subdir', type=str, 
                    default="data/10_26",
                    help='Subdirectory containing data')
    args = parser.parse_args()
    
    converter = DTUConverter(
        base_dir=args.base_dir,
        subdir=args.subdir
    )
    converter.convert()

if __name__ == "__main__":
    main()