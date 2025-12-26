#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
import argparse
import open3d as o3d
import os

class LidarTransform:
    """雷达坐标转换类"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        
    def _read_pose_file(self, pose_file: Path) -> np.ndarray:
        """读取位姿文件
        
        Args:
            pose_file: 位姿文件路径
            
        Returns:
            位姿矩阵列表 [N, 4, 4]
        """
        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                # 读取12个数字
                values = [float(x) for x in line.strip().split()]
                if len(values) != 12:
                    continue
                    
                # 转换为4x4矩阵
                mat = np.eye(4, dtype=np.float32)
                mat[:3, :4] = np.array(values).reshape(3, 4)
                poses.append(mat)
        
        return np.array(poses)
    
    def transform_point_cloud(self, points: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """转换点云到世界坐标系
        
        Args:
            points: 点云数据 [N, 3]
            pose: 位姿矩阵 [4, 4]
            
        Returns:
            转换后的点云 [N, 3]
        """
        # 转换点云坐标系
        points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)  # [N, 4]
        
        pose = (pose @ points_homo.T).T
        
        return pose[:, :3]
    
    def ply2npz(self, ply_dir: Path, npz_dir: Path):
        """将PLY点云转换为局部坐标系的NPZ光线格式
        
        Args:
            ply_dir: 点云文件目录
            npz_dir: 输出NPZ目录
        """
        npz_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理每个点云文件
        ply_files = sorted(ply_dir.glob('*.ply'))
        for ply_file in ply_files:
            # 读取点云
            ply = o3d.io.read_point_cloud(str(ply_file))
            points = np.asarray(ply.points)
            
            # 计算距离（深度）
            ranges = np.linalg.norm(points, axis=1)
    
            # 计算局部坐标系下的光线方向（归一化）
            non_zero_ranges = ranges > 1e-6 
            rays_d_local = np.zeros_like(points)
            rays_d_local[non_zero_ranges] = points[non_zero_ranges] / ranges[non_zero_ranges, np.newaxis]
            
            # 光线起点在局部坐标系原点
            rays_o_local = np.zeros((points.shape[0], 3), dtype=np.float32)

            ray_output_path = os.path.join(npz_dir, f'{ply_file.stem}.npz')
            print(ray_output_path)
            np.savez_compressed(
                ray_output_path,
                rays_o=rays_o_local,
                rays_d=rays_d_local.astype(np.float32),
                ranges=ranges.astype(np.float32)
            )
            
    def process_point_clouds(self, ply_dir: Path, pose_file: Path, output_dir: Path):
        """处理点云文件
        
        Args:
            ply_dir: 点云文件目录
            pose_file: 位姿文件路径
            output_dir: 输出目录
        """
        # 读取位姿
        poses = self._read_pose_file(pose_file)
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理每个点云文件
        ply_files = sorted(ply_dir.glob('*.ply'))  # 假设使用ply格式
        for i, ply_file in enumerate(ply_files):
            if i >= len(poses):
                break
                
            # 读取点云
            ply = o3d.io.read_point_cloud(str(ply_file))
            points = np.asarray(ply.points)
            
            # 转换到世界坐标系
            points_world = self.transform_point_cloud(points, poses[i])
            
            # 保存转换后的点云
            ply_world = o3d.geometry.PointCloud()
            ply_world.points = o3d.utility.Vector3dVector(points_world)
            output_file = output_dir / ply_file.name
            o3d.io.write_point_cloud(str(output_file), ply_world)
            
            print(f'Processed {ply_file.name}')

def main():
    parser = argparse.ArgumentParser(description='Transform LiDAR point clouds to world coordinate')
    parser.add_argument('--base_dir', type=str, 
                        default="/home/hqlab/workspace/dataset/parkinglot",
                        help='Base directory')
    parser.add_argument('--ply_dir', type=str, 
                        default="data/bright_expose/lidar",
                        help='Point cloud directory relative to base_dir')
    parser.add_argument('--pose_file', type=str, 
                        default="data/bright_expose/poses_adjusted.txt",
                        help='Pose file path relative to base_dir')
    parser.add_argument('--output_dir', type=str, 
                        default="data/bright_expose/lidar_world",
                        help='Output directory relative to base_dir')
    parser.add_argument('--npz_dir', type=str, 
                        default="data/bright_expose/lidar_npz",
                        help='Point cloud directory relative to base_dir')
    
    args = parser.parse_args()
    
    transformer = LidarTransform(args.base_dir)
    ply_dir = Path(args.base_dir) / args.ply_dir
    pose_file = Path(args.base_dir) / args.pose_file
    output_dir = Path(args.base_dir) / args.output_dir
    npz_dir = Path(args.base_dir) / args.npz_dir
    
    transformer.process_point_clouds(ply_dir, pose_file, output_dir)
    transformer.ply2npz(ply_dir, npz_dir)  # 不再需要pose_file参数

if __name__ == "__main__":
    main()
