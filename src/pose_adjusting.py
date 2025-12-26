#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from typing import List, Optional

class PoseAdjuster:
    """用于处理和调整雷达位姿的类"""
    
    def __init__(self, base_dir: str, subdir: str = "data/test1"):
        """
        初始化位姿调整器
        
        Args:
            base_dir: 基础目录路径
            subdir: 数据子目录路径
        """
        self.base_dir = Path(base_dir) / subdir
        self.load_poses()
        
    def load_poses(self, pose_file: str = "lidar_poses.txt") -> Optional[List[np.ndarray]]:
        """
        从文件中读取雷达位姿数据
        
        Args:
            pose_file: 位姿文件名
            
        Returns:
            位姿矩阵列表，如果读取失败则返回None
        """
        pose_path = self.base_dir / pose_file
        
        if not pose_path.exists():
            print(f"警告: 未找到位姿文件: {pose_path}")
            return None
            
        poses = []
        try:
            with open(pose_path, 'r') as f:
                for line in f:
                    # 将每行数据转换为12个浮点数
                    values = [float(x) for x in line.strip().split()]
                    if len(values) != 12:
                        print(f"警告: 位姿数据格式错误: {line}")
                        continue
                        
                    # 将数据重塑为3x4矩阵
                    pose_matrix = np.array(values).reshape(3, 4)
                    
                    # 转换为4x4齐次变换矩阵
                    transform = np.eye(4)
                    transform[:3, :] = pose_matrix
                    
                    poses.append(transform)
                self.poses = poses
                    
            return poses
            
        except Exception as e:
            print(f"读取位姿文件时出错: {e}")
            return None
            
    def adjust_poses(self) -> List[np.ndarray]:
        """
        通过计算最佳拟合平面的法向量，构建旋转矩阵将轨迹调整到水平面
            
        Returns:
            调整后的位姿矩阵列表
        """
        if not self.poses:
            return []
            
        # 提取所有位姿的位置点
        points = np.array([pose[:3, 3] for pose in self.poses])
        
        # 计算点云的质心
        centroid = np.mean(points, axis=0)
        
        # 构建协方差矩阵
        covariance_matrix = np.cov(points.T)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # 最小特征值对应的特征向量就是平面法向量
        normal = eigenvectors[:, 0]
        
        # 计算从法向量到Z轴的旋转矩阵
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(normal, z_axis)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # 计算旋转角度
        cos_theta = np.dot(normal, z_axis)
        theta = np.arccos(cos_theta)
        
        # 构建旋转矩阵（Rodrigues公式）
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - cos_theta) * (K @ K)
        
        # 构建4x4变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = R
        
        # 应用变换矩阵到所有位姿
        # 检查第一个pose的(2,2)是否接近-1
        first_pose = transform @ self.poses[0]
        if first_pose[2,2] < 0:
            # 如果接近-1,则将transform绕X轴旋转180度
            rot_x = np.array([[1, 0, 0, 0],
                            [0, -1, 0, 0], 
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])
            transform = rot_x @ transform
        
        self.poses = [transform @ pose for pose in self.poses]
        
        # 写出调整后的位姿
        output_path = self.base_dir / "poses_adjusted.txt"
        try:
            with open(output_path, "w") as f:
                for pose in self.poses:
                    # 只输出前12个数字
                    pose_str = " ".join([f"{x:.6f}" for x in pose[:3,:4].flatten()])
                    f.write(f"{pose_str}\n")
            print(f"已将调整后的位姿写入: {output_path}")
        except Exception as e:
            print(f"写入位姿文件时出错: {e}")
            
        return self.poses
            
        
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='调整位姿到水平面')
    parser.add_argument('--base_dir', type=str, 
                    default="/home/hqlab/workspace/dataset/parkinglot",
                    help='基础目录路径')
    parser.add_argument('--subdir', type=str, 
                    default="data/test1",
                    help='包含数据的子目录')
    args = parser.parse_args()

    # 创建PoseAdjuster实例并调用
    adjuster = PoseAdjuster(args.base_dir, args.subdir)
    adjuster.adjust_poses()


