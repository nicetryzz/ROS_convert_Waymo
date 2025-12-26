#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import sqlite3
import collections
import quaternion
import shutil
from pathlib import Path
import argparse
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import pickle
import cv2

class ColmapConvertor:
    """用于将数据集转换为COLMAP格式的转换器"""

    def __init__(self):
        """初始化转换器"""
        self.start_frame = 0
        self.end_frame = 0
        self.sampling_rate = 1.0
        self.image_list: Dict[str, List[str]] = {}
        
        # 相机列表
        self.camera_list = [
            'camera_FRONT', 'camera_FRONT_LEFT', 'camera_FRONT_RIGHT',
            'camera_BACK', 'camera_BACK_LEFT', 'camera_BACK_RIGHT'
        ]
        

    def set_frame_range(self, start_frame: int, end_frame: int) -> None:
        """设置要处理的帧范围
        
        Args:
            start_frame: 起始帧
            end_frame: 结束帧
        """
        if start_frame < 0 or end_frame < start_frame:
            raise ValueError("无效的帧范围设置")
        self.start_frame = start_frame
        self.end_frame = end_frame

    def set_sampling_rate(self, rate: float) -> None:
        """设置采样率
        
        Args:
            rate: 采样率(Hz)
            
        Raises:
            ValueError: 当采样率小于等于0时
        """
        if rate <= 0:
            raise ValueError("采样率必须大于0")
        self.sampling_rate = rate

    def _load_scenario_data(self, root_dir: str) -> dict:
        """加载scenario.pt文件中的数据
        
        Args:
            root_dir: 根目录路径
            
        Returns:
            包含场景数据的字典
            
        Raises:
            FileNotFoundError: 当scenario.pt文件不存在时
            ValueError: 当文件格式无效时
        """
        scenario_path = Path(root_dir) / 'scenario.pt'
        if not scenario_path.exists():
            raise FileNotFoundError(f"未找到scenario.pt文件: {scenario_path}")
            
        try:
            with open(scenario_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise ValueError(f"无法加载scenario.pt文件: {e}")

    def sample_images(self, root_dir: str) -> Dict[str, Dict[str, Dict]]:
        """根据设定的参数采样图像和对应的参数
        
        Args:
            root_dir: 根目录路径
            
        Returns:
            采样后的图像路径和参数字典
            格式为: {
                'camera_name': {
                    'image_paths': [image_paths],
                    'extrinsics': [4x4_matrices],
                    'intrinsic': [3x3_matrix],
                    'distortion': [5_coefficients]
                }
            }
            
        Raises:
            ValueError: 当输入参数无效时
        """
        root_dir = Path(root_dir)
        if not root_dir.exists():
            raise ValueError(f"根目录不存在: {root_dir}")
            
        # 加载场景数据
        scenario_data = self._load_scenario_data(root_dir)
        
        result = {}
        img_path = root_dir / 'images'
        
        # 处理每个相机
        for camera in tqdm(self.camera_list, desc="处理相机"):
            camera_path = img_path / camera
            if not camera_path.exists():
                continue
                
            # 获取相机参数
            if camera not in scenario_data['observers']:
                continue
                
            camera_data = scenario_data['observers'][camera]['data']
            result[camera] = self._process_camera_data(
                camera_path, camera_data
            )
                
        return result

    def _process_camera_data(self, camera_path: Path, camera_data: dict) -> Dict[str, Dict]:
        """处理单个相机的数据
        
        Args:
            camera_path: 相机目录路径
            camera_data: 相机参数数据
            
        Returns:
            处理后的相机数据字典
        """
        result = {}
        camera_extrinsics = camera_data['c2w']
        camera_intrinsic = camera_data['intr'][0]
        camera_distortion = camera_data['distortion'][0]
        
                
        # 获取并采样图像
        images = sorted([f for f in camera_path.glob("*.png")])
        valid_images = images[self.start_frame:self.end_frame+1]
        valid_extrinsics = camera_extrinsics[self.start_frame:self.end_frame+1]
        
        interval = int(self.sampling_rate)
        sampled_images = valid_images[::interval]
        sampled_extrinsics = valid_extrinsics[::interval]
        
        result = {
            'image_paths': [str(img) for img in sampled_images],
            'extrinsics': sampled_extrinsics,
            'intrinsic': camera_intrinsic,
            'distortion': camera_distortion
        }
            
        return result

    def write_colmap_format(self, camera_data: Dict, output_dir: str) -> None:
        """将相机参数写入COLMAP格式的txt文件
        
        Args:
            camera_data: sample_images返回的相机数据
            output_dir: 输出目录路径
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 写入cameras.txt
        self._write_cameras(output_dir / 'cameras.txt', camera_data)
        
        # 写入images.txt
        self._write_images(output_dir / 'images.txt', camera_data)
        
        # 写入空的points3D.txt
        self._write_points3d(output_dir / 'points3D.txt')
        
        print(f"已写入COLMAP格式文件到: {output_dir}")

    def _write_cameras(self, output_path: Path, camera_data: Dict) -> None:
        """写入cameras.txt文件"""
        with output_path.open('w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            
            camera_id = 1
            for camera_name in camera_data.keys():
                data = camera_data[camera_name]
                # 获取图像尺寸
                img = Image.open(data['image_paths'][0])
                width, height = img.size
                img.close()
                
                # 提取内参
                K = data['intrinsic']
                params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]  # fx, fy, cx, cy
                params_str = ' '.join(map(str, params))
                
                f.write(f"{camera_id} PINHOLE {width} {height} {params_str}\n")
                camera_id += 1

    def _write_images(self, output_path: Path, camera_data: Dict) -> None:
        """写入images.txt文件"""
        with output_path.open('w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            image_id = 1
            camera_id = 1
            for camera_name in camera_data.keys():
                data = camera_data[camera_name]
                for image_path, extrinsic in zip(data['image_paths'], data['extrinsics']):
                # 计算COLMAP格式的外参
                    R = extrinsic[:3, :3]
                    t = extrinsic[:3, 3]
                    R_inv = R.T
                    t_inv = -R.T @ t
                    
                    # 转换为四元数
                    q = quaternion.from_rotation_matrix(R_inv)
                    
                    f.write(f"{image_id} {q.w} {q.x} {q.y} {q.z} "
                            f"{t_inv[0]} {t_inv[1]} {t_inv[2]} {camera_id} "
                            f"{Path(image_path).name}\n\n")
                    
                    image_id += 1
                camera_id += 1

    def _write_points3d(self, output_path: Path) -> None:
        """写入空的points3D.txt文件"""
        with output_path.open('w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")

    def prepare_colmap_workspace(self, base_dir: str, root_dir: str, mask_dir: str = None) -> None:
        """准备COLMAP工作空间，组织图像和参数文件
        
        Args:
            base_dir: COLMAP工作空间根目录
            root_dir: 原始图像根目录
            mask_dir: 相机掩码目录，默认为None
        """
        base_dir = Path(base_dir)
        images_dir = base_dir / 'images'
        sparse_dir = base_dir / 'sparse'
        masks_dir = base_dir / 'masks' if mask_dir else None
        
        # 创建目录
        images_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)
        if masks_dir:
            masks_dir.mkdir(parents=True, exist_ok=True)
        
        # 采样图像和参数
        print("采样图像和参数...")
        camera_data = self.sample_images(root_dir)
        
        # 复制图像和创建对应的掩码
        print("复制图像文件...")
        for camera_name in camera_data.keys():
            data = camera_data[camera_name]
            for i, src_path in enumerate(data['image_paths']):
                # 创建新的文件名并复制图像
                dst_name = f"{camera_name}_{Path(src_path).name}"
                dst_path = images_dir / dst_name
                shutil.copy2(src_path, dst_path)
                
                # 如果提供了掩码目录，创建对应的掩码文件
                if mask_dir:
                    mask_src = Path(mask_dir) / f"{camera_name}.png"
                    if mask_src.exists():
                        mask_dst = masks_dir / f"{dst_name}.png"
                        # 读取掩码并反转颜色
                        mask = cv2.imread(str(mask_src), cv2.IMREAD_GRAYSCALE)
                        mask = 255 - mask  # 反转黑白
                        cv2.imwrite(str(mask_dst), mask)
                
                # 更新路径
                data['image_paths'][i] = str(dst_path)
        
        # 写入COLMAP格式文件
        print("写入COLMAP格式文件...")
        self.write_colmap_format(camera_data, sparse_dir)
        
        print(f"\nCOLMAP工作空间准备完成：")
        print(f"- 图像目录：{images_dir}")
        print(f"- 掩码目录：{masks_dir if mask_dir else '未创建'}")
        print(f"- 稀疏重建目录：{sparse_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将数据集转换为COLMAP格式')
    parser.add_argument('--base_dir', type=str, 
                      default='/home/hqlab/workspace/reconstruction/result/final_result/parkinglot/scene_0/colmap',
                      help='COLMAP工作空间目录')
    parser.add_argument('--root_dir', type=str,
                      default='/home/hqlab/workspace/dataset/parkinglot/data/test1',
                      help='原始图像目录')
    parser.add_argument('--start_frame', type=int, default=60,
                      help='起始帧')
    parser.add_argument('--end_frame', type=int, default=400,
                      help='结束帧')
    parser.add_argument('--sampling_rate', type=float, default=5.0,
                      help='采样率(Hz)')
    parser.add_argument('--mask_dir', type=str,
                      default='/home/hqlab/workspace/dataset/parkinglot/ego_masks',
                      help='相机掩码目录')
    args = parser.parse_args()
    
    # 创建转换器实例
    convertor = ColmapConvertor()
    
    # 设置参数
    convertor.set_frame_range(args.start_frame, args.end_frame)
    convertor.set_sampling_rate(args.sampling_rate)
    
    # 准备COLMAP工作空间
    convertor.prepare_colmap_workspace(
        base_dir=args.base_dir,
        root_dir=args.root_dir,
        mask_dir=args.mask_dir
    )

if __name__ == "__main__":
    main()
# python src/colmap_convertor.py --base_dir /home/hqlab/workspace/reconstruction/result/final_result/parkinglot/scene0/colmap --root_dir /home/hqlab/workspace/dataset/carla_data/dumper/2024_12_29_13_59_40 --start_frame 60 --end_frame 400 --sampling_rate 5.0 --mask_dir /home/hqlab/workspace/dataset/parkinglot/ego_masks