#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from pathlib import Path
from typing import Dict, List, Optional
import argparse

from depth_anything_v2.dpt import DepthAnythingV2

class DepthEstimator:
    """用于估计图像深度的类"""
    
    # 相机目录名称
    CAMERA_DIRS = [
        "camera_FRONT", "camera_FRONT_RIGHT", "camera_BACK_RIGHT",
        "camera_BACK", "camera_BACK_LEFT", "camera_FRONT_LEFT"
    ]

    def __init__(self, base_dir: str, input_subdir: str, output_subdir: str, 
                 repo_path: str, input_size: int = 518, encoder: str = 'vitl', 
                 pred_only: bool = False, grayscale: bool = False):
        """
        初始化深度估计器
        
        Args:
            base_dir: 基础目录路径
            input_subdir: 输入数据的相对路径
            output_subdir: 输出数据的相对路径
            repo_path: Depth-Anything-V2仓库路径
            input_size: 输入图像大小
            encoder: 编码器类型 ['vits', 'vitb', 'vitl', 'vitg']
            pred_only: 是否只保存深度预测结果
            grayscale: 是否使用灰度图显示
        """
        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / input_subdir
        self.output_dir = self.base_dir / output_subdir
        self.repo_path = repo_path
        self.input_size = input_size
        self.pred_only = pred_only
        self.grayscale = grayscale
        
        # 初始化模型
        self.device = self._get_device()
        self.model = self._init_model(encoder)
        self.cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        
        # 创建输出目录
        self._create_directories()

    def _get_device(self) -> str:
        """获取计算设备"""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def _init_model(self, encoder: str) -> DepthAnythingV2:
        """初始化深度估计模型"""
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        model = DepthAnythingV2(**model_configs[encoder])
        checkpoint_path = Path(self.repo_path) / 'checkpoints' / f'depth_anything_v2_{encoder}.pth'
        model.load_state_dict(torch.load(str(checkpoint_path), map_location='cpu'))
        return model.to(self.device).eval()

    def _create_directories(self) -> None:
        """创建输出目录结构"""
        for camera_dir in self.CAMERA_DIRS:
            output_path = self.output_dir / camera_dir
            output_path.mkdir(parents=True, exist_ok=True)

    def process_single_image(self, image_path: Path, output_path: Path) -> None:
        """处理单张图像"""
        print(f'Processing: {image_path}')
        raw_image = cv2.imread(str(image_path))
        depth = self.model.infer_image(raw_image, self.input_size)
        
        # 处理深度值
        zero_count = np.sum(depth == 0)
        processed_depth = np.zeros_like(depth)
        
        if zero_count > 0:
            mask = depth > 0.1
            processed_depth[mask] = 1.0 / depth[mask]
        else:
            processed_depth = 1.0 / depth
        
        print(f"处理后深度范围: min={processed_depth.min():.3f}, max={processed_depth.max():.3f}")
        
        # 保存深度图
        depth_output_path = output_path.with_suffix('.npz')
        np.savez_compressed(depth_output_path, depth=processed_depth.astype(np.float32))
        
        # 保存可视化结果
        if not self.pred_only:
            self._save_visualization(raw_image, processed_depth, output_path)

    def _save_visualization(self, raw_image: np.ndarray, depth: np.ndarray, output_path: Path) -> None:
        """保存深度图可视化结果"""
        depth_vis = (depth * 255.0).astype(np.uint8)
        if self.grayscale:
            depth_vis = np.repeat(depth_vis[..., np.newaxis], 3, axis=-1)
        else:
            depth_vis = (self.cmap(depth_vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([raw_image, split_region, depth_vis])
        cv2.imwrite(str(output_path), combined_result)

    def process_dataset(self) -> None:
        """处理整个数据集"""
        for camera_dir in self.CAMERA_DIRS:
            input_path = self.input_dir / camera_dir
            if not input_path.exists():
                print(f"警告: 目录不存在 {input_path}")
                continue
                
            print(f"处理相机 {camera_dir} 的数据...")
            for img_path in input_path.glob('*.png'):
                output_path = self.output_dir / camera_dir / img_path.name
                self.process_single_image(img_path, output_path)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Depth Estimation')
    parser.add_argument('--base_dir', type=str, default="/home/hqlab/workspace/dataset/parkinglot",
                        help='Base directory for both input and output data.')
    parser.add_argument('--input_subdir', type=str, default="data/10_26/images",
                        help='Subdirectory under base_dir for input images.')
    parser.add_argument('--output_subdir', type=str, default="data/10_26/depths",
                        help='Subdirectory under base_dir for output depth maps.')
    parser.add_argument('--repo_path', type=str, default="/home/hqlab/workspace/depth_estimation/Depth-Anything-V2",
                        help='Path to the Depth-Anything-V2 repository.')
    parser.add_argument('--input_size', type=int, default=518,
                        help='Input image size for depth estimation.')
    parser.add_argument('--encoder', type=str, default='vitl',
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Encoder type for depth estimation.')
    parser.add_argument('--pred-only', action='store_true',
                        help='Only save depth prediction.')
    parser.add_argument('--grayscale', action='store_true',
                        help='Use grayscale visualization.')
    
    args = parser.parse_args()
    
    estimator = DepthEstimator(
        base_dir=args.base_dir,
        input_subdir=args.input_subdir,
        output_subdir=args.output_subdir,
        repo_path=args.repo_path,
        input_size=args.input_size,
        encoder=args.encoder,
        pred_only=args.pred_only,
        grayscale=args.grayscale
    )
    estimator.process_dataset()

if __name__ == "__main__":
    main()