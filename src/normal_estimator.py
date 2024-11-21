#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional
import argparse

class NormalEstimator:
    """用于估计图像法向量的类"""
    
    # 相机目录名称
    CAMERA_DIRS = [
        "camera_FRONT", "camera_FRONT_RIGHT", "camera_BACK_RIGHT",
        "camera_BACK", "camera_BACK_LEFT", "camera_FRONT_LEFT"
    ]

    def __init__(self, base_dir: str, input_subdir: str, output_subdir: str):
        """
        初始化法向量估计器
        
        Args:
            base_dir: 基础目录路径
            input_subdir: 输入数据的相对路径
            output_subdir: 输出数据的相对路径
        """
        # 设置HuggingFace镜像
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / input_subdir
        self.output_dir = self.base_dir / output_subdir
        
        # 初始化模型
        self.model = self._init_model()
        
        # 创建输出目录
        self._create_directories()

    def _init_model(self):
        """初始化法向量估计模型"""
        return torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True)

    def _create_directories(self) -> None:
        """创建输出目录结构"""
        for camera_dir in self.CAMERA_DIRS:
            output_path = self.output_dir / camera_dir
            output_path.mkdir(parents=True, exist_ok=True)

    def process_single_image(self, image_path: Path, output_path: Path) -> None:
        """处理单张图像"""
        print(f'Processing: {image_path}')
        
        # 读取图像
        input_image = Image.open(image_path).convert("RGB")
        
        # 估计法向量
        normal_image = self.model(input_image)
        
        # 将PIL图像转换为numpy数组 (H, W, 3)，范围[-1, 1]
        normal_array = np.array(normal_image)
        normal_array = 255 - normal_array
        normal_image = Image.fromarray(normal_array)
        normal_image.save(str(output_path))
        print(f'Saved normal map to: {output_path}')

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
    parser = argparse.ArgumentParser(description='Normal Estimation')
    parser.add_argument('--base_dir', type=str, default="/home/hqlab/workspace/dataset/parkinglot",
                        help='Base directory for both input and output data.')
    parser.add_argument('--input_subdir', type=str, default="data/10_26/images",
                        help='Subdirectory under base_dir for input images.')
    parser.add_argument('--output_subdir', type=str, default="data/10_26/normals",
                        help='Subdirectory under base_dir for output normal maps.')
    
    args = parser.parse_args()
    
    estimator = NormalEstimator(
        base_dir=args.base_dir,
        input_subdir=args.input_subdir,
        output_subdir=args.output_subdir
    )
    estimator.process_dataset()

if __name__ == "__main__":
    main()