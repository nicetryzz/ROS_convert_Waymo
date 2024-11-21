#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import itertools
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import mmcv
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor, inference_segmentor

import dinov2.eval.segmentation.models
import dinov2.eval.segmentation_m2f.models.segmentors
import dinov2.eval.segmentation.utils.colormaps as colormaps

import cv2

class SemanticSegmentation:
    """用于进行语义分割的类"""

    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
    CHECKPOINT_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f.pth"

    # ego car的标签值和颜色
    EGO_CAR_LABEL = 150
    EGO_CAR_COLOR = np.array([255, 0, 0], dtype=np.uint8)  # 红色
    
    DATASET_COLORMAPS = {
        "ade20k": colormaps.ADE20K_COLORMAP,
        "voc2012": colormaps.VOC2012_COLORMAP,
    }

    def __init__(self, base_dir: str, input_subdir: str, output_subdir: str, 
                 repo_path: str, max_size: int = 1024, save_visualization: bool = True):
        """
        初始化语义分割器
        
        Args:
            base_dir: 基础目录路径
            input_subdir: 输入数据的相对路径
            output_subdir: 输出数据的相对路径
            repo_path: DINOv2代码仓库路径
            max_size: 输入图像的最大尺寸
            save_visualization: 是否保存可视化结果
        """
        # 添加DINOv2到Python路径
        sys.path.append(repo_path)
        
        self.base_dir = Path(base_dir)
        self.input_dir = self.base_dir / input_subdir
        self.output_dir = self.base_dir / output_subdir
        self.max_size = max_size
        
        # 初始化模型
        self.model = self._init_model()
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_visualization = save_visualization

    def _load_config_from_url(self, url: str) -> str:
        """从URL加载配置文件"""
        import urllib.request
        with urllib.request.urlopen(url) as f:
            return f.read().decode()

    def _init_model(self):
        """初始化DINOv2语义分割模型"""
        
        # 从URL加载配置
        cfg_str = self._load_config_from_url(self.CONFIG_URL)
        cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")
        
        # 初始化模型并加载检查点
        model = init_segmentor(cfg)
        load_checkpoint(model, self.CHECKPOINT_URL, map_location="cpu")
        model.cuda()
        model.eval()
        return model

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """预处理图像"""
        if max(image.size) > self.max_size:
            ratio = self.max_size / max(image.size)
            new_size = tuple(int(x * ratio) for x in image.size)
            image = image.resize(new_size, Image.LANCZOS)
        return image

    def _get_ego_car_mask(self, camera_name: str) -> np.ndarray:
        """获取指定相机的ego car mask
        
        Args:
            camera_name: 相机名称
            
        Returns:
            ego car的二值mask
        """
        # 从base_dir下的ego_masks目录读取mask
        mask_path = Path(self.base_dir) / "ego_masks" / f"{camera_name}.png"
        if not mask_path.exists():
            print(f"警告: 未找到相机 {camera_name} 的ego mask: {mask_path}")
            return None
            
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return (mask > 127).astype(np.uint8)

    def _restore_original_size(self, segmentation: np.ndarray, original_size: tuple) -> np.ndarray:
        """
        将分割结果还原到原始图像尺寸
        
        Args:
            segmentation: 分割结果
            original_size: 原始图像尺寸 (width, height)
            
        Returns:
            还原尺寸后的分割结果
        """
        if segmentation.shape[:2] != (original_size[1], original_size[0]):
            segmentation = cv2.resize(
                segmentation, 
                original_size,  # (width, height)
                interpolation=cv2.INTER_NEAREST
            )
        return segmentation

    def _refine_ego_car_mask(self, segmentation: np.ndarray, ego_mask: np.ndarray, 
                            kernel_size: int = 5) -> np.ndarray:
        """优化ego car区域的标注
        
        Args:
            segmentation: 分割结果
            ego_mask: 原始ego car mask
            kernel_size: 形态学操作的核大小
            
        Returns:
            优化后的分割结果
        """
        # 创建形态学操作的核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 对ego mask进行膨胀操作，处理边界区域
        dilated_mask = cv2.dilate(ego_mask, kernel, iterations=3)
        
        # 获取膨胀区域（边界区域）
        boundary_region = dilated_mask & (~ego_mask)
        
        # 在边界区域中，如果像素与ego car区域相邻，则也标记为ego car
        segmentation_refined = segmentation.copy()
        segmentation_refined[ego_mask > 0] = self.EGO_CAR_LABEL
        segmentation_refined[boundary_region > 0] = self.EGO_CAR_LABEL
        
        return segmentation_refined

    def process_single_image(self, image_path: Path, output_path: Path, ego_mask: np.ndarray) -> None:
        """处理单张图像
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            ego_mask: ego car的二值mask
        """
        print(f'Processing: {image_path}')
        
        # 读取图像并获取原始尺寸
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        
        # 预处理图像
        image = self._preprocess_image(image)
        array = np.array(image)[:, :, ::-1]  # BGR
        
        # 进行分割
        segmentation_logits = inference_segmentor(self.model, array)[0]
        
        # 还原到原始尺寸
        segmentation_logits = self._restore_original_size(segmentation_logits, original_size)
        
        # 优化ego car区域
        segmentation_logits = self._refine_ego_car_mask(segmentation_logits, ego_mask)
        
        # 创建输出目录
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存原始标签
        label_path = output_path.with_suffix('.npz')
        np.savez_compressed(str(label_path), segmentation_logits.astype(np.uint8))
        
        # 可选：保存可视化结果
        if self.save_visualization:
            # 获取基础colormap并扩展以包含ego car的颜色
            colormap = self.DATASET_COLORMAPS["ade20k"].copy()
            
            # 确保colormap足够长
            if len(colormap) <= self.EGO_CAR_LABEL + 2:
                colormap.extend([[0, 0, 0]] * (self.EGO_CAR_LABEL + 2 - len(colormap)))
            
            # 设置ego car的颜色
            colormap[self.EGO_CAR_LABEL + 1] = self.EGO_CAR_COLOR.tolist()
            
            # 创建可视化图像
            colormap_array = np.array(colormap, dtype=np.uint8)
            segmentation_values = colormap_array[segmentation_logits + 1]
            
            vis_image = Image.fromarray(segmentation_values)
            vis_image.save(str(output_path))
            print(f'Saved visualization to: {output_path}')
        
        print(f'Saved semantic segmentation labels to: {label_path}')

    def process_dataset(self) -> None:
        """处理整个数据集"""
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 处理每个相机目录
        for camera_dir in ["camera_FRONT", "camera_FRONT_RIGHT", "camera_BACK_RIGHT",
                        "camera_BACK", "camera_BACK_LEFT", "camera_FRONT_LEFT"]:
            print(f"\n处理相机 {camera_dir}")
            
            # 创建相机输出目录
            camera_output_dir = self.output_dir / camera_dir
            camera_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取图像路径并排序
            image_paths = sorted(list((self.input_dir / camera_dir).glob("*.png"))) # 只取前10张
            
            if not image_paths:
                print(f"警告: 未找到相机 {camera_dir} 的图像")
                continue
            
            # 获取ego car mask
            ego_mask = self._get_ego_car_mask(camera_dir)
            if ego_mask is None:
                continue
            
            # 处理每张图片
            for img_path in image_paths:
                output_path = camera_output_dir / img_path.name
                self.process_single_image(img_path, output_path, ego_mask)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Semantic Segmentation using DINOv2')
    parser.add_argument('--base_dir', type=str, default="/home/hqlab/workspace/dataset/parkinglot",
                        help='Base directory for both input and output data.')
    parser.add_argument('--input_subdir', type=str, default="data/10_26/images",
                        help='Subdirectory under base_dir for input images.')
    parser.add_argument('--output_subdir', type=str, default="data/10_26/semantic",
                        help='Subdirectory under base_dir for output semantic segmentation.')
    parser.add_argument('--repo_path', type=str, default="/home/hqlab/workspace/base_model/dinov2",
                        help='Path to the DINOv2 repository.')
    parser.add_argument('--max_size', type=int, default=1024,
                        help='Maximum size for input images.')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization results.')
    
    args = parser.parse_args()
    
    segmenter = SemanticSegmentation(
        base_dir=args.base_dir,
        input_subdir=args.input_subdir,
        output_subdir=args.output_subdir,
        repo_path=args.repo_path,
        max_size=args.max_size,
        save_visualization=args.save_vis
    )
    segmenter.process_dataset()

if __name__ == "__main__":
    main()