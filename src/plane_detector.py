#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from pathlib import Path
import argparse
from scipy import ndimage
from utils import *
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from multiprocessing import Pool

class PlaneDetector:
    """平面检测器类"""
    
    # 语义标签定义
    SEMANTIC_LABELS = {
        # 需要处理的对象
        'process': [
            0,   # WALL
            3,   # FLOOR
            42,  # COLUMN
            14,  # DOOR
            43,  # SIGNBOARD
            5,   # CEILING
        ],
        # 不需要处理的对象
        'ignore': [
            20,  # CAR
            150, # EGO_CAR
            82,  # LIGHT
        ]
    }
    
    def __init__(self, base_dir: str, input_subdir: str, output_subdir: str, 
                 save_visualization: bool = True):
        """初始化平面检测器
        
        Args:
            base_dir: 基础目录
            input_subdir: 输入子目录，包含images/depths/normals/masks
            output_subdir: 输出子目录
            save_visualization: 是否保存可视化结果
        """
        self.base_dir = Path(base_dir)
        self.save_visualization = save_visualization
        
        # 设置输入输出路径
        self.image_dir = self.base_dir / input_subdir / "images"
        self.depth_dir = self.base_dir / input_subdir / "depths"
        self.normal_dir = self.base_dir / input_subdir / "normals"
        self.mask_dir = self.base_dir / input_subdir / "masks"
        self.output_dir = self.base_dir / output_subdir
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_single_image(self, image_path: Path) -> None:
        """处理单张图像
        
        Args:
            image_path: 输入图像路径
        """
        print(f'Processing: {image_path}')
        
        # 获取对应的深度图、法向量图和语义分割结果路径
        camera_dir = image_path.parent.name
        frame_name = image_path.stem
        
        depth_path = self.depth_dir / camera_dir / f"{frame_name}.npz"
        normal_path = self.normal_dir / camera_dir / f"{frame_name}.png"
        mask_path = self.mask_dir / camera_dir / f"{frame_name}.npz"
        
        # 读取所有数据
        image = cv2.imread(str(image_path))
        depth_data = np.load(str(depth_path))
        depth = depth_data['depth']
        depth_data.close()
        
        normal = cv2.imread(str(normal_path))
        normal = (normal.astype(np.float32) / 127.5) - 1.0
        
        # normal = self._depth_to_normals(depth)
        
        mask_data = np.load(str(mask_path))
        segmentation = mask_data['arr_0']
        mask_data.close()
        
        # 创建输出路径
        output_path = self.output_dir / camera_dir / f"{frame_name}.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 进行平面检测
        planes = self._detect_planes(image, depth, normal, segmentation)
        
        # 保存结果
        np.savez_compressed(str(output_path), planes=planes)
        
        # 可选：保存可视化结果
        if self.save_visualization:
            vis_path = output_path.with_suffix('.png')
            vis_image = self._visualize_planes(planes)
            cv2.imwrite(str(vis_path), vis_image)
    
    def process_dataset(self, use_multiprocess=True) -> None:
        """处理整个数据集
        
        Args:
            use_multiprocess: 是否使用多进程处理
        """
        print("开始处理数据集...")
        
        # 收集所有需要处理的图像路径
        all_images = []
        for camera_dir in ["camera_FRONT", "camera_FRONT_RIGHT", "camera_BACK_RIGHT",
                          "camera_BACK", "camera_BACK_LEFT", "camera_FRONT_LEFT"]:
            image_path = self.image_dir / camera_dir
            if not image_path.exists():
                print(f"警告: 目录不存在 {image_path}")
                continue
            all_images.extend(sorted(image_path.glob('*.png')))
        
        if use_multiprocess:
            # 多进程处理
            num_processes = max(1, multiprocessing.cpu_count() - 2)  # 预留2个CPU核心
            print(f"使用 {num_processes} 个进程进行处理")
            
            with Pool(processes=num_processes) as pool:
                pool.map(self.process_single_image, all_images)
        else:
            # 单进程处理
            print("使用单进程处理")
            for image_path in all_images:
                self.process_single_image(image_path)
    
    def _filter_invalid_regions(self, segmentation: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """过滤无效区域，包括语义和深度
        
        Args:
            segmentation: 语义分割结果 [H, W]
            depth: 深度图 [H, W]
            
        Returns:
            过滤后的mask，True表示需要保留的区域
        """
        # 创建有效区域mask
        valid_mask = np.zeros_like(segmentation, dtype=bool)
        
        # 标记需要处理的区域
        for label in self.SEMANTIC_LABELS['process']:
            valid_mask |= (segmentation == label)
        
        return valid_mask
    
    def _depth_to_normals(self, depth, fx=1211.792358, fy=1202.317627, cx=1017.378282, cy=787.396111):
        h, w = depth.shape
        
        u_map = np.ones((h, 1)) * np.arange(1, w + 1) - cx  # u-u0
        v_map = np.arange(1, h + 1).reshape(h, 1) * np.ones((1, w)) - cy  # v-v0
        
        Gu, Gv = get_DAG_filter(depth)
        est_nx = Gu * fx
        est_ny = Gv * fy
        est_nz = -(depth + v_map * Gv + u_map * Gu)
        est_normal = cv2.merge((est_nx, est_ny, est_nz))
        
        est_normal = vector_normalization(est_normal)
        
        est_normal = MRF_optim(depth, est_normal)
        
        return est_normal
    
    def _compute_depth_second_derivative(self, depth: np.ndarray) -> np.ndarray:
        """计算深度图的二阶导数
        
        Args:
            depth: 深度图 [H, W]
        Returns:
            二阶导数图 [H, W]
        """
        # 使用Sobel算子计算一阶导数
        dx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        
        # 对一阶导数再次求导得到二阶导数
        dxx = cv2.Sobel(dx, cv2.CV_64F, 1, 0, ksize=3)
        dyy = cv2.Sobel(dy, cv2.CV_64F, 0, 1, ksize=3)
        
        # 合并水平和垂直方向的二阶导数
        second_derivative = np.sqrt(dxx**2 + dyy**2)
        return second_derivative
    
    def _segment_by_two_stage(self, normals: np.ndarray, depth: np.ndarray, 
                             segmentation: np.ndarray, valid_mask: np.ndarray,
                             angle_threshold: float = 17,
                             coarse_curvature_threshold: float = 0,
                             fine_curvature_threshold: float = 0.05,
                             min_region_size: int = 7500) -> np.ndarray:
        """两阶段分割方法
        1. 仅使用法向量进行区域生长得到初步分割
        2. 根据语义标签重新分配平面：
           - 墙面和地面直接合并
           - 其他物体（如柱子）使用法向量和二阶导数进行区域生长
        """
        h, w = normals.shape[:2]
        visited = ~valid_mask
        
        # 计算深度图的二阶导数（供第二阶段使用）
        depth_curvature = self._compute_depth_second_derivative(depth)
        
        def get_neighbors(y, x):
            neighbors = []
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    neighbors.append((ny, nx))
            return neighbors
        
        def check_consistency(seed_y, seed_x, ny, nx, curvature_threshold):
            # 检查法向量和二阶导数一致性
            seed_normal = normals[seed_y, seed_x]
            current_normal = normals[ny, nx]
            dot_product = np.clip(np.dot(seed_normal, current_normal), -1.0, 1.0)
            angle = np.arccos(dot_product) * 180 / np.pi
            
            if curvature_threshold > 0:
                # 检查二阶导数
                curvature = depth_curvature[ny, nx]
                return angle < angle_threshold and curvature < curvature_threshold
            else:
                return angle < angle_threshold
        
        # 第一阶段：仅使用法向量的区域生长
        initial_segments = np.zeros((h, w), dtype=int)
        current_label = 0
        
        points = [(y, x) for y in range(h) for x in range(w) if not visited[y, x]]
        for y, x in points:
            if visited[y, x]:
                continue
            
            queue = [(y, x)]
            region = [(y, x)]
            visited[y, x] = True
            
            while queue:
                cy, cx = queue.pop(0)
                for ny, nx in get_neighbors(cy, cx):
                    if check_consistency(y, x, ny, nx, coarse_curvature_threshold):
                        queue.append((ny, nx))
                        region.append((ny, nx))
                        visited[ny, nx] = True
            
            if len(region) >= min_region_size:
                current_label += 1
                for ry, rx in region:
                    initial_segments[ry, rx] = current_label
        
        # return initial_segments
         
        # 第二阶段：根据语义重新分配
        final_segments = np.zeros_like(initial_segments)
        current_label = 0
        visited = ~valid_mask  # 重置访问标记
        
        # 定义需要直接合并的语义类别
        merge_semantics = {0, 3}  # WALL和FLOOR
        
        # 处理需要直接合并的语义类别
        for semantic_label in merge_semantics:
            # 获取当前语义类别的所有初始平面标签
            semantic_mask = (segmentation == semantic_label)
            initial_labels = np.unique(initial_segments[semantic_mask & (initial_segments > 0)])
            
            # 对每个初始平面分别处理
            for initial_label in initial_labels:
                mask = (segmentation == semantic_label) & (initial_segments == initial_label)
                if np.any(mask) and mask.sum() >= min_region_size:
                    current_label += 1
                    final_segments[mask] = current_label
        
        # 处理需要进行二次区域生长的类别
        other_semantics = set(self.SEMANTIC_LABELS['process']) - merge_semantics
        
        # 对其他语义类别进行二次区域生长
        for semantic_label in other_semantics:
            # 获取当前语义类别的所有未访问点
            semantic_points = [(y, x) for y in range(h) for x in range(w) 
                             if not visited[y, x] and segmentation[y, x] == semantic_label]
            
            # 按二阶导数大小排序，优先从平坦区域开始生长
            semantic_points.sort(key=lambda p: depth_curvature[p[0], p[1]])
            
            for y, x in semantic_points:
                if visited[y, x]:
                    continue
                
                queue = [(y, x)]
                region = [(y, x)]
                visited[y, x] = True
                
                while queue:
                    cy, cx = queue.pop(0)
                    for ny, nx in get_neighbors(cy, cx):
                        # 确保邻居点属于同一语义类别
                        if segmentation[ny, nx] == semantic_label:
                            if check_consistency(y, x, ny, nx, fine_curvature_threshold):
                                queue.append((ny, nx))
                                region.append((ny, nx))
                                visited[ny, nx] = True
                
                if len(region) >= min_region_size:
                    current_label += 1
                    for ry, rx in region:
                        final_segments[ry, rx] = current_label
        
        return final_segments
    
    def _detect_planes(self, image: np.ndarray, depth: np.ndarray, 
                      normal: np.ndarray, segmentation: np.ndarray,) -> tuple:
        """检测平面
        
        Args:
            image: RGB图像 [H, W, 3]
            depth: 深度图 [H, W]
            normal: 法向量图 [H, W, 3]
            segmentation: 语义分割结果 [H, W]
            
        Returns:
            (refined_segments, plane_info): 优化后的分割结果和平面信息
        """
        # 1. 过滤无效区域
        valid_mask = self._filter_invalid_regions(segmentation, depth)

        
        # 2. 使用区域生长进行分割
        segments = self._segment_by_two_stage(normal, depth, segmentation, valid_mask)
        
        # 3. 使用深度信息优化分割结果
        # refined_segments = self._refine_by_depth(segments, depth)
        
        return segments
    
    def _visualize_planes(self, segments: np.ndarray) -> np.ndarray:
        """可视化检测到的平面
        
        Args:
            segments: 分割结果 [H, W]，每个区域有唯一的标签
            
        Returns:
            可视化结果图像 [H, W, 3]
        """
        # 为每个分割区域随机分配颜色
        n_segments = segments.max() + 1
        colors = np.random.randint(0, 255, (n_segments + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # 背景为黑色
        
        # 创建彩色分割图
        colored_segments = colors[segments]
        
        return colored_segments

def main():
    parser = argparse.ArgumentParser(description='Detect planes from images')
    parser.add_argument('--base_dir', type=str, 
                        default="/home/hqlab/workspace/dataset/parkinglot",
                        help='Base directory')
    parser.add_argument('--input_subdir', type=str, 
                        default="data/10_26",
                        help='Input subdirectory')
    parser.add_argument('--output_subdir', type=str, 
                        default="data/10_26/planes",
                        help='Output subdirectory')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization results')
    
    args = parser.parse_args()
    
    detector = PlaneDetector(args.base_dir, args.input_subdir, args.output_subdir, 
                           save_visualization=args.save_vis)
    detector.process_dataset(use_multiprocess=True)
    # detector = PlaneDetector(args.base_dir, args.input_subdir, args.output_subdir, 
    #                         save_visualization=True)
    # detector.process_single_image(Path("/home/hqlab/workspace/dataset/parkinglot/data/10_26/images/camera_BACK_LEFT/000276.png"))
if __name__ == "__main__":
    main()