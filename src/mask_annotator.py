#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from pathlib import Path
import argparse

class MaskAnnotator:
    """交互式mask标注工具"""
    
    def __init__(self, image_dir: str, output_dir: str, max_display_size: int = 1200):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_display_size = max_display_size
        
        # 标注状态
        self.points = []
        self.current_image = None
        self.original_image = None  # 保存原始图像
        self.scale_factor = 1.0    # 缩放比例
        self.window_name = "Mask Annotator"
        self.polygons = []  # 存储多个多边形
        self.current_polygon = []  # 当前正在标注的多边形
        self.snap_threshold = 10  # 吸附阈值（像素）
        
    def _resize_image(self, image):
        """调整图像大小以适应显示"""
        h, w = image.shape[:2]
        if max(h, w) > self.max_display_size:
            scale = self.max_display_size / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            resized = cv2.resize(image, new_size)
            return resized, scale
        return image.copy(), 1.0
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 转换到原始图像坐标
            original_x = int(x / self.scale_factor)
            original_y = int(y / self.scale_factor)
            
            # 吸附到图像边缘
            h, w = self.original_image.shape[:2]
            if original_x < self.snap_threshold:
                original_x = 0
            elif original_x > w - self.snap_threshold:
                original_x = w
            if original_y < self.snap_threshold:
                original_y = 0
            elif original_y > h - self.snap_threshold:
                original_y = h
                
            self.current_polygon.append((original_x, original_y))
            self.update_display()
            
    def update_display(self):
        """更新显示"""
        if self.current_image is None:
            return
            
        # 创建显示图像的副本
        display = self.current_image.copy()
        
        # 绘制已完成的多边形
        for polygon in self.polygons:
            display_points = np.array([(int(x * self.scale_factor), int(y * self.scale_factor)) 
                                     for x, y in polygon])
            cv2.fillPoly(display, [display_points], (0, 100, 0))  # 半透明填充
            cv2.polylines(display, [display_points], True, (0, 255, 0), 2)
        
        # 绘制当前多边形的点和线
        if self.current_polygon:
            # 绘制点
            for point in self.current_polygon:
                display_x = int(point[0] * self.scale_factor)
                display_y = int(point[1] * self.scale_factor)
                cv2.circle(display, (display_x, display_y), 3, (0, 255, 0), -1)
            
            # 绘制线
            if len(self.current_polygon) > 1:
                display_points = np.array([(int(x * self.scale_factor), int(y * self.scale_factor)) 
                                         for x, y in self.current_polygon])
                cv2.polylines(display, [display_points], False, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, display)
        
    def generate_mask(self, camera_name: str):
        """为指定相机生成mask"""
        # 获取所有图像路径
        image_paths = sorted(list((self.image_dir / camera_name).glob("*.png")))
        if not image_paths:
            print(f"未找到相机 {camera_name} 的图像")
            return
            
        # 当前图像索引
        current_image_idx = 0
        
        def load_current_image():
            """加载当前索引的图像"""
            self.original_image = cv2.imread(str(image_paths[current_image_idx]))
            self.current_image, self.scale_factor = self._resize_image(self.original_image)
            print(f"加载图像: {image_paths[current_image_idx].name}")
        
        # 加载第一张图像
        load_current_image()
        
        # 创建窗口和鼠标回调
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n=== 标注说明 ===")
        print("- 左键点击添加点")
        print("- 按'c'清除当前多边形")
        print("- 按'n'完成当前多边形并开始新的多边形")
        print("- 按'f'完成所有标注")
        print("- 按'q'跳过当前相机")
        print("- 按'z'撤销上一个点")
        print("- 按'b'撤销上一个多边形")
        print("- 按'r'切换到下一张图像")
        print("- 靠近边缘会自动吸附")
        
        while True:
            self.update_display()
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('c'):  # 清除当前多边形
                self.current_polygon = []
                self.update_display()
            elif key == ord('z'):  # 撤销上一个点
                if self.current_polygon:
                    self.current_polygon.pop()
                    print("撤销上一个点")
                    self.update_display()
            elif key == ord('b'):  # 撤销上一个多边形
                if self.polygons:
                    removed_polygon = self.polygons.pop()
                    print(f"撤销上一个多边形 (点数: {len(removed_polygon)})")
                    self.update_display()
            elif key == ord('r'):  # 切换到下一张图像
                current_image_idx = (current_image_idx + 1) % len(image_paths)
                load_current_image()
                print(f"切换到图像 {current_image_idx + 1}/{len(image_paths)}")
            elif key == ord('n'):  # 完成当前多边形，开始新的
                if len(self.current_polygon) > 2:
                    self.polygons.append(self.current_polygon)
                    self.current_polygon = []
                    print("完成当前多边形，开始新的多边形")
                    self.update_display()
            elif key == ord('f'):  # 完成所有标注
                if self.current_polygon and len(self.current_polygon) > 2:
                    self.polygons.append(self.current_polygon)
                
                if self.polygons:
                    # 生成mask（使用原始图像尺寸）
                    h, w = self.original_image.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # 填充所有多边形
                    for polygon in self.polygons:
                        points = np.array(polygon)
                        cv2.fillPoly(mask, [points], 255)
                    
                    # 保存mask
                    output_path = self.output_dir / f"{camera_name}.png"
                    cv2.imwrite(str(output_path), mask)
                    print(f"已保存mask到 {output_path}")
                    break
            elif key == ord('q'):  # 跳过
                print(f"跳过相机 {camera_name}")
                break
                
        cv2.destroyWindow(self.window_name)
        self.current_polygon = []
        self.polygons = []
        self.current_image = None
        self.original_image = None

def main():
    parser = argparse.ArgumentParser(description='Mask annotation tool')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing camera images')
    parser.add_argument('--image_subdir', type=str, default="data/10_26/images",
                        help='Subdirectory containing camera images')
    parser.add_argument('--output_subdir', type=str, default="ego_masks",
                        help='Output directory for masks')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    image_dir = base_dir / args.image_subdir
    output_dir = base_dir / args.output_subdir
    
    annotator = MaskAnnotator(image_dir, output_dir)
    
    # 处理每个相机
    camera_dirs = [
        "camera_FRONT", "camera_FRONT_RIGHT", "camera_BACK_RIGHT",
        "camera_BACK", "camera_BACK_LEFT", "camera_FRONT_LEFT"
    ]
    
    for camera in camera_dirs:
        print(f"\n处理相机 {camera}")
        annotator.generate_mask(camera)

if __name__ == "__main__":
    main()