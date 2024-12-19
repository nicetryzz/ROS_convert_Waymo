#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import cv2
from cv_bridge import CvBridge
import os
from pathlib import Path
from sensor_msgs.msg import CompressedImage, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from collections import defaultdict
import bisect
from typing import Dict, List, Tuple, Optional
import argparse
import json

class RosbagReader:
    """用于读取和处理ROS包数据的类"""
    
    # 相机话题名称映射到目录名称
    CAMERA_MAPPING = {
        '/cameras/front/image_color/compressed': 'camera_FRONT',
        '/cameras/front_right/image_color/compressed': 'camera_FRONT_RIGHT',
        '/cameras/rear_right/image_color/compressed': 'camera_BACK_RIGHT',
        '/cameras/rear/image_color/compressed': 'camera_BACK',
        '/cameras/rear_left/image_color/compressed': 'camera_BACK_LEFT',
        '/cameras/front_left/image_color/compressed': 'camera_FRONT_LEFT'
    }

    def __init__(self, bag_file: str, output_dir: str):
        """
        初始化RosbagReader
        
        Args:
            bag_file: rosbag文件路径
            output_dir: 输出目录路径
        """
        self.bag_file = Path(bag_file)
        self.output_dir = Path(output_dir)
        self.bridge = CvBridge()
        
        # 创建输出目录结构
        self.images_dir = self.output_dir / 'images'
        self.pointcloud_dir = self.output_dir / 'lidar'
        self.time_stamp_dir = self.output_dir / 'time'
        self.timestamp_dict = defaultdict(list)
        self._create_directories()

    def _create_directories(self) -> None:
        """创建必要的输出目录结构"""
        # 创建主输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建点云数据目录
        self.pointcloud_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建所有相机目录
        for camera_dir in self.CAMERA_MAPPING.values():
            camera_path = self.images_dir / camera_dir
            camera_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def find_closest_timestamp(target_time: int, timestamps: List[int]) -> int:
        """
        找到最接近的时间戳
        
        Args:
            target_time: 目标时间戳
            timestamps: 时间戳列表
            
        Returns:
            最接近的时间戳
        """
        idx = bisect.bisect_left(timestamps, target_time)
        if idx == 0:
            return timestamps[0]
        if idx == len(timestamps):
            return timestamps[-1]
        before = timestamps[idx - 1]
        after = timestamps[idx]
        return after if after - target_time < target_time - before else before

    def _collect_timestamps(self) -> Tuple[Dict[str, List], List, List]:
        """
        第一遍遍历：收集所有时间戳
        
        Returns:
            相机数据、点云时间戳和点云数据的元组
        """
        camera_data = defaultdict(list)
        pointcloud_timestamps = []
        pointcloud_data = []
        
        print("第一遍遍历：收集所有时间戳...")
        with rosbag.Bag(self.bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(
                topics=list(self.CAMERA_MAPPING.keys()) + ['/pandar_points']
            ):
                timestamp = t.to_nsec()
                if topic in self.CAMERA_MAPPING:
                    camera_data[topic].append((timestamp, msg))
                elif topic == '/pandar_points':
                    pointcloud_timestamps.append(timestamp)
                    pointcloud_data.append((timestamp, msg))
        
        # 对数据进行排序
        for topic in camera_data:
            camera_data[topic].sort(key=lambda x: x[0])
        pointcloud_data.sort(key=lambda x: x[0])
        self.timestamp_dict['pointcloud'] = pointcloud_timestamps
        
        return camera_data, pointcloud_timestamps, pointcloud_data

    def _save_pointcloud(self, points: np.ndarray, filename: str) -> None:
        """
        保存点云数据为PLY格式
        
        Args:
            points: 点云数据数组
            filename: 输出文件名
        """
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float intensity\n")
            f.write("end_header\n")
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]} {point[3]}\n")

    def process(self) -> None:
        """处理rosbag文件并提取数据"""
        # 收集时间戳
        camera_data, pc_timestamps, pointcloud_data = self._collect_timestamps()
        
        # 统计时间差
        total_time_diff = defaultdict(float)
        frame_count = 0
        
        # 处理每一帧数据
        print("开始处理点云和图像配对...")
        for frame_idx, (pc_timestamp, pc_msg) in enumerate(pointcloud_data):
            try:
                frame_number = f"{frame_idx:06d}"
                
                # 处理点云数据
                points = pc2.read_points(pc_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
                cloud = np.array(list(points))  # [N, 4] (x, y, z, intensity)
                
                # Waymo到P的坐标系转换矩阵
                # 从 (y_back, x_left, -z_down) 到 (x_front, y_left, z_up)
                pandar_to_waymo = np.array([
                    [0, -1, 0, 0],   # y_back -> x_front
                    [1, 0, 0, 0],  # x_left -> y_left
                    [0, 0, 1, 0],   # -z_down -> -z_down
                    [0, 0, 0, 1]
                ], dtype=np.float32)

                # 分离点云坐标和强度
                xyz = cloud[:, :3]
                intensity = cloud[:, 3]

                # 转换为齐次坐标
                xyz_homogeneous = np.hstack([xyz, np.ones((xyz.shape[0], 1))])  # [N, 4]

                # 应用坐标系转换
                xyz_waymo = (pandar_to_waymo @ xyz_homogeneous.T).T  # [N, 4]

                # 重新组合转换后的坐标和强度
                cloud = np.column_stack([xyz_waymo[:, :3], intensity])
                
                pc_filename = self.pointcloud_dir / f"{frame_number}.ply"
                self._save_pointcloud(cloud, str(pc_filename))
                print(f"保存点云: {pc_filename}")
                
                # 处理图像数据
                self._process_images(frame_number, pc_timestamp, camera_data, total_time_diff)
                
                frame_count += 1
                
            except Exception as e:
                print(f"处理数据时出错: {e}")
                continue
        
        # 打印统计信息
        self._print_statistics(total_time_diff, frame_count)
        
        # 保存时间戳
        self._save_timestamps()

    def _process_images(self, frame_number: str, pc_timestamp: int, 
                       camera_data: Dict, total_time_diff: Dict) -> None:
        """处理图像数据"""
        for topic in self.CAMERA_MAPPING:
            camera_timestamps = [x[0] for x in camera_data[topic]]
            closest_ts = self.find_closest_timestamp(pc_timestamp, camera_timestamps)
            matched_msg = next(msg for ts, msg in camera_data[topic] if ts == closest_ts)
            
            # 处理图像
            np_arr = np.frombuffer(matched_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # 保存图像
            camera_dir = self.CAMERA_MAPPING[topic]
            save_dir = self.images_dir / camera_dir
            img_filename = save_dir / f"{frame_number}.png"
            cv2.imwrite(str(img_filename), cv_image)
            
            # 计算时间差
            time_diff = abs(pc_timestamp - closest_ts) / 1e9
            total_time_diff[camera_dir] += time_diff
            print(f"保存图像: {img_filename} (时间差: {time_diff:.3f}秒)")
            
            # 记录图像时间戳
            self.timestamp_dict[camera_dir].append(closest_ts)

    def _print_statistics(self, total_time_diff: Dict, frame_count: int) -> None:
        """打印统计信息"""
        print("\n平均时间差统计：")
        for camera_dir in self.CAMERA_MAPPING.values():
            avg_time_diff = total_time_diff[camera_dir] / frame_count
            print(f"{camera_dir}: {avg_time_diff:.3f}秒")

    def _save_timestamps(self) -> None:
        """将时间戳保存到JSON文件"""
        self.time_stamp_dir.mkdir(parents=True, exist_ok=True)
        
        for key, timestamps in self.timestamp_dict.items():
            timestamp_file = self.time_stamp_dir / f"{key}.json"
            
            # 将时间戳转换为字符串格式
            timestamps_str = [str(ts) for ts in timestamps]
            
            with open(timestamp_file, 'w') as f:
                json.dump(timestamps_str, f, indent=2)
            
            print(f"{key}的时间戳已保存到: {timestamp_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Process a ROS bag file.")
    parser.add_argument('--base_dir', type=str, default="/home/hqlab/workspace/dataset/parkinglot",
                        help='Base directory for both input and output data.')
    parser.add_argument('--bag_subdir', type=str, default="raw_data/cql_circle_2024-10-26-01-28-40.bag",
                        help='Subdirectory under base_dir for the ROS bag file.')
    parser.add_argument('--output_subdir', type=str, default="data/10_26",
                        help='Subdirectory under base_dir for output data.')
    
    args = parser.parse_args()
    
    # 构建完整的输入和输出路径
    bag_file = Path(args.base_dir) / args.bag_subdir
    output_dir = Path(args.base_dir) / args.output_subdir
    
    reader = RosbagReader(str(bag_file), str(output_dir))
    reader.process()

if __name__ == "__main__":
    main()