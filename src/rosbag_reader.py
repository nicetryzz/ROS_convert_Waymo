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
from typing import Dict, List, Tuple, Optional, Any
import argparse
import json

class RosbagReader:
    """
    用于读取和处理ROS包数据的类。
    采用两遍式处理和即时处理策略，并支持对有效帧进行降采样。
    """
    
    CAMERA_MAPPING = {
        '/cameras/front/image_color/compressed': 'camera_FRONT',
        '/cameras/front_right/image_color/compressed': 'camera_FRONT_RIGHT',
        '/cameras/rear_right/image_color/compressed': 'camera_BACK_RIGHT',
        '/cameras/rear/image_color/compressed': 'camera_BACK',
        '/cameras/rear_left/image_color/compressed': 'camera_BACK_LEFT',
        '/cameras/front_left/image_color/compressed': 'camera_FRONT_LEFT'
    }

    def __init__(self, bag_file: str, output_dir: str, downsample_rate: int = 1):
        self.bag_file = Path(bag_file)
        self.output_dir = Path(output_dir)
        self.bridge = CvBridge()
        self.downsample_rate = downsample_rate
        
        self.images_dir = self.output_dir / 'images_all'
        self.pointcloud_dir = self.output_dir / 'lidar_all'
        self.time_stamp_dir = self.output_dir / 'time'
        self.output_info_dir = self.output_dir / 'info'
        self.timestamp_dict = {}
        self._create_directories()

    def _create_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pointcloud_dir.mkdir(parents=True, exist_ok=True)
        self.time_stamp_dir.mkdir(parents=True, exist_ok=True)
        self.output_info_dir.mkdir(parents=True, exist_ok=True)
        for camera_dir in self.CAMERA_MAPPING.values():
            (self.images_dir / camera_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def find_closest_timestamp_info(target_time: int, timestamps_info: List[Tuple[int, Any]]) -> Tuple[int, Any]:
        timestamps = [ts for ts, _ in timestamps_info]
        idx = bisect.bisect_left(timestamps, target_time)
        if idx == 0: return timestamps_info[0]
        if idx == len(timestamps): return timestamps_info[-1]
        before_ts, _ = timestamps_info[idx - 1]
        after_ts, _ = timestamps_info[idx]
        return timestamps_info[idx] if after_ts - target_time < target_time - before_ts else timestamps_info[idx - 1]

    @staticmethod
    def _find_closest_next_info(target_time: int, camera_infos: List[Tuple[int, Any]], tolerance: int) -> Optional[Tuple[int, Any]]:
        future_frames = [info for info in camera_infos if info[1] >= target_time]
        if not future_frames: return None
        closest_next_info = min(future_frames, key=lambda x: x[1])
        if (closest_next_info[1] - target_time) > tolerance: return None
        return closest_next_info

    def _plan_extraction(self) -> Tuple[Dict[Any, Tuple], List[int], Dict, Dict]:
        """
        第一遍 (Pass 1): 扫描元数据，执行同步和过滤，然后进行降采样，最后创建“行动地图”。
        """
        print("第一遍 (Pass 1): 收集元数据并构建提取计划...")
        # 1. & 2. & 3. 收集元数据, 寻找同步起点, 计算相对时间戳 (与之前版本相同)
        temp_camera_infos = defaultdict(list)
        temp_lidar_infos = []
        msg_idx = 0
        with rosbag.Bag(self.bag_file, 'r') as bag:
            topics = list(self.CAMERA_MAPPING.keys()) + ['/pandar_points']
            for topic, msg, t in bag.read_messages(topics=topics):
                header_nsec = msg.header.stamp.to_nsec()
                if topic in self.CAMERA_MAPPING: temp_camera_infos[topic].append((header_nsec, t.to_nsec()))
                else: temp_lidar_infos.append((header_nsec, t.to_nsec(), msg_idx)); msg_idx += 1
        for topic in temp_camera_infos: temp_camera_infos[topic].sort(key=lambda x: x[0])
        temp_lidar_infos.sort(key=lambda x: x[0])
        print("寻找同步起始点...")
        tolerance = int(0.1 * 1e9)
        first_frame_infos = {}
        while True:
            if not temp_lidar_infos: raise RuntimeError("无法找到有效的同步起始点。")
            matched = True
            _, first_lidar_ts, _ = temp_lidar_infos[0]
            for topic, infos in temp_camera_infos.items():
                closest_info = self._find_closest_next_info(first_lidar_ts, infos, tolerance)
                if closest_info is None: temp_lidar_infos.pop(0); matched = False; break
                else: first_frame_infos[topic] = closest_info
            if matched: print("成功找到同步起始点！"); break
        start_lidar_header_ts, _, _ = temp_lidar_infos[0]
        final_camera_infos = defaultdict(list)
        relative_timestamps_map = {}
        for topic, infos in temp_camera_infos.items():
            start_cam_info = first_frame_infos[topic]
            start_cam_header_ts, _ = start_cam_info
            start_idx = infos.index(start_cam_info)
            final_camera_infos[topic] = infos[start_idx:]
            for header_nsec, bag_time in final_camera_infos[topic]: relative_timestamps_map[bag_time] = header_nsec - start_cam_header_ts
        bag_time_to_original_idx = {info[1]: info[2] for info in temp_lidar_infos}
        synced_lidar_bag_times = [info[1] for info in temp_lidar_infos]
        for header_nsec, bag_time, _ in temp_lidar_infos: relative_timestamps_map[bag_time] = header_nsec - start_lidar_header_ts
        
        # 4. 找出所有高质量的有效帧
        print("过滤低质量同步帧...")
        valid_frames_plan = []
        max_time_diff_sec = 0.02
        for lidar_bag_time in synced_lidar_bag_times:
            pc_relative_ts = relative_timestamps_map[lidar_bag_time]
            is_frame_valid = True
            potential_image_entries = {}
            for topic, cam_infos in final_camera_infos.items():
                cam_relative_infos = [(relative_timestamps_map[bt], bt) for _, bt in cam_infos]
                closest_cam_relative_ts, closest_cam_bag_time = self.find_closest_timestamp_info(pc_relative_ts, cam_relative_infos)
                time_diff = abs(pc_relative_ts - closest_cam_relative_ts) / 1e9
                if time_diff > max_time_diff_sec:
                    is_frame_valid = False
                    print(f"-> 丢弃雷达帧 (rel_ts {pc_relative_ts}): 与 {self.CAMERA_MAPPING[topic]} 时间差过大 ({time_diff:.4f}s > {max_time_diff_sec}s)")
                    break
                potential_image_entries[topic] = {'bag_time': closest_cam_bag_time, 'relative_ts': closest_cam_relative_ts, 'time_diff_sec': time_diff}
            
            if is_frame_valid:
                valid_frames_plan.append({
                    'lidar_bag_time': lidar_bag_time,
                    'image_plans': potential_image_entries
                })
        
        print(f"找到 {len(valid_frames_plan)} 个有效帧。")

        # 5. 执行降采样
        if self.downsample_rate > 1:
            downsampled_plan = valid_frames_plan[::self.downsample_rate]
            print(f"按频率 {self.downsample_rate} 进行降采样后，将保存 {len(downsampled_plan)} 帧。")
        else:
            downsampled_plan = valid_frames_plan

        # 6. 基于降采样后的计划，构建最终的“行动地图”
        print("构建最终行动地图...")
        extraction_map = {}
        final_lidar_timestamps = []
        final_image_timestamps = defaultdict(list)
        saved_lidar_original_indices = []

        for output_frame_idx, frame_plan in enumerate(downsampled_plan):
            frame_number = f"{output_frame_idx:06d}"
            lidar_bag_time = frame_plan['lidar_bag_time']
            pc_relative_ts = relative_timestamps_map[lidar_bag_time]
            
            # 添加雷达计划
            pc_savename = self.pointcloud_dir / f"{frame_number}.ply"
            extraction_map[lidar_bag_time] = ('lidar', str(pc_savename), {'relative_ts': pc_relative_ts})
            final_lidar_timestamps.append(pc_relative_ts)
            saved_lidar_original_indices.append(bag_time_to_original_idx[lidar_bag_time])

            # 添加图像计划
            for topic, image_plan in frame_plan['image_plans'].items():
                camera_dir = self.CAMERA_MAPPING[topic]
                img_savename = self.images_dir / camera_dir / f"{frame_number}.png"
                extraction_map[image_plan['bag_time']] = ('image', str(img_savename), {'topic': topic, 'time_diff_sec': image_plan['time_diff_sec']})
                final_image_timestamps[camera_dir].append(image_plan['relative_ts'])

        return extraction_map, saved_lidar_original_indices, final_lidar_timestamps, final_image_timestamps

    def _process_and_save_pointcloud(self, msg: PointCloud2, save_path: str):
        points = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        cloud = np.array(list(points))
        pandar_to_waymo = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        xyz_h = np.hstack([cloud[:, :3], np.ones((cloud.shape[0], 1))])
        xyz_w = (pandar_to_waymo @ xyz_h.T).T
        cloud = np.column_stack([xyz_w[:, :3], cloud[:, 3]])
        with open(save_path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(cloud)}\n")
            f.write("property float x\nproperty float y\nproperty float z\nproperty float intensity\nend_header\n")
            np.savetxt(f, cloud, fmt='%f %f %f %f')
        print(f"  保存点云: {save_path}")

    def _process_and_save_image(self, msg: CompressedImage, save_path: str, time_diff: float):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(save_path, cv_image)
        print(f"  保存图像: {save_path} (时间差: {time_diff:.4f}s)")

    def process(self) -> None:
        extraction_map, lidar_indices, lidar_ts, image_ts = self._plan_extraction()
        total_time_diff = defaultdict(float)
        frame_count = len(lidar_ts)
        if frame_count == 0:
            print("\n警告: 经过滤和降采样后，没有符合条件的帧可以处理。")
            return
        print(f"\n第二遍 (Pass 2): 即时处理并保存 {len(extraction_map)} 个规划好的消息...")
        processed_count = 0
        with rosbag.Bag(self.bag_file, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=list(self.CAMERA_MAPPING.keys()) + ['/pandar_points']):
                if t.to_nsec() in extraction_map:
                    msg_type, save_path, extra_info = extraction_map[t.to_nsec()]
                    if msg_type == 'lidar': self._process_and_save_pointcloud(msg, save_path)
                    elif msg_type == 'image':
                        time_diff = extra_info['time_diff_sec']
                        self._process_and_save_image(msg, save_path, time_diff)
                        camera_dir = self.CAMERA_MAPPING[extra_info['topic']]
                        total_time_diff[camera_dir] += time_diff
                    processed_count += 1
                    if processed_count == len(extraction_map):
                        print("所有规划的消息已处理完毕，提前结束遍历。")
                        break
        self._print_statistics(total_time_diff, frame_count)
        self.timestamp_dict['pointcloud'] = lidar_ts
        for cam_name, ts_list in image_ts.items(): self.timestamp_dict[cam_name] = sorted(list(set(ts_list)))
        self._save_timestamps()
        self._save_indices(lidar_indices)
        
    def _print_statistics(self, total_time_diff: Dict, frame_count: int):
        if frame_count == 0:
            print("\n没有处理任何帧，无法统计。")
            return
        print("\n平均时间差统计：")
        for camera_dir in self.CAMERA_MAPPING.values():
            if camera_dir in total_time_diff:
                avg_time_diff = total_time_diff[camera_dir] / frame_count
                print(f"{camera_dir}: {avg_time_diff:.4f}秒")

    def _save_timestamps(self):
        self.time_stamp_dir.mkdir(parents=True, exist_ok=True)
        for key, timestamps in self.timestamp_dict.items():
            ts_file = self.time_stamp_dir / f"{key}.json"
            with open(ts_file, 'w') as f:
                json.dump([str(ts) for ts in timestamps], f, indent=2)
            print(f"{key}的时间戳已保存到: {ts_file}")
    
    def _save_indices(self, indices: List[int]):
        indices_file = self.output_info_dir / "lidar_original_indices.json"
        with open(indices_file, 'w') as f:
            json.dump(indices, f, indent=2)
        print(f"\n已保存 {len(indices)} 个雷达帧的原始索引到: {indices_file}")

def main():
    parser = argparse.ArgumentParser(description="Process a ROS bag file with filtering and downsampling.")
    parser.add_argument('--base_dir', type=str, default="/media/hqlab/b9795ed4-8132-4cfb-8048-e3705c71f80b/home/hqlab/byd_dataset_ws")
    parser.add_argument('--bag_subdir', type=str, default="2025-07-17-15-48-45.bag")
    parser.add_argument('--output_subdir', type=str, default="data/long_test")
    parser.add_argument('--downsample', type=int, default=5,
                        help='Rate for downsampling the valid frames. e.g., 5 means saving 1 out of every 5 valid frames. Default is 1 (no downsampling).')
    args = parser.parse_args()
    
    bag_file = Path(args.base_dir) / args.bag_subdir
    output_dir = Path(args.base_dir) / args.output_subdir
    
    if not bag_file.exists():
        print(f"错误: Rosbag 文件未找到 at {bag_file}")
        return
        
    print(f"输入 Rosbag: {bag_file}")
    print(f"输出目录: {output_dir}")
    print(f"时间同步过滤阈值: 0.02 秒")
    if args.downsample > 1:
        print(f"降采样频率: 每 {args.downsample} 帧有效数据取 1 帧")
    
    # 将 downsample 参数传递给 RosbagReader
    reader = RosbagReader(str(bag_file), str(output_dir), downsample_rate=args.downsample)
    reader.process()

if __name__ == "__main__":
    main()