#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shutil
from pathlib import Path
import argparse
import json
from collections import defaultdict

class UnifiedSampler:
    """
    一个统一的采样器，用于从已提取的数据中，
    按指定范围和采样率，同步采样图像、点云和SLAM位姿。
    """
    
    CAMERAS = {
        'camera_FRONT', 'camera_FRONT_RIGHT', 'camera_BACK_RIGHT',
        'camera_BACK', 'camera_BACK_LEFT', 'camera_FRONT_LEFT'
    }

    def __init__(self, source_dir: str, pose_file: str, sampling_rate: int, 
                 start_frame: int, end_frame: int, only_pose: bool):
        """
        初始化 UnifiedSampler
        
        Args:
            source_dir (str): 包含images_all, lidar_all, info的源数据目录。
            pose_file (str): LeGO-LOAM生成的完整位姿文件路径。
            sampling_rate (int): 采样率，即每N帧取一帧。
            start_frame (int): 采样的起始帧编号（包含）。
            end_frame (int): 采样的终止帧编号（包含）。-1代表直到最后一帧。
        """
        # --- 基础路径设置 ---
        self.source_dir = Path(source_dir)
        self.pose_file_path = Path(self.source_dir / pose_file)
        
        # --- 采样参数 ---
        self.sampling_rate = sampling_rate
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.only_pose = only_pose
        
        # --- 自动推断源和目标路径 ---
        self.source_images_dir = self.source_dir / 'images_all'
        self.source_lidar_dir = self.source_dir / 'lidar_all'
        self.source_time_dir = self.source_dir / 'time'
        self.index_file_path = self.source_dir / 'info' / 'lidar_original_indices.json'
        
        self.target_rgb_dir = self.source_dir / 'images'
        self.target_lidar_dir = self.source_dir / 'lidar'
        self.target_time_dir = self.source_dir / 'time'
        # 输出的位姿文件将保存在源目录的根下
        self.output_pose_path = self.source_dir / 'lidar_poses.txt'
        
        # 打印初始化信息
        self._print_initial_info()

    def _print_initial_info(self):
        print(f"\n--- 统一采样器初始化 ---")
        print(f"源数据目录: '{self.source_dir}'")
        print(f"源位姿文件: '{self.pose_file_path}'")
        print(f"索引映射文件: '{self.index_file_path}'")
        print("-" * 20)
        print(f"采样范围: 从帧 {self.start_frame} 到 {'最后一帧' if self.end_frame == -1 else self.end_frame}。")
        print(f"采样率: 每 {self.sampling_rate} 帧取 1 帧。")
        print("-" * 20)
        print(f"输出图像 -> '{self.target_rgb_dir}'")
        print(f"输出点云 -> '{self.target_lidar_dir}'")
        print(f"输出位姿 -> '{self.output_pose_path}'")
        print("-" * 40)

    def _create_target_directories(self) -> None:
        """创建所有采样输出目录"""
        self.target_rgb_dir.mkdir(parents=True, exist_ok=True)
        self.target_lidar_dir.mkdir(parents=True, exist_ok=True)
        for camera_name in self.CAMERAS:
            (self.target_rgb_dir / camera_name).mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """执行统一的采样流程"""
        # --- 0. 健全性检查 ---
        if (not self.source_lidar_dir.exists() or not self.index_file_path.exists() or not self.pose_file_path.exists()) and not self.only_pose:
            print("错误：缺少必要的源文件或目录。请确保以下路径均正确：")
            print(f"  - 点云目录: {self.source_lidar_dir}")
            print(f"  - 索引文件: {self.index_file_path}")
            print(f"  - 位姿文件: {self.pose_file_path}")
            return
        
        # --- 1. 创建目标目录 ---
        self._create_target_directories()
        
        original_timestamps = {}
        timestamp_keys = list(self.CAMERAS) + ['pointcloud']
        for key in timestamp_keys:
            ts_file = self.source_time_dir / f"{key}.json"
            if ts_file.exists():
                with open(ts_file, 'r') as f:
                    original_timestamps[key] = json.load(f)
                print(f"  已加载 '{ts_file.name}'")
            else:
                print(f"  [警告] 时间戳文件未找到: {ts_file}")
        
        # --- 2. 加载原始索引映射文件 ---
        with open(self.index_file_path, 'r') as f:
            # 这是从rosbag_reader提取出的所有帧，到它们在原始bag中索引的映射
            original_indices_map = json.load(f)
            
        if self.only_pose:
            print(f"\n阶段二：根据 {len(original_indices_map)} 个索引采样位姿文件...")
            with open(self.pose_file_path, 'r') as f_in, open(self.output_pose_path, 'w') as f_out:
                poses_written = 0
                for line_num, line_content in enumerate(f_in):
                    if line_num in original_indices_map:
                        f_out.write(line_content)
                        poses_written += 1
            return

        # --- 3. 采样图像和点云，同时收集需要保留的位姿索引 ---
        lidar_files = sorted(list(self.source_lidar_dir.glob('*.ply')))
        total_frames = len(lidar_files)

        start_index = self.start_frame
        end_index = total_frames if self.end_frame == -1 or self.end_frame >= total_frames else self.end_frame + 1

        if start_index >= end_index:
            print(f"错误：起始帧 ({start_index}) 必须小于终止帧 ({end_index-1})。")
            return
            
        print(f"\n阶段一：采样图像和点云，并收集位姿索引...")
        pose_indices_to_keep = set() # 用集合来存储需要保留的位姿行号，查找效率高
        sampled_timestamps = defaultdict(list)
        sampled_count = 0
        for i in range(start_index, end_index, self.sampling_rate):
            # 获取原始帧的文件名
            original_frame_name = lidar_files[i].stem
            
            # 复制点云
            source_lidar_path = lidar_files[i]
            target_lidar_path = self.target_lidar_dir / f"{sampled_count:06d}.ply"
            shutil.copy2(source_lidar_path, target_lidar_path)
            
            # 复制图像
            for camera_dir_name in self.CAMERAS:
                source_img_path = self.source_images_dir / camera_dir_name / f"{original_frame_name}.png"
                if source_img_path.exists():
                    target_img_path = self.target_rgb_dir / camera_dir_name / f"{sampled_count:06d}.png"
                    shutil.copy2(source_img_path, target_img_path)
            
            for key, ts_list in original_timestamps.items():
                if i < len(ts_list):
                    sampled_timestamps[key].append(ts_list[i])

            # 关键一步：记录下这一帧对应的原始rosbag索引号
            original_bag_index = original_indices_map[i]
            pose_indices_to_keep.add(original_bag_index)
            
            print(f"  处理帧 {i} -> 输出帧 {sampled_count} (原始bag索引: {original_bag_index})")
            sampled_count += 1
        
        print(f"阶段一完成！采样了 {sampled_count} 帧图像和点云。")
        
        # --- 4. 根据收集到的索引，采样位姿文件 ---
        print(f"\n阶段二：根据 {len(pose_indices_to_keep)} 个索引采样位姿文件...")
        with open(self.pose_file_path, 'r') as f_in, open(self.output_pose_path, 'w') as f_out:
            poses_written = 0
            for line_num, line_content in enumerate(f_in):
                if line_num in pose_indices_to_keep:
                    f_out.write(line_content)
                    poses_written += 1
        
        print("阶段二完成！")
        
        print(f"\n阶段三：保存采样后的时间戳文件...")
        for key, ts_list in sampled_timestamps.items():
            target_ts_file = self.target_time_dir / f"{key}.json"
            with open(target_ts_file, 'w') as f:
                json.dump(ts_list, f, indent=2)
            print(f"  已保存 {len(ts_list)} 个时间戳到 '{target_ts_file.name}'")
        
        if poses_written != sampled_count:
            print(f"  [警告] 采样出的位姿数量 ({poses_written}) 与图像/点云数量 ({sampled_count}) 不匹配！请检查LeGO-LOAM是否处理了所有雷达帧。")
        else:
            print(f"  成功筛选出 {poses_written} 条位姿。")
        
        print(f"\n所有采样任务完成！")


def main():
    """主函数，用于解析命令行参数并启动统一采样器。"""
    parser = argparse.ArgumentParser(
        description="统一采样器：同步采样图像、点云和SLAM位姿。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--source_dir', type=str, required=True,
                        help='包含 images_all, lidar_all, info 的源数据目录路径。')
    parser.add_argument('--pose_file', type=str, default="aft_map_data.txt",
                        help='LeGO-LOAM生成的完整位姿文件路径 (例如: full_poses.txt)。')
    parser.add_argument('--sampling_rate', type=int, default=5,
                        help='采样率，即每N帧取一帧。')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='采样的起始帧编号（从0开始）。')
    parser.add_argument('--end_frame', type=int, default=-1,
                        help='采样的终止帧编号。使用-1表示直到最后一帧。')
    parser.add_argument('--only_pose', type=bool, default=True,
                        help='只过滤位姿态信息，图像已经进行降采样')
    
    args = parser.parse_args()
    
    sampler = UnifiedSampler(
        source_dir=args.source_dir,
        pose_file=args.pose_file,
        sampling_rate=args.sampling_rate,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        only_pose = args.only_pose
    )
    sampler.run()

if __name__ == "__main__":
    main()