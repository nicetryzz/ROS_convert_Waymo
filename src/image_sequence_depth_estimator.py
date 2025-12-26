import cv2
import glob
import matplotlib
import numpy as np
import json
import os
import torch
from pathlib import Path
import imageio.v2 as imageio
import argparse
import natsort # 用于自然排序
from typing import List, Tuple, Optional # Added Tuple
import sys
sys.path.append("/home/hqlab/workspace/depth_estimation/Video-Depth-Anything")

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video
from util import depth_to_normal, load_camera_params, inv_depth_to_normal, inv_depth_to_pseudo_depth, inv_depth_to_normal_and_confidence, normal_to_uint8

class ImageSequenceDepthGenerator:
    """
    从多个相机子目录的图像序列生成深度视频的类。
    """
    # 相机目录名称列表
    CAMERA_DIRS = [
        "camera_FRONT", "camera_FRONT_RIGHT", "camera_BACK_RIGHT",
        "camera_BACK", "camera_BACK_LEFT", "camera_FRONT_LEFT"
    ]

    def __init__(self, base_input_dir: str, base_output_dir: str, encoder: str = 'vitl',
                 input_size: int = 518, max_res: int = 1280, target_fps: int = 24,
                 max_frames_per_camera: int = -1,
                 fp32: bool = False, grayscale: bool = False,
                 save_npz: bool = False, save_exr: bool = False,
                 save_input_frames_video: bool = False,
                 model_checkpoint_path: Optional[str] = None):
        """
        初始化图像序列深度生成器。

        Args:
            base_input_dir (str): 包含相机子目录 (例如 "camera_FRONT") 的基础输入文件夹路径。
            base_output_dir (str): 保存输出文件的基础目录。输出将按相机保存在子目录中。
            encoder (str): 模型编码器类型 ('vits' 或 'vitl')。
            input_size (int): 模型推理的输入图像大小。
            max_res (int): 读取图像时的最大分辨率，超过则缩放。
            target_fps (int): 输出视频的帧率。
            max_frames_per_camera (int): 每个相机序列要处理的最大图像数量。-1表示处理所有。
            fp32 (bool): 是否使用 FP32 进行模型推理。
            grayscale (bool): 是否将深度视频保存为灰度。
            save_npz (bool): 是否将深度数据保存为 .npz 文件。
            save_exr (bool): 是否将深度数据保存为 .exr 文件序列。
            save_input_frames_video (bool): 是否将输入的图像序列也保存为一个视频。
            model_checkpoint_path (Optional[str]): VideoDepthAnything 模型的权重文件路径。
        """
        self.base_input_dir = Path(base_input_dir)
        self.base_output_dir = Path(base_output_dir)
        self.encoder = encoder
        self.input_size = input_size
        self.max_res = max_res
        self.target_fps = float(target_fps)
        self.max_frames_per_camera = max_frames_per_camera
        self.fp32 = fp32
        self.grayscale = grayscale
        self.save_npz = save_npz
        self.save_exr = save_exr
        self.save_input_frames_video = save_input_frames_video

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")

        # 创建基础输出目录 (如果不存在)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化模型 (与之前相同)
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        if encoder not in model_configs:
            raise ValueError(f"不支持的编码器: {encoder}. 可选项: {list(model_configs.keys())}")
        self.model = VideoDepthAnything(**model_configs[encoder])
        if model_checkpoint_path is None:
            model_checkpoint_path = f'./checkpoints/video_depth_anything_{encoder}.pth'
        if not Path(model_checkpoint_path).exists():
            print(f"警告：模型权重文件未找到 '{model_checkpoint_path}'。将使用骨架模型（如果适用）。")
        try:
            self.model.load_state_dict(torch.load(model_checkpoint_path, map_location='cpu'), strict=True)
        except FileNotFoundError:
            print(f"错误：模型权重文件 '{model_checkpoint_path}' 未找到。")
        except RuntimeError as e:
            print(f"加载模型权重时出错: {e}.")
        self.model = self.model.to(self.device).eval()

    def _read_and_prepare_frames_for_camera(self, current_camera_input_dir: Path) -> Tuple[Optional[np.ndarray], List[str], float]:
        """
        从特定相机的文件夹读取、排序和预处理图像帧。
        返回:
            - 一个包含所有帧的NumPy数组 (N, H, W, C) 或 None。
            - 处理后的原始图片文件名列表 (用于保存npz)。
            - 目标FPS。
        """
        image_files_paths_obj = []
        supported_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        for ext in supported_extensions:
            image_files_paths_obj.extend(list(current_camera_input_dir.glob(ext)))

        if not image_files_paths_obj:
            print(f"在 '{current_camera_input_dir}' 中没有找到支持的图片文件。")
            return None, [], self.target_fps

        # 使用Path对象的stem进行排序，保留Path对象列表
        sorted_image_files_paths_obj = natsort.natsorted(image_files_paths_obj, key=lambda p: p.stem)
        
        original_filenames_processed = [] # 存储将要处理的原始文件名 (不含扩展名)

        if self.max_frames_per_camera > 0 and len(sorted_image_files_paths_obj) > self.max_frames_per_camera:
            print(f"由于 max_frames_per_camera={self.max_frames_per_camera} 的限制，将处理前 {self.max_frames_per_camera} 张图片。")
            sorted_image_files_paths_obj = sorted_image_files_paths_obj[:self.max_frames_per_camera]
        
        num_files_to_process = len(sorted_image_files_paths_obj)
        print(f"找到并排序了 {num_files_to_process} 张图片。正在加载...")

        frames_list = []
        for i, f_path_obj in enumerate(sorted_image_files_paths_obj):
            print(f"\r加载图片: {i+1}/{num_files_to_process} ({f_path_obj.name})", end="")
            try:
                frame = cv2.imread(str(f_path_obj))
                if frame is None:
                    print(f"\n警告：无法读取图片 {f_path_obj}，跳过。")
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]
                if self.max_res > 0 and max(h, w) > self.max_res:
                    scale = self.max_res / max(h, w)
                    new_h, new_w = int(round(h * scale)), int(round(w * scale))
                    if new_h > 0 and new_w > 0:
                        frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    else:
                        print(f"\n警告: 缩放后尺寸无效 ({new_w}x{new_h}) for {f_path_obj}。跳过缩放。")
                frames_list.append(frame_rgb)
                original_filenames_processed.append(f_path_obj.stem) # 保存不带扩展名的文件名
            except Exception as e:
                print(f"\n加载或处理图片 {f_path_obj} 时出错: {e}")
        print("\n图片加载完成。")

        if not frames_list:
            print("未能成功加载任何图片帧。")
            return None, [], self.target_fps
        try:
            frames_array = np.stack(frames_list, axis=0)
        except ValueError as e:
            print(f"\n错误：无法将帧堆叠成数组。所有图片必须具有相同的尺寸。错误信息: {e}")
            return None, [], self.target_fps
        return frames_array, original_filenames_processed, self.target_fps

    def process_all_camera_sequences(self):
        """
        处理 CAMERA_DIRS 中定义的每个相机子目录的图像序列。
        """
        for camera_name in self.CAMERA_DIRS:
            current_camera_input_dir = self.base_input_dir / camera_name
            current_camera_output_dir = self.base_output_dir / camera_name

            print(f"\n{'='*20} 开始处理相机: {camera_name} {'='*20}")
            print(f"输入目录: {current_camera_input_dir}")
            print(f"输出目录: {current_camera_output_dir}")

            if not current_camera_input_dir.is_dir():
                print(f"警告：相机输入目录 '{current_camera_input_dir}' 不存在或不是一个目录，跳过此相机。")
                continue

            # 为此相机创建特定的输出目录
            current_camera_output_dir.mkdir(parents=True, exist_ok=True)

            frames_array, original_filenames, effective_fps = self._read_and_prepare_frames_for_camera(current_camera_input_dir) # Get original filenames
            
            if frames_array is None or frames_array.shape[0] == 0:
                print(f"相机 '{camera_name}' 没有帧可处理。继续下一个相机。")
                continue

            print(f"开始对相机 '{camera_name}' ({frames_array.shape[0]} 帧) 进行深度推断...")
            depths_output_array, output_fps = self.model.infer_video_depth(
                frames_array, effective_fps, input_size=self.input_size, device=self.device, fp32=self.fp32
            )
            print(f"相机 '{camera_name}' 深度推断完成。")

            if depths_output_array is None or depths_output_array.shape[0] == 0 or depths_output_array.shape[0] != len(original_filenames):
                print(f"相机 '{camera_name}' 的深度推断结果与输入帧数不匹配或无效。")
                continue
            
            # 使用相机名称和固定序列名称作为输出文件的基础名称
            output_base_name = f"{camera_name}_sequence"

            depth_vis_video_path = current_camera_output_dir / f"{output_base_name}_depth_visualized.mp4"
            print(f"准备保存 '{camera_name}' 的深度可视化视频到: {depth_vis_video_path}")
            save_video(depths_output_array, str(depth_vis_video_path), fps=output_fps, 
                       is_depths=True, grayscale=self.grayscale)

            if self.save_input_frames_video:
                input_frames_video_path = current_camera_output_dir / f"{output_base_name}_input_frames.mp4"
                print(f"准备保存 '{camera_name}' 的输入帧视频到: {input_frames_video_path}")
                save_video(frames_array, str(input_frames_video_path), fps=output_fps, is_depths=False)

            if self.save_npz:
                # Create a subdirectory for individual NPZ files for this camera
                npz_output_subdir = os.path.join(os.path.dirname(self.base_output_dir) , "video_depth" , camera_name)
                npz_output_subdir = Path(npz_output_subdir)
                npz_output_subdir.mkdir(parents=True, exist_ok=True)
                normal_output_subdir = os.path.join(os.path.dirname(self.base_output_dir) , "video_normal" , camera_name)
                normal_output_subdir = Path(normal_output_subdir)
                normal_output_subdir.mkdir(parents=True, exist_ok=True)
                json_path = os.path.join( "/home/hqlab/workspace/dataset/parkinglot", "extrinsic" , "camera" , f"{camera_name}.json")
                print(f"准备将 '{camera_name}' 的逐帧深度数据保存到目录: {npz_output_subdir}")
                
                extrinsic, intrinsic, distortion = load_camera_params(json_path)
                
                for i in range(depths_output_array.shape[0]):
                    original_frame_stem = original_filenames[i] # This is already stem (name without ext)
                    depth_frame_data = depths_output_array[i, :, :] # Single depth map (H, W)
                    # 模型输出的是单目无尺度深度的“倒数”提示：即本身就是逆深度 ρ
                    inv_depth = depth_frame_data.astype(np.float32)
                    # 在地下车库等低纹理、光照复杂环境下，使用单位法向+置信度
                    n_unit, n_conf = inv_depth_to_normal_and_confidence(inv_depth, intrinsic, smooth_ksize=5)
                    normal_frame_data = normal_to_uint8(n_unit)
                    # 同时导出数值稳定的“伪深度”，便于下游使用/可视化
                    pseudo_depth = inv_depth_to_pseudo_depth(
                        inv_depth,
                        method='convex',  # 单调压缩，避免极远处爆炸
                        a=0.1             # 你提到的 1/(0.1 + 0.9*ρ) 等价形式
                    )
                    
                    # Save with original filename (stem) + .npz
                    individual_npz_path = npz_output_subdir / f"{original_frame_stem}.npz"
                    individual_normal_path = normal_output_subdir / f"{original_frame_stem}.png"
                    
                    try:
                        # 保存：逆深度、伪深度、单位法向、法向置信度，以及可视化法向PNG
                        np.savez_compressed(
                            individual_npz_path,
                            inv_depth=inv_depth.astype(np.float32),
                            pseudo_depth=pseudo_depth.astype(np.float32),
                            normal_unit=n_unit.astype(np.float32),
                            normal_confidence=n_conf.astype(np.float32)
                        )
                        imageio.imwrite(individual_normal_path, normal_frame_data)
                        print(f"\r为 '{camera_name}' 已保存NPZ: {i+1}/{len(original_filenames)} ({individual_npz_path.name})", end="")
                    except Exception as e:
                        print(f"\n为 '{camera_name}' 保存 .npz 文件 '{individual_npz_path}' 时出错: {e}")
                print(f"\n'{camera_name}' 的所有NPZ文件保存完成。")


            if self.save_exr:
                depth_exr_dir = current_camera_output_dir / f"{output_base_name}_depths_exr"
                depth_exr_dir.mkdir(parents=True, exist_ok=True)
                print(f"准备将 '{camera_name}' 的EXR格式深度帧保存到: {depth_exr_dir}")
                try:
                    import OpenEXR
                    import Imath
                    for i in range(depths_output_array.shape[0]):
                        depth_frame_float32 = depths_output_array[i, :, :]
                        if not isinstance(depth_frame_float32, np.ndarray) or depth_frame_float32.dtype != np.float32:
                            depth_frame_float32 = np.array(depth_frame_float32, dtype=np.float32)
                        if depth_frame_float32.ndim != 2:
                             print(f"警告：相机 '{camera_name}' 的EXR深度帧 {i} 格式不正确，跳过。")
                             continue
                        output_exr_path = depth_exr_dir / f"frame_{i:05d}.exr"
                        header = OpenEXR.Header(depth_frame_float32.shape[1], depth_frame_float32.shape[0])
                        header["channels"] = {"Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
                        exr_file = OpenEXR.OutputFile(str(output_exr_path), header)
                        exr_file.writePixels({"Z": depth_frame_float32.tobytes()})
                        exr_file.close()
                        print(f"\r为 '{camera_name}' 已保存EXR帧: {i+1}/{depths_output_array.shape[0]}", end="")
                    print(f"\n'{camera_name}' 的EXR帧保存完成。")
                except ImportError:
                    print(f"错误 (相机 {camera_name}): 保存 .exr 文件需要 OpenEXR-python 包。")
                except Exception as e:
                    print(f"为 '{camera_name}' 保存 .exr 文件时出错: {e}")
            
            print(f"--- 相机: {camera_name} 处理完成 ---")
        print(f"\n{'='*20} 所有相机处理完成 {'='*20}")
        print(f"所有输出文件已保存在基础目录: {self.base_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="从多个相机子目录的图像序列生成深度视频。")
    parser.add_argument('--base_input_dir', type=str, required=True,
                        help="包含相机子目录 (例如 'camera_FRONT') 的基础输入文件夹路径。")
    parser.add_argument('--base_output_dir', type=str, required=True,
                        help="保存输出文件的基础目录。输出将按相机保存在子目录中。")
    # 其他参数与之前版本类似，但描述可能需要微调
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'], help="模型编码器类型。")
    parser.add_argument('--model_checkpoint', type=str, default="/home/hqlab/workspace/depth_estimation/Video-Depth-Anything/checkpoints/video_depth_anything_vitl.pth", help="模型权重文件路径。")
    parser.add_argument('--input_size', type=int, default=518, help="模型输入图像大小。")
    parser.add_argument('--max_res', type=int, default=2048, help="图像最大分辨率。")
    parser.add_argument('--target_fps', type=int, default=24, help="输出视频帧率。")
    parser.add_argument('--max_frames_per_camera', type=int, default=-1, help="每个相机序列处理的最大图像数 (-1 无限制)。")
    parser.add_argument('--fp32', action='store_true', help="使用 FP32 推理。")
    parser.add_argument('--grayscale', action='store_true', help="深度视频使用灰度。")
    parser.add_argument('--save_npz', action='store_true', help="保存深度为 .npz。")
    parser.add_argument('--save_exr', action='store_true', help="保存深度为 .exr 序列。")
    parser.add_argument('--save_input_frames_video', action='store_true', help="保存输入帧视频。")

    args = parser.parse_args()

    # (依赖检查与之前版本相同)
    if args.save_exr:
        try: import OpenEXR
        except ImportError: print("错误: --save_exr 需要 OpenEXR-python。请运行 'pip install OpenEXR'。"); return
    try: import natsort
    except ImportError: print("错误: 此脚本需要 'natsort'。请运行 'pip install natsort'。"); return
        
    generator = ImageSequenceDepthGenerator(
        base_input_dir=args.base_input_dir,
        base_output_dir=args.base_output_dir,
        encoder=args.encoder,
        input_size=args.input_size,
        max_res=args.max_res,
        target_fps=args.target_fps,
        max_frames_per_camera=args.max_frames_per_camera,
        fp32=args.fp32,
        grayscale=args.grayscale,
        save_npz=args.save_npz,
        save_exr=args.save_exr,
        save_input_frames_video=args.save_input_frames_video,
        model_checkpoint_path=args.model_checkpoint
    )
    generator.process_all_camera_sequences()

if __name__ == "__main__":
    main()