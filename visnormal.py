import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

def analyze_normal_map(image_path: str):
    """
    读取并分析法向量PNG图，输出其法向量模长的统计数据和分布直方图。

    Args:
        image_path (str): 法向量贴图文件的路径。
    """
    try:
        # --- 1. 读取图像并转换为NumPy数组 ---
        print(f"正在读取图像: {image_path}...")
        img = Image.open(image_path).convert('RGB')
        # 将图像数据转换为浮点数类型的NumPy数组，方便计算
        pixel_array = np.array(img, dtype=np.float32)
        
        # --- 2. 将像素值 [0, 255] 映射到向量分量 [-1, 1] ---
        # 使用向量化操作，效率远高于逐像素循环
        print("正在将像素值重新映射到向量空间 [-1, 1]...")
        vectors = (pixel_array / 255.0) * 2.0 - 1.0
        
        # --- 3. 计算每个向量的模长 ---
        # np.linalg.norm沿最后一个轴(axis=2, 即RGB通道)计算欧几里得范数
        print("正在计算所有法向量的模长...")
        magnitudes = np.linalg.norm(vectors, axis=2)
        
        # --- 4. 统计分析模长分布 ---
        print("\n--- 模长统计分析结果 ---")
        max_mag = np.max(magnitudes)
        min_mag = np.min(magnitudes)
        mean_mag = np.mean(magnitudes)
        median_mag = np.median(magnitudes)
        std_dev = np.std(magnitudes)
        
        print(f"最大模长 (Max Magnitude):    {max_mag:.6f}")
        print(f"最小模长 (Min Magnitude):    {min_mag:.6f}")
        print(f"平均模长 (Mean Magnitude):    {mean_mag:.6f}")
        print(f"模长中位数 (Median Magnitude): {median_mag:.6f}")
        print(f"模长标准差 (Std Deviation):  {std_dev:.6f}")
        
        # 检查有多少比例的向量是近似单位向量 (例如，模长在 0.99 到 1.01 之间)
        unit_vectors_ratio = np.mean((magnitudes >= 0.99) & (magnitudes <= 1.01))
        print(f"近似单位向量的比例 (模长在 [0.99, 1.01] 内): {unit_vectors_ratio:.2%}")

        # --- 5. 可视化模长分布 ---
        print("\n正在生成模长分布直方图...")
        plt.style.use('ggplot')
        plt.figure(figsize=(12, 7))
        
        # 使用 flatten() 将2D的模长数组转换为1D数组以便绘制直方图
        # bins 参数可以调整直方图的精细度
        plt.hist(magnitudes.flatten(), bins=200, density=True, alpha=0.75, label='实际分布')
        
        plt.title('法向量模长分布直方图 (Normal Vector Magnitude Distribution)', fontsize=16)
        plt.xlabel('法向量模长 (Magnitude)', fontsize=12)
        plt.ylabel('频率密度 (Density)', fontsize=12)
        
        # 在图上标记出关键统计数据
        plt.axvline(mean_mag, color='r', linestyle='dashed', linewidth=2, label=f'平均值: {mean_mag:.4f}')
        plt.axvline(1.0, color='g', linestyle='solid', linewidth=2, label='理想值 (1.0)')
        
        plt.legend()
        plt.grid(True)
        plt.xlim(min_mag - 0.01, max_mag + 0.01) # 设置一个合适的x轴范围
        plt.show()

    except FileNotFoundError:
        print(f"错误: 文件未找到 '{image_path}'", file=sys.stderr)
    except Exception as e:
        print(f"处理图像时发生错误: {e}", file=sys.stderr)

if __name__ == '__main__':
    # --- 使用说明 ---
    # 1. 将你的法向量贴图文件路径替换下面的字符串
    # 2. 确保文件与此脚本在同一目录，或者提供完整路径
    # 例如: "C:/Users/YourUser/Desktop/my_normal_map.png"
    target_image_path = '/home/hqlab/workspace/dataset/parkinglot/data/20000/video_normal/camera_FRONT/000047.png' # <--- 修改这里!

    if target_image_path == 'path/to/your/normal_map.png':
        print("请在脚本中修改 'target_image_path' 变量，指向你的法向量PNG文件。")
    else:
        analyze_normal_map(target_image_path)