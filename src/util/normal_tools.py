import cv2
import numpy as np
import json

kernel_Gx = np.array([[0, 0, 0],
                      [-1, 0, 1],
                      [0, 0, 0]])

kernel_Gy = np.array([[0, -1, 0],
                      [0, 0, 0],
                      [0, 1, 0]])

cp2tv_Gx = np.array([[0, 0, 0],
                     [0, -1, 1],
                     [0, 0, 0]])

cp2tv_Gy = np.array([[0, 0, 0],
                     [0, -1, 0],
                     [0, 1, 0]])

lap_ker_alpha = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]])

lap_ker_beta = np.array([[-1, -1, -1],
                         [-1, 8, -1],
                         [-1, -1, -1]])

lap_ker_gamma = np.array([[0.25, 0.5, 0.25],
                          [0.5, -3, 0.5],
                          [0.25, 0.5, 0.25]])

gradient_l = np.array([[-1, 1, 0]])
gradient_r = np.array([[0, -1, 1]])
gradient_u = np.array([[-1],
                       [1],
                       [0]])
gradient_d = np.array([[0],
                       [-1],
                       [1]])

laplace_hor = np.array([[-1, 2, -1]])

laplace_ver = np.array([[-1],
                        [2],
                        [-1]])


def vector_normalization(normal, eps=1e-8):
    mag = np.linalg.norm(normal, axis=2)
    normal /= (np.expand_dims(mag, axis=2) + eps)
    return normal

def soft_min(laplace_map, base, direction):
    """

    :param laplace_map: the horizontal laplace map or vertical laplace map, shape = [vMax, uMax]
    :param base: the base of the exponent operation
    :param direction: 0 for horizontal, 1 for vertical
    :return: weighted map (lambda 1,2 or 3,4)
    """
    h, w = laplace_map.shape
    eps = 1e-8  # to avoid division by zero

    lap_power = np.power(base, -laplace_map)
    if direction == 0:  # horizontal
        lap_pow_l = np.hstack([np.zeros((h, 1)), lap_power[:, :-1]])
        lap_pow_r = np.hstack([lap_power[:, 1:], np.zeros((h, 1))])
        return (lap_pow_l + eps * 0.5) / (eps + lap_pow_l + lap_pow_r), \
               (lap_pow_r + eps * 0.5) / (eps + lap_pow_l + lap_pow_r)

    elif direction == 1:  # vertical
        lap_pow_u = np.vstack([np.zeros((1, w)), lap_power[:-1, :]])
        lap_pow_d = np.vstack([lap_power[1:, :], np.zeros((1, w))])
        return (lap_pow_u + eps / 2) / (eps + lap_pow_u + lap_pow_d), \
               (lap_pow_d + eps / 2) / (eps + lap_pow_u + lap_pow_d)


def get_filter(Z, cp2tv=False):
    """get partial u, partial v"""
    if cp2tv:
        Gu = cv2.filter2D(Z, -1, cp2tv_Gx)
        Gv = cv2.filter2D(Z, -1, cp2tv_Gy)
    else:
        Gu = cv2.filter2D(Z, -1, kernel_Gx) / 2
        Gv = cv2.filter2D(Z, -1, kernel_Gy) / 2
    return Gu, Gv


def get_DAG_filter(Z, base=np.e, lap_conf='1D-DLF'):
    # calculate gradients along four directions
    grad_l = cv2.filter2D(Z, -1, gradient_l)
    grad_r = cv2.filter2D(Z, -1, gradient_r)
    grad_u = cv2.filter2D(Z, -1, gradient_u)
    grad_d = cv2.filter2D(Z, -1, gradient_d)

    # calculate laplace along 2 directions
    if lap_conf == '1D-DLF':
        lap_hor = abs(grad_l - grad_r)
        lap_ver = abs(grad_u - grad_d)
    elif lap_conf == 'DLF-alpha':
        lap_hor = abs(cv2.filter2D(Z, -1, lap_ker_alpha))
        lap_ver = abs(cv2.filter2D(Z, -1, lap_ker_alpha))
    elif lap_conf == 'DLF-beta':
        lap_hor = abs(cv2.filter2D(Z, -1, lap_ker_beta))
        lap_ver = abs(cv2.filter2D(Z, -1, lap_ker_beta))
    elif lap_conf == 'DLF-gamma':
        lap_hor = abs(cv2.filter2D(Z, -1, lap_ker_gamma))
        lap_ver = abs(cv2.filter2D(Z, -1, lap_ker_gamma))
    else:
        raise ValueError

    lambda_map1, lambda_map2 = soft_min(lap_hor, base, 0)
    lambda_map3, lambda_map4 = soft_min(lap_ver, base, 1)

    eps = 1e-8
    thresh = base
    lambda_map1[lambda_map1 / (lambda_map2 + eps) > thresh] = 1
    lambda_map2[lambda_map1 / (lambda_map2 + eps) > thresh] = 0
    lambda_map1[lambda_map2 / (lambda_map1 + eps) > thresh] = 0
    lambda_map2[lambda_map2 / (lambda_map1 + eps) > thresh] = 1

    lambda_map3[lambda_map3 / (lambda_map4 + eps) > thresh] = 1
    lambda_map4[lambda_map3 / (lambda_map4 + eps) > thresh] = 0
    lambda_map3[lambda_map4 / (lambda_map3 + eps) > thresh] = 0
    lambda_map4[lambda_map4 / (lambda_map3 + eps) > thresh] = 1

    # lambda_maps = [lambda_map1, lambda_map2, lambda_map3, lambda_map4]
    Gu = lambda_map1 * grad_l + lambda_map2 * grad_r
    Gv = lambda_map3 * grad_u + lambda_map4 * grad_d
    return Gu, Gv


def MRF_optim(depth, n_est, lap_conf='DLF-alpha'):
    h, w = depth.shape
    n_x, n_y, n_z = n_est[:, :, 0], n_est[:, :, 1], n_est[:, :, 2]
    # =====================optimize the normal with MRF=============================
    if lap_conf == '1D-DLF':
        Z_laplace_hor = abs(cv2.filter2D(depth, -1, laplace_hor))
        Z_laplace_ver = abs(cv2.filter2D(depth, -1, laplace_ver))

        # [x-1,y] [x+1,y] [x,y-1] [x,y+1], [x,y]
        Z_laplace_stack = np.array((np.hstack((np.inf * np.ones((h, 1)), Z_laplace_hor[:, :-1])),
                                    np.hstack((Z_laplace_hor[:, 1:], np.inf * np.ones((h, 1)))),
                                    np.vstack((np.inf * np.ones((1, w)), Z_laplace_ver[:-1, :])),
                                    np.vstack((Z_laplace_ver[1:, :], np.inf * np.ones((1, w)))),
                                    (Z_laplace_hor + Z_laplace_ver) / 2))
    else:
        if lap_conf == 'DLF-alpha':
            Z_laplace = abs(cv2.filter2D(depth, -1, lap_ker_alpha))
        elif lap_conf == 'DLF-beta':
            Z_laplace = abs(cv2.filter2D(depth, -1, lap_ker_beta))
        elif lap_conf == 'DLF-gamma':
            Z_laplace = abs(cv2.filter2D(depth, -1, lap_ker_gamma))
        else:
            raise ValueError
        Z_laplace_stack = np.array((np.hstack((np.inf * np.ones((h, 1)), Z_laplace[:, :-1])),
                                    np.hstack((Z_laplace[:, 1:], np.inf * np.ones((h, 1)))),
                                    np.vstack((np.inf * np.ones((1, w)), Z_laplace[:-1, :])),
                                    np.vstack((Z_laplace[1:, :], np.inf * np.ones((1, w)))),
                                    Z_laplace))

    # best_loc_map: 0 for left, 1 for right, 2 for up, 3 for down, 4 for self
    best_loc_map = np.argmin(Z_laplace_stack, axis=0)
    Nx_t_stack = np.array((np.hstack((np.zeros((h, 1)), n_x[:, :-1])),
                           np.hstack((n_x[:, 1:], np.zeros((h, 1)))),
                           np.vstack((np.zeros((1, w)), n_x[:-1, :])),
                           np.vstack((n_x[1:, :], np.zeros((1, w)))),
                           n_x)).reshape(5, -1)
    Ny_t_stack = np.array((np.hstack((np.zeros((h, 1)), n_y[:, :-1])),
                           np.hstack((n_y[:, 1:], np.zeros((h, 1)))),
                           np.vstack((np.zeros((1, w)), n_y[:-1, :])),
                           np.vstack((n_y[1:, :], np.zeros((1, w)))),
                           n_y)).reshape(5, -1)
    Nz_t_stack = np.array((np.hstack((np.zeros((h, 1)), n_z[:, :-1])),
                           np.hstack((n_z[:, 1:], np.zeros((h, 1)))),
                           np.vstack((np.zeros((1, w)), n_z[:-1, :])),
                           np.vstack((n_z[1:, :], np.zeros((1, w)))),
                           n_z)).reshape(5, -1)

    n_x = Nx_t_stack[best_loc_map.reshape(-1), np.arange(h * w)].reshape(h, w)
    n_y = Ny_t_stack[best_loc_map.reshape(-1), np.arange(h * w)].reshape(h, w)
    n_z = Nz_t_stack[best_loc_map.reshape(-1), np.arange(h * w)].reshape(h, w)
    n_est = cv2.merge((n_x, n_y, n_z))
    return n_est

def depth_to_normal(depth, intrinsic):
    h, w = depth.shape
    u_map = np.ones((h, 1)) * np.arange(1, w + 1) - intrinsic[0, 2]  # u-u0
    v_map = np.arange(1, h + 1).reshape(h, 1) * np.ones((1, w)) - intrinsic[1, 2]  # v-v0
    
    Gu, Gv = get_filter(depth)
    est_nx = Gu * intrinsic[0, 0]
    est_ny = Gv * intrinsic[1, 1]
    est_nz = -(depth + v_map * Gv + u_map * Gu)
    est_nx = est_nx.astype(np.float32)
    est_ny = est_ny.astype(np.float32)
    est_nz = est_nz.astype(np.float32)
    
    est_normal = cv2.merge((est_nx, est_ny, est_nz))
    
    est_normal = vector_normalization(est_normal)
    
    # est_normal = MRF_optim(depth, est_normal)
    return ((est_normal + 1) * 0.5 * 255).astype(np.uint8)

def inv_depth_to_normal(inv_depth, intrinsic, smooth_ksize: int = 0):
    """
    从无尺度的单目逆深度 ρ=1/Z 直接计算法向量（尺度不敏感）。

    公式（与 depth_to_normal 等价但避免 1/ρ 数值不稳）：
      n ∝ [ fx * ∂ρ/∂u,  fy * ∂ρ/∂v,  ρ - (u-cx)∂ρ/∂u - (v-cy)∂ρ/∂v ]

    Args:
        inv_depth (np.ndarray HxW): 单目逆深度（可任意尺度）。
        intrinsic (np.ndarray 3x3): 相机内参。
        smooth_ksize (int): 可选的高斯平滑核（奇数，>0 时启用），降低噪声。

    Returns:
        np.ndarray HxWx3 uint8: 可视化法向量。
    """
    inv_depth = inv_depth.astype(np.float32)
    # 可选平滑以抑制高频噪声（地下车库低纹理区域常见）
    if smooth_ksize and smooth_ksize >= 3 and smooth_ksize % 2 == 1:
        inv_depth = cv2.GaussianBlur(inv_depth, (smooth_ksize, smooth_ksize), 0)

    h, w = inv_depth.shape
    u_map = np.ones((h, 1), dtype=np.float32) * np.arange(1, w + 1, dtype=np.float32) - intrinsic[0, 2]
    v_map = np.arange(1, h + 1, dtype=np.float32).reshape(h, 1) * np.ones((1, w), dtype=np.float32) - intrinsic[1, 2]

    # 对 ρ 求偏导
    Gu, Gv = get_filter(inv_depth)

    est_nx = intrinsic[0, 0] * Gu
    est_ny = intrinsic[1, 1] * Gv
    est_nz = inv_depth - (u_map * Gu + v_map * Gv)

    est_nx = est_nx.astype(np.float32)
    est_ny = est_ny.astype(np.float32)
    est_nz = est_nz.astype(np.float32)

    # 为与 depth_to_normal 的方向一致，需要整体取反（两者相差一个负的标量因子）
    est_normal = -cv2.merge((est_nx, est_ny, est_nz))
    est_normal = vector_normalization(est_normal)
    return ((est_normal + 1) * 0.5 * 255).astype(np.uint8)

def inv_depth_to_pseudo_depth(inv_depth: np.ndarray,
                              method: str = 'convex',
                              epsilon: float = None,
                              max_depth: float = None,
                              a: float = 0.1) -> np.ndarray:
    """
    将逆深度 ρ 映射为数值稳定、可保存/可视化的“伪深度”。

    注意：单目逆深度无绝对尺度，任何映射只保证单调性与数值稳定；
         若后续需要“度量深度”，需额外做尺度对齐（例如用已知物理尺寸、LiDAR 对齐等）。

    方法：
      - method='reciprocal': depth = 1 / max(ρ, epsilon)。避免 ρ→0 时爆炸，epsilon 可用分位数自适应。
      - method='convex'    : depth = 1 / (a + (1-a)*ρ)。单调“压缩”大深度，最大深度≈1/a；你提到的 1/(0.1+0.9*1/z)
                             在 ρ=1/z 的情形下等价于此（a=0.1）。

    参数：
      - inv_depth: ρ（HxW, float32）
      - epsilon: reciprocal 模式下用来下限裁剪 ρ 的阈值；None 时取 max(1e-6, P1(ρ))
      - max_depth: 额外限制最大深度（可选）；当提供时，会将 depth = min(depth, max_depth)
      - a: convex 模式的常数 a∈(0,1)，a 越大，最大深度 1/a 越小、压缩越强。

    返回：depth（HxW, float32）
    """
    ρ = inv_depth.astype(np.float32)
    if method == 'reciprocal':
        if epsilon is None:
            # 用 1% 分位数抑制极小值，再与 1e-6 取较大者
            eps_auto = float(np.percentile(ρ[~np.isnan(ρ)], 1)) if np.isfinite(ρ).any() else 1e-6
            epsilon = max(1e-6, eps_auto)
        depth = 1.0 / np.maximum(ρ, epsilon)
    elif method == 'convex':
        a = float(np.clip(a, 1e-6, 0.999999))
        depth = 1.0 / (a + (1.0 - a) * ρ)
    else:
        raise ValueError(f"Unknown method: {method}")

    if max_depth is not None and np.isfinite(max_depth):
        depth = np.minimum(depth, float(max_depth))
    return depth.astype(np.float32)

def inv_depth_to_normal_and_confidence(inv_depth: np.ndarray,
                                       intrinsic: np.ndarray,
                                       smooth_ksize: int = 0):
    """
    从逆深度计算：
      - 单位法向 n_unit（HxWx3, float32，模长=1）
      - 置信度 conf（HxW, float32 ∈[0,1]），基于梯度强度的稳健分位归一化

    conf 反映局部几何信号强度（梯度越强、越可靠），不会随单位化丢失。
    """
    inv_depth = inv_depth.astype(np.float32)
    if smooth_ksize and smooth_ksize >= 3 and smooth_ksize % 2 == 1:
        inv_depth = cv2.GaussianBlur(inv_depth, (smooth_ksize, smooth_ksize), 0)

    h, w = inv_depth.shape
    u_map = np.ones((h, 1), dtype=np.float32) * np.arange(1, w + 1, dtype=np.float32) - intrinsic[0, 2]
    v_map = np.arange(1, h + 1, dtype=np.float32).reshape(h, 1) * np.ones((1, w), dtype=np.float32) - intrinsic[1, 2]

    Gu, Gv = get_filter(inv_depth)
    nx = intrinsic[0, 0] * Gu
    ny = intrinsic[1, 1] * Gv
    nz = inv_depth - (u_map * Gu + v_map * Gv)

    # 符号对齐到与 depth_to_normal 一致
    nx, ny, nz = -nx, -ny, -nz

    vec = cv2.merge((nx.astype(np.float32), ny.astype(np.float32), nz.astype(np.float32)))
    mag = np.linalg.norm(vec, axis=2)
    n_unit = vec / (mag[..., None] + 1e-8)

    # 以梯度强度生成置信度，并用稳健分位归一化到 [0,1]
    g = np.sqrt((intrinsic[0, 0] * Gu) ** 2 + (intrinsic[1, 1] * Gv) ** 2).astype(np.float32)
    valid = np.isfinite(g)
    if valid.any():
        p5 = float(np.percentile(g[valid], 5))
        p95 = float(np.percentile(g[valid], 95))
    else:
        p5, p95 = 0.0, 1.0
    conf = (g - p5) / (max(1e-6, p95 - p5))
    conf = np.clip(conf, 0.0, 1.0)

    # 低值/无效逆深度区域置信度降为0
    conf[~np.isfinite(inv_depth) | (inv_depth <= 0)] = 0.0
    return n_unit.astype(np.float32), conf.astype(np.float32)

def normal_to_uint8(n_unit: np.ndarray) -> np.ndarray:
    """将单位法向[-1,1]映射到可视化的uint8 RGB。"""
    return ((np.clip(n_unit, -1.0, 1.0) + 1.0) * 0.5 * 255.0).astype(np.uint8)

def load_camera_params(json_path) :
        """加载相机参数（内参和外参）
        
        Args:
            json_path: 相机参数json
            
        Returns:
            tuple[np.ndarray, np.ndarray]: 
                - extrinsic: [4, 4] 相机外参矩阵
                - intrinsic: [3, 3] 相机内参矩阵
        """
        
        # 读取JSON文件
        with open(json_path, 'r') as f:
            params = json.load(f)
        
        # 解析外参矩阵
        extrinsic = np.array(params["extrinsic"]).reshape(4, 4)
        
        # 解析内参矩阵
        intrinsic = np.array(params["intrinsic"]).reshape(3, 3)
        
        # 解析畸变系数
        distortion = np.array(params["distortion"])
        
        return extrinsic, intrinsic, distortion