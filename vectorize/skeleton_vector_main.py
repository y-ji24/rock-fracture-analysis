"""
================================================================================
 岩石裂隙二值mask图片骨架化与迹线矢量化
================================================================================
 参考论文：
   - Chen J. 2022, JRMGE (Journal of Rock Mechanics and Geotechnical Engineering)
   - Qiu Y. 2025 处理流程

 处理流程：
   1. 读取二值mask → 形态学开运算去噪
   2. Zhang-Suen骨架化 (skimage.morphology.skeletonize)
   3. 节点检测：端点 / 分支交点 / 普通中间点
   4. 端点追踪 → 获取独立裂缝迹线像素序列
   5. Ramer-Douglas-Peucker 多段线矢量化简化
   6. 可视化：骨架 + 红色矢量化多段线叠加
   7. JSON 输出裂缝迹线信息
   8. 批量处理 input_mask 下全部 PNG 图片
   9. 控制台打印统计信息
  10. 异常捕获与警告
  11. 新手友好注释，变量名通俗易懂
  12. 可配置参数置顶

 依赖：numpy, scikit-image, opencv-python
 安装：pip install numpy scikit-image opencv-python
================================================================================
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import cv2
from skimage.morphology import skeletonize, binary_opening, disk
from skimage import io, img_as_ubyte

# ============================================================
#  ★ 可配置参数（放在最上方，方便调整）
# ============================================================

# RDP (Ramer–Douglas–Peucker) 简化阈值，单位：像素
# 对应论文中的 Dmax —— 值越大，简化越激进，多段线顶点越少
DP_EPSILON = 2.0

# 最少迹线像素点数：像素点少于该值的碎片迹线将被丢弃
MIN_TRACE_PIXELS = 5

# 形态学开运算的圆形结构元素半径，单位：像素
# 用于消除细小噪声毛刺；值越大去噪越强，但也可能损伤有效裂缝
MORPH_KERNEL_SIZE = 2

# ============================================================
#  ★ 路径设置（全部使用相对路径，基于脚本所在目录）
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input_mask"
OUTPUT_SKELETON_DIR = PROJECT_ROOT / "output_skeleton"
OUTPUT_VECTOR_IMG_DIR = PROJECT_ROOT / "output_vector_img"
OUTPUT_VECTOR_JSON_DIR = PROJECT_ROOT / "output_vector_json"
TEMP_DEBUG_DIR = PROJECT_ROOT / "temp_debug"

# 确保所有输出目录存在
for directory in [OUTPUT_SKELETON_DIR, OUTPUT_VECTOR_IMG_DIR,
                  OUTPUT_VECTOR_JSON_DIR, TEMP_DEBUG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
#  工具函数
# ============================================================

def load_grayscale_mask(image_path):
    """
    读取PNG图片并转换为二值灰度图。
    白色像素（值>127）视为裂缝前景，黑色为背景。

    参数:
        image_path: PNG文件路径 (Path 或 str)

    返回:
        binary_mask: 二值numpy数组 (dtype=bool)，True=裂缝
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    binary_mask = image > 127
    return binary_mask


def apply_morphological_opening(binary_mask, kernel_radius):
    """
    形态学开运算：先腐蚀后膨胀，用于消除细小噪声毛刺。
    使用圆形结构元素（disk），保持裂缝各向同性。

    参数:
        binary_mask: 输入二值图像 (bool)
        kernel_radius: 结构元素半径（像素）

    返回:
        cleaned_mask: 开运算后的二值图像 (bool)
    """
    if kernel_radius <= 0:
        return binary_mask
    structuring_element = disk(kernel_radius)
    cleaned_mask = binary_opening(binary_mask, structuring_element)
    return cleaned_mask


def extract_skeleton(binary_mask):
    """
    使用 Zhang-Suen 算法提取单像素宽度骨架。
    调用 skimage.morphology.skeletonize。

    参数:
        binary_mask: 二值图像 (bool)

    返回:
        skeleton: 二值骨架图像 (bool)
    """
    skeleton = skeletonize(binary_mask)
    return skeleton


def get_skeleton_pixel_coords(skeleton):
    """
    获取骨架图中所有前景像素的坐标列表。

    参数:
        skeleton: 二值骨架图像 (bool)

    返回:
        pixels: [(row, col), ...] 列表
    """
    rows, cols = np.where(skeleton)
    return list(zip(rows.tolist(), cols.tolist()))


def count_8_neighbors(skeleton, row, col):
    """
    计算像素 (row, col) 在骨架图中的 8-邻域前景像素数量。
    8-邻域：上、下、左、右 + 四个对角方向。

    这是论文中链码节点检测的基础：
      邻域数 = 1 → 端点
      邻域数 = 2 → 普通中间点
      邻域数 ≥ 3 → 分支交点
    """
    height, width = skeleton.shape
    count = 0
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width:
                if skeleton[nr, nc]:
                    count += 1
    return count


def get_8_neighbor_list(skeleton, row, col):
    """
    获取像素 (row, col) 在骨架图中所有 8-邻域前景像素的坐标列表。

    返回:
        neighbors: [(row, col), ...] 列表
    """
    height, width = skeleton.shape
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width:
                if skeleton[nr, nc]:
                    neighbors.append((nr, nc))
    return neighbors


def classify_skeleton_nodes(skeleton):
    """
    遍历骨架图所有像素，根据 8-邻域前景像素数将每个像素分类为：
      - 'endpoint'  (端点)：邻域数 = 1
      - 'ordinary'  (普通中间点)：邻域数 = 2
      - 'junction'  (分支交点)：邻域数 ≥ 3
      邻域数 = 0 的孤立像素也归类为端点（罕见情况）。

    参数:
        skeleton: 二值骨架图像 (bool)

    返回:
        node_types:  字典 {(row, col): 'endpoint'|'ordinary'|'junction'}
        endpoints:   [(row, col), ...] 端点坐标列表
        junctions:   [(row, col), ...] 分支交点坐标列表
    """
    node_types = {}
    endpoints = []
    junctions = []

    pixel_coords = get_skeleton_pixel_coords(skeleton)

    for row, col in pixel_coords:
        neighbor_count = count_8_neighbors(skeleton, row, col)

        if neighbor_count <= 1:
            node_types[(row, col)] = 'endpoint'
            endpoints.append((row, col))
        elif neighbor_count == 2:
            node_types[(row, col)] = 'ordinary'
        else:
            node_types[(row, col)] = 'junction'
            junctions.append((row, col))

    return node_types, endpoints, junctions


def trace_from_endpoint(start_pixel, skeleton, node_types, visited_non_junction):
    """
    从端点出发，沿骨架追踪连通像素，直到遇到分支交点或另一端点。

    追踪逻辑（参考论文链码追踪）：
      1. 从端点开始，沿唯一邻居方向前进
      2. 每步选择唯一的未访问普通邻居
      3. 遇到分支交点 → 停止（交点被多条迹线共享）
      4. 遇到另一端点 → 停止（完整裂缝段）

    参数:
        start_pixel: (row, col) 起始端点坐标
        skeleton:    二值骨架图像
        node_types:  像素类型字典
        visited_non_junction: 已访问的非交点像素集合（会被原地修改）

    返回:
        trace: [(row, col), ...] 迹线像素坐标序列
    """
    trace = [start_pixel]
    visited_non_junction.add(start_pixel)
    current = start_pixel

    while True:
        all_neighbors = get_8_neighbor_list(skeleton, current[0], current[1])

        # 过滤出既未访问过、也不是交点的邻居
        unvisited_ordinary = []
        for nb in all_neighbors:
            if nb in visited_non_junction:
                continue
            if node_types.get(nb) == 'junction':
                continue
            unvisited_ordinary.append(nb)

        # 如果当前像素是交点且不是起点，则停止追踪
        if node_types.get(current) == 'junction' and current != start_pixel:
            break

        if not unvisited_ordinary:
            # 没有可用的普通邻居，检查是否有交点邻居
            junction_neighbors = [nb for nb in all_neighbors
                                  if node_types.get(nb) == 'junction']
            if junction_neighbors:
                # 把交点作为迹线的终点（但不标记为已访问，方便其他迹线共享）
                trace.append(junction_neighbors[0])
            break

        # 选择第一个（也是唯一一个）未访问普通邻居前进
        next_pixel = unvisited_ordinary[0]
        trace.append(next_pixel)
        visited_non_junction.add(next_pixel)
        current = next_pixel

        # 如果到达另一端点，完成追踪
        if node_types.get(current) == 'endpoint' and current != start_pixel:
            break

    return trace


def extract_all_traces(skeleton, node_types, endpoints):
    """
    从所有端点出发，提取全部独立裂缝迹线。
    过滤掉像素点少于 MIN_TRACE_PIXELS 的噪声碎片。

    参数:
        skeleton:   二值骨架图像
        node_types: 像素类型字典
        endpoints:  端点坐标列表

    返回:
        traces: [{'trace_id': int, 'pixels': [(row, col), ...]}, ...]
    """
    visited_non_junction = set()
    traces = []
    trace_id = 0

    for endpoint in endpoints:
        if endpoint in visited_non_junction:
            continue

        raw_pixels = trace_from_endpoint(
            endpoint, skeleton, node_types, visited_non_junction
        )

        if len(raw_pixels) < MIN_TRACE_PIXELS:
            continue

        trace_id += 1
        traces.append({
            'trace_id': trace_id,
            'pixels': raw_pixels,
        })

    return traces


# ============================================================
#  Ramer–Douglas–Peucker (RDP) 多段线简化
# ============================================================

def perpendicular_distance(point, line_start, line_end):
    """
    计算点到直线（由 line_start 和 line_end 定义）的垂直距离。
    使用 2D 叉积公式：|(B-A) × (P-A)| / |B-A|

    参数:
        point:      (row, col) 待计算点
        line_start: (row, col) 线段起点
        line_end:   (row, col) 线段终点

    返回:
        distance:   垂直距离（浮点数）
    """
    a = np.array(line_start, dtype=float)
    b = np.array(line_end, dtype=float)
    p = np.array(point, dtype=float)

    if np.array_equal(a, b):
        return np.linalg.norm(p - a)

    ab = b - a
    ap = p - a
    cross = abs(ab[0] * ap[1] - ab[1] * ap[0])
    distance = cross / np.linalg.norm(ab)
    return distance


def rdp_simplify(points, epsilon):
    """
    Ramer–Douglas–Peucker 算法：递归简化多段线。
    对应论文中的 Dmax 像素阈值。

    算法流程：
      1. 找到距离首尾连线最远的点
      2. 若最远距离 > epsilon，在该点处分裂，递归处理两段
      3. 若最远距离 ≤ epsilon，丢弃所有中间点，仅保留首尾

    参数:
        points:  [(row, col), ...] 原始像素坐标序列
        epsilon: 简化阈值（像素）

    返回:
        simplified: [(row, col), ...] 简化后的关键点列表
    """
    if len(points) < 3:
        return list(points)

    first = points[0]
    last = points[-1]

    max_distance = 0.0
    max_index = 0

    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], first, last)
        if dist > max_distance:
            max_distance = dist
            max_index = i

    if max_distance > epsilon:
        left_part = rdp_simplify(points[:max_index + 1], epsilon)
        right_part = rdp_simplify(points[max_index:], epsilon)
        return left_part[:-1] + right_part
    else:
        return [first, last]


# ============================================================
#  可视化
# ============================================================

def create_visualization(skeleton, dp_traces):
    """
    创建可视化图像：
      - 白色背景
      - 骨架用浅灰色绘制
      - RDP 简化后的多段线用红色绘制

    参数:
        skeleton:  二值骨架图像 (bool)
        dp_traces: 简化后迹线 [{'trace_id': int, 'dp_points': [...]}, ...]

    返回:
        vis_img:   RGB 彩色图像 (numpy uint8, shape: H×W×3)
    """
    height, width = skeleton.shape
    vis_img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 用浅灰色绘制骨架
    skeleton_mask = skeleton > 0
    vis_img[skeleton_mask] = (200, 200, 200)

    # 用红色绘制 RDP 简化后的多段线
    for dp_trace in dp_traces:
        dp_points = dp_trace['dp_points']
        if len(dp_points) < 2:
            continue
        pts = np.array([[c, r] for (r, c) in dp_points], dtype=np.int32)
        cv2.polylines(vis_img, [pts], isClosed=False,
                      color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    return vis_img


# ============================================================
#  JSON 输出
# ============================================================

def traces_to_json_format(traces, dp_traces):
    """
    将原始迹线和简化迹线组装为 JSON 可序列化的字典列表。
    每条记录包含：
      - trace_id:  迹线编号
      - raw_pixels: 原始像素坐标 [[col, row], ...]（x, y 格式）
      - dp_points:  RDP 简化关键点 [[col, row], ...]（x, y 格式）

    坐标统一使用 (col, row) 即 (x, y) 格式，符合常规几何坐标系习惯。
    """
    records = []
    for trace, dp_trace in zip(traces, dp_traces):
        record = {
            'trace_id': trace['trace_id'],
            'raw_pixels': [[c, r] for (r, c) in trace['pixels']],
            'dp_points': [[c, r] for (r, c) in dp_trace['dp_points']],
            'raw_pixel_count': len(trace['pixels']),
            'dp_point_count': len(dp_trace['dp_points']),
        }
        records.append(record)
    return records


# ============================================================
#  主处理流程：处理单张图片
# ============================================================

def process_single_image(image_path):
    """
    对单张PNG图片执行完整的骨架化 → 矢量化处理流程。

    步骤：
      1. 读取二值mask
      2. 形态学开运算去噪
      3. Zhang-Suen 骨架化
      4. 节点检测（端点/分支交点）
      5. 端点追踪提取迹线
      6. RDP 多段线简化
      7. 可视化输出
      8. JSON 输出

    参数:
        image_path: PNG文件路径
    """
    image_name = image_path.stem
    print(f"\n{'='*60}")
    print(f"  [处理中] {image_path.name}")
    print(f"{'='*60}")

    # ---------- 步骤1：读取二值mask ----------
    binary_mask = load_grayscale_mask(image_path)
    print(f"  图片尺寸: {binary_mask.shape[1]} × {binary_mask.shape[0]}")
    print(f"  裂缝像素数: {np.sum(binary_mask)}")

    # ---------- 步骤2：形态学开运算去噪 ----------
    cleaned_mask = apply_morphological_opening(binary_mask, MORPH_KERNEL_SIZE)
    print(f"  开运算后裂缝像素数: {np.sum(cleaned_mask)}")

    # ---------- 步骤3：Zhang-Suen 骨架化 ----------
    skeleton = extract_skeleton(cleaned_mask)
    skeleton_pixel_count = np.sum(skeleton)
    print(f"  骨架像素数: {skeleton_pixel_count}")

    # 保存骨架图片到 output_skeleton 目录
    skeleton_img = img_as_ubyte(skeleton)
    skeleton_output_path = OUTPUT_SKELETON_DIR / f"{image_name}_skeleton.png"
    cv2.imwrite(str(skeleton_output_path), skeleton_img)
    print(f"  骨架已保存: {skeleton_output_path.name}")

    # ---------- 步骤4：节点检测 ----------
    node_types, endpoints, junctions = classify_skeleton_nodes(skeleton)
    print(f"  检测到端点: {len(endpoints)} 个")
    print(f"  检测到分支交点: {len(junctions)} 个")

    # ---------- 步骤5：端点追踪提取迹线 ----------
    traces = extract_all_traces(skeleton, node_types, endpoints)
    print(f"  过滤后有效裂缝条数: {len(traces)} 条")
    print(f"  (过滤阈值: 最少 {MIN_TRACE_PIXELS} 像素)")

    if len(traces) == 0:
        print(f"  ⚠ 警告: 未检测到有效裂缝迹线，跳过后续处理")
        return

    # ---------- 步骤6：RDP 多段线简化 ----------
    dp_traces = []
    for trace in traces:
        raw_pixels = trace['pixels']
        dp_points = rdp_simplify(raw_pixels, DP_EPSILON)
        dp_traces.append({
            'trace_id': trace['trace_id'],
            'dp_points': dp_points,
        })
        print(f"  迹线 #{trace['trace_id']:03d}: "
              f"原始 {len(raw_pixels):4d} 像素 → "
              f"DP简化 {len(dp_points):4d} 关键点")

    # ---------- 步骤7：可视化 ----------
    vis_img = create_visualization(skeleton, dp_traces)
    vis_output_path = OUTPUT_VECTOR_IMG_DIR / f"{image_name}_vector.png"
    cv2.imwrite(str(vis_output_path), vis_img)
    print(f"  矢量化可视化已保存: {vis_output_path.name}")

    # ---------- 步骤8：JSON 输出 ----------
    json_records = traces_to_json_format(traces, dp_traces)
    json_output = {
        'image_name': image_path.name,
        'image_size': {'width': binary_mask.shape[1],
                       'height': binary_mask.shape[0]},
        'parameters': {
            'dp_epsilon': DP_EPSILON,
            'min_trace_pixels': MIN_TRACE_PIXELS,
            'morph_kernel_size': MORPH_KERNEL_SIZE,
        },
        'statistics': {
            'endpoint_count': len(endpoints),
            'junction_count': len(junctions),
            'valid_trace_count': len(traces),
        },
        'traces': json_records,
    }
    json_output_path = OUTPUT_VECTOR_JSON_DIR / f"{image_name}_traces.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    print(f"  JSON结果已保存: {json_output_path.name}")
    print(f"{'='*60}")


# ============================================================
#  批量处理入口
# ============================================================

def main():
    """
    批量读取 input_mask 下全部 PNG 图片，逐张处理。
    包含异常捕获，单张图片失败不影响其他图片。
    """
    print("=" * 60)
    print("  岩石裂隙骨架化与迹线矢量化")
    print("  参考: Chen J. 2022 JRMGE / Qiu Y. 2025")
    print("=" * 60)
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  输入目录:   {INPUT_DIR}")
    print(f"  配置参数:")
    print(f"    DP_EPSILON        = {DP_EPSILON}")
    print(f"    MIN_TRACE_PIXELS  = {MIN_TRACE_PIXELS}")
    print(f"    MORPH_KERNEL_SIZE = {MORPH_KERNEL_SIZE}")
    print("=" * 60)

    png_files = sorted(INPUT_DIR.glob("*.png"))
    png_files += sorted(INPUT_DIR.glob("*.PNG"))
    png_files = list(dict.fromkeys(png_files))

    if not png_files:
        print(f"\n  ⚠ 警告: 在 {INPUT_DIR} 中未找到PNG图片！")
        print(f"  请将裂缝mask图片放入 input_mask 文件夹后重新运行。")
        return

    print(f"\n  找到 {len(png_files)} 张PNG图片，开始批量处理...\n")

    success_count = 0
    fail_count = 0

    for i, png_path in enumerate(png_files, start=1):
        print(f"\n  [{i}/{len(png_files)}]")
        try:
            process_single_image(png_path)
            success_count += 1
        except Exception as e:
            fail_count += 1
            print(f"  ✗ 处理失败: {png_path.name}")
            print(f"    错误信息: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  批量处理完成！")
    print(f"  成功: {success_count} 张 | 失败: {fail_count} 张")
    print(f"{'='*60}")
    print(f"  输出目录:")
    print(f"    骨架图片:      {OUTPUT_SKELETON_DIR}")
    print(f"    矢量化可视化:  {OUTPUT_VECTOR_IMG_DIR}")
    print(f"    JSON结果:      {OUTPUT_VECTOR_JSON_DIR}")
    print(f"    调试文件:      {TEMP_DEBUG_DIR}")
    print(f"{'='*60}")


# ============================================================
#  程序入口
# ============================================================
if __name__ == "__main__":
    main()