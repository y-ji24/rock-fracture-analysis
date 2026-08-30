# -*- coding: utf-8 -*-
"""
岩体裂隙 S-RQD 快速评估程序（纯 OpenCV 版，srqd_assessment_opencv.py）
========================================================================
不依赖深度学习模型，仅用 OpenCV 传统方法识别裂隙：
    灰度化 -> 高斯滤波 -> Canny 边缘 -> 形态学闭运算/膨胀 -> 连通域去噪
随后交互完成：比例尺标定 -> 扫描线选择 -> 计算 S-RQD。

【S-RQD 算法（本程序重点）】
    沿扫描线（测线）采样，标记每个像素是否位于裂隙上，得到一系列
    “裂隙区间”；相邻裂隙区间中点之间的距离即为“完整岩体段长度”
    （间距）。按 Deere(1964) 的标准定义：
        S-RQD = 100 × Σ(完整岩体段长度 ≥ 阈值 10cm) / 扫描线总长度
    另附“个数口径”参考值：间距 ≥ 10cm 的个数 / 总间距个数 × 100%。

运行：python srqd_assessment_opencv.py
依赖：opencv-python、numpy、tkinter（Anaconda 自带）
"""
import os
import sys
import traceback

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# ----------------------------- 可调参数 -----------------------------
CANNY_LOW = 50             # Canny 低阈值
CANNY_HIGH = 150           # Canny 高阈值
GAUSSIAN_KERNEL = (5, 5)   # 高斯滤波核
MORPH_KERNEL_SIZE = 5      # 形态学闭运算核
MORPH_CLOSE_ITER = 2       # 闭运算次数
DILATE_ITER = 1            # 膨胀次数
MIN_AREA = 20              # 连通域最小面积（去噪点）
RQD_THRESHOLD_CM = 10.0    # S-RQD 有效间距阈值 (cm)

WIN_MAIN = "S-RQD Assessment - Main Window"
WIN_MASK = "Fracture Mask"


def _setup_console_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _draw_instruction(img, text, color=(0, 255, 255)):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA)


def _resize_for_display(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale < 1.0:
        disp = cv2.resize(img, (max(1, int(round(w * scale))),
                               max(1, int(round(h * scale)))),
                          interpolation=cv2.INTER_AREA)
    else:
        disp = img.copy()
    return disp, scale


def _to_orig(display_pt, display_img, window_name, base_scale):
    x, y = display_pt
    cur_scale = base_scale
    try:
        _rx, _ry, rw, _rh = cv2.getWindowImageRect(window_name)
        if rw > 0:
            cur_scale = base_scale * (display_img.shape[1] / float(rw))
    except cv2.error:
        pass
    return int(round(x / cur_scale)), int(round(y / cur_scale))


def _screen_size(root):
    try:
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    except Exception:
        sw, sh = 1600, 900
    return max(480, sw - 120), max(360, sh - 240)


def _wait_any_key(window_name):
    while True:
        key = cv2.waitKey(100) & 0xFF
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
        if key != 255:
            break


class _LinePointCollector:
    """鼠标点采集器：左键加点，右键/ESC 取消，关闭窗口视为取消。"""

    def __init__(self, window_name, display_img, labels, color):
        self.window_name = window_name
        self.image = display_img
        self.labels = labels
        self.color = color
        self.points = []
        self.finished = False
        self.cancelled = False

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.finished:
            self.points.append((x, y))
            self._draw_marker()
            if len(self.points) < 2:
                _draw_instruction(self.image,
                                  "Click point %s" % self.labels[1], self.color)
            cv2.imshow(self.window_name, self.image)
            if len(self.points) >= 2:
                self.finished = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.cancelled = True

    def _draw_marker(self):
        idx = len(self.points) - 1
        x, y = self.points[idx]
        cv2.circle(self.image, (x, y), 5, self.color, -1)
        cv2.putText(self.image, self.labels[idx], (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2, cv2.LINE_AA)
        if len(self.points) == 2:
            cv2.line(self.image, self.points[0], self.points[1], self.color, 2)

    def run(self):
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        _draw_instruction(self.image, "Click point %s (left mouse)"
                          % self.labels[0], self.color)
        cv2.imshow(self.window_name, self.image)
        while not self.finished and not self.cancelled:
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                self.cancelled = True
                break
            try:
                if cv2.getWindowProperty(self.window_name,
                                         cv2.WND_PROP_VISIBLE) < 1:
                    self.cancelled = True
                    break
            except cv2.error:
                self.cancelled = True
                break
        return None if self.cancelled else self.points


def load_image(root=None):
    """文件对话框加载图片（兼容中文路径），取消返回 None。"""
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    try:
        path = filedialog.askopenfilename(
            title="选择岩体露头照片", parent=root,
            filetypes=[("图片文件", "*.jpg *.jpeg *.png"), ("所有文件", "*.*")])
    finally:
        try:
            root.attributes("-topmost", False)
        except Exception:
            pass
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError("文件不存在：%s" % path)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError("无法读取图片：%s" % path)
    print("已加载图片：%s（%d x %d）" % (path, img.shape[1], img.shape[0]))
    return img


# ============================ 裂隙识别（OpenCV 传统方法） ============================

def detect_fractures_opencv(img):
    """改进的裂隙识别：CLAHE增强 + 自适应形态学 + 形状过滤"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ----- 1. 对比度增强 -----
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # ----- 2. 高斯滤波（去噪）-----
    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # ----- 3. Canny 边缘检测（阈值调低，配合增强后更灵敏）-----
    edges = cv2.Canny(blurred, 30, 100)   # 原为 (50,150)

    # ----- 4. 形态学操作 -----
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # 先开运算去除孤立小噪点
    opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    # 闭运算连接断裂的裂隙段
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 轻度膨胀，使裂隙线条更连续
    mask = cv2.dilate(closed, kernel, iterations=1)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    # ----- 5. 连通域分析 + 形状过滤（去除圆形/块状噪声）-----
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        clean = np.zeros_like(mask)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 10:          # 滤除极小噪声
                continue

            # 提取当前连通域轮廓
            cnt_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(cnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = contours[0]

            # 最小外接矩形，计算长宽比
            rect = cv2.minAreaRect(cnt)
            (w, h) = rect[1]
            if w == 0 or h == 0:
                continue
            aspect_ratio = max(w, h) / min(w, h)

            # 裂隙通常细长，保留长宽比 >= 2.0 的连通域
            if aspect_ratio >= 2.0:
                clean[labels == i] = 255

        mask = clean

    # ----- 6. 生成叠加显示图 -----
    overlay = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    return mask, overlay


# ============================ 比例尺标定 / 扫描线 / S-RQD ============================

def calibrate_scale(root, overlay):
    """比例尺标定：点击比例尺两端，输入实际距离(cm)，返回 cm/像素。"""
    disp, _ = _resize_for_display(overlay, *_screen_size(root))
    win = WIN_MAIN
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp.shape[1], disp.shape[0])
    cv2.imshow(win, disp)
    print("请在图像窗口中点击比例尺的两个端点（左键添加点，右键/ESC 取消）。")

    collector = _LinePointCollector(win, disp, ("P1", "P2"), (0, 255, 0))
    display_pts = collector.run()
    if display_pts is None:
        print("已取消比例尺标定。")
        cv2.destroyWindow(win)
        return None

    disp_scale = _resize_for_display(overlay, *_screen_size(root))[1]
    p1 = _to_orig(display_pts[0], disp, win, disp_scale)
    p2 = _to_orig(display_pts[1], disp, win, disp_scale)
    dist_px = np.linalg.norm(np.array(p1) - np.array(p2))

    answer = simpledialog.askfloat(
        "比例尺标定", "两点间距 %.1f 像素，请输入实际距离（cm）：" % dist_px,
        parent=root, minvalue=0.0)
    if answer is None or answer <= 0:
        print("未输入有效距离，取消标定。")
        cv2.destroyWindow(win)
        return None
    scale_cm_per_px = answer / dist_px
    print("比例尺标定完成：%.4f cm/像素（%.1f px = %.1f cm）。"
          % (scale_cm_per_px, dist_px, answer))
    cv2.destroyWindow(win)
    return scale_cm_per_px


def select_scanline(root, overlay, mask):
    """扫描线选择：点击测线两端，返回裂隙区间列表(像素下标)与扫描线长度。"""
    max_w, max_h = _screen_size(root)
    disp, _ = _resize_for_display(overlay, max_w, max_h)
    win = WIN_MAIN
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp.shape[1], disp.shape[0])
    cv2.imshow(win, disp)
    print("请在图像窗口中点击测线两端（左键添加点，右键/ESC 取消）。")

    collector = _LinePointCollector(win, disp, ("Start", "End"), (0, 255, 255))
    display_pts = collector.run()
    if display_pts is None:
        print("已取消扫描线选择。")
        cv2.destroyWindow(win)
        return None

    disp_scale = _resize_for_display(overlay, max_w, max_h)[1]
    p1 = _to_orig(display_pts[0], disp, win, disp_scale)
    p2 = _to_orig(display_pts[1], disp, win, disp_scale)
    scan_len_px = np.linalg.norm(np.array(p1) - np.array(p2))

    # DDA 采样：沿扫描线逐像素判断是否落在裂隙(mask=255)上
    x0, y0 = p1
    x1, y1 = p2
    dx, dy = x1 - x0, y1 - y0
    steps = max(abs(dx), abs(dy), 1)
    xs = np.linspace(x0, x1, int(steps) + 1).astype(int)
    ys = np.linspace(y0, y1, int(steps) + 1).astype(int)
    on_fracture = []
    for xi, yi in zip(xs, ys):
        if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]:
            on_fracture.append(mask[yi, xi] == 255)
        else:
            on_fracture.append(False)

    # 连续 True 段 -> 裂隙区间 [(起点下标, 终点下标), ...]
    intervals = []
    i = 0
    while i < len(on_fracture):
        if on_fracture[i]:
            j = i
            while j < len(on_fracture) and on_fracture[j]:
                j += 1
            intervals.append((i, j - 1))
            i = j
        else:
            i += 1
    cv2.destroyWindow(win)
    print("扫描线长度：%.0f 像素，与 %d 条裂隙相交。"
          % (scan_len_px, len(intervals)))
    return intervals, scan_len_px


def compute_srqd(intervals, scan_len_px, scale_cm_per_px):
    """S-RQD 计算（核心算法）。

    相邻裂隙区间中点之间的距离 = 完整岩体段长度（cm）。

    标准口径（Deere）：
        S-RQD = 100 × Σ(完整岩体段长度 >= RQD_THRESHOLD_CM) / 扫描线总长度
    参考口径（个数占比）：
        S-RQD' = 100 × (完整岩体段长度 >= 阈值 的个数) / 总个数

    返回 (srqd_std, srqd_count, gaps_cm)
    """
    gaps_cm = []
    if len(intervals) >= 2:
        mids = [(a + b) / 2.0 for a, b in intervals]
        for k in range(1, len(mids)):
            gaps_cm.append(abs(mids[k] - mids[k - 1]) * scale_cm_per_px)

    if not gaps_cm or scan_len_px <= 0:
        return 0.0, 0.0, gaps_cm

    # 标准：达标完整岩体段长度之和 / 扫描线总长度
    srqd_std = 100.0 * sum(g for g in gaps_cm
                           if g >= RQD_THRESHOLD_CM) / (scan_len_px * scale_cm_per_px)
    # 参考：达标间距个数 / 总间距个数
    srqd_count = 100.0 * sum(1 for g in gaps_cm
                             if g >= RQD_THRESHOLD_CM) / len(gaps_cm)
    return srqd_std, srqd_count, gaps_cm


# ============================== 主程序 ==============================

def main():
    _setup_console_utf8()
    root = tk.Tk()
    root.withdraw()
    try:
        root.update()
        print("=" * 60)
        print("岩体裂隙 S-RQD 快速评估程序（纯 OpenCV 版）")
        print("=" * 60)

        print("步骤 1/4：请在弹出的文件对话框中选择岩体露头照片（jpg/png）。")
        img = load_image(root)
        if img is None:
            print("未选择图片，程序退出。")
            return

        print("步骤 2/4：正在识别裂隙（Canny + 形态学）...")
        mask, overlay = detect_fractures_opencv(img)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
        print("裂隙识别完成：共检测到 %d 条裂隙轮廓。"
              % len(contours))

        max_w, max_h = _screen_size(root)
        disp, _ = _resize_for_display(overlay, max_w, max_h)
        mask_view, _ = _resize_for_display(
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
            max(480, max_w // 2), max_h)
        cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WIN_MASK, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(WIN_MAIN, disp.shape[1], disp.shape[0])
            cv2.resizeWindow(WIN_MASK, mask_view.shape[1], mask_view.shape[0])
        except cv2.error:
            pass
        cv2.imshow(WIN_MAIN, disp)
        cv2.imshow(WIN_MASK, mask_view)

        print("步骤 3/4：请完成比例尺标定与扫描线选择。")
        scale = calibrate_scale(root, overlay)
        if scale is None:
            print("未完成比例尺标定，程序退出。")
            return
        scan = select_scanline(root, overlay, mask)
        if scan is None:
            print("未完成扫描线选择，程序退出。")
            return
        intervals, scan_len_px = scan

        print("步骤 4/4：计算 S-RQD 指标...")
        srqd_std, srqd_count, gaps_cm = compute_srqd(intervals, scan_len_px, scale)
        print("=" * 60)
        print("评估结果")
        print("  扫描线长度：%.1f cm（%.0f 像素）"
              % (scan_len_px * scale, scan_len_px))
        print("  相交裂隙数：%d 条" % len(intervals))
        if gaps_cm:
            print("  相邻裂隙间距(cm)：%s"
                  % ", ".join("%.1f" % g for g in gaps_cm))
        else:
            print("  相邻裂隙间距(cm)：无（裂隙不足 2 条）")
        print("  S-RQD（标准·长度口径）= %.1f %%" % srqd_std)
        print("  S-RQD（参考·个数口径）= %.1f %%" % srqd_count)
        print("  阈值：%.0f cm" % RQD_THRESHOLD_CM)
        print("=" * 60)

        messagebox.showinfo("S-RQD 评估结果",
                            "S-RQD（长度口径）= %.1f %%\nS-RQD（个数口径）= %.1f %%"
                            % (srqd_std, srqd_count),
                            parent=root)
        print("按任意键关闭图像窗口并退出...")
        _wait_any_key(WIN_MAIN)
        _wait_any_key(WIN_MASK)
    except Exception as exc:
        print("程序出错：%s" % exc)
        traceback.print_exc()
        messagebox.showerror("错误", "程序出错：%s" % exc, parent=root)
    finally:
        cv2.destroyAllWindows()
        root.destroy()


if __name__ == "__main__":
    main()

