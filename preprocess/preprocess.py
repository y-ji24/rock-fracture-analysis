# -*- coding: utf-8 -*-
"""
地质照片批量预处理：统一比例尺与画幅
======================================
参照物：图中的地质锤（锤长 = 柄端到锤头端的长度）
  1. 识别地质锤（按优先级依次尝试）：
     方案一 模板匹配 —— ref/ 中干净无背景的锤子图，多角度多尺度匹配（主方案）
     方案二 LSD 直线 —— 长直线 + 端点钢头 + 柄侧暗度评分
     方案三 颜色连通域 —— 手柄颜色 + 钢头 + 细长形状
  2. 缩放：使地质锤长度 = HAMMER_TARGET_PX 像素（不考虑倾斜，不做旋转校正）
  3. 画幅：输出 OUTPUT_SIZE x OUTPUT_SIZE，地质锤居中，空白处白色填充

用法：
  python preprocess.py             # 处理 data/ 下所有图片 -> result/
  python preprocess.py --debug     # 同时在 debug/ 输出识别过程图，便于检查
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "result"
DEBUG_DIR = BASE_DIR / "debug"
TEMPLATE_DIR = BASE_DIR / "ref"

HAMMER_TARGET_PX = 300.0   # 缩放后地质锤的目标长度（像素）
OUTPUT_SIZE = 1024         # 输出画幅（正方形）
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
TEMPLATE_SCORE_MIN = 0.7   # 模板匹配接受阈值（已含尺度惩罚），低于则转备选方案


# ================= 方案一：模板匹配 =================

_tmpl_cache = None
_rot_cache = {}


def _load_template():
    """加载 ref/ 中预处理好的竖直模板 hammer_aligned.png。

    该模板由 prepare_ref.py 生成：手柄轴严格竖直，竖直高度 = 柄端到锤头端的长度。
    长度对所有旋转角度是同一个常量（刚体旋转不改变沿柄轴的长度）。
    若无对齐模板则用原图按 minAreaRect 粗略摆正兜底。
    返回 (bgr, mask, 柄端到锤头端长度)。
    """
    global _tmpl_cache
    if _tmpl_cache is not None:
        return _tmpl_cache
    aligned = TEMPLATE_DIR / "hammer_aligned.png"
    candidates = ([aligned] if aligned.exists() else []) + [
        f for f in sorted(TEMPLATE_DIR.iterdir())
        if f.suffix.lower() == ".png" and f.name != aligned.name]
    tmpl = None
    for f in candidates:
        img = imread_unchanged(f)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            continue
        alpha = img[:, :, 3]
        ys, xs = np.where(alpha > 10)
        if len(ys) < 1000:
            continue
        crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        if f != aligned:   # 兜底：粗略摆正
            rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
            (rw, rh), ang = rect[1], rect[2]
            rotate = -ang if rh > rw else 90.0 - ang
            h, w = crop.shape[:2]
            m = cv2.getRotationMatrix2D((w / 2, h / 2), rotate, 1.0)
            crop = cv2.warpAffine(crop, m, (w, h), flags=cv2.INTER_LINEAR,
                                  borderValue=(0, 0, 0, 0))
            al = crop[:, :, 3]
            ys, xs = np.where(al > 127)
            crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        # 竖直高度 = 柄端到锤头端长度（常量，与后续旋转角度无关）
        tmpl = (crop[:, :, :3], (crop[:, :, 3] > 127).astype(np.uint8) * 255,
                crop.shape[0])
        break
    _tmpl_cache = tmpl
    return tmpl


def _rotate_expand(tpl, msk, angle):
    """旋转模板并扩展画布（避免大角度时被裁掉），按掩码重新裁紧。"""
    h, w = tpl.shape[:2]
    rad = np.deg2rad(angle)
    cs, sn = abs(np.cos(rad)), abs(np.sin(rad))
    nw, nh = int(w * cs + h * sn) + 2, int(w * sn + h * cs) + 2
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    ri = cv2.warpAffine(tpl, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    rm = cv2.warpAffine(msk, m, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    ys, xs = np.where(rm > 127)
    if len(ys) < 500:
        return None, None
    return (ri[ys.min():ys.max() + 1, xs.min():xs.max() + 1],
            rm[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def _get_rotated(angle):
    """按角度取旋转后的模板（跨图片缓存）。"""
    if angle not in _rot_cache:
        t = _load_template()
        if t is None:
            return None, None
        _rot_cache[angle] = _rotate_expand(t[0], t[1], angle)
    return _rot_cache[angle]


def _match_scale_range(img, tpl, msk, s_lo, s_hi, s_step, both=False):
    """
    在指定尺度范围内匹配。尺度惩罚：小模板的归一化相关虚高
    （M 个位置中随机最大相关 ≈ √(2lnM/N)），减去该项使不同尺度得分可比。
    惩罚分把峰推向大尺度、原始分把峰略推向小尺度——both=True 时同时返回
    两者，取平均尺度可抵消偏置。返回 (score, scale, loc, tw, th)；
    both=True 时返回 ((惩罚最佳), (原始最佳))。
    """
    h_img, w_img = img.shape[:2]
    best_pen = (-1.0, 1.0, (0, 0), 0, 0)
    best_raw = (-1.0, 1.0, (0, 0), 0, 0)
    s = s_lo
    while s <= s_hi:
        tw, th = int(tpl.shape[1] * s), int(tpl.shape[0] * s)
        if 6 <= tw <= w_img and 6 <= th <= h_img:
            ts = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
            m2 = cv2.resize(msk, (tw, th), interpolation=cv2.INTER_NEAREST)
            n_valid = int((m2 > 0).sum())
            if n_valid > 0:
                res = cv2.matchTemplate(img, ts, cv2.TM_CCORR_NORMED, mask=m2)
                mx = res.max()
                if mx > best_raw[0]:
                    ry, rx = np.unravel_index(res.argmax(), res.shape)
                    best_raw = (mx, s, (int(rx), int(ry)), tw, th)
                pen = mx - math.sqrt(2.0 * math.log(float(res.size)) / n_valid)
                if pen > best_pen[0]:
                    ry, rx = np.unravel_index(res.argmax(), res.shape)
                    best_pen = (pen, s, (int(rx), int(ry)), tw, th)
        s *= s_step
    if both:
        return best_pen, best_raw
    return best_pen


def detect_by_template(img):
    """模板匹配主方案。返回 (info, debug) 或 (None, None)。"""
    t = _load_template()
    if t is None:
        return None, None
    tlen = float(t[2])               # 柄端到锤头端长度（模板竖直高度，常量）

    # ---- 粗搜：缩小图，角度步长 10°，尺度步长 1.3 ----
    f = 1.0
    work = img
    if max(img.shape[:2]) > 900:
        f = 900.0 / max(img.shape[:2])
        work = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    h_w, w_w = work.shape[:2]
    min_axis = min(h_w, w_w)

    best = None
    for ang in range(0, 360, 10):
        rt, rm = _get_rotated(ang)
        if rt is None:
            continue
        s_lo = min_axis * 0.10 / tlen
        s_hi = min_axis * 0.95 / tlen
        score, s, loc, tw, th = _match_scale_range(work, rt, rm,
                                                   s_lo, s_hi, 1.3)
        if best is None or score > best[0]:
            best = (score, ang, s, loc, tw, th)
    if best is None or best[0] < TEMPLATE_SCORE_MIN:
        return None, None

    # ---- 精搜1：半分辨率局部窗口定角度/尺度，角度 ±10° 步长 5°，尺度步长 1.05 ----
    score0, ang0, s0, loc0, tw0, th0 = best
    f2 = 0.5
    img2 = cv2.resize(img, None, fx=f2, fy=f2, interpolation=cv2.INTER_AREA)
    cx = (loc0[0] + tw0 / 2) * f2 / f
    cy = (loc0[1] + th0 / 2) * f2 / f
    length_est = tlen * s0 * f2 / f                 # 粗估锤长（精搜图像素）
    win = int(min(length_est * 2.2, min(img2.shape[:2])))
    x0 = int(np.clip(cx - win / 2, 0, img2.shape[1] - win))
    y0 = int(np.clip(cy - win / 2, 0, img2.shape[0] - win))
    sub = img2[y0:y0 + win, x0:x0 + win]

    refined = None
    for ang in (ang0 - 10, ang0 - 5, ang0, ang0 + 5, ang0 + 10):
        rt, rm = _get_rotated(ang % 360)
        if rt is None:
            continue
        score, s, loc, tw, th = _match_scale_range(
            sub, rt, rm, length_est * 0.75 / tlen, length_est * 1.25 / tlen, 1.05)
        if refined is None or score > refined[0]:
            refined = (score, ang % 360, s, (loc[0] + x0, loc[1] + y0), tw, th)

    score, ang, s, loc, tw, th = refined
    if score < TEMPLATE_SCORE_MIN:
        return None, None
    # ---- 精搜2：全分辨率小窗口精确定位，尺度 ±20% 步长 1.02 ----
    # 同时求惩罚峰与原始峰，取平均尺度抵消两者的反向偏置；
    # 窗口须覆盖精搜1 可能的偏置
    rt, rm = _get_rotated(ang)
    length_est2 = tlen * s / f2                      # 锤长（原图像素）
    cx2 = (loc[0] + tw / 2) / f2
    cy2 = (loc[1] + th / 2) / f2
    win2 = int(min(length_est2 * 1.6, min(img.shape[:2])))
    x2 = int(np.clip(cx2 - win2 / 2, 0, img.shape[1] - win2))
    y2 = int(np.clip(cy2 - win2 / 2, 0, img.shape[0] - win2))
    sub2 = img[y2:y2 + win2, x2:x2 + win2]
    res_pen, res_raw = _match_scale_range(
        sub2, rt, rm, length_est2 * 0.80 / tlen, length_est2 * 1.20 / tlen, 1.02,
        both=True)
    if res_raw[0] < TEMPLATE_SCORE_MIN:
        return None, None
    s2 = (res_pen[1] + res_raw[1]) / 2.0            # 平均尺度，抵消偏置
    score2, _, loc2, tw2, th2 = res_raw             # 定位用原始分的最佳位置
    score, ang, s, loc, tw, th = score2, ang, s2, \
        (loc2[0] + x2, loc2[1] + y2), tw2, th2

    length = tlen * s                               # 锤长（原图像素）
    hc_x = loc[0] + tw / 2
    hc_y = loc[1] + th / 2
    rad = np.deg2rad(ang)
    # 柄轴方向（中心→柄端）：模板竖直时柄端在下，旋转 ang 后为 (sinθ, cosθ)
    u = np.array([math.sin(rad), math.cos(rad)])    # 仅用于调试画线
    info = dict(cx=hc_x, cy=hc_y, length=length, u=u, method="template",
                score=score, ang=ang)
    debug = dict(loc=loc, tw=tw, th=th, score=score, ang=ang, method="template")
    return info, debug


# ================= 方案二：LSD 直线 + 钢头 + 暗度 =================

def norm_ang(d):
    """角度差归一到 [0, 90] 度。"""
    return abs((d + 90.0) % 180.0 - 90.0)


def lsd_segments(img):
    """LSD 直线段检测，返回线段字典列表。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    segs = []
    if lines is None:
        return segs
    for x1, y1, x2, y2 in lines.reshape(-1, 4).astype(np.float64):
        dx, dy = x2 - x1, y2 - y1
        ln = math.hypot(dx, dy)
        if ln < 40:
            continue
        ux, uy = dx / ln, dy / ln
        segs.append(dict(x1=x1, y1=y1, x2=x2, y2=y2, len=ln,
                         ux=ux, uy=uy,
                         ang=math.degrees(math.atan2(dy, dx)),
                         mx=(x1 + x2) / 2, my=(y1 + y2) / 2))
    return segs


def merge_collinear(segs, ang_tol=6.0, lat_tol=10.0, gap_tol=60.0):
    """合并同一直线上被打断的线段（手柄可能被手或遮挡物断开）。"""
    out = []
    unused = segs[:]
    while unused:
        a = unused.pop(0)
        lane = [a]
        changed = True
        while changed:  # 收集同一车道上的线段
            changed = False
            for b in unused[:]:
                if norm_ang(a["ang"] - b["ang"]) > ang_tol:
                    continue
                for r in lane:
                    rx, ry, rux, ruy = r["x1"], r["y1"], r["ux"], r["uy"]
                    d1 = abs((b["x1"] - rx) * (-ruy) + (b["y1"] - ry) * rux)
                    d2 = abs((b["x2"] - rx) * (-ruy) + (b["y2"] - ry) * rux)
                    if max(d1, d2) <= lat_tol:
                        lane.append(b)
                        unused.remove(b)
                        changed = True
                        break
        # 沿车道方向投影并合并区间
        ux, uy = lane[0]["ux"], lane[0]["uy"]
        pts = np.array([[s["x1"], s["y1"]] for s in lane]
                       + [[s["x2"], s["y2"]] for s in lane], dtype=np.float64)
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        intervals = []
        for s in lane:
            t1 = (s["x1"] - cx) * ux + (s["y1"] - cy) * uy
            t2 = (s["x2"] - cx) * ux + (s["y2"] - cy) * uy
            intervals.append((min(t1, t2), max(t1, t2)))
        intervals.sort()
        merged = [intervals[0]]
        for lo, hi in intervals[1:]:
            if lo <= merged[-1][1] + gap_tol:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        for lo, hi in merged:
            if hi - lo < 40:
                continue
            p1 = np.array([cx, cy]) + np.array([ux, uy]) * lo
            p2 = np.array([cx, cy]) + np.array([ux, uy]) * hi
            out.append(dict(x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1],
                            len=hi - lo, ux=ux, uy=uy, ang=lane[0]["ang"],
                            mx=(p1[0] + p2[0]) / 2, my=(p1[1] + p2[1]) / 2))
    return out


def _bright_mask(img):
    """高亮低饱和像素（钢头/金属）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 150)


def _head_evidence(bright, c_lat, u, t_lo, t_hi, band):
    """
    沿柄轴在两端外侧找钢头（紧凑的亮金属块），扩展 t_lo/t_hi。
    返回 (t_lo, t_hi, found)。限制轴向/侧向跨度与对准轴，排除大片亮岩石与天空。
    """
    h, w = bright.shape
    ys, xs = np.where(bright)
    t = xs * u[0] + ys * u[1]
    lat = -xs * u[1] + ys * u[0]
    found = False
    for side, sign in ((t_hi, +1), (t_lo, -1)):
        in_band = np.abs(lat - c_lat) < band
        beyond = (t > side) & (t < side + 320) if sign > 0 \
            else (t < side) & (t > side - 320)
        sel = in_band & beyond
        if sel.sum() < 250:
            continue
        mask = np.zeros((h, w), np.uint8)
        mask[ys[sel], xs[sel]] = 255
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            continue
        i = 1 + int(np.argmax(st[1:, 4]))   # 带内最大的亮连通块 = 钢头
        comp = lab == i
        tt, ll = t[comp[ys, xs]], lat[comp[ys, xs]]
        if tt.size < 250:
            continue
        tspan = tt.max() - tt.min()
        lspan = ll.max() - ll.min()
        lcenter = (ll.max() + ll.min()) / 2
        if not (15 <= tspan <= 320 and lspan <= 150
                and abs(lcenter - c_lat) <= band):
            continue
        side = tt.max() if sign > 0 else tt.min()
        if sign > 0:
            t_hi = side
        else:
            t_lo = side
        found = True
    return t_lo, t_hi, found


def _side_darkness(img, s, u):
    """线段两侧各约 25px 处的平均亮度（V），取更暗的一侧，越暗越接近 1。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vals = []
    for off in (25, -25):
        x1, y1 = s["x1"] - off * u[1], s["y1"] + off * u[0]
        x2, y2 = s["x2"] - off * u[1], s["y2"] + off * u[0]
        mask = np.zeros(gray.shape, np.uint8)
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 9)
        if mask.any():
            vals.append(float(gray[mask > 0].mean()))
    if not vals:
        return 0.0
    return float(np.clip(1.0 - min(vals) / 150.0, 0.0, 1.0))


def detect_by_edges(img):
    """
    备选方案一：LSD 直线段（合并共线断段）作为锤柄候选，
    评分 = 长度 x 钢头证据 x 柄侧暗度，取最高分。
    返回 (info, debug) 或 (None, None)。
    """
    h, w = img.shape[:2]
    segs = merge_collinear(lsd_segments(img), gap_tol=110.0)
    if not segs:
        return None, None
    bright = _bright_mask(img)
    min_len = min(h, w) * 0.06
    best = None
    for s in segs:
        if s["len"] < min_len:
            continue
        u = np.array([s["ux"], s["uy"]])
        c_lat = -s["x1"] * s["uy"] + s["y1"] * s["ux"]
        t1 = s["x1"] * u[0] + s["y1"] * u[1]
        t2 = s["x2"] * u[0] + s["y2"] * u[1]
        t_lo, t_hi = min(t1, t2), max(t1, t2)
        t_lo, t_hi, found = _head_evidence(bright, c_lat, u, t_lo, t_hi, 50.0)
        length = t_hi - t_lo
        dark = _side_darkness(img, s, u)
        score = length * (1 + 2.0 * int(found)) * (1 + 0.5 * dark)
        tc = (t_lo + t_hi) / 2.0
        cx = tc * u[0] - c_lat * u[1]
        cy = tc * u[1] + c_lat * u[0]
        cand = dict(cx=cx, cy=cy, length=length, u=u, line=s,
                    t_lo=t_lo, t_hi=t_hi, found=found,
                    score=score, method="edges")
        if best is None or cand["score"] > best["score"]:
            best = cand
    if best is None:
        return None, None
    debug = dict(segs=segs, line=best["line"], method="edges")
    return best, debug


# ================= 方案三：颜色连通域 =================

def handle_color_masks(img):
    """常见地质锤手柄颜色掩码（橙红/黄/蓝/木色），OpenCV HSV: H 0-179。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return {
        "orange_red": (((h <= 20) | (h >= 165)) & (s > 100) & (v > 60)),
        "yellow": ((h >= 18) & (h <= 42) & (s > 90) & (v > 120)),
        "blue": ((h >= 95) & (h <= 130) & (s > 50) & (v > 50)),
        "wood": ((h >= 8) & (h <= 35) & (s > 40) & (s < 160) & (v > 50) & (v < 200)),
    }


def steel_mask(img):
    """钢头掩码：低饱和 + 高亮 + 低纹理（局部亮度方差小）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sq_mean = cv2.blur(gray * gray, (21, 21))
    mean = cv2.blur(gray, (21, 21))
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0))
    return (s < 70) & (v > 165) & (std < 28)


def elongated_components(mask, img):
    """从掩码中提取细长连通域，返回候选列表（得分降序）。"""
    h, w = img.shape[:2]
    min_len = min(h, w) * 0.06
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    cands = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 800 or bw < 8 or bh < 8:
            continue
        ys, xs = np.where(labels == i)
        rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        rl, rs = sorted(rect[1], reverse=True)
        if rl < min_len or rl / max(rs, 1e-3) < 2.0:
            continue
        touch_edge = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        score = (rl / rs) * area * (0.5 if touch_edge else 1.0)
        cands.append(dict(rect=rect, area=area, ratio=rl / rs,
                          length=rl, score=score))
    cands.sort(key=lambda c: -c["score"])
    return cands


def measure_along_axis(mask, rect):
    """沿矩形两轴投影掩码，取跨度大的轴向，返回 (cx, cy, length, u)。"""
    (rcx, rcy), (rw, rh), ang = rect
    u1 = np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))])
    u2 = np.array([-u1[1], u1[0]])
    ys, xs = np.where(mask)
    band = max(min(rw, rh) * 2.0, 20)
    best = None
    for u in (u1, u2):
        t = (xs - rcx) * u[0] + (ys - rcy) * u[1]
        d = np.abs(-(xs - rcx) * u[1] + (ys - rcy) * u[0])
        keep = d < band
        if keep.sum() < 10:
            continue
        span = t[keep].max() - t[keep].min()
        if best is None or span > best[0]:
            best = (span, u, t[keep])
    if best is None:
        return rcx, rcy, max(rw, rh), u1
    span, u, t = best
    tc = (t.min() + t.max()) / 2.0
    hc = np.array([rcx, rcy]) + u * tc
    return float(hc[0]), float(hc[1]), float(span), u


def detect_by_color(img):
    """备选方案二：手柄颜色 + 钢头 + 细长形状。返回 (info, debug) 或 (None, None)。"""
    sources = handle_color_masks(img)
    steel = steel_mask(img)
    sources["steel"] = steel
    best, best_key = None, None
    for key, mask in sources.items():
        combined = cv2.dilate((mask | steel).astype(np.uint8),
                              np.ones((9, 9), np.uint8)) > 0
        cands = elongated_components(combined, img)
        if cands and (best is None or cands[0]["score"] > best["score"]):
            best, best_key = cands[0], key
    if best is None:
        return None, None
    full_mask = sources[best_key] | steel
    cx, cy, length, u = measure_along_axis(full_mask, best["rect"])
    info = dict(cx=cx, cy=cy, length=length, u=u, method="color")
    debug = dict(rect=best["rect"], combined=full_mask, key=best_key,
                 method="color")
    return info, debug


def detect_hammer(img):
    """依次尝试 模板匹配 -> LSD 直线 -> 颜色连通域。"""
    for fn in (detect_by_template, detect_by_edges, detect_by_color):
        info, debug = fn(img)
        if info is not None:
            return info, debug
    return None, None


# ================= 几何变换 =================

def transform(img, cx, cy, scale):
    """缩放 + 平移：地质锤中心映射到画布中心，画布外以白色填充（含剪裁/扩图）。"""
    m = np.array([[scale, 0, OUTPUT_SIZE / 2 - scale * cx],
                  [0, scale, OUTPUT_SIZE / 2 - scale * cy]], dtype=np.float64)
    flags = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, m, (OUTPUT_SIZE, OUTPUT_SIZE), flags=flags,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def fallback_transform(img):
    """未识别到地质锤：整图等比缩放居中，白色填充。"""
    h, w = img.shape[:2]
    scale = min(OUTPUT_SIZE / w, OUTPUT_SIZE / h)
    return transform(img, w / 2, h / 2, scale), scale


# ================= 调试可视化 =================

def draw_debug(img, info, dbg):
    vis = img.copy()
    if dbg["method"] == "template":
        loc, tw, th = dbg["loc"], dbg["tw"], dbg["th"]
        p1 = (int(loc[0]), int(loc[1]))
        p2 = (int(loc[0] + tw), int(loc[1] + th))
        cv2.rectangle(vis, p1, p2, (0, 255, 0), 4)
        label = f"template score={dbg['score']:.2f} ang={dbg['ang']}"
    elif dbg["method"] == "edges":
        for s in dbg["segs"]:
            cv2.line(vis, (int(s["x1"]), int(s["y1"])),
                     (int(s["x2"]), int(s["y2"])), (0, 180, 0), 1)
        s = dbg["line"]
        cv2.line(vis, (int(s["x1"]), int(s["y1"])),
                 (int(s["x2"]), int(s["y2"])), (0, 255, 0), 4)
        label = f"edges len={info['length']:.0f}px"
    else:
        box = cv2.boxPoints(dbg["rect"]).astype(np.int32)
        cv2.drawContours(vis, [box], 0, (0, 255, 0), 4)
        label = f"color len={info['length']:.0f}px"
    u, ln = info["u"], info["length"]
    p1 = np.array([info["cx"], info["cy"]]) - u * ln / 2
    p2 = np.array([info["cx"], info["cy"]]) + u * ln / 2
    cv2.line(vis, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 0, 0), 4)
    cv2.circle(vis, (int(info["cx"]), int(info["cy"])), 12, (0, 0, 255), -1)
    cv2.putText(vis, f"{label} len={ln:.0f}px", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
    return vis


# ================= 读写（兼容中文路径） =================

def imread_safe(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imread_unchanged(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def imwrite_safe(path, img):
    ext = Path(path).suffix.lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(str(path))
    return ok


# ================= 主流程 =================

def process_one(path, debug=False):
    img = imread_safe(path)
    if img is None:
        print(f"[跳过] 无法读取: {path.name}")
        return None

    info, dbg = detect_hammer(img)
    if info is None:
        print(f"[警告] {path.name}: 未识别到地质锤，按整图缩放居中处理")
        out, scale = fallback_transform(img)
    else:
        scale = HAMMER_TARGET_PX / info["length"]
        out = transform(img, info["cx"], info["cy"], scale)
        print(f"[完成] {path.name}: 原图 {img.shape[1]}x{img.shape[0]}, "
              f"锤长 {info['length']:.0f}px -> {HAMMER_TARGET_PX:.0f}px "
              f"(缩放 {scale:.3f}), 锤中心 ({info['cx']:.0f},{info['cy']:.0f}) "
              f"[{info['method']}]")
        if debug:
            vis = draw_debug(img, info, dbg)
            imwrite_safe(DEBUG_DIR / f"{path.stem}_debug.png", vis)

    out_path = RESULT_DIR / f"{path.stem}{path.suffix.lower()}"
    if not imwrite_safe(out_path, out):
        print(f"[失败] {path.name}: 保存失败")
        return None
    return out


def preprocess(image_path, output_path):
    """单张图片处理接口（兼容模块调用）：识别地质锤并保存到 output_path。"""
    img = imread_safe(Path(image_path))
    if img is None:
        return False
    info, _ = detect_hammer(img)
    if info is None:
        out, _ = fallback_transform(img)
    else:
        out = transform(img, info["cx"], info["cy"],
                        HAMMER_TARGET_PX / info["length"])
    return imwrite_safe(Path(output_path), out)


def main():
    ap = argparse.ArgumentParser(description="地质照片统一比例尺与画幅")
    ap.add_argument("--debug", action="store_true", help="输出识别过程图到 debug/")
    ap.add_argument("--data", default=str(DATA_DIR), help="输入目录")
    ap.add_argument("--out", default=str(RESULT_DIR), help="输出目录")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in data_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in IMG_EXTS)
    if not files:
        print(f"{data_dir} 下没有图片")
        return
    print(f"共 {len(files)} 张图片，目标：锤长 {HAMMER_TARGET_PX:.0f}px，"
          f"画幅 {OUTPUT_SIZE}x{OUTPUT_SIZE}，开始处理...")

    ok = 0
    for f in files:
        try:
            if process_one(f, args.debug) is not None:
                ok += 1
        except Exception as e:  # 单张失败不影响其余
            print(f"[失败] {f.name}: {e}")
    print(f"处理完成: {ok}/{len(files)}，结果保存在 {out_dir}")


if __name__ == "__main__":
    main()
