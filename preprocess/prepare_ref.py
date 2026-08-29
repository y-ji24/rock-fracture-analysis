# -*- coding: utf-8 -*-
"""
预处理 ref 模板：把锤子旋转到严格竖直并裁紧，保存为 ref/hammer_aligned.png。
处理后保证：竖直方向的高度 = 柄端到锤头端的长度。

对齐方法：只取"手柄段"的窄行（行宽 < 220px 的下半段），用行中心拟合柄轴，
迭代旋转（扩展画布，避免裁切）直至柄轴竖直。不用 minAreaRect/PCA——
锤头的方头+弯尖嘴质量不对称，会把整体惯量轴/外接矩形带偏。
"""
import math
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
REF_DIR = BASE / "ref"
OUT_NAME = "hammer_aligned.png"


def handle_axis(mask):
    """手柄段行中心拟合的柄轴与水平方向夹角（度）。"""
    ys, xs = np.where(mask > 127)
    rows = {}
    for y, x in zip(ys, xs):
        rows.setdefault(int(y), []).append(x)
    yy = np.array(sorted(rows))
    widths = np.array([max(rows[y]) - min(rows[y]) for y in yy])
    centers = np.array([np.mean(rows[y]) for y in yy])
    y0, y1 = yy[0], yy[-1]
    m = (yy > y0 + (y1 - y0) * 0.45) & (widths < 220)   # 下半段的窄行 = 手柄
    if m.sum() < 50:
        m = widths < 220
    p = np.polyfit(yy[m], centers[m], 1)
    return math.degrees(math.atan2(1.0, p[0]))


def rotate_expand(img, angle):
    """旋转并扩展画布，避免大角度旋转裁掉内容。"""
    h, w = img.shape[:2]
    rad = np.deg2rad(angle)
    cs, sn = abs(np.cos(rad)), abs(np.sin(rad))
    nw, nh = int(w * cs + h * sn) + 2, int(w * sn + h * cs) + 2
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderValue=(0, 0, 0, 0))


def crop_tight(img):
    al = img[:, :, 3]
    ys, xs = np.where(al > 127)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def main():
    src = None
    for f in sorted(REF_DIR.iterdir()):
        if f.suffix.lower() == ".png" and f.name != OUT_NAME:
            src = f
            break
    if src is None:
        print("ref/ 中没有可处理的 PNG")
        return

    data = np.fromfile(str(src), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    img = crop_tight(img)

    for it in range(6):
        a = handle_axis(img[:, :, 3])
        err = a - 90.0
        print(f"  迭代{it}: 柄轴角度 {a:.3f}°，偏差 {err:+.3f}°")
        if abs(err) < 0.3:
            break
        rot = -err                        # 猜测旋转方向
        cand = crop_tight(rotate_expand(img, rot))
        if abs(handle_axis(cand[:, :, 3]) - 90.0) > abs(err):
            rot = -rot                    # 方向反了，取反
            cand = crop_tight(rotate_expand(img, rot))
        img = cand

    a = handle_axis(img[:, :, 3])
    print(f"最终柄轴角度: {a:.3f}°（目标 90°）")
    print(f"对齐后图像尺寸 (高x宽): {img.shape[0]} x {img.shape[1]}")
    print(f"竖直高度 = 柄端到锤头端长度 = {img.shape[0]} px")

    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(REF_DIR / OUT_NAME))
    print("已保存:", REF_DIR / OUT_NAME)


if __name__ == "__main__":
    main()
