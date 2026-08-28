# 岩石裂隙识别与岩体质量评价

基于深度学习裂隙识别与迹线分析的野外岩体质量快速评价方法研究
（地质实习 · 本科创新课题）

## 技术流程
野外岩石照片 → 图片预处理 → 深度学习裂隙识别 → 主裂隙识别 → 骨架化 → 矢量化（裂缝迹线） → S-RQD 计算 → 岩体质量评价

## 目录结构
| 目录 | 模块 | 说明 |
|---|---|---|
| preprocess/ | 图片预处理 | 裁剪、去噪、光照校正、数据增强 |
| detection/ | 深度学习裂隙识别 | 调用 GeoFractNet 预训练模型，输出裂缝 mask |
| main_fracture/ | 主裂隙识别 | 从 mask 中筛选主裂隙，去除碎小噪声 |
| skeleton/ | 骨架化 | 裂缝 mask → 单像素骨架（scikit-image） |
| vectorize/ | 矢量化 | 骨架 → 裂缝迹线 polyline（OpenCV Douglas-Peucker） |
| srqd/ | S-RQD 计算 | 扫描线统计裂缝间距 → S-RQD 值 |
| data/ | 数据 | raw/ 原始照片（不入库）、samples/ 示例图 |
| docs/ | 文档 | 技术流程说明、组会记录 |
| scripts/ | 脚本 | 一键运行完整流程 |



