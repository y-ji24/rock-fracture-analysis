# 图片预处理：统一比例尺与画幅

以照片中的**地质锤**为参照物，把野外岩石照片统一为相同比例尺与画幅，
作为深度学习裂隙识别的标准化输入。

处理流程：
1. 识别地质锤（锤长 = 柄端到锤头端的长度）
   - 主方案：`ref/` 中干净模板的多角度多尺度模板匹配
   - 备选：LSD 直线检测（长直线 + 钢头 + 柄侧暗度）、手柄颜色连通域
   - 全部失败：整图缩放居中，并在日志中给出警告
2. 缩放：使地质锤长度 = `HAMMER_TARGET_PX`（默认 300 像素）
3. 画幅：输出 `OUTPUT_SIZE` × `OUTPUT_SIZE`（默认 1024），锤子居中，空白处白色填充

## 使用

```bash
# 处理 preprocess/data/ 下所有图片，结果保存到 preprocess/result/
python preprocess/preprocess.py

# 指定输入/输出目录
python preprocess/preprocess.py --data <图片目录> --out <输出目录>

# 同时输出识别过程图到 preprocess/debug/，用于检查识别效果
python preprocess/preprocess.py --debug

# 单张图片（模块调用）
python -c "from preprocess import preprocess; preprocess('a.jpg', 'b.jpg')"
```

## 目录

| 文件 | 说明 |
|---|---|
| `preprocess.py` | 主脚本：识别地质锤 + 缩放 + 画幅统一 |
| `prepare_ref.py` | 模板预处理：把 ref 模板旋转到严格竖直（竖直高度 = 柄到头的长度） |
| `ref/hammer_aligned.png` | 对齐后的锤子模板（运行时使用） |
| `ref/` 其余 PNG | 原始模板图；更换模板后重新运行 `prepare_ref.py` |

## 参数

`preprocess.py` 顶部可调：`HAMMER_TARGET_PX`（目标锤长，默认 300）、
`OUTPUT_SIZE`（画幅边长，默认 1024）、`TEMPLATE_SCORE_MIN`（模板匹配接受阈值）。

## 依赖

opencv-python、numpy
