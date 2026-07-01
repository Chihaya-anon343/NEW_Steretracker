# Steretracker — GPNP 双目视觉跟踪器

基于 **YOLO 检测 → ROI 面积策略分发 → 特征提取 → 视差 → 位姿解算** 流水线的 C++17 双目视觉定位系统，面向无人机视觉定位场景。

**技术栈**: C++17 · OpenCV 4.x · Eigen 3.x · ONNX Runtime

> 项目由 Python 单文件 (~1250 行) 重构为模块化 C++ 架构 (~23 个源文件)。算法逻辑与 Python 版**完全等价**，详见 [PORTING_REPORT.md](./PORTING_REPORT.md)。

---

## 1. 整体流程

```
                          ┌─────────────────┐
                          │ tracker_config.json │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  加载双目图像     │
                          └────────┬────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │      逐 帧 循 环             │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  ① YOLO ONNX 目标检测        │
                    │     输出 左右 ROI 矩形        │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  ② ROI 面积判定 + 策略分发    │
                    │   FeatureExtractorFactory    │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     ┌────────▼────────┐  ┌───────▼────────┐  ┌────────▼────────┐
     │  TinyTarget     │  │  BinaryCorner  │  │  AKAZE_GPNP     │
     │  ROI ≤ 800 px²  │  │ 801~40000 px²  │  │ ≥40001 / 无检测  │
     │  → solvePnP     │  │ → GPNP (LM)    │  │ → GPNP (LM)     │
     └────────┬────────┘  └───────┬────────┘  └────────┬────────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  ③ 输出位姿 [R|t] + 日志      │
                    │     可视化调试图像 (可选)      │
                    └─────────────────────────────┘
```

---

## 2. 策略分发机制

系统根据 YOLO 检测到的 ROI 面积自动选择特征提取策略：

| ROI 面积 | 策略 | 提取器 | 位姿解算 |
|----------|------|--------|----------|
| ≤ 800 px² | TinyTarget | `TinyTargetExtractor` | `cv::solvePnP(ITERATIVE)` |
| 801 ~ 40000 px² | BinaryCorner | `BinaryCornerExtractor` | `GPnPSolver` (Eigen LM) |
| ≥ 40001 px² / 无检测回退 | AKAZE_GPNP | `AkazeGpnpExtractor` | `GPnPSolver` (Eigen LM) |

分发函数 `createFeatureExtractor()` 位于 `src/feature/FeatureExtractorFactory.cpp`，运行时通过 `StereoTracker::setExtractor()` 热切换提取器。

**ROI 外扩**：在送入提取器前，BinaryCorner 和 TinyTarget 策略会外扩 ROI 边距（`roi_pad_pixels`），为角点提取提供周围上下文像素。

---

## 3. 策略一：TinyTarget（ROI ≤ 800 px²）

**适用场景**：远距离微小矩形目标（如 4 角点标定板）。

**模块映射**：`TinyTargetExtractor` (`src/feature/TinyTargetExtractor.cpp`)

### 3.1 完整流水线

```
┌──────────────────────────────────────────────────────────────┐
│                    TinyTarget 流水线                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  左 ROI 灰度图                   右 ROI 灰度图                 │
│       │                              │                        │
│       ▼                              ▼                        │
│  ┌─────────────┐              ┌─────────────┐                │
│  │ Otsu 二值化  │              │ Otsu 二值化  │               │
│  └──────┬──────┘              └──────┬──────┘                │
│         │                            │                        │
│         ▼                            ▼                        │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 模板匹配 (IoU)      │    │ 模板匹配 (IoU)      │            │
│  │ 确定旋转角度         │    │ 确定旋转角度         │            │
│  │ (0°/90°/180°/270°) │    │ (0°/90°/180°/270°) │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           ▼                         ▼                         │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 超分辨率放大 (×4)    │    │ 超分辨率放大 (×4)    │            │
│  │ → GaussianBlur     │    │ → GaussianBlur     │            │
│  │ → Otsu → 形态学     │    │ → Otsu → 形态学     │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           ▼                         ▼                         │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 连通域评分           │    │ 连通域评分           │            │
│  │ (面积/矩形度/        │    │ (面积/矩形度/        │            │
│  │  中心距/长宽比)      │    │  中心距/长宽比)      │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           ▼                         ▼                         │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ minAreaRect → 4顶点│    │ minAreaRect → 4顶点│            │
│  │ cornerSubPix 亚像素 │    │ cornerSubPix 亚像素 │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           ▼                         ▼                         │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 角度对齐 (模板角序)  │    │ 角度对齐 (模板角序)  │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           │    ┌────────────────────┘                         │
│           │    │                                              │
│           ▼    ▼                                              │
│  ┌───────────────────────────────────────────┐               │
│  │ 视差计算: disparity = -(left.x - right.x)  │               │
│  └────────────────────┬──────────────────────┘               │
│                       │                                       │
│                       ▼                                       │
│  ┌───────────────────────────────────────────┐               │
│  │ cv::solvePnP(ITERATIVE)                    │               │
│  │ 4 个图像角点 ↔ 4 个正方形物点                │               │
│  │ (物点间距 = square_size_m)                 │               │
│  └────────────────────┬──────────────────────┘               │
│                       │                                       │
│                       ▼                                       │
│                ┌─────────────┐                               │
│                │ 输出 [R|t]   │                               │
│                └─────────────┘                               │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 位姿解算

TinyTarget **不走 GPNP 优化**，而是直接用 OpenCV 的 `cv::solvePnP(cv::SOLVEPNP_ITERATIVE)` 求解 PnP：

- 4 个图像角点 ←→ 4 个正方形物点（边长 = `square_size_m` 米）
- 物点坐标固定：`(0,0,0)`, `(s,0,0)`, `(s,s,0)`, `(0,s,0)`
- 无初值要求，直接迭代求解

---

## 4. 策略二：BinaryCorner（ROI 801 ~ 40000 px²）

**适用场景**：中等尺寸多边形目标（如 10 角点标识板）。

**模块映射**：`BinaryCornerExtractor` → `InitialPnPSolver` → `GPnPSolver`

### 4.1 完整流水线

```
┌──────────────────────────────────────────────────────────────┐
│                   BinaryCorner 流水线                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  左 ROI 灰度图                   右 ROI 灰度图                 │
│       │                              │                        │
│       ▼                              ▼                        │
│  ┌─────────────┐              ┌─────────────┐                │
│  │ Otsu 二值化  │              │ Otsu 二值化  │               │
│  └──────┬──────┘              └──────┬──────┘                │
│         │                            │                        │
│         ▼                            ▼                        │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 保留最大连通域       │    │ 保留最大连通域       │            │
│  │ → 填充孔洞          │    │ → 填充孔洞          │            │
│  │ → 形态学平滑         │    │ → 形态学平滑         │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           ▼                         ▼                         │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 模板匹配 (IoU)      │    │ 模板匹配 (IoU)      │            │
│  │ 确定旋转角度         │    │ 确定旋转角度         │            │
│  │ (从文件名解析)       │    │ (从文件名解析)       │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           ▼                         ▼                         │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 旋转回正             │    │ 旋转回正             │            │
│  │ → 提取最大轮廓       │    │ → 提取最大轮廓       │            │
│  │ → approxPolyDP      │    │ → approxPolyDP      │            │
│  │   二分搜索 N 角点    │    │   二分搜索 N 角点    │            │
│  │ → 逆旋转回原角度     │    │ → 逆旋转回原角度     │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           ▼                         ▼                         │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ 模板角点重排序       │    │ 模板角点重排序       │            │
│  │ (reorderByGeometry  │    │ (reorderByGeometry  │            │
│  │  极角对齐)           │    │  极角对齐)           │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           │    ┌────────────────────┘                         │
│           │    │                                              │
│           ▼    ▼                                              │
│  ┌───────────────────────────────────────────┐               │
│  │ 视差计算: disparity = -(left.x - right.x)  │               │
│  │ (全图坐标下直接计算)                        │               │
│  └────────────────────┬──────────────────────┘               │
│                       │                                       │
│                       ▼                                       │
│  ┌───────────────────────────────────────────┐               │
│  │ 位姿解算                                    │               │
│  │                                             │               │
│  │  ┌─ 首帧 ──────────────────────────────┐   │               │
│  │  │ ① InitialPnP (RANSAC PnP + 精化)    │   │               │
│  │  │   成功 → GPNP warm-start             │   │               │
│  │  │   失败 → 视差估算深度 (500mm 默认)    │   │               │
│  │  └─────────────────────────────────────┘   │               │
│  │  ┌─ 后续帧 ────────────────────────────┐   │               │
│  │  │ GPNP 优化 (Eigen Levenberg-Marquardt)│   │               │
│  │  │ 初值 = 上一帧位姿                      │   │               │
│  │  └─────────────────────────────────────┘   │               │
│  └────────────────────┬──────────────────────┘               │
│                       │                                       │
│                       ▼                                       │
│                ┌─────────────┐                               │
│                │ 输出 [R|t]   │                               │
│                └─────────────┘                               │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 位姿解算

**首帧**：
1. 若 `use_initial_pnp=true`，运行 `InitialPnPSolver`（RANSAC PnP 300 次迭代 + ITERATIVE 精化），成功后作为 GPNP 的 warm-start
2. 若 InitialPnP 失败或 `use_initial_pnp=false`，则由视差中位数估算深度 `depth = f·b/median_disp`，clamp 至 [50, 5000] mm 作为初值 `(0,0,depth)`

**后续帧**：直接用上一帧位姿作为 GPNP 初值

**GPNP 求解器** (`GPnPSolver`)：基于 **Eigen Levenberg-Marquardt** 非线性优化，最小化双目交叉射线（cross-product）残差：

```
残差 = cross(P_c1 - origin, direction)   // 左射线部分
     + cross(P_c1 - t_rl_cam, direction) // 右射线部分 (变换到左相机坐标系)
```

优化参数 `[qx, qy, qz, qw, tx, ty, tz]`（7 维），Jacobian 由 Eigen 自动数值差分计算。

---

## 5. 策略三：AKAZE_GPNP（ROI ≥ 40001 px² / 无检测回退）

**适用场景**：大尺寸 / 全图纹理丰富的目标（传统 AKAZE 特征匹配方案）。

**模块映射**：`AkazeGpnpExtractor` → `OpticalFlowTracker` → `StereoProjector` → `TemplateMatcher` → `MadDisparityFilter` → `InitialPnPSolver` → `GPnPSolver`

### 5.1 完整流水线

```
┌──────────────────────────────────────────────────────────────┐
│                    AKAZE_GPNP 流水线                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  左灰度图 (resize×scale)          右灰度图 (resize×scale)     │
│       │                              │                        │
│       ▼                              ▼                        │
│  ┌────────────────────┐    ┌────────────────────┐            │
│  │ AKAZE 特征提取      │    │ AKAZE 特征提取      │            │
│  │ • cv::AKAZE::create │    │ • cv::AKAZE::create │            │
│  │ • detectAndCompute  │    │ • detectAndCompute  │            │
│  │ • N×61 二值描述子   │    │ • 坐标还原 ÷scale   │            │
│  └────────┬───────────┘    └────────┬───────────┘            │
│           │                         │                         │
│           ▼                         │                         │
│  ┌──────────────────────────────────────────┐                │
│  │ LK 金字塔光流 L→R (OpticalFlowTracker)    │                │
│  │ • winSize=21×21, maxLevel=3               │                │
│  │ • Forward-Backward 一致性校验 (fb<1.0px)  │                │
│  │ • MAD 视差异常点滤波                        │                │
│  │   (|dx - median| < 3 × max(MAD, 1.0))     │                │
│  │ • 劣化处理：匹配点 < 3 → 回退到未滤波数据  │                │
│  └────────────────────┬─────────────────────┘                │
│                       │                                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────┐                │
│  │ 立体投影 (StereoProjector)                │                │
│  │ • 视差 → 深度: depth = f·b / |disp|       │                │
│  │ • 左相机射线 → 3D 点 → 右相机投影          │                │
│  │ • 有效性检查: disparity > 0, Pz > 0       │                │
│  └────────────────────┬─────────────────────┘                │
│                       │                                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────┐                │
│  │ 模板匹配 (TemplateMatcher)                │                │
│  │ ┌─ Stage 1: Ratio Test ────────────────┐ │                │
│  │ │ KNN k=2, threshold=0.75              │ │                │
│  │ │ <4 matches → return                  │ │                │
│  │ └──────────────────────────────────────┘ │                │
│  │ ┌─ Stage 2: Cross-Check ───────────────┐ │                │
│  │ │ 模板→图像方向 Ratio Test + 对称性验证  │ │                │
│  │ │ <4 matches → return                  │ │                │
│  │ └──────────────────────────────────────┘ │                │
│  │ ┌─ Stage 3: Homography RANSAC ─────────┐ │                │
│  │ │ cv::findHomography(RANSAC, 5.0px)     │ │                │
│  │ │ 保留 inlier 内点                       │ │                │
│  │ └──────────────────────────────────────┘ │                │
│  └────────────────────┬─────────────────────┘                │
│                       │                                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────┐                │
│  │ MAD 视差滤波 (MadDisparityFilter)         │                │
│  │ 将投影步骤的 valid_mask 与 MAD 过滤结果同步 │                │
│  └────────────────────┬─────────────────────┘                │
│                       │                                       │
│                       ▼                                       │
│  ┌──────────────────────────────────────────┐                │
│  │ 位姿解算                                   │                │
│  │ ┌─ 首帧 ───────────────────────────────┐ │                │
│  │ │ InitialPnP (RANSAC PnP 300+精化)      │ │                │
│  │ │ 成功 → GPNP warm-start                │ │                │
│  │ │ 跳过 → 原始 GPNP (t=[0,0,5000]mm)     │ │                │
│  │ └──────────────────────────────────────┘ │                │
│  │ ┌─ 后续帧 ─────────────────────────────┐ │                │
│  │ │ GPNP (Eigen LM) warm-start = 上一帧   │ │                │
│  │ └──────────────────────────────────────┘ │                │
│  └────────────────────┬─────────────────────┘                │
│                       │                                       │
│                       ▼                                       │
│                ┌─────────────┐                               │
│                │ 输出 [R|t]   │                               │
│                └─────────────┘                               │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 关键模块说明

| 模块 | 文件 | 职责 |
|------|------|------|
| `AkazeGpnpExtractor` | `src/feature/AkazeGpnpExtractor.cpp` | AKAZE 提取 + 光流 + 投影 + 模板匹配一次性完成 |
| `OpticalFlowTracker` | `src/feature/OpticalFlowTracker.cpp` | LK 金字塔光流 + FB 一致性检查，`fb_err < 1.0px` |
| `MadDisparityFilter` | `src/feature/MadDisparityFilter.cpp` | 中值绝对偏差（MAD）异常值剔除，`dx_thresh = 3×max(MAD,1.0)` |
| `StereoProjector` | `src/stereo/StereoProjector.cpp` | 视差 → 深度 → 右图投影，输出 valid_mask |
| `TemplateMatcher` | `src/matching/TemplateMatcher.cpp` | Ratio Test → Cross-Check → Homography RANSAC 三阶段级联 |
| `InitialPnPSolver` | `src/pose/InitialPnPSolver.cpp` | RANSAC PnP (300 iter, 8.0px) → ITERATIVE 精化 |
| `GPnPSolver` | `src/pose/GPnPSolver.cpp` | Eigen LM 优化，交叉射线残差，7 维参数空间 |

### 5.3 位姿有效性检查

GPNP 输出位姿需通过以下全部校验：

1. `t[2] > 0` — 相机在目标平面前方
2. `10 < |t| < 20000` mm — 深度在合理范围
3. 所有分量有限（无 NaN/Inf）
4. 旋转矩阵有限

---

## 6. 三策略对比总结

| 维度 | TinyTarget | BinaryCorner | AKAZE_GPNP |
|------|-----------|-------------|------------|
| **ROI 阈值** | ≤ 800 px² | 801 ~ 40000 px² | ≥ 40001 px² |
| **特征提取器** | `TinyTargetExtractor` | `BinaryCornerExtractor` | `AkazeGpnpExtractor` |
| **特征类型** | 4 角点 (minAreaRect) | N 角点 (approxPolyDP) | AKAZE 关键点 + 61 维二值描述子 |
| **左右匹配** | 模板匹配 L↔R | 模板匹配 L↔R | LK 光流 L→R + FB 校验 |
| **视差滤波** | 无 | 无 | MAD (median ± 3σ) |
| **位姿解算** | `cv::solvePnP(ITERATIVE)` | `GPnPSolver` (Eigen LM) | `GPnPSolver` (Eigen LM) |
| **物点来源** | 4 正方形顶点 (固定) | 模板角点 (像素→实际尺寸) | 模板 AKAZE 关键点 (像素→mm) |
| **首帧初值** | 无需 | InitialPnP / 深度估算 | InitialPnP / 默认 5000mm |
| **可视化** | 标准 solvePnP 输出 | 5 面板 (二值/轴系/模板/立体/重投影) | 4 面板 (特征/立体/模板/坐标轴) |

---

## 7. 核心数据结构

所有模块间数据传递使用强类型结构体，定义于 [`include/common/Types.hpp`](include/common/Types.hpp)：

| 结构体 | 用途 |
|--------|------|
| `StereoCameraParams` | 双目相机内外参 (K, R_rl, t_rl, focal_length, baseline) |
| `TrackerConfig` | 跟踪器配置 (scale, gpnp_min_pts, use_initial_pnp, LK 参数) |
| `PipelineResult` | 单帧完整输出: 特征、光流、投影、匹配、位姿、计时 |
| `TrackResult` | 光流跟踪结果: 左右匹配点、视差、FB 统计 |
| `ProjectionResult` | 立体投影结果: 投影点、valid_mask |
| `MatchResult` | 模板匹配结果: good_matches、匹配统计 |
| `PoseEstimate` | 位姿估计: R, t, success, num_points |
| `GPNPMonitor` | GPNP 优化诊断: 成本、迭代、失败原因 |
| `LogEntry` | 单帧日志: 特征数、匹配数、视差中位数、耗时 |
| `TrackingState` | 帧间状态: 上帧位姿缓存、帧计数、日志列表 |
| `RoiRect` | ROI 矩形: (x, y, width, height) |
| `MadFilterResult` | MAD 滤波输出: 过滤后点集、filter_mask、劣化标志 |

---

## 8. 配置文件

配置文件位于 [`config/tracker_config.json`](config/tracker_config.json)，结构如下：

```jsonc
{
  "camera": {
    "fx": 1400.0,          // 焦距 × 像素/mm
    "fy": 1400.0,
    "cx": 960.0,           // 主点坐标
    "cy": 540.0,
    "baseline_mm": 120.0   // 双目基线 (mm)
  },
  "strategies": {
    "akaze_gpnp": {        // AKAZE_GPNP 策略
      "template_path": "data/模板/akaze_template.png",
      "template_real_width_mm": 200.0,
      "template_real_height_mm": 200.0,
      "scale": 0.5,        // AKAZE 图像缩放因子
      "gpnp_min_pts": 4,   // GPNP 最少匹配点数
      "use_initial_pnp": true
    },
    "binary_corner": {     // BinaryCorner 策略
      "corners": 10,       // 目标角点数
      "kernel_size": 3,
      "corner_scale": 0.5,
      "target_width": 200,
      "target_height": 200,
      "pixel_to_meter_scale": 0.001,
      "roi_pad_pixels": 3,  // ROI 外扩像素
      "otsu_ratio": 0.5,
      "template_dir": "data/NewMuBan/"
    },
    "tiny_target": {       // TinyTarget 策略
      "target_width": 40,
      "target_height": 40,
      "scale_factor": 4.0,  // 超分辨率放大倍率
      "square_size_m": 0.05,// 正方形边长 (米)
      "roi_pad_pixels": 5,
      "template_dir": "data/小图/"
    }
  },
  "yolo": {                // YOLO 目标检测
    "model_path": "best.onnx",
    "conf_threshold": 0.5,
    "target_class_id": 0,
    "roi_expand_ratio": 0.1,
    "roi_min_size": 100
  },
  "input": {
    "left": "data/left.png",
    "right": "data/right.png"
  },
  "output": {
    "visualize": true
  }
}
```

---

## 9. 构建与运行

### 依赖

| 库 | 用途 |
|----|------|
| OpenCV 4.x (`opencv2/core`, `features2d`, `calib3d`, `imgproc`, `imgcodecs`) | AKAZE、光流、PnP、图像处理 |
| Eigen 3.x | 线性代数、GPNP LM 优化 |
| ONNX Runtime | YOLO 模型推理 |

### 编译

```bash
git clone https://github.com/Chihaya-anon343/Steretracker.git
cd Steretracker
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### 运行

```bash
# 使用默认配置文件 config/tracker_config.json
./build/Steretracker

# 指定配置文件
./build/Steretracker path/to/config.json
```

### 输出

- **控制台**: 每帧处理摘要（特征点、匹配、投影、视差中位数、GPNP 状态、耗时）
- **日志表**: 所有帧的详细统计（`tracker.printLogs()`）
- **output/ 目录**: 可视化调试图像（若 `visualize=true`）
  - AKAZE: `akaze_***.png`
  - BinaryCorner: `binary_corner_***.png`（5 面板）
  - TinyTarget: `tiny_target_***.png`

---

## 10. 项目目录结构

```
Steretracker/
├── README.md                          # 项目说明（本文件）
├── PORTING_REPORT.md                  # Python→C++ 移植报告
├── FEATURE_EXTRACTION_SPEC.md         # 特征提取策略详细设计文档
├── CMakeLists.txt                     # CMake 构建配置
├── main.cpp                           # 程序入口（YOLO + 策略分发 + 逐帧循环）
├── best.onnx                          # YOLO ONNX 模型
│
├── config/
│   └── tracker_config.json            # 默认配置文件
│
├── data/                              # 测试图像与模板
│   ├── 大图/                          # AKAZE 场景
│   ├── 中图/                          # BinaryCorner 场景
│   ├── 小图/                          # TinyTarget 场景
│   ├── NewMuBan/                      # BinaryCorner 模板库
│   ├── NewMuBan(reordered)/
│   └── rotated/
│
├── include/
│   ├── common/
│   │   ├── Types.hpp                  # 12 个强类型结构体
│   │   ├── Config.hpp                 # 配置校验与工厂函数
│   │   └── GeometryUtils.hpp          # 几何工具函数 (inline)
│   ├── detection/
│   │   ├── YoloDetector.hpp           # ONNX YOLO 推理
│   │   ├── YoloRoiProvider.hpp        # 检测→ROI 转换 + ROI 配置
│   │   └── RoiGenerator.hpp           # ROI 生成器接口
│   ├── feature/
│   │   ├── FeatureExtractor.hpp       # 特征提取器抽象基类
│   │   ├── AkazeGpnpExtractor.hpp     # AKAZE 提取 + 光流 + 投影 + 匹配
│   │   ├── BinaryCornerExtractor.hpp  # 二值轮廓角点提取
│   │   ├── TinyTargetExtractor.hpp    # 微小矩形目标角点提取
│   │   ├── FeatureExtractorFactory.hpp# 策略工厂（按 ROI 面积分发）
│   │   ├── OpticalFlowTracker.hpp     # LK 光流 + FB 校验
│   │   └── MadDisparityFilter.hpp     # MAD 视差滤波
│   ├── matching/
│   │   └── TemplateMatcher.hpp        # 三阶段模板匹配
│   ├── stereo/
│   │   └── StereoProjector.hpp        # 视差→深度→投影
│   ├── pose/
│   │   ├── InitialPnPSolver.hpp       # RANSAC+ITERATIVE 初始 PnP
│   │   └── GPnPSolver.hpp            # Eigen LM GPNP 非线性优化
│   ├── visualization/
│   │   └── Visualizer.hpp            # 4 面板调试图像
│   ├── tracker/
│   │   └── StereoTracker.hpp         # 协调器（薄封装，~200 行）
│   └── utils/
│       └── Timing.hpp                 # 计时工具
│
├── src/                               # 对应 .cpp 实现
│   ├── detection/  (YoloDetector + YoloRoiProvider + RoiGenerator)
│   ├── feature/    (AkazeGpnpExtractor + BinaryCornerExtractor
│   │                + TinyTargetExtractor + FeatureExtractorFactory
│   │                + OpticalFlowTracker + MadDisparityFilter)
│   ├── matching/   (TemplateMatcher)
│   ├── stereo/     (StereoProjector)
│   ├── pose/       (InitialPnPSolver + GPnPSolver)
│   ├── visualization/ (Visualizer)
│   └── tracker/    (StereoTracker)
│
└── test/
    └── test_stereo_tracker.cpp         # 单元测试