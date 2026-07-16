# Project Structure

本文档说明 `fake_video_structural_anomaly` 当前工程目录及主要文件用途。目录中的
虚拟环境、缓存、模型权重、视频、抽帧结果和实验输出不逐文件展开，因为它们属于本地
运行资源或可重复生成的产物。

## 1. Overall Layout

```text
fake_video_structural_anomaly/
├── configs/                    # 实验配置和尺度先验
├── data/                       # 合成、手工及本地视频输入
├── docs/                       # 开发日志和工程文档
├── examples/                   # 可直接运行的最小示例
├── scripts/                    # 数据构建、批量实验和诊断脚本
├── src/
│   ├── semantic3d/             # 当前主要实现
│   └── structural_anomaly/     # 早期轻量实现，保留兼容和对照
├── tests/                      # pytest 自动化测试
├── outputs/                    # 实验生成结果，不提交 Git
├── checkpoints/               # 本地模型权重，不提交 Git
├── .venv/                      # 项目 Python 环境，不提交 Git
├── .gitignore                  # Git 忽略规则
├── CHANGELOG.md                # 面向版本的变化摘要
├── README.md                   # 项目入口说明
└── pyproject.toml              # Python 包、依赖和 pytest 配置
```

## 2. Core Package: `src/semantic3d`

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | `semantic3d` 公共 API 入口，使用延迟导入暴露主要数据结构和函数。 |
| `scale_depth.py` | 实现对象等效投影尺度、比例空间与 log-space 的尺度—深度残差 `R_sd`，以及帧内两两残差矩阵。 |
| `scale_prior.py` | 读取尺度先验 YAML，标准化类别名，解析 exact/alias/reliable/missing 四种先验状态。 |
| `strict_scale_prior.py` | 读取冻结的严格物理尺度先验，仅允许通过 `reliable_single` 自动审核的类别参与主 `R_sd`。 |
| `empirical_pair_prior.py` | 预留经验类别对混合分布先验的抽象接口；严格物理基线不会实例化或自动回退到该接口。 |
| `projected_measurement.py` | 实现归一化 bbox 高度/宽度、等效直径和 bbox 对角线四类二维投影量及 compatibility group。 |
| `physical_prior_gate.py` | 使用置信度、边界、尺寸、面积、宽高比、深度和可选轨迹稳定性执行启发式观测质量门控。 |
| `dimension_aligned_scale_depth.py` | 实现 strict v2 先验解析、同维度兼容检查、条件门控和 ratio/log dimension-aligned `R_sd`。 |
| `rsd_v2_error_audit.py` | 提供 strict v2 只读误差归因所需的稳定 pair ID、显式公式复算、安全 bbox 裁剪和多种对象区域深度统计。 |
| `keypoint_provider.py` | 定义统一二维关键点结构和 Base/Mock/Ultralytics 人体姿态 provider；按已有 person bbox 匹配姿态，不实现三维反投影。 |
| `pose_applicability.py` | 判断 person bbox 是否可代表完整直立身高，并对 cup upright height 执行尺寸、边界、稳定性和深度质量门控。 |
| `depth_provider.py` | 定义深度 provider；包含测试用 mock depth、真实单目相对深度推理、深度方向转换、bbox 区域中位深度统计和深度图保存。 |
| `depth_temporal_consistency.py` | 实现跨帧深度—投影联合一致性残差 `R_depth_cons`、transition 校验、轨迹/片段汇聚及 CSV 驱动可视化。 |
| `object_association.py` | 合并重叠 clip 中的重复帧，并按类别、IoU 和中心距离将跨帧 detection 关联成唯一 track。 |
| `observations.py` | 定义可 JSON 序列化的对象、帧、片段和残差结果 dataclass；可选保存二维关键点、provider、全局帧号和 person track id。 |
| `io.py` | 保存和读取 frame/clip observation 及 clip residual result JSON，自动创建父目录。 |
| `data_structures.py` | 定义早期通用 `FrameObservation`、`ClipObservation`、`ObjectTrack` 和 `ResidualReport` 数据结构。 |
| `interfaces.py` | 定义分割、深度、光流和跟踪 provider 的抽象接口，不绑定具体模型。 |
| `providers.py` | 定义对象 provider 基类，以及用于闭环测试的 `MockObjectProvider` 和 mock 深度规则。 |
| `real_object_provider.py` | 使用 Ultralytics YOLO 进行真实目标检测，将 bbox 面积近似为 mask area，并输出统一对象 observation。 |
| `provider_registry.py` | 按名称创建 `mock` 或 `real_detector` 对象 provider。 |
| `build_observations.py` | 从图像、对象 provider 和深度 provider 构建 frame observation，再组织为 clip observation。 |
| `video_preprocess.py` | 使用 OpenCV 抽帧，并按 `clip_len` 与 `stride` 构建滑动窗口片段。 |
| `aggregation.py` | 提供 mean、max、top-k mean、mask 区域和点集残差汇聚函数。 |
| `multilevel_residuals.py` | 定义对象 mask、对象级残差、对象对残差和 clip 汇总，并完成多粒度残差组装。 |
| `residual_fusion.py` | 定义六类残差值/权重、归一化、加权融合和视频级 mean/max/top-k 风险评分。 |
| `visualization.py` | 绘制对象残差热力图、对象对关系图、mask 热力图和多粒度综合图。 |

当前深度约定：项目内 `depth` 数值越大表示对象越远。真实单目模型输出的是相对深度，
不是米制绝对深度；当前真实实验默认使用反向后的 `real_depth_invert`。

## 3. Compatibility Package: `src/structural_anomaly`

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 暴露早期原型 API。 |
| `scale_depth.py` | 早期支持 NumPy/可选 Torch Tensor 的 `R_sd` 实现。 |
| `fusion.py` | 早期 `ResidualBundle` 和简单加权融合接口。 |

新实验主要使用 `semantic3d`。该目录暂时保留，避免旧示例或外部调用失效。

## 4. Configuration

| 文件 | 作用 |
| --- | --- |
| `configs/scale_priors.yaml` | 保存物体物理尺度区间、`reliable` 标记和类别 alias，是 `R_sd` 的先验来源。 |
| `configs/scale_priors_candidates.yaml` | 独立于评估视频的物理尺度资料候选库，记录统一特征尺度、来源、估计方法和审核输入。 |
| `configs/scale_priors_strict_v1.yaml` | 冻结的 strict physical v1；只启用 `reliable_single`，后续修订必须新建 v2。 |
| `configs/scale_priors_strict_v2.yaml` | 冻结的 dimension-aligned strict v2；分别标记 strict_high、conditional、pose-sensitive 和 unsupported。 |
| `configs/projected_measurement_rules.yaml` | 固定四种投影量公式、compatibility group 和通用几何有效性阈值。 |
| `configs/structural_residual_eval.yaml` | 固定 manifest 先导评估的视频采样、真实 provider、深度方向、关联与残差参数。 |
| `pyproject.toml` | 定义项目包信息、基础依赖、YOLO/depth 可选依赖、源码目录和 pytest 配置。 |
| `.gitignore` | 排除 `.venv`、缓存、模型权重、视频和 `outputs` 等本地大文件。 |
| `.vscode/settings.json` | 当前工作区的 VS Code 配置。 |

## 5. Experiment Scripts: `scripts`

### Environment and Smoke Tests

| 文件 | 作用 |
| --- | --- |
| `run_tests.sh` | 使用项目 `.venv` 运行测试的统一入口。 |
| `check_ultralytics_env.py` | 检查 Python、Ultralytics、Torch、CUDA 和 YOLO 权重状态。 |
| `yolo_video_smoke_test.py` | 对少量视频帧运行 YOLO，打印 detection 并保存带框图片。 |
| `check_depth_model_env.py` | 检查 Torch、Transformers、Pillow 和 NumPy 深度推理环境。 |
| `depth_smoke_test.py` | 对图片或视频帧运行真实单目深度模型，输出深度统计和深度可视化。 |

### Observation Construction

| 文件 | 作用 |
| --- | --- |
| `build_observations_from_video.py` | 使用 mock provider 完成视频抽帧、clip 划分和 observation JSON 构建。 |
| `build_real_object_observations_from_video.py` | 使用 YOLO 和可选真实/mock/none 深度 provider 构建真实 observation JSON 与深度图。 |

### Scale-Depth Residual

| 文件 | 作用 |
| --- | --- |
| `run_scale_depth_experiment.py` | 对 30 个合成案例批量计算 ratio/log `R_sd`、分类指标、CSV 和柱状图。 |
| `run_manual_observation_rsd.py` | 对手工 observation 样例批量计算 `R_sd` 并输出 CSV/PNG。 |
| `run_observation_rsd_pipeline.py` | 读取 clip observation，通过尺度先验 resolver 安全计算真实帧对象对 `R_sd` 和 clip score。 |

### Depth Direction and Prior Maintenance

| 文件 | 作用 |
| --- | --- |
| `compare_depth_direction_rsd.py` | 对单个视频比较 no-depth、no-invert 和 invert-depth 三种 `R_sd` 结果。 |
| `batch_compare_depth_direction_rsd.py` | 批量处理视频目录，汇总深度方向结果并调用尺度先验覆盖统计。 |
| `visualize_depth_direction_debug.py` | 对照显示原帧、两种深度方向、对象深度和 `R_sd` 关系边。 |
| `report_missing_scale_priors.py` | 扫描 observation 类别，统计 exact、alias、missing 和 unreliable 覆盖状态。 |
| `generate_scale_prior_candidates.py` | 从缺失报告生成待人工审核的候选 YAML，不直接修改主先验配置。 |
| `audit_physical_scale_priors.py` | 将候选资料展开为 source report，并按可复现规则生成可靠性 audit report。 |
| `freeze_strict_scale_priors.py` | 从候选和审核结果生成只写一次的严格先验版本，拒绝覆盖已有冻结文件。 |
| `run_strict_rsd_baseline.py` | 以显式冻结先验运行 strict `R_sd`，保留每个候选 pair 的 NaN、skip reason、覆盖率和解释。 |
| `run_strict_rsd_v2.py` | 执行 dimension-aligned v2，分别汇聚 strict_high 与 conditional_physical 证据。 |
| `compare_strict_rsd_versions.py` | 比较 v1、v2 strict_high、v2 conditional 和 v2 all-available 的 pair/video 覆盖。 |
| `audit_strict_rsd_v2_errors.py` | 对指定视频类别对执行帧级去重、公式核验、深度/投影策略对比、轨迹级诊断和可解释图；不修改冻结先验或正式 `R_sd`。 |
| `run_rsd_applicability_gate_eval.py` | 使用真实人体关键点和 cup 质量门控比较 strict v2 前后证据覆盖；失败 residual 保持 NaN。 |

### Temporal Depth Consistency

| 文件 | 作用 |
| --- | --- |
| `run_depth_consistency_pipeline.py` | 去重全局帧、关联对象、计算 transition `R_depth_cons`，输出 pair/track/clip CSV 和关联结果。 |
| `run_depth_consistency_sensitivity.py` | 扫描 tolerance，分析阈值对有效异常响应的影响。 |
| `generate_depth_consistency_perturbations.py` | 对指定轨迹注入深度、投影面积或联合扰动。 |
| `run_depth_consistency_perturbation_experiment.py` | 批量运行扰动实验，验证残差对控制变量的响应。 |
| `visualize_depth_consistency_analysis.py` | 从 transition CSV 绘制 raw residual、阈值后 residual 和综合分析图。 |

### Manifest-driven Pilot Evaluation

| 文件 | 作用 |
| --- | --- |
| `build_video_manifest.py` | 从现有 real/fake 视频目录生成、验证 6-video manifest 和 2-video smoke manifest。 |
| `run_batch_structural_residual_eval.py` | 按 manifest 逐视频运行 `R_sd`、`R_depth_cons` 并输出 video/clip/quality 表。 |
| `analyze_structural_residual_distributions.py` | 生成残差分布、有效证据、尺度先验覆盖与 track quality 图。 |

## 6. Runnable Examples: `examples`

| 文件 | 作用 |
| --- | --- |
| `_bootstrap.py` | 让示例优先使用项目 `.venv`，并设置项目源码路径。 |
| `demo_scale_depth.py` | 最早的足球—大象尺度深度 demo。 |
| `demo_scale_depth_residual.py` | 展示合理与异常对象对的 ratio/log `R_sd` 详细计算。 |
| `demo_visualize_scale_depth.py` | 用合成 mask 输出 `R_sd` 对象图、关系图和热力图。 |
| `demo_residual_fusion.py` | 展示多个残差项的归一化、融合和视频评分。 |
| `demo_multilevel_residuals.py` | 构造对象级、对象对级和 clip 级残差汇聚结果。 |
| `demo_multilevel_visualization.py` | 生成对象残差图、对象对关系图和 16:9 综合图。 |
| `demo_observation_io.py` | 保存/读取两帧 clip JSON，并从读取结果计算 `R_sd`。 |
| `demo_manual_observation_rsd.py` | 读取一个手工 observation，打印对象几何和 `R_sd` 并画关系图。 |
| `demo_video_observation_pipeline.py` | 自动生成小视频，演示 mock 视频 observation 完整闭环。 |
| `demo_real_object_observation_pipeline.py` | 演示真实视频经 YOLO 生成 observation 并进入 `R_sd` 管线。 |
| `demo_real_depth_observation_pipeline.py` | 演示 YOLO + 真实单目相对深度 + bbox 中位深度 + `R_sd` 闭环。 |
| `demo_depth_consistency.py` | 用稳定、合理接近、深度跳变和投影跳变轨迹展示 `R_depth_cons`。 |

## 7. Tests: `tests`

| 文件 | 验证范围 |
| --- | --- |
| `_bootstrap.py` | 测试环境和 `src` 导入路径初始化。 |
| `test_scale_depth.py` | `R_sd` 区间、ratio/log 行为、非法输入和 pairwise 矩阵。 |
| `test_scale_depth_experiment.py` | 30 案例批量实验、CSV、PNG、指标和非法案例处理。 |
| `test_scale_prior.py` | exact/alias/missing/unreliable 解析、覆盖报告及 candidate YAML。 |
| `test_residual_fusion.py` | 残差独立字段、归一化、加权融合和视频风险评分。 |
| `test_multilevel_residuals.py` | mask/点集汇聚、对象级与对象对级残差及 clip 汇总。 |
| `test_visualization.py` | 三类多粒度图、空输入、自动建目录和阈值边显示。 |
| `test_data_structures.py` | 通用帧、clip、track 和 report 数据结构校验。 |
| `test_observations_io.py` | Observation JSON 的帧/clip/结果往返读写和必填字段错误。 |
| `test_manual_observation_rsd.py` | 手工 observation 读取、合理/异常分离及 CSV/PNG。 |
| `test_video_observation_pipeline.py` | 视频抽帧、clip 窗口、mock provider 和 observation 管线。 |
| `test_real_object_provider.py` | YOLO provider 导入、bbox 面积、label mapping、registry 和样本推理。 |
| `test_depth_provider.py` | mock/真实深度接口、bbox median、无效 bbox、depth 写回 observation。 |
| `test_depth_direction_comparison.py` | 单视频 no-depth/no-invert/invert 比较流程。 |
| `test_depth_direction_batch.py` | 多视频深度方向批处理、汇总 CSV 和可视化。 |
| `test_object_association.py` | IoU/中心匹配、track 唯一性、重复帧去重和旧 JSON 兼容。 |
| `test_depth_temporal_consistency.py` | `R_depth_cons` 公式、有效/无效 transition、CSV、去重和绘图一致性。 |
| `test_depth_consistency_sensitivity.py` | tolerance 扫描结果和阈值建议。 |
| `test_depth_consistency_perturbations.py` | 深度/面积/联合扰动及非目标轨迹不变性。 |
| `test_batch_structural_residual_eval.py` | 实际视频目录、manifest、正式配置、NaN 证据语义、批量表和六张分析图。 |
| `test_strict_rsd_baseline.py` | 严格物理先验审核、exact/alias、missing/unreliable NaN、覆盖聚合、解释、绘图和经验接口隔离。 |
| `test_projected_measurement.py` | 四类投影量、compatibility group、非法 bbox 和 NaN 行为。 |
| `test_physical_prior_gate.py` | 通用观测门控的通过、低置信度、边界截断、宽高比和无效深度行为。 |
| `test_dimension_aligned_rsd.py` | 同维度计算、跨维度跳过、conditional tier、gate 失败和姿态敏感先验。 |
| `test_strict_rsd_v2.py` | v1 哈希冻结、v2 独立配置、tier 分离汇聚、目录隔离和 v1/v2 对比产物。 |
| `test_rsd_v2_error_audit.py` | pair ID、重叠 clip 去重、公式/交换不变性、深度策略、NaN、诊断图、报告和 v1/v2 哈希冻结。 |
| `test_pose_applicability.py` | 直立/坐姿/弯身/缺踝/边界/低置信度、cup 质量、NaN、旧 JSON、真实姿态 smoke 和冻结哈希。 |

## 8. Data: `data`

```text
data/
├── manifests/
│   ├── pilot_real_fake.csv             # 4 real + 2 fake 先导评估清单
│   └── pilot_smoke_2video.csv          # 1 real + 1 fake smoke 清单
├── demo_observations/
│   └── clip_001.json                  # 两帧 observation IO 示例
├── manual_observations/
│   ├── manual_001...json              # person-car 合理案例
│   ├── manual_002...json              # cup-person 合理案例
│   ├── manual_003...json              # soccer-elephant 合理案例
│   ├── manual_004...json              # soccer-elephant 异常案例
│   └── manual_005...json              # cup-car 异常案例
├── synthetic_observations/
│   ├── case_001...case_012             # 合理案例
│   ├── case_013...case_022             # 异常案例
│   ├── case_023...case_026             # 边界案例
│   └── case_027...case_030             # 非法输入案例
├── tests_videos/
│   ├── tests_real_videos/real_1...4.mp4
│   └── tests_fake_videos/fake_1...2.mp4
├── videos/test_real.mp4               # 单视频真实闭环输入
└── demo_videos/demo.mp4               # 自动/手工 demo 视频
```

视频文件由 `.gitignore` 排除。迁移服务器时需要通过数据盘、SCP 或对象存储单独传输。

## 9. Generated Outputs: `outputs`

| 目录 | 内容 |
| --- | --- |
| `frames/` | OpenCV 抽出的原始视频帧。 |
| `depth_maps/` | 深度数组 `.npy` 和可视化图片。 |
| `observations/` | mock observation clip JSON。 |
| `real_observations/` | YOLO detection observation JSON。 |
| `real_observations_depth/` | YOLO + 深度写回后的 observation JSON。 |
| `observations_with_tracks/` | 去重并分配 `track_id` 后的 observation。 |
| `results/` | `R_sd`、`R_depth_cons`、阈值、尺度先验覆盖和实验汇总 CSV/JSON/YAML。 |
| `visualizations/` | 残差柱状图、轨迹图、热力图、关系图和调试图。 |
| `test_videos_batch/` | 六段真实/伪造测试视频的分视频中间产物。 |
| `evaluation/pilot_smoke/` | 1 real + 1 fake 的真实 provider smoke 结果。 |
| `evaluation/pilot_6video/` | 4 real + 2 fake 的 per-video、per-clip、质量表和分布分析。 |
| `evaluation/rsd_strict_baseline/` | 冻结 strict v1 的逐 pair 诊断、视频覆盖、解释、运行哈希和四张严格基线图。 |
| `evaluation/rsd_strict_v2/` | dimension-aligned v2 的逐 pair/tier 结果、运行哈希、覆盖对比表和版本比较图。 |
| `evaluation/rsd_strict_v2_error_audit/` | `real_3` person-cup 高残差的逐帧、深度策略、投影策略、轨迹级和最终误差归因产物。 |
| `evaluation/rsd_strict_v2_applicability_gate/` | 六视频门控前后 pair/video 表、关键点 JSON、real_3 叠加图和覆盖/skip reason 图。 |
| `yolo_smoke_test*/` | YOLO smoke test 的带框帧。 |

`outputs/` 是可重复生成的实验产物，已整体排除在 Git 之外。论文使用的最终结果应另行
归档，并记录生成命令、配置、代码 commit 和数据版本。

## 10. Local-Only Directories

| 目录 | 作用 |
| --- | --- |
| `.venv/` | 项目独立 Python/Conda 环境及安装依赖。 |
| `checkpoints/` | `yolov8n.pt` 等本地模型权重。 |
| `.git/` | Git 提交历史、分支和远程配置。 |
| `.pytest_cache/`、`__pycache__/` | pytest/Python 自动生成缓存。 |
| `src/*.egg-info/` | 本地安装 Python 包时生成的元数据。 |

这些目录不属于论文方法源码，不应提交到 Git。

## 11. Main Processing Flow

```text
video
  -> video_preprocess.py (frames/clips)
  -> real_object_provider.py (YOLO objects)
  -> depth_provider.py (monocular relative depth)
  -> build_observations.py + observations.py + io.py (observation JSON)
  -> scale_prior.py + scale_depth.py (frame-level object-pair R_sd)
  -> object_association.py (cross-frame tracks)
  -> depth_temporal_consistency.py (transition-level R_depth_cons)
  -> aggregation/residual_fusion (clip/video scores)
  -> visualization.py and experiment scripts (CSV/PNG)
```

目前工程已经具备真实 `R_sd` 与 `R_depth_cons` 的实验闭环，但尚不是完整的伪造视频
分类模型。`R_flow`、`R_track`、`R_occ`、`R_corr` 仍属于后续研究项。
