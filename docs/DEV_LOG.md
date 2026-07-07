# Development Log

本文件用于记录项目开发过程中的关键修改，包括每次新增、删除、修改的文件、修改目的、验证命令、验证结果、当前限制和下一步计划。

每次新增记录请使用如下模板：

````markdown
## YYYY-MM-DD - 修改主题

### 目的
说明这次修改是为了解决什么问题，或者为了推进哪一阶段实验。

### 新增文件
- path/to/file.py：说明作用

### 修改文件
- path/to/file.py：说明修改内容

### 删除文件
- path/to/file.py：说明删除原因
- 无

### 主要变化
- 变化 1
- 变化 2
- 变化 3

### 验证命令
```bash
pytest ...
python ...
```

### 验证结果
- 说明测试是否通过、输出文件是否生成、关键指标是否符合预期。

### 当前限制
- 限制 1
- 限制 2

### 下一步计划
- 计划 1
- 计划 2
````

## 2026-07-07 - 建立开发日志与变更记录机制

### 目的
为后续论文写作、代码维护、服务器迁移和实验复现建立统一记录机制，避免项目逐步迭代后无法追踪每次文件变化、实验目的、验证命令和当前限制。

### 新增文件
- `docs/DEV_LOG.md`：记录每次开发过程中的新增、修改、删除文件、验证命令、验证结果、当前限制和下一步计划。
- `CHANGELOG.md`：记录项目版本级变更摘要，便于迁移、汇报和论文实验阶段回溯。

### 修改文件
- 无

### 删除文件
- 无

### 主要变化
- 新增开发日志模板，后续每次工程迭代可直接复制填写。
- 新增当前阶段的项目变更摘要，覆盖 R_sd、YOLO observation、尺度先验 alias 和深度 provider 闭环。
- 明确开发日志和 changelog 的用途区别：前者记录过程细节，后者记录版本摘要。

### 验证命令
```bash
ls docs/DEV_LOG.md CHANGELOG.md
```

### 验证结果
- 日志文件和变更记录文件已创建。

### 当前限制
- 该日志是从当前阶段开始补建的，早期迭代记录是阶段性归纳，不是逐次 commit 级记录。
- 后续每次新增模块、实验脚本或测试后，需要手动继续追加记录。

### 下一步计划
- 每次完成新实验闭环后，在本文件追加一条开发记录。
- 在提交 Git 前同步更新 `CHANGELOG.md`，保持服务器迁移和论文复现实验说明一致。

## 2026-07-07 - 当前原型阶段汇总记录

### 目的
对当前项目已经完成的关键工程闭环做一次阶段性归档，便于后续继续接入真实深度估计、光流、点轨迹和更多三维结构残差。

### 新增文件
- `configs/scale_priors.yaml`：维护精确类别尺度先验、类别别名和粗类别映射。
- `src/semantic3d/scale_prior.py`：实现 `ScalePriorResolver`，支持 exact prior、alias prior 和 unknown label 安全跳过。
- `src/semantic3d/depth_provider.py`：实现 `BaseDepthProvider`、`MockDepthProvider`、`RealDepthProvider` 预留接口和 bbox 区域深度统计函数。
- `scripts/check_ultralytics_env.py`：检查 ultralytics、torch、CUDA 和 YOLO 权重环境。
- `scripts/yolo_video_smoke_test.py`：对真实视频执行 YOLO 检测 smoke test，并保存检测框图像。
- `examples/demo_real_depth_observation_pipeline.py`：演示真实视频、YOLO、mock depth、observation JSON 和 R_sd pipeline 的完整闭环。
- `tests/test_depth_provider.py`：测试 mock depth map、bbox median depth、无效 bbox 和 observation depth 写入逻辑。
- `tests/test_scale_prior.py`：测试类别尺度先验解析、alias 映射和 unknown label 安全处理。

### 修改文件
- `src/semantic3d/real_object_provider.py`：接入 ultralytics YOLO，输出真实检测对象的 label、bbox、bbox_area、confidence 和临时 depth。
- `src/semantic3d/provider_registry.py`：支持 `mock` 和 `real_detector` provider 选择。
- `src/semantic3d/build_observations.py`：支持可选 depth provider，将 depth map 中 bbox 区域 median depth 写入对象 observation。
- `scripts/build_real_object_observations_from_video.py`：支持真实检测 provider、depth provider、保存 depth map 和生成 clip observation。
- `scripts/run_observation_rsd_pipeline.py`：使用 `ScalePriorResolver`，对 unknown label 和缺失尺度先验的对象对安全跳过并输出统计日志。
- `src/semantic3d/__init__.py`：导出新增的尺度先验和深度相关接口。
- `pyproject.toml`：补充 PyYAML 和 YOLO 可选依赖。
- `.gitignore`：忽略 checkpoints、outputs、视频文件和模型权重，避免误传大文件或未公开资源。

### 删除文件
- 无

### 主要变化
- 已完成对象级尺度-深度一致性残差 `R_sd` 的单样例、批量合成、手工 observation、真实视频 mock observation 和真实 YOLO observation 闭环。
- YOLO 检测结果可以进入 observation JSON，当前 mask_area 使用 bbox_area 近似。
- depth 字段已支持由 depth map 的 bbox 区域 median 统计得到，不再只能使用固定 `default_depth`。
- 尺度先验系统已从逐类别硬编码升级为“精确类别 + alias 映射 + unknown label 安全跳过”。
- 输出结果包括 CSV、PNG 可视化和 observation JSON，便于后续论文实验展示。

### 验证命令
```bash
pytest tests/test_depth_provider.py -v
pytest tests/test_scale_prior.py -v
pytest -q
python scripts/check_ultralytics_env.py
python scripts/yolo_video_smoke_test.py --video_path data/videos/test_real.mp4 --model_path checkpoints/yolov8n.pt --output_dir outputs/yolo_smoke_test --max_frames 5 --confidence_threshold 0.3 --device cpu
python examples/demo_real_depth_observation_pipeline.py
```

### 验证结果
- `tests/test_depth_provider.py` 已通过。
- 当前项目全量测试已通过。
- YOLO smoke test 可在真实视频上检测目标并输出带检测框图像。
- real depth demo 可生成 `outputs/real_observations_depth/`、`outputs/depth_maps/`、`outputs/results/real_depth_rsd_results.csv` 和 `outputs/visualizations/real_depth_rsd_scores.png`。

### 当前限制
- `RealDepthProvider` 仍是预留接口，尚未真正接入 Depth Anything、MiDaS 或其他深度估计模型。
- 当前 `MockDepthProvider` 生成的是合成相对深度图，只用于验证管线，不代表真实场景几何深度。
- 真实检测阶段暂时使用 bbox_area 近似 mask_area，尚未接入实例分割 mask。
- 当前 R_sd 只覆盖尺度-深度投影几何一致性，还没有完整接入光流、点轨迹、遮挡和空间对应残差。
- 单目相对深度的尺度不确定性后续需要通过归一化、时序稳定性或相机/场景约束进一步处理。

### 下一步计划
- 接入真实深度估计 provider，并将相对深度稳定写入 observation JSON。
- 接入实例分割 mask，替代 bbox_area 近似。
- 继续实现 `R_flow`、`R_track`、`R_occ`、`R_corr` 等三维结构残差项。
- 设计真实/伪造视频数据集的 observation 生成和实验评估协议。
