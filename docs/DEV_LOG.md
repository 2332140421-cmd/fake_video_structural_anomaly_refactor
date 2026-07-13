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

## 2026-07-08 - 核查 RealDepthProvider 真实深度接入状态

### 目的
核查 `src/semantic3d/depth_provider.py` 中的 `RealDepthProvider` 是否已经真正调用真实深度估计模型，避免在论文记录、实验说明或项目汇报中把 mock depth 或接口占位误写成真实深度估计接入。

### 新增文件
- 无

### 修改文件
- `docs/DEV_LOG.md`：补充 RealDepthProvider 状态审计记录，明确当前完成的是 depth provider 最小闭环，不是真实深度模型接入。

### 删除文件
- 无

### 主要变化
- 确认 `MockDepthProvider` 会根据图像尺寸生成合成相对深度图，用于测试 bbox 区域 median depth 写入 observation JSON。
- 确认 `RealDepthProvider` 当前是可插拔接口占位：只有在外部传入带 `predict_depth(frame_path)` 方法的模型对象时才会调用该对象。
- 确认当前工程没有真正接入 Depth Anything、MiDaS 或其他具体真实深度估计模型。
- 确认当前已经完成的是：视频帧 -> mock depth map -> bbox median depth -> ObjectObservationJSON.depth -> R_sd pipeline 的最小闭环。

### 验证命令
```bash
sed -n '1,240p' src/semantic3d/depth_provider.py
rg -n "RealDepthProvider|MockDepthProvider|Depth Anything|MiDaS|real_depth|mock depth" docs/DEV_LOG.md src/semantic3d/depth_provider.py CHANGELOG.md
```

### 验证结果
- `RealDepthProvider.__init__` 在 `model is None` 时抛出 `RuntimeError`，提示使用 `mock_depth` 或安装深度模型依赖。
- `RealDepthProvider.predict_depth` 只调用注入模型的 `predict_depth` 方法，不包含任何真实模型加载逻辑。
- 当前没有接入具体真实深度模型；尚未下载、加载或调用 Depth Anything、MiDaS 等模型。

### 当前限制
- 当前 `outputs/depth_maps/` 中的 depth map 来自 `MockDepthProvider` 的合成梯度，不是真实深度估计结果。
- 当前 `real_depth` 参数只能作为预留入口使用，如果没有注入或实现具体模型，会给出清晰错误。
- 因为没有真实深度模型，当前真实视频上的 R_sd 仍不能代表真实三维尺度-深度一致性实验结论，只能验证数据管线。

### 下一步计划
- 明确选择一个真实深度估计模型，例如 Depth Anything 或 MiDaS。
- 在 `RealDepthProvider` 中实现模型加载、图像预处理、推理和 depth map 后处理。
- 增加真实深度 provider 的 smoke test，验证输出 shape、数值范围和 bbox median depth 是否合理。
- 在论文实验记录中区分 “mock depth 管线验证” 和 “真实深度估计实验”。

## 2026-07-08 - 接入真实 RealDepthProvider

### 目的
将 `RealDepthProvider` 从接口占位推进为真实单目深度估计 provider，使真实视频帧可以经过 YOLO bbox 检测、Depth Anything 深度估计、bbox 区域 median depth 统计，再写入 `ObjectObservationJSON.depth` 并进入 `R_sd` pipeline。

### 新增文件
- `scripts/check_depth_model_env.py`：检查 Python、torch、CUDA、transformers、PIL 和 numpy 环境，不强制重装 torch。
- `scripts/depth_smoke_test.py`：对单张图片或视频前若干帧运行真实深度估计，保存 `.npy` depth map 和 PNG 深度可视化图。

### 修改文件
- `src/semantic3d/depth_provider.py`：实现基于 `transformers.pipeline("depth-estimation")` 的 `RealDepthProvider`，支持 Depth Anything V2 Small、depth 归一化、NaN/inf 清理、原图尺寸 resize 和 `invert_depth`。
- `src/semantic3d/build_observations.py`：保存 depth map 时同时输出 `.npy` 和 PNG 可视化。
- `scripts/build_real_object_observations_from_video.py`：新增 `--depth_model_name` 和 `--invert_depth`，`--depth_provider real_depth` 时初始化真实深度 provider。
- `examples/demo_real_depth_observation_pipeline.py`：将 demo 从 `mock_depth` 改为 `real_depth`。
- `tests/test_depth_provider.py`：增加真实 depth provider import、smoke、bbox depth 写入等测试。
- `src/semantic3d/real_object_provider.py`：让真实检测 provider 的可保留标签兼容 `scale_priors.yaml` 中的 alias，避免 `mouse`、`remote`、`dining_table` 在进入 R_sd pipeline 前被错误跳过。
- `pyproject.toml`：增加 `pillow` 依赖和 `depth` 可选依赖组。

### 删除文件
- 无

### 主要变化
- `RealDepthProvider` 现在会真正调用 transformers 深度估计模型，不再只是占位接口。
- 默认命令中传入的 `depth-anything/Depth-Anything-V2-Small` 会自动映射到 transformers 兼容模型 `depth-anything/Depth-Anything-V2-Small-hf`。
- 项目内部保持统一约定：depth 数值越大表示对象越远；如果模型输出 inverse depth/disparity，可以通过 `--invert_depth` 显式反转。
- `depth_provider=none` 仍保留默认固定 depth；`mock_depth` 仍保留合成相对深度图；`real_depth` 使用真实单目深度估计模型。
- 当前真实深度输出是单目相对深度，经过归一化后写入 observation，不是绝对米制深度。

### 验证命令
```bash
python scripts/check_depth_model_env.py
python scripts/depth_smoke_test.py --video_path data/videos/test_real.mp4 --model_name depth-anything/Depth-Anything-V2-Small --output_dir outputs/depth_smoke_test --max_frames 3 --device cpu
python scripts/build_real_object_observations_from_video.py --video_path data/videos/test_real.mp4 --output_dir outputs/real_observations_depth --max_frames 20 --clip_len 8 --stride 4 --object_provider real_detector --model_path checkpoints/yolov8n.pt --confidence_threshold 0.3 --depth_provider real_depth --depth_model_name depth-anything/Depth-Anything-V2-Small --depth_output_dir outputs/depth_maps --save_depth_maps --device cpu
python scripts/run_observation_rsd_pipeline.py --observation_dir outputs/real_observations_depth --output_csv outputs/results/real_depth_rsd_results.csv --visualization_path outputs/visualizations/real_depth_rsd_scores.png
pytest tests/test_depth_provider.py -v
pytest -q
```

### 验证结果
- `scripts/check_depth_model_env.py` 通过，当前环境中 `transformers.pipeline`、PIL、numpy 和 torch 可导入。
- `scripts/depth_smoke_test.py` 成功生成 3 帧真实 depth map，输出 `.npy` 和 PNG，depth map shape 为 `(1280, 720)`，数值范围归一化到 `[1, 10]`，无 NaN/inf。
- `scripts/build_real_object_observations_from_video.py --depth_provider real_depth` 成功生成 4 个 clip observation JSON。
- `outputs/depth_maps/` 中已保存真实 depth `.npy` 和 PNG 可视化文件。
- `outputs/real_observations_depth/test_real_real_detector_clip_000.json` 中对象 depth 不再全部等于 `default_depth=5.0`，当前样例 depth 范围约为 `4.3774` 到 `8.3442`，24 个对象有 24 个不同 depth 值。
- 对照结果：`depth_provider=none` 时 24 个对象 depth 全为 `5.0`；`mock_depth` 范围约为 `5.3667` 到 `7.5632`；`real_depth` 范围约为 `4.3774` 到 `8.3442`。
- `scripts/run_observation_rsd_pipeline.py` 成功输出 `outputs/results/real_depth_rsd_results.csv` 和 `outputs/visualizations/real_depth_rsd_scores.png`。
- 当前真实视频的 `R_sd_log` 仍为 0，原因是检测对象组合在当前粗尺度先验与估计深度下仍落在合理区间，不再是 depth 全部相同导致。
- `pytest tests/test_depth_provider.py -v` 通过。
- `pytest -q` 通过，结果为 `78 passed, 1 warning`。warning 来自当前 NVIDIA driver 过旧导致 CUDA 不可用，CPU 推理不受影响。

### 当前限制
- Depth Anything V2 Small 输出的是相对深度，不是绝对米制深度。
- 当前对 depth 方向不做自动判断，需要用户根据模型输出约定和实验可视化决定是否使用 `--invert_depth`。
- 当前 depth map 经过 `[1, 10]` 归一化，适合稳定计算相对深度比，但不能直接解释为真实距离。
- 当前仍使用 YOLO bbox area 近似 `mask_area`，没有接入真实实例分割 mask。
- 当前真实视频样例没有触发非零 `R_sd`，后续需要构造或收集尺度-深度不一致的视频片段进行消融验证。

### 下一步计划
- 对 Depth Anything 的深度方向进行定性校验：近处物体的 depth 是否更小、远处物体的 depth 是否更大。
- 增加 `--invert_depth` 前后 R_sd 对比实验，确认方向选择对残差的影响。
- 使用手工标注或可控场景视频构造正负样例，做 `none`、`mock_depth`、`real_depth` 的消融实验。
- 接入实例分割 mask，用 mask 区域 median depth 替代 bbox 区域 median depth。
- 在论文中明确说明当前使用的是单目相对深度，而不是米制深度。

## 2026-07-08 - 深度方向校验与 R_sd 对比实验

### 目的
验证单目相对深度输出方向是否符合项目内部约定，并比较 `no_depth`、`real_depth_no_invert`、`real_depth_invert` 三种模式对对象 depth 分布、R_sd、R_sd_log 和 clip_score 的影响。项目内部约定为：depth 数值越大表示对象越远。

### 新增文件
- `scripts/compare_depth_direction_rsd.py`：对同一视频自动运行 `no_depth`、`real_depth_no_invert`、`real_depth_invert` 三种模式，分别生成 observation、R_sd CSV、clip-score 图和汇总 CSV。
- `scripts/visualize_depth_direction_debug.py`：将原始帧、YOLO bbox、对象 depth、no-invert depth heatmap、invert depth heatmap 和 R_sd 对象对连线绘制到 debug PNG。
- `tests/test_depth_direction_comparison.py`：测试 compare 脚本、summary CSV、默认 depth、depth 变化、debug 可视化和空 R_sd CSV 处理。

### 修改文件
- `docs/DEV_LOG.md`：追加本次深度方向校验与 R_sd 对比实验记录。

### 删除文件
- 无

### 主要变化
- 新增统一对比输出目录 `outputs/depth_direction_comparison/`。
- 三种模式分别输出：
  - `no_depth_observations/`
  - `real_depth_no_invert_observations/`
  - `real_depth_invert_observations/`
  - 对应的 `*_rsd_results.csv`
  - `depth_direction_summary.csv`
- 新增人工判断提示：近处对象应有较小 depth，远处对象应有较大 depth；若相反，则优先考虑 `--invert_depth`。
- 可视化脚本输出到 `outputs/visualizations/depth_direction_debug/`，便于人工检查 depth 方向。

### 验证命令
```bash
python scripts/compare_depth_direction_rsd.py --video_path data/videos/test_real.mp4 --model_path checkpoints/yolov8n.pt --depth_model_name depth-anything/Depth-Anything-V2-Small --max_frames 20 --clip_len 8 --stride 4 --confidence_threshold 0.3 --device cpu
python scripts/visualize_depth_direction_debug.py --no_invert_observation_dir outputs/depth_direction_comparison/real_depth_no_invert_observations --invert_observation_dir outputs/depth_direction_comparison/real_depth_invert_observations --no_invert_depth_dir outputs/depth_direction_comparison/real_depth_no_invert_depth_maps --invert_depth_dir outputs/depth_direction_comparison/real_depth_invert_depth_maps --no_invert_rsd_csv outputs/depth_direction_comparison/real_depth_no_invert_rsd_results.csv --invert_rsd_csv outputs/depth_direction_comparison/real_depth_invert_rsd_results.csv --output_dir outputs/visualizations/depth_direction_debug --max_frames 10
pytest tests/test_depth_direction_comparison.py -v
pytest -q
```

### 验证结果
- `compare_depth_direction_rsd.py` 成功生成三种模式的 observation 和 R_sd CSV。
- `depth_direction_summary.csv` 已生成，当前真实视频统计如下：
  - `no_depth`：99 个对象，depth 全为 `5.0`，unique depth count 为 1。
  - `real_depth_no_invert`：99 个对象，depth 范围约为 `3.8048` 到 `8.3864`，unique depth count 为 63。
  - `real_depth_invert`：99 个对象，depth 范围约为 `2.6136` 到 `7.1952`，unique depth count 为 63。
- 三种模式都计算了 109 个对象对；当前视频的 `R_sd_log` 和 clip_score 均为 0，说明该视频没有触发尺度-深度异常。
- `visualize_depth_direction_debug.py` 成功生成 10 张 debug PNG。
- `pytest tests/test_depth_direction_comparison.py -v` 通过。
- `pytest -q` 通过，结果为 `84 passed, 1 warning`。warning 来自当前 NVIDIA driver 过旧导致 CUDA 不可用，CPU 推理不受影响。

### 当前限制
- 当前 depth 仍是单目相对深度，不是绝对米制深度。
- `no_invert` 与 `invert` 哪个更符合真实远近关系，需要结合 debug PNG 进行人工判断。
- 当前真实视频检测对象主要为桌面类和手持小物体，未触发非零 R_sd，不足以证明异常检测能力。
- 当前仍基于 YOLO bbox 区域取 median depth，尚未接入实例分割 mask 区域深度。

### 下一步计划
- 人工检查 `outputs/visualizations/depth_direction_debug/` 中的可视化图，确定是否默认启用 `--invert_depth`。
- 在正常视频和异常/伪造视频上分别运行方向对比，观察 `R_sd_log` 和 clip_score 是否有区分度。
- 设计 `none`、`real_depth_no_invert`、`real_depth_invert` 的消融实验表格。
- 深度方向确认后，可以继续实现 `R_depth_cons`，用于深度时序一致性残差；但应与当前对象级 `R_sd` 保持概念区分。

## 2026-07-08 - real_2 至 real_4 真实视频参数测试

### 目的
使用 `data/real_videos/real_2.mp4`、`real_3.mp4` 和 `real_4.mp4` 测试真实 YOLO 检测、Depth Anything 相对深度、深度方向对比和 R_sd 输出参数，检查不同真实视频是否能产生有效对象对和 residual rows。

### 新增文件
- 无

### 修改文件
- `configs/scale_priors.yaml`：新增 `tabletop_object`、`plant`、`traffic_light` 粗类别尺度先验，并将 `cup`、`vase`、`potted_plant`、`traffic light` 映射到对应粗类别，避免真实视频中常见桌面物体被 R_sd pipeline 跳过。
- `docs/DEV_LOG.md`：追加本次真实视频参数测试记录。

### 删除文件
- 无

### 主要变化
- `real_2.mp4` 在默认置信度 0.3 下没有 YOLO 检测对象，因此额外使用 0.01 阈值进行诊断测试。
- `real_3.mp4` 检测到 `person`、`cup`、`dining_table`，补充 alias 后可计算 96 个对象对。
- `real_4.mp4` 检测到 `vase`、`cup`，补充 alias 后可计算 31 个对象对。
- 三段视频均生成了 no-depth、real-depth no-invert、real-depth invert 三种模式的 summary CSV 和 debug PNG。

### 验证命令
```bash
python scripts/compare_depth_direction_rsd.py --video_path data/real_videos/real_2.mp4 --model_path checkpoints/yolov8n.pt --depth_model_name depth-anything/Depth-Anything-V2-Small --max_frames 20 --clip_len 8 --stride 4 --confidence_threshold 0.01 --device cpu --output_dir outputs/depth_direction_comparison_real_2_conf001
python scripts/compare_depth_direction_rsd.py --video_path data/real_videos/real_3.mp4 --model_path checkpoints/yolov8n.pt --depth_model_name depth-anything/Depth-Anything-V2-Small --max_frames 20 --clip_len 8 --stride 4 --confidence_threshold 0.3 --device cpu --output_dir outputs/depth_direction_comparison_real_3
python scripts/compare_depth_direction_rsd.py --video_path data/real_videos/real_4.mp4 --model_path checkpoints/yolov8n.pt --depth_model_name depth-anything/Depth-Anything-V2-Small --max_frames 20 --clip_len 8 --stride 4 --confidence_threshold 0.3 --device cpu --output_dir outputs/depth_direction_comparison_real_4
pytest tests/test_scale_prior.py tests/test_depth_direction_comparison.py -q
```

### 验证结果
- `real_2_conf001`：53 个对象，26 个对象对；`real_depth_no_invert` 的 `R_sd_log max` 约为 `2.2038`，`real_depth_invert` 的 `R_sd_log max` 约为 `0.3366`。
- `real_3`：96 个对象，96 个对象对；`real_depth_no_invert` 的 `R_sd_log max` 约为 `1.3217`，`real_depth_invert` 的 `R_sd_log max` 为 `0.0`。
- `real_4`：63 个对象，31 个对象对；三种模式 `R_sd_log max` 均为 `0.0`。
- `pytest tests/test_scale_prior.py tests/test_depth_direction_comparison.py -q` 通过，结果为 `13 passed`。

### 当前限制
- `real_2.mp4` 需要降低检测阈值到 `0.01` 才能得到对象；低置信度检测结果应谨慎用于论文定量实验。
- `real_4.mp4` 中 no-invert 和 invert 的 depth 范围都较窄，R_sd 无明显变化。
- `traffic_light`、`plant`、`tabletop_object` 的尺度先验目前是粗粒度经验范围，后续需要根据实验对象类别继续校准。

### 下一步计划
- 人工查看 `outputs/visualizations/depth_direction_debug_real_*/`，判断 no-invert 与 invert 哪个更符合真实远近关系。
- 将 `real_2` 的低阈值结果与正常阈值结果分开记录，不混入正式定量实验。
- 准备包含明确正常/异常标签的视频集合，再统计 R_sd 阈值分类指标。

## 2026-07-08 - 确认后续实验默认启用 invert depth

### 目的
根据 `real_2`、`real_3` 和 `real_4` 的 depth direction debug 可视化结果，确认后续真实视频实验中的深度方向设置，使项目内部 depth 约定保持一致：depth 数值越大表示对象越远。

### 新增文件
- 无

### 修改文件
- `docs/DEV_LOG.md`：追加后续实验默认启用 `--invert_depth` 的实验约定。

### 删除文件
- 无

### 主要变化
- 经人工查看 `outputs/visualizations/depth_direction_debug_real_*/` 后，确认当前 Depth Anything 输出更适合在项目中使用 `--invert_depth`。
- 后续真实视频 observation 构建命令默认加入 `--invert_depth`。
- 论文实验记录中应明确：单目深度模型输出经过方向反转，以满足“depth 数值越大表示越远”的项目内部约定。

### 验证命令
```bash
python scripts/compare_depth_direction_rsd.py --video_path data/real_videos/real_3.mp4 --model_path checkpoints/yolov8n.pt --depth_model_name depth-anything/Depth-Anything-V2-Small --max_frames 20 --clip_len 8 --stride 4 --confidence_threshold 0.3 --device cpu --output_dir outputs/depth_direction_comparison_real_3
python scripts/visualize_depth_direction_debug.py --no_invert_observation_dir outputs/depth_direction_comparison_real_3/real_depth_no_invert_observations --invert_observation_dir outputs/depth_direction_comparison_real_3/real_depth_invert_observations --no_invert_depth_dir outputs/depth_direction_comparison_real_3/real_depth_no_invert_depth_maps --invert_depth_dir outputs/depth_direction_comparison_real_3/real_depth_invert_depth_maps --no_invert_rsd_csv outputs/depth_direction_comparison_real_3/real_depth_no_invert_rsd_results.csv --invert_rsd_csv outputs/depth_direction_comparison_real_3/real_depth_invert_rsd_results.csv --output_dir outputs/visualizations/depth_direction_debug_real_3 --max_frames 10
```

### 验证结果
- `real_2`：`real_depth_no_invert` 的 `R_sd_log max` 约为 `2.2038`，`real_depth_invert` 的 `R_sd_log max` 约为 `0.3366`。
- `real_3`：`real_depth_no_invert` 的 `R_sd_log max` 约为 `1.3217`，`real_depth_invert` 的 `R_sd_log max` 为 `0.0`。
- `real_4`：no-invert 与 invert 均为 `0.0`，对方向判断帮助较小。
- 人工检查 debug 可视化后，后续默认采用 `--invert_depth`。

### 当前限制
- `--invert_depth` 是基于当前视频和可视化的实验约定，后续更换深度模型时需要重新校验方向。
- 当前深度仍是单目相对深度，不是米制绝对深度。
- `real_2` 的有效检测需要低置信度阈值，正式实验中应谨慎使用。

### 下一步计划
- 将后续真实视频 observation 构建脚本命令统一加入 `--invert_depth`。
- 在 README 或实验说明中补充 depth 方向约定。
- 开始准备正常/异常视频集合，进行 R_sd 阈值评估和消融实验。

## 2026-07-08 - 尺度先验覆盖增强与缺失先验统计机制

### 目的
真实视频测试中发现 YOLO 能检测到 `cup`、`vase`、`potted_plant`、`traffic_light` 等常见类别，但旧版尺度先验配置没有完整覆盖，导致 R_sd pipeline 安全跳过对象对。本次修改目标是增强尺度先验覆盖、显式标注先验可靠性，并新增自动报告工具，方便后续维护 scale prior 库。

### 新增文件
- `scripts/report_missing_scale_priors.py`：扫描 observation JSON，统计每个检测类别的 scale prior 覆盖状态，输出 exact、alias、missing、unreliable 报告 CSV。

### 修改文件
- `configs/scale_priors.yaml`：将 scale prior 扩展为包含 `reliable` 字段的结构；新增 `cup`、`traffic_light` 等稳定尺度类别；将 `vase`、`potted_plant` 标记为 `reliable: false`。
- `src/semantic3d/scale_prior.py`：新增 `ScalePriorRecord`，升级 `ScalePriorResolver`，支持 label 标准化、exact prior、alias prior、missing prior 和 unreliable prior 状态。
- `scripts/run_observation_rsd_pipeline.py`：使用 resolver 的可靠性状态，默认跳过 missing 和 unreliable 对象，并输出更细粒度统计。
- `tests/test_scale_prior.py`：新增 exact、alias、missing、unreliable、报告 CSV 和 R_sd 跳过 unreliable 类别的测试。
- `tests/test_real_object_provider.py`：更新 unknown label 测试，避免使用现在已经有 prior 的 `traffic light` 作为未知类别。
- `src/semantic3d/__init__.py`：导出 `ScalePriorRecord`。
- `docs/DEV_LOG.md`：追加本次尺度先验覆盖增强记录。

### 删除文件
- 无

### 主要变化
- 精确类别优先：`person`、`car`、`chair`、`table`、`cup`、`bottle`、`traffic_light` 等可直接命中 exact prior。
- alias 用于名称归一化或少量粗类别兜底：例如 `dining table -> table`、`sports ball -> ball`、`mouse -> handheld_object`。
- `vase` 和 `potted_plant` 由于尺度变化较大，虽然有尺度范围记录，但默认 `reliable: false`，不参与主 R_sd 计算。
- R_sd pipeline 新增统计项：`total_objects`、`exact_prior_objects`、`alias_prior_objects`、`skipped_missing_prior_objects`、`skipped_unreliable_prior_objects`、`total_candidate_pairs`、`computed_pairs`、`skipped_pairs_missing_prior`、`skipped_pairs_unreliable_prior`。

### 验证命令
```bash
pytest tests/test_scale_prior.py -v
python scripts/report_missing_scale_priors.py --observation_dir outputs/real_observations_depth --scale_prior_path configs/scale_priors.yaml --output_csv outputs/results/missing_scale_prior_report.csv
python scripts/run_observation_rsd_pipeline.py --observation_dir outputs/real_observations_depth --output_csv outputs/results/real_depth_rsd_results.csv --visualization_path outputs/visualizations/real_depth_rsd_scores.png
python scripts/report_missing_scale_priors.py --observation_dir outputs/depth_direction_comparison_real_4/real_depth_invert_observations --scale_prior_path configs/scale_priors.yaml --output_csv outputs/results/missing_scale_prior_report_real_4.csv
python scripts/run_observation_rsd_pipeline.py --observation_dir outputs/depth_direction_comparison_real_4/real_depth_invert_observations --output_csv outputs/results/real_4_invert_rsd_reliable_results.csv --visualization_path outputs/visualizations/real_4_invert_rsd_reliable_scores.png
pytest -q
```

### 验证结果
- `pytest tests/test_scale_prior.py -v` 通过。
- `outputs/real_observations_depth` 的报告显示：`mouse`、`dining_table`、`remote` 均通过 alias 映射，无 missing/unreliable。
- 对 `outputs/real_observations_depth` 运行 R_sd：99 个对象，109 个候选对象对，109 个对象对参与计算。
- `real_4` 的报告显示：`cup` 为 exact 且 reliable，`vase` 为 unreliable。
- 对 `real_4` invert observation 运行 R_sd：63 个对象，其中 31 个 `cup` exact，32 个 `vase` 因 unreliable 跳过；31 个候选对象对因包含 unreliable prior 被跳过，因此 CSV 无 residual rows。这是预期行为。
- 全量测试通过：`85 passed, 1 warning`。warning 来自当前 NVIDIA driver 过旧导致 CUDA 不可用，CPU 推理不受影响。

### 当前限制
- `cup`、`traffic_light` 等新增尺度先验仍是粗粒度经验范围，后续需要结合实验数据校准。
- `vase`、`potted_plant` 默认不参与 R_sd，不能将其 skipped 结果作为主实验异常结论。
- `report_missing_scale_priors.py` 只能报告 observation 中已经出现的类别，不能替代完整类别库设计。

### 下一步计划
- 定期对真实 observation 运行 `report_missing_scale_priors.py`，维护 missing/unreliable 类别清单。
- 对论文实验中高频出现的稳定尺度类别逐步补充 exact prior。
- 对尺度变化大的类别继续保留 `reliable: false`，必要时只在消融实验中启用。

## 2026-07-08 - 尺度先验半自动维护机制

### 目的
进一步规范 scale prior 的维护流程：不再每遇到新 YOLO 类别就直接写入主配置，而是先扫描 observation 中的类别覆盖情况，统计 exact、alias、missing、unreliable 状态，再对高频 missing/unreliable 类别生成待人工审核的候选 YAML。

### 新增文件
- `scripts/generate_scale_prior_candidates.py`：读取 missing prior report，生成 `candidate_scale_priors` YAML，供人工审核；不会自动修改 `configs/scale_priors.yaml`。

### 修改文件
- `scripts/report_missing_scale_priors.py`：新增 `--min_count` 参数，支持只报告出现次数达到阈值的类别。
- `tests/test_scale_prior.py`：新增候选 YAML 生成测试，并验证 candidates 不会污染主 `scale_priors.yaml`。
- `docs/DEV_LOG.md`：追加本次尺度先验半自动维护机制记录。

### 删除文件
- 无

### 主要变化
- `report_missing_scale_priors.py` 输出字段包括：`label`、`count`、`has_exact_prior`、`has_alias_prior`、`resolved_label`、`reliable`、`status`。
- `generate_scale_prior_candidates.py` 只为 `missing` 和 `unreliable` 类别生成候选项。
- `missing` 类别候选项默认 `action: review`，提示人工补精确先验或映射到粗类别。
- `unreliable` 类别候选项默认 `action: review_or_skip`，提示尺度变化大，除非人工确认，否则建议跳过。
- 主配置 `configs/scale_priors.yaml` 不包含 `candidate_scale_priors`，候选文件只保存在 `outputs/results/`。

### 验证命令
```bash
python scripts/report_missing_scale_priors.py --observation_dir outputs/real_observations_depth --scale_prior_path configs/scale_priors.yaml --output_csv outputs/results/missing_scale_prior_report.csv --min_count 1
python scripts/generate_scale_prior_candidates.py --missing_report_csv outputs/results/missing_scale_prior_report.csv --output_yaml outputs/results/scale_prior_candidates.yaml --min_count 2
python scripts/report_missing_scale_priors.py --observation_dir outputs/depth_direction_comparison_real_4/real_depth_invert_observations --scale_prior_path configs/scale_priors.yaml --output_csv outputs/results/missing_scale_prior_report_real_4.csv --min_count 1
python scripts/generate_scale_prior_candidates.py --missing_report_csv outputs/results/missing_scale_prior_report_real_4.csv --output_yaml outputs/results/scale_prior_candidates_real_4.yaml --min_count 2
pytest tests/test_scale_prior.py -v
pytest -q
```

### 验证结果
- 对 `outputs/real_observations_depth` 生成报告：`mouse`、`dining_table`、`remote` 均通过 alias 覆盖，无 missing/unreliable；因此 `scale_prior_candidates.yaml` 中候选数为 0。
- 对 `real_4` invert observation 生成报告：`cup` 为 exact/reliable，`vase` 为 unreliable。
- `outputs/results/scale_prior_candidates_real_4.yaml` 生成 1 个候选项：`vase`，`action: review_or_skip`，`reliable: false`。
- `pytest tests/test_scale_prior.py -v` 通过。
- 全量测试通过：`85 passed, 1 warning`。

### 当前限制
- 候选 YAML 只给出人工审核模板，不会自动推断真实物理尺度。
- 仅凭 2D YOLO observation 不能可靠估计真实物体尺寸；尺度先验仍需要人工、文献或 3D/RGB-D 数据支持。
- 对尺度变化大的类别，即使高频出现，也不应直接加入主 R_sd 实验。

### 下一步计划
- 建立 scale prior 审核流程：先 report，再 candidates，再人工确认，最后合并到主配置。
- 后续可从 RGB-D / 3D 标注数据集中统计类别真实尺寸分布，生成候选 min/max。
- 为论文实验固定一版 `scale_priors.yaml`，并记录版本与变更原因。

## 2026-07-13 - 真实 R_sd 闭环批量验证与深度方向检查

### 目的
继续完善真实视频、YOLO 目标检测、单目深度估计、observation JSON、尺度-深度残差 `R_sd` 的闭环验证。重点检查 `RealDepthProvider` 是否真正调用真实单目深度模型，并对多段真实视频批量比较 no-invert 与 invert-depth 两种深度方向；同时补充 scale prior 覆盖统计和候选 YAML 生成验证。

### 新增文件
- `scripts/batch_compare_depth_direction_rsd.py`：批量扫描视频目录，对每段视频复用单视频 depth direction 对比流程，生成总汇总 CSV、debug PNG、scale prior 覆盖报告和候选 YAML。
- `tests/test_depth_direction_batch.py`：使用临时合成视频和 mock provider 验证批量脚本、汇总 CSV、debug PNG、scale prior report 与 candidates 文件能正确生成。

### 修改文件
- `docs/DEV_LOG.md`：追加本次真实 R_sd 闭环验证记录。

### 删除文件
- 无

### 主要变化
- 已核对 `src/semantic3d/depth_provider.py`：`RealDepthProvider` 会加载 `transformers.pipeline("depth-estimation", model=..., device=...)`，并调用该 pipeline 对输入帧预测深度图；因此它不是 `MockDepthProvider`，也不是固定 `default_depth` 占位。
- `RealDepthProvider` 当前使用的是单目深度估计输出，属于相对深度，不是米制绝对距离；项目内部约定 `depth` 数值越大表示对象越远。
- 批量脚本默认只把 `real_depth_no_invert` 和 `real_depth_invert` 写入 `depth_direction_batch_summary.csv`；单视频工作目录中仍保留 `no_depth` 作为对照基线。
- 批量脚本会调用 `report_missing_scale_priors.py` 和 `generate_scale_prior_candidates.py`，统计 exact、alias、missing、unreliable 四类类别覆盖情况，并生成只供人工审核的 candidate YAML。
- 当前仓库没有 `data/videos/depth_check/` 目录；本次真实批量验证使用已有 `data/real_videos/real_1.mp4` 至 `real_4.mp4`。后续如果把视频放入 `data/videos/depth_check/`，可直接使用同一脚本的默认路径。

### 验证命令
```bash
python scripts/check_depth_model_env.py
python scripts/batch_compare_depth_direction_rsd.py --video_dir data/real_videos --output_dir outputs/depth_direction_batch --summary_csv outputs/results/depth_direction_batch_summary.csv --visualization_dir outputs/visualizations/depth_direction_batch --scale_prior_report_csv outputs/results/depth_direction_scale_prior_coverage.csv --scale_prior_candidates_yaml outputs/results/depth_direction_scale_prior_candidates.yaml --max_frames 3 --clip_len 3 --stride 3 --object_provider real_detector --real_depth_backend real_depth --model_path checkpoints/yolov8n.pt --confidence_threshold 0.3 --device cpu --max_debug_frames 1
pytest tests/test_depth_direction_batch.py -v
```

### 验证结果
- `python scripts/check_depth_model_env.py` 通过：当前使用 `.venv/bin/python`，`transformers.pipeline`、PIL、numpy 和 torch 均可用；CUDA 当前不可用，因此真实深度推理走 CPU。
- 批量真实视频输出已生成：
  - `outputs/results/depth_direction_batch_summary.csv`
  - `outputs/visualizations/depth_direction_batch/real_1/depth_direction_frame_000000.png`
  - `outputs/visualizations/depth_direction_batch/real_2/depth_direction_frame_000000.png`
  - `outputs/visualizations/depth_direction_batch/real_3/depth_direction_frame_000000.png`
  - `outputs/visualizations/depth_direction_batch/real_4/depth_direction_frame_000000.png`
  - `outputs/results/depth_direction_scale_prior_coverage.csv`
  - `outputs/results/depth_direction_scale_prior_candidates.yaml`
- 在本次 `max_frames=3` 的小规模真实验证中，`real_3` 的 `real_depth_no_invert` 平均 `R_sd_log` 约为 1.21，`real_depth_invert` 平均 `R_sd_log` 约为 0.34，支持后续默认使用 `--invert_depth` 的约定。
- `real_2` 前 3 帧未检测到可用对象，因此没有有效 R_sd 行；这是数据/检测结果为空导致的正常跳过。
- `real_4` 中 `vase` 被标记为 `unreliable`，因此涉及该类别的对象对被安全跳过；candidate YAML 中生成了 `vase` 的 `review_or_skip` 候选记录。
- scale prior 覆盖报告显示：`cup`、`person` 为 exact；`dining_table`、`mouse` 为 alias；`vase` 为 unreliable；本次批量样本中没有 missing 类别。

### 当前限制
- 单目深度输出是相对深度，不能解释为米制真实距离；不同视频、不同帧之间的绝对尺度不可直接比较。
- 当前 `R_sd` 依赖 YOLO bbox_area 近似 mask_area，尚未接入实例分割 mask。
- 当前只验证对象级尺度-深度投影几何一致性，尚未实现 `R_depth_cons`、`R_track`、`R_occ`。
- `data/videos/`、`outputs/`、`checkpoints/` 仍应保持 git 忽略，不应提交真实视频、模型权重或实验输出。

### 下一步计划
- 可以进入 `R_depth_cons` 的最小闭环设计：先基于同一对象或同一 track 的相邻帧相对深度变化做时序一致性残差，不要混淆为 `R_sd`。
- 在实现 `R_depth_cons` 前，建议先补充轻量 object track / association 机制，用于跨帧关联 YOLO 检测对象。
- 继续用 `batch_compare_depth_direction_rsd.py` 对更多 `data/videos/depth_check/` 视频做 no-invert/invert 对比，固定论文实验中的 depth direction 设置。
- 持续通过 coverage report 和 candidate YAML 维护 `scale_priors.yaml`，不自动把 missing/unreliable 类别加入主实验。

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

## 2026-07-13 - 跨帧对象关联与 R_depth_cons 最小闭环

### 目的
在真实 `R_sd` 闭环之后，继续实现跨帧对象关联和深度时序一致性残差 `R_depth_cons`。本阶段只验证对象轨迹上的相对深度与投影尺度时序一致性，不实现 `R_track`、`R_occ`、`R_corr`，也不把 `R_depth_cons` 与 `R_sd` 融合成最终检测分数。

### 新增文件
- `src/semantic3d/object_association.py`：实现轻量级 `ObjectAssociator`，基于 canonical label、bbox IoU、归一化中心距离和 bbox 面积比例为跨帧对象分配 `track_id`。
- `src/semantic3d/depth_temporal_consistency.py`：实现帧级参考深度、`R_depth_cons` 计算、轨迹级/片段级汇聚和轨迹可视化。
- `scripts/run_depth_consistency_pipeline.py`：读取 observation JSON，执行对象关联，保存带 `track_id` 的 observation，并输出 pair/track/clip 三类独立 CSV。
- `examples/demo_depth_consistency.py`：构造 stable、consistent approaching、depth jump、projection jump 四类合成轨迹，演示 `R_depth_cons` 行为。
- `tests/test_object_association.py`：测试同类高 IoU 关联、不同类别不关联、大位移不关联、新对象建轨、空帧、重复帧去重和旧 JSON 兼容。
- `tests/test_depth_temporal_consistency.py`：测试 `R_depth_cons` 公式、无效输入跳过、无 scale prior 类别参与、轨迹/clip 汇聚、pipeline CSV 和 PNG 输出。

### 修改文件
- `src/semantic3d/observations.py`：为 `ObjectObservationJSON` 增加可选字段 `track_id` 和 `canonical_label`，并保持旧 JSON 向后兼容。
- `src/semantic3d/__init__.py`：导出对象关联和深度时序一致性相关 API。
- `docs/DEV_LOG.md`：追加本次开发记录。

### 删除文件
- 无

### 核心公式
```text
p_t = sqrt(mask_area_t / frame_area_t)
z_rel_t = object_depth_t / frame_depth_reference_t
g_t = log(z_rel_t + eps) + log(p_t + eps)
raw_residual = abs(g_current - g_previous)
R_depth_cons = max(0, raw_residual - tolerance)
confidence_weight = sqrt(conf_previous * conf_current)
weighted_residual = confidence_weight * R_depth_cons
```

### 主要变化
- `R_depth_cons` 与对象对级 `R_sd` 分开计算、分开保存、分开汇聚，当前不做融合。
- `R_depth_cons` 不依赖 scale prior，因此没有尺度先验或 `unreliable` 的类别也可以参与时序残差计算。
- 帧级参考深度优先来自 depth map 的有限正值 median；没有 depth map 时，退化为当前帧有效 object.depth 的 median。
- `ObjectAssociator` 会先按 `frame_index` 去重和排序，适配滑动窗口 clip 中重复帧的情况。
- pipeline 会将带 `track_id` 的 observation 另存到 `outputs/observations_with_tracks/`，不覆盖原始 observation。
- demo 中 stable 和合理靠近轨迹残差较低；depth jump 和 projection jump 产生明显较高残差。

### 验证命令
```bash
pytest tests/test_object_association.py -v
pytest tests/test_depth_temporal_consistency.py -v
python examples/demo_depth_consistency.py
python scripts/run_depth_consistency_pipeline.py --observation_dir outputs/depth_direction_batch --depth_map_dir outputs/depth_direction_batch --output_pair_csv outputs/results/depth_consistency_pairs.csv --output_track_csv outputs/results/depth_consistency_tracks.csv --output_clip_csv outputs/results/depth_consistency_clips.csv --iou_threshold 0.1 --center_distance_threshold 0.25 --max_frame_gap 1 --tolerance 0.10
pytest -q
```

### 验证结果
- 新增单元测试通过：`tests/test_object_association.py` 与 `tests/test_depth_temporal_consistency.py` 共 20 个测试通过。
- 全量测试通过：`109 passed, 1 warning`。warning 来自当前机器 CUDA 驱动版本，实际流程使用 CPU，不影响结果。
- `examples/demo_depth_consistency.py` 输出：
  - stable_object：残差序列为 0；
  - approaching_object_consistent：残差序列为 0；
  - depth_jump_anomaly：最大残差约 0.686；
  - projection_jump_anomaly：最大残差约 0.912。
- 真实 observation pipeline 输出：
  - `outputs/results/depth_consistency_pairs.csv`
  - `outputs/results/depth_consistency_tracks.csv`
  - `outputs/results/depth_consistency_clips.csv`
  - `outputs/observations_with_tracks/`
  - `outputs/visualizations/depth_consistency_tracks.png`
- 在当前 `outputs/depth_direction_batch` 小样本上，pipeline 统计为：`total_frames=12`、`total_objects=23`、`total_tracks=8`、`valid_transitions=15`、`skipped_invalid_depth=0`、`skipped_missing_reference=0`、`skipped_frame_gap=0`。
- 当前真实小样本 `mean_R_depth_cons=0`、`max_R_depth_cons=0`，表示在当前阈值、前三帧和现有检测结果上没有发现明显深度时序跳变。

### 当前限制
- 单目深度仍是相对深度，不是米制绝对深度；跨视频或跨场景不能直接比较绝对 depth 数值。
- 当前使用 YOLO bbox_area 近似 mask_area，检测框抖动会影响 projection scale。
- 当前对象关联是几何启发式 tracker，不包含 ReID，不处理长时间遮挡后重新出现。
- 帧级参考深度只能减弱单目深度的全局漂移，不能彻底消除模型帧间不稳定。
- 物体姿态变化、非刚体形变、相机 zoom、错误关联都会影响 `R_depth_cons`，不能夸大为严格三维恢复。
- `outputs/`、`data/videos/`、`checkpoints/` 仍应保持 git 忽略，不提交实验输出、真实视频或模型权重。

### 下一步计划
- 可以先增强 tracking 稳定性，例如引入更稳健的 IoU/运动预测或后续接入真实 tracker，再继续扩大真实视频验证。
- 如果 `R_depth_cons` 继续稳定，可以开始设计 `R_occ` 的最小闭环，但仍应保持各残差项独立输出，等消融实验阶段再讨论融合。
- 在论文实验中分别报告 `R_sd` 与 `R_depth_cons`，避免把对象对尺度-深度一致性和单对象深度时序一致性混淆。

## 2026-07-13 - 修复 R_depth_cons 关联去重与 CSV 对齐可视化

### 目的
修复真实视频 depth consistency 可视化中同一 `track_id` 在同一 `frame_index` 上出现多个点的问题，并确认 `R_depth_cons` 的残差计算、CSV 输出和可视化完全对齐。重点排查重复帧、跨视频 track id 冲突、错误 object association、残差对齐错误和 invalid transition 被绘制为 0 的风险。

### 新增文件
- 无

### 修改文件
- `src/semantic3d/object_association.py`：新增 `AssociationDiagnostics`，记录 `duplicate_frame_count`、`duplicate_track_frame_count` 和 `one_to_many_assignment_count`；关联后统计单帧内重复 track。
- `src/semantic3d/depth_temporal_consistency.py`：新增 `depth_consistency_plot_series_from_csv` 和 `save_depth_consistency_tracks_plot_from_csv`，可视化只从 pair CSV 的 valid transition 绘制，invalid transition 不再被画成 0。
- `scripts/run_depth_consistency_pipeline.py`：先按视频全局 `frame_index` 合并去重，再关联；输出每个视频一份全局去重后的 tracked observation；打印新增诊断统计；绘图改为读取 `depth_consistency_pairs.csv`。
- `tests/test_object_association.py`：新增同一帧内同一 track 不能分配给多个 detection 的测试。
- `tests/test_depth_temporal_consistency.py`：新增重叠 clip 去重、残差公式、invalid 不绘制为 0、图中 residual 数量等于 CSV valid transition 数量等测试。
- `src/semantic3d/__init__.py`：导出新增诊断和 CSV 可视化 API。
- `docs/DEV_LOG.md`：追加本次修复记录。

### 删除文件
- 无

### 主要变化
- 可视化分组键从单独 `track_id` 修正为 `video_id:track_id`，避免不同视频中同名 track 被画成同一条曲线。
- `outputs/observations_with_tracks/` 现在每个视频只保存一份全局去重 JSON，例如 `real_1_associated_tracks.json`，避免重叠 clip 中重复帧造成 `(video_id, frame_index, track_id)` 重复。
- `R_depth_cons` 图中的 residual 点只来自 `depth_consistency_pairs.csv` 中 `valid=True` 的 transition，且位于 `current_frame_index`；没有前驱帧的第一个轨迹点不会被填成 residual=0。
- pipeline 对 valid transition 断言：`residual = max(0, raw_residual - tolerance)`。
- 新增统计输出：`total_unique_frames`、`duplicate_frame_count`、`duplicate_track_frame_count`、`one_to_many_assignment_count`、`mean_raw_residual`。

### 验证命令
```bash
python scripts/run_depth_consistency_pipeline.py --observation_dir outputs/depth_direction_batch --depth_map_dir outputs/depth_direction_batch --output_pair_csv outputs/results/depth_consistency_pairs.csv --output_track_csv outputs/results/depth_consistency_tracks.csv --output_clip_csv outputs/results/depth_consistency_clips.csv --iou_threshold 0.1 --center_distance_threshold 0.25 --max_frame_gap 1 --tolerance 0.10
pytest tests/test_object_association.py tests/test_depth_temporal_consistency.py -v
pytest -q
```

### 验证结果
- 真实 pipeline 统计：
  - `total_unique_frames: 12`
  - `total_objects: 23`
  - `total_tracks: 8`
  - `duplicate_frame_count: 24`
  - `duplicate_track_frame_count: 0`
  - `one_to_many_assignment_count: 0`
  - `valid_transitions: 15`
  - `skipped_invalid_depth: 0`
  - `skipped_missing_reference: 0`
  - `skipped_frame_gap: 0`
  - `mean_raw_residual: 0.009277536099532236`
  - `mean_R_depth_cons: 0.0`
  - `max_R_depth_cons: 0.0`
- 手工检查 `outputs/observations_with_tracks/`：共 23 个 tracked objects，`duplicate_track_frame_count=0`。
- 手工检查 `outputs/results/depth_consistency_pairs.csv`：15 条 valid transition，公式错误数为 0。
- CSV 可视化序列检查：绘图 groups 为 8，residual 点数为 15，等于 CSV 中 valid transition 数量。
- 相关测试通过：`tests/test_object_association.py` 与 `tests/test_depth_temporal_consistency.py` 共 24 个测试通过。
- 全量测试通过：`113 passed, 1 warning`。

### 当前限制
- `duplicate_frame_count=24` 说明输入目录中包含 no-depth、no-invert、invert-depth 等多套 observation 或重叠 clip 的重复帧；pipeline 已在视频全局层面去重，但这个统计仍应保留用于审计数据来源。
- 当前真实小样本 `raw_residual` 都低于 `tolerance=0.10`，因此 `R_depth_cons=0` 是公式阈值裁剪结果，不代表 relative_depth 与 projection_scale 完全没有变化。
- 当前 association 仍是几何启发式，尚未引入 ReID 或运动模型；复杂遮挡或误检仍可能导致错误轨迹。
- 单目深度仍是相对深度，不是米制绝对深度。

### 下一步计划
- 在更长视频和更大帧数上复测 `R_depth_cons`，观察 `raw_residual` 分布后再调 `tolerance`。
- 可增加 `raw_residual` 可视化或分布直方图，辅助选择阈值。
- 在进入 `R_occ` 前，优先增强 tracking 稳定性或接入更可靠的 tracker。

## 2026-07-13 - R_depth_cons 阈值敏感性与可控扰动验证

### 目的
完善真实 `R_depth_cons` 闭环验证，严格区分 `no_depth`、`real_depth_no_invert`、`real_depth_invert` 等深度模式，避免不同 observation 混用；同时补充 raw residual、阈值裁剪 residual 和可控扰动实验，用于判断 `R_depth_cons` 是否对深度跳变、投影尺度跳变具有响应。

### 新增文件
- `scripts/run_depth_consistency_sensitivity.py`：批量扫描多个 tolerance，输出阈值敏感性 CSV、折线图和初步阈值建议 JSON。
- `scripts/generate_depth_consistency_perturbations.py`：基于已有 tracked observation 生成可控扰动样例，包括 depth、mask_area、bbox 和组合扰动。
- `scripts/run_depth_consistency_perturbation_experiment.py`：自动生成多组扰动并运行 `R_depth_cons` pipeline，输出响应曲线和结果 CSV。
- `scripts/visualize_depth_consistency_analysis.py`：从 `depth_consistency_pairs.csv` 生成 raw residual、thresholded residual 和综合分析图。
- `tests/test_depth_consistency_sensitivity.py`：测试深度模式隔离、raw/threshold 可视化和 tolerance 响应。
- `tests/test_depth_consistency_perturbations.py`：测试可控扰动是否改变目标 transition，且不会修改原始 observation。

### 修改文件
- `scripts/run_depth_consistency_pipeline.py`：新增 `--depth_mode`、raw/thresholded/combined 可视化输出参数；按 depth mode 过滤 observation；当 auto 模式发现多个深度模式时直接报错，避免静默混用。
- `src/semantic3d/depth_temporal_consistency.py`：新增从 pair CSV 绘制 raw residual 与 thresholded residual 的函数；可视化只绘制 `valid=True` 的 transition，不把 invalid transition 填成 0。
- `src/semantic3d/__init__.py`：导出新增的 depth consistency 可视化函数。
- `docs/DEV_LOG.md`：追加本次实验记录。

### 删除文件
- 无

### 主要变化
- pipeline 当前支持显式深度模式：`no_depth`、`real_depth_no_invert`、`real_depth_invert`。推荐真实单目深度闭环使用 `real_depth_invert`，因为项目内部约定 depth 数值越大表示越远。
- `raw_residual` 表示未经过 tolerance 裁剪的深度时序几何变化量；`R_depth_cons = max(0, raw_residual - tolerance)`。
- 新增输出图：
  - `outputs/visualizations/depth_consistency_raw_residual.png`
  - `outputs/visualizations/depth_consistency_thresholded_residual.png`
  - `outputs/visualizations/depth_consistency_combined_analysis.png`
  - `outputs/visualizations/depth_consistency_tolerance_sensitivity.png`
  - `outputs/visualizations/depth_consistency_perturbation_response.png`
- 新增输出表：
  - `outputs/results/depth_consistency_sensitivity/depth_consistency_sensitivity.csv`
  - `outputs/results/depth_consistency_threshold_recommendations.json`
  - `outputs/results/depth_consistency_perturbation_results.csv`
- 可控扰动结果显示，深度扰动、投影面积扰动和组合扰动都会使 `raw_residual` 与 `R_depth_cons` 上升，且组合扰动响应最大。

### 验证命令
```bash
python scripts/run_depth_consistency_pipeline.py \
  --observation_dir outputs/depth_direction_batch \
  --depth_map_dir outputs/depth_direction_batch \
  --depth_mode real_depth_invert \
  --output_pair_csv outputs/results/depth_consistency_pairs.csv \
  --output_track_csv outputs/results/depth_consistency_tracks.csv \
  --output_clip_csv outputs/results/depth_consistency_clips.csv \
  --tolerance 0.10

python scripts/visualize_depth_consistency_analysis.py \
  --pair_csv outputs/results/depth_consistency_pairs.csv \
  --output_dir outputs/visualizations \
  --tolerance 0.10

python scripts/run_depth_consistency_sensitivity.py \
  --observation_dir outputs/depth_direction_batch \
  --depth_map_dir outputs/depth_direction_batch \
  --depth_mode real_depth_invert \
  --tolerances 0 0.005 0.01 0.02 0.03 0.05 0.10 \
  --output_dir outputs/results/depth_consistency_sensitivity \
  --visualization_dir outputs/visualizations

python scripts/run_depth_consistency_perturbation_experiment.py \
  --input_observation_dir outputs/depth_direction_batch \
  --depth_map_dir outputs/depth_direction_batch \
  --output_dir outputs/depth_consistency_perturbation_experiment \
  --depth_mode real_depth_invert \
  --tolerance 0.02

pytest tests/test_depth_consistency_sensitivity.py tests/test_depth_consistency_perturbations.py -q
pytest tests/test_depth_consistency_sensitivity.py tests/test_depth_consistency_perturbations.py tests/test_depth_temporal_consistency.py tests/test_object_association.py -q
pytest -q
```

### 验证结果
- 显式 `real_depth_invert` pipeline：
  - `total_unique_frames: 12`
  - `total_objects: 23`
  - `total_tracks: 8`
  - `duplicate_frame_count: 0`
  - `duplicate_track_frame_count: 0`
  - `one_to_many_assignment_count: 0`
  - `valid_transitions: 15`
  - `skipped_invalid_depth: 0`
  - `skipped_missing_reference: 0`
  - `mean_raw_residual: 0.009277536099532236`
  - `mean_R_depth_cons: 0.0`
  - `max_R_depth_cons: 0.0`
- depth mode auto 在同一目录发现多个深度模式时会报错，避免 no-depth/no-invert/invert observation 被静默混合。
- tolerance 敏感性：
  - `tolerance=0.0`：15/15 个 transition 非零，`mean_R=0.009278`，`max_R=0.029746`
  - `tolerance=0.005`：9/15 个 transition 非零，`mean_R=0.005277`
  - `tolerance=0.01`：3/15 个 transition 非零，`mean_R=0.003290`
  - `tolerance=0.02`：3/15 个 transition 非零，`mean_R=0.001290`
  - `tolerance>=0.03`：当前小样本上全部被阈值裁剪为 0
- 当前小样本 raw residual 的 `p95_raw_residual` 约为 `0.02844`。因此 `0.10` 对当前真实小样本过大，后续 debug 可先使用 `0.02` 或围绕 `0.02-0.03` 观察响应；论文最终阈值不能在测试集上调。
- 可控扰动实验：
  - `original`: raw=0.027881, R=0.007881
  - `depth_scale_1_2`: raw=0.210202, R=0.190202
  - `depth_scale_1_5`: raw=0.433346, R=0.413346
  - `depth_scale_2_0`: raw=0.721027, R=0.701027
  - `mask_area_scale_1_2`: raw=0.119042, R=0.099042
  - `mask_area_scale_1_5`: raw=0.230613, R=0.210613
  - `mask_area_scale_2_0`: raw=0.374454, R=0.354454
  - `combined_inconsistent_1_5`: raw=0.636077, R=0.616077
  - `combined_inconsistent_2_0`: raw=1.067600, R=1.047600
- 新增测试通过：`9 passed`。
- 相关 depth consistency 与 association 测试通过：`33 passed`。
- 全量测试通过：`122 passed, 1 warning`。

### 当前限制
- 当前单目深度是相对深度，不是米制绝对深度；只能在同一视频/片段内通过相对变化参与约束。
- `real_depth_invert` 是当前项目约定下的推荐方向，但仍应在新数据集上抽样复核深度方向。
- `R_depth_cons` 当前只验证单对象跨帧深度-投影尺度一致性，不包含 `R_track`、`R_occ` 或 `R_corr`。
- 当前扰动实验是可控合成扰动，用于验证残差响应方向，不等价于真实伪造视频 benchmark 结论。
- 输出目录 `outputs/`、真实视频目录 `data/videos/`、模型权重目录 `checkpoints/` 不应加入 git。

### 下一步计划
- 可以进入 `R_depth_cons` 更大规模真实/伪造样本评估阶段，先固定 `real_depth_invert`，并在验证集上选择 tolerance。
- 进入 `R_depth_cons` 后，建议先保留 raw residual 与 thresholded residual 双输出，避免阈值过大掩盖有效信号。
- 暂时不建议立即实现 `R_occ` 或 `R_track`，应先扩大 `R_depth_cons` 的视频数量并确认 tracking 稳定性。

## 2026-07-13 - 测试视频目录整理与 6 段真实/伪造视频闭环测试

### 目的
基于新整理的测试视频目录，对 4 段真实视频和 2 段 AI 伪造视频运行真实 YOLO 检测、真实单目相对深度、`R_sd` 和 `R_depth_cons` 闭环，确认当前 pipeline 能在分组测试目录上稳定运行。

### 新增文件
- 无

### 修改文件
- `docs/DEV_LOG.md`：追加本次测试记录。

### 删除文件
- 无

### 主要变化
- 输入视频目录采用：
  - `data/tests_videos/tests_real_videos/`：`real_1.mp4` 到 `real_4.mp4`
  - `data/tests_videos/tests_fake_videos/`：`fake_1.mp4`、`fake_2.mp4`
- 输出统一整理到：
  - `outputs/test_videos_batch/real/`
  - `outputs/test_videos_batch/fake/`
  - `outputs/results/test_videos_*.csv`
  - `outputs/visualizations/test_videos_*`
- 对 6 个视频分别生成 no-depth、real-depth-no-invert、real-depth-invert 的 observation 和 `R_sd` summary。
- 采用 `real_depth_invert` 对 6 个视频合并运行 `R_depth_cons`。
- 生成 6 视频总览：
  - `outputs/results/test_videos_depth_direction_summary.csv`
  - `outputs/results/test_videos_overview_summary.csv`
  - `outputs/visualizations/test_videos_overview_scores.png`
- 生成合并 scale-prior 覆盖报告：
  - `outputs/results/test_videos_scale_prior_coverage.csv`
  - `outputs/results/test_videos_scale_prior_candidates.yaml`

### 验证命令
```bash
python scripts/batch_compare_depth_direction_rsd.py \
  --video_dir data/tests_videos/tests_real_videos \
  --output_dir outputs/test_videos_batch/real \
  --summary_csv outputs/results/test_real_depth_direction_batch_summary.csv \
  --visualization_dir outputs/visualizations/test_videos_depth_direction/real \
  --scale_prior_report_csv outputs/results/test_real_scale_prior_coverage.csv \
  --scale_prior_candidates_yaml outputs/results/test_real_scale_prior_candidates.yaml \
  --max_frames 20 \
  --clip_len 8 \
  --stride 4 \
  --confidence_threshold 0.3 \
  --device cpu \
  --object_provider real_detector \
  --real_depth_backend real_depth

python scripts/batch_compare_depth_direction_rsd.py \
  --video_dir data/tests_videos/tests_fake_videos \
  --output_dir outputs/test_videos_batch/fake \
  --summary_csv outputs/results/test_fake_depth_direction_batch_summary.csv \
  --visualization_dir outputs/visualizations/test_videos_depth_direction/fake \
  --scale_prior_report_csv outputs/results/test_fake_scale_prior_coverage.csv \
  --scale_prior_candidates_yaml outputs/results/test_fake_scale_prior_candidates.yaml \
  --max_frames 20 \
  --clip_len 8 \
  --stride 4 \
  --confidence_threshold 0.3 \
  --device cpu \
  --object_provider real_detector \
  --real_depth_backend real_depth

python scripts/run_depth_consistency_pipeline.py \
  --observation_dir outputs/test_videos_batch \
  --depth_map_dir outputs/test_videos_batch \
  --depth_mode real_depth_invert \
  --output_pair_csv outputs/results/test_videos_depth_consistency_pairs.csv \
  --output_track_csv outputs/results/test_videos_depth_consistency_tracks.csv \
  --output_clip_csv outputs/results/test_videos_depth_consistency_clips.csv \
  --associated_observation_dir outputs/observations_with_tracks/test_videos \
  --visualization_path outputs/visualizations/test_videos_depth_consistency_tracks.png \
  --raw_residual_visualization_path outputs/visualizations/test_videos_depth_consistency_raw_residual.png \
  --thresholded_residual_visualization_path outputs/visualizations/test_videos_depth_consistency_thresholded_residual.png \
  --combined_visualization_path outputs/visualizations/test_videos_depth_consistency_combined_analysis.png \
  --tolerance 0.02

python scripts/report_missing_scale_priors.py \
  --observation_dir outputs/test_videos_batch \
  --scale_prior_path configs/scale_priors.yaml \
  --output_csv outputs/results/test_videos_scale_prior_coverage.csv \
  --min_count 1

python scripts/generate_scale_prior_candidates.py \
  --missing_report_csv outputs/results/test_videos_scale_prior_coverage.csv \
  --output_yaml outputs/results/test_videos_scale_prior_candidates.yaml \
  --min_count 2

pytest tests/test_scale_prior.py tests/test_depth_temporal_consistency.py tests/test_object_association.py -q
```

### 验证结果
- 6 视频合并 `R_depth_cons`：
  - `total_unique_frames: 120`
  - `total_objects: 281`
  - `total_tracks: 25`
  - `duplicate_frame_count: 72`
  - `duplicate_track_frame_count: 0`
  - `one_to_many_assignment_count: 0`
  - `valid_transitions: 256`
  - `skipped_invalid_depth: 0`
  - `skipped_missing_reference: 0`
  - `mean_raw_residual: 0.011043929473725053`
  - `mean_R_depth_cons: 0.0034791441638102848`
  - `max_R_depth_cons: 0.1649093772740953`
- `real_depth_invert` 下的 `R_sd` 总览：
  - `real_1`：99 objects，109 R_sd rows，`clip_score_max=0`
  - `real_2`：0 objects，0 R_sd rows
  - `real_3`：96 objects，96 R_sd rows，`clip_score_max=0.7582102150235432`
  - `real_4`：63 objects，0 R_sd rows，主要因为 `vase` 为 unreliable 类别
  - `fake_1`：100 objects，32 R_sd rows，`clip_score_max=0`
  - `fake_2`：87 objects，0 R_sd rows，主要因为 `book` 缺失 scale prior
- `R_depth_cons` 按视频总览：
  - `fake_1`：58 valid transitions，`max_R_depth_cons=0.01994488469513735`
  - `fake_2`：47 valid transitions，`max_R_depth_cons=0.1649093772740953`
  - `real_1`：57 valid transitions，`max_R_depth_cons=0.10875878424809572`
  - `real_2`：0 valid transitions
  - `real_3`：57 valid transitions，`max_R_depth_cons=0.029826955260595662`
  - `real_4`：37 valid transitions，`max_R_depth_cons=0`
- scale-prior 覆盖：
  - exact：`cup`、`person`
  - alias：`dining_table -> table`、`mouse -> handheld_object`、`remote -> handheld_object`
  - missing：`book`、`couch`
  - unreliable：`vase`
- 相关测试通过：`tests/test_scale_prior.py`、`tests/test_depth_temporal_consistency.py`、`tests/test_object_association.py`。

### 当前限制
- `real_2` 没有检测到可参与实验的对象，不能用于当前对象级残差结论。
- `fake_2` 中 `book` 高频出现，但当前无 scale prior，因此 `R_sd` 无法使用这些对象对；需要后续人工审核是否加入 `book` 先验或继续跳过。
- `fake_1` 中 `couch` 高频出现，但当前无 scale prior；如果后续论文实验包含室内场景，建议优先审核 `couch/sofa` 是否作为可靠尺度类别。
- `real_4` 的 `vase` 当前设置为 `reliable=false`，不参与主实验结论。
- 当前测试只说明 pipeline 能跑通并给出残差响应，不能直接作为真实/伪造分类性能结论。

### 下一步计划
- 补充或审核 `book`、`couch/sofa` 的尺度先验策略，决定是否纳入 `R_sd` 主实验。
- 对当前 6 个视频人工检查 debug PNG，确认 `real_depth_invert` 的深度方向仍符合“depth 越大越远”的项目约定。
- 若继续做真实/伪造对比，应扩大视频数量并固定验证集阈值，避免用这 6 个样本直接调参得出论文结论。

## 2026-07-13 - 尺度先验人工维护策略代执行与 couch/book 更新

### 目的
将后续 scale prior 维护明确为工程侧代执行：由项目维护者根据公开尺寸资料和类别稳定性判断补充尺度先验，用户不需要逐项手工估计“计算机视角”的图像尺度。当前先处理 6 视频测试中高频出现的 `couch`、`book` 和已存在的 `vase`。

### 新增文件
- 无

### 修改文件
- `configs/scale_priors.yaml`：新增 `couch` 可靠尺度先验，新增 `book` 非可靠尺度先验，并添加 `sofa -> couch` alias。
- `tests/test_scale_prior.py`：新增 `couch` exact prior、`sofa` alias 和 `book` unreliable 的测试。
- `docs/DEV_LOG.md`：追加本次 scale prior 维护记录。

### 删除文件
- 无

### 主要变化
- `couch`：
  - `min: 1.20`
  - `max: 3.20`
  - `reliable: true`
  - 作为大件家具参与 `R_sd`。
- `sofa`：
  - 通过 alias 映射到 `couch`。
- `book`：
  - `min: 0.12`
  - `max: 0.45`
  - `reliable: false`
  - 只进入覆盖统计和候选审核，默认不参与主 `R_sd`，避免高变化类别污染结果。
- `vase`：
  - 保持 `reliable: false`，默认不参与主 `R_sd`。

### 验证命令
```bash
python scripts/report_missing_scale_priors.py \
  --observation_dir outputs/test_videos_batch \
  --scale_prior_path configs/scale_priors.yaml \
  --output_csv outputs/results/test_videos_scale_prior_coverage.csv \
  --min_count 1

python scripts/generate_scale_prior_candidates.py \
  --missing_report_csv outputs/results/test_videos_scale_prior_coverage.csv \
  --output_yaml outputs/results/test_videos_scale_prior_candidates.yaml \
  --min_count 2

pytest tests/test_scale_prior.py -q

python scripts/run_observation_rsd_pipeline.py \
  --observation_dir outputs/test_videos_batch/fake/fake_1/real_depth_invert_observations \
  --output_csv outputs/results/fake_1_real_depth_invert_rsd_after_scale_prior_update.csv \
  --visualization_path outputs/visualizations/fake_1_real_depth_invert_rsd_after_scale_prior_update.png
```

### 验证结果
- scale-prior 覆盖报告更新后：
  - `couch` 从 `missing` 变为 `exact`，`reliable=True`。
  - `book` 从 `missing` 变为 `unreliable`，`reliable=False`。
  - `vase` 仍为 `unreliable`。
- 候选 YAML 更新后剩余 2 个需要审核/跳过的类别：`book`、`vase`。
- `tests/test_scale_prior.py` 通过。
- `fake_1` 的 `real_depth_invert` R_sd 重算后：
  - `total_objects: 100`
  - `exact_prior_objects: 68`
  - `alias_prior_objects: 32`
  - `computed_pairs: 108`
  - `skipped_missing_prior_objects: 0`
  - `skipped_unreliable_prior_objects: 0`
  - R_sd 行数从原先 32 增至 108。
  - 当前最大 `R_sd_log` 仍为 0，表示加入 `couch` 后，当前视频中的 cup/table/couch 尺度-深度关系仍未触发尺度-深度异常。

### 当前限制
- `couch` 的尺度范围是粗粒度家具尺度，不是精确三维模型尺寸；它适合原型实验，但论文中应表述为 coarse object scale prior。
- `book` 尺度变化较大，且 YOLO 可能把平面纹理误检为 book，因此暂时不纳入主残差。
- 公开资料尺寸只能提供初始范围，后续更严格实验仍应结合数据集统计或人工抽样复核。

### 下一步计划
- 后续遇到高频缺失类别时，由工程侧继续按“公开尺寸资料 + 类别稳定性 + 检测可靠性”维护 `configs/scale_priors.yaml`。
- 对 `book`、`vase` 暂时不作为主实验结论来源；若论文实验必须使用室内场景，可单独做可靠性消融。
