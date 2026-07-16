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

## 2026-07-14 - 4 real + 2 fake 结构残差先导评估闭环

### 目的
在不移动、不复制现有视频的前提下，以 manifest 统一管理
`data/tests_videos/tests_real_videos/` 中 4 个真实视频和
`data/tests_videos/tests_fake_videos/` 中 2 个 AI 伪造视频，完成第一轮可复现的
`R_sd + R_depth_cons` 小规模先导评估。重点检查有效证据覆盖、尺度先验覆盖、跨帧关联
质量和真实/伪造残差的初步分布，不使用当前 2 个伪造视频确定论文最终阈值。

### 新增文件
- `configs/structural_residual_eval.yaml`：固定先导实验的视频采样、YOLO、真实单目相对深度、depth direction、尺度先验和 `R_depth_cons` 参数。
- `scripts/build_video_manifest.py`：从现有 real/fake 视频目录生成并验证 6-video 和 2-video smoke manifest。
- `scripts/run_batch_structural_residual_eval.py`：按 manifest 逐视频运行 observation、对象关联、`R_sd`、`R_depth_cons` 和质量汇总。
- `scripts/analyze_structural_residual_distributions.py`：生成六张分布/证据/覆盖/轨迹质量图和探索性分组摘要。
- `tests/test_batch_structural_residual_eval.py`：验证实际目录、配置约束、manifest、重复 ID、NaN 语义、批量输出和分析图。
- `data/manifests/pilot_real_fake.csv`：4 real + 2 fake 的 `val` manifest。
- `data/manifests/pilot_smoke_2video.csv`：`real_1 + fake_1` 的 smoke manifest。

### 修改文件
- `docs/DEV_LOG.md`：追加本次先导评估记录。
- `docs/PROJECT_STRUCTURE.md`：补充 manifest 驱动的批量评估相关文件和输出目录说明。

### 删除文件
- 无

### 数据规模
- 真实视频：4 个，`real_1.mp4` 至 `real_4.mp4`，`label=0`。
- AI 伪造视频：2 个，`fake_1.mp4`、`fake_2.mp4`，`label=1`。
- split：全部为 `val`。
- 每个视频本轮最多读取前 20 帧，共得到 120 个视频级唯一帧。
- clip 只用于定位和片段汇总，不作为独立视频样本扩大样本量。

### 运行参数
- Python：项目环境 `.venv/bin/python`。
- `object_provider=real_detector`。
- YOLO 权重：`checkpoints/yolov8n.pt`。
- `confidence_threshold=0.30`。
- `depth_provider=real_depth`。
- 深度模型：`depth-anything/Depth-Anything-V2-Small`。
- `invert_depth=true`，`depth_mode=real_depth_invert`。
- 项目约定：depth 数值越大表示越远。
- 单目深度是相对深度，不是米制绝对深度。
- `clip_len=8`，`stride=4`。
- `R_depth_cons tolerance=0.02`，仅为 debug 值，不是论文最终阈值。

### 主要变化
- 空 `R_sd` pair 或空 depth transition 不再汇总为 0，而是写入 `NaN`。
- 每个视频独立保存 observation、depth map、关联轨迹和原始残差，避免跨视频污染。
- 新增 `per_video_features.csv`、`per_clip_features.csv`、`quality_report.csv` 和 `scale_prior_coverage.csv`。
- 新增 `R_sd` 分布、raw `R_depth_cons` 分布、有效 pair 数量、有效 transition 数量、尺度先验覆盖和 track quality 六张图。
- 不计算 AUC、accuracy、最优阈值，不使用 clip 数量替代视频样本量。

### 验证命令
```bash
.venv/bin/python scripts/build_video_manifest.py

.venv/bin/python scripts/run_batch_structural_residual_eval.py \
  --manifest data/manifests/pilot_smoke_2video.csv \
  --config configs/structural_residual_eval.yaml \
  --output_dir outputs/evaluation/pilot_smoke \
  --overwrite

.venv/bin/python scripts/run_batch_structural_residual_eval.py \
  --manifest data/manifests/pilot_real_fake.csv \
  --config configs/structural_residual_eval.yaml \
  --output_dir outputs/evaluation/pilot_6video \
  --overwrite

.venv/bin/python scripts/analyze_structural_residual_distributions.py \
  --evaluation_dir outputs/evaluation/pilot_6video

.venv/bin/python -m pytest tests/test_batch_structural_residual_eval.py \
  tests/test_scale_prior.py tests/test_depth_provider.py \
  tests/test_real_object_provider.py tests/test_object_association.py \
  tests/test_depth_temporal_consistency.py -q

.venv/bin/python -m pytest -q
```

### 验证结果
- smoke：2/2 处理成功，0 个程序失败；两个视频均有有效 `R_sd` pair 和 depth transition。
- pilot：6/6 完成，0 个程序失败；状态为 `ok` 3 个、`partial_evidence` 2 个、`no_evidence` 1 个。
- 总对象 observation：281。
- 有效 `R_sd` pair：200，覆盖 3/6 个视频。
- 有效 `R_depth_cons` transition：256，覆盖 5/6 个视频。
- 可靠尺度先验对象：225/281，实例级覆盖率约 80.1%。
- 所有视频 `duplicate_track_frame_count=0`、`one_to_many_assignment_count=0`。
- 所有视频 `skipped_invalid_depth=0`、`skipped_missing_reference=0`。
- `real_2` 没有检测对象，因此两类残差均为 `NaN`，状态为 `no_evidence`。
- `real_4` 的 vase 为 `reliable=false`，没有有效 `R_sd` pair，但有 37 个有效 transition。
- `fake_2` 的 book 为 `reliable=false`，没有有效 `R_sd` pair，但有 47 个有效 transition。
- 相关测试：54 passed，1 个非阻塞 CUDA 驱动警告。
- 全量测试：127 passed，1 个非阻塞 CUDA 驱动警告。

### 初步分布趋势
- 真实视频（按有证据的视频级统计）：`rsd_log_mean=0.170851`，组内最大
  `rsd_log_max=0.758210`，视频级 `rsd_log_topk_mean` 均值为 `0.377970`。
- 伪造视频：只有 `fake_1` 有 `R_sd` 证据，其 mean/max/top-k 均为 0；`fake_2` 为
  `NaN`，因此当前 `R_sd` 没有呈现“伪造高于真实”的趋势，`real_3` 反而有较高响应，
  需要人工检查检测、深度和尺度先验是否造成真实视频误报。
- raw `R_depth_cons` 的视频级 mean：real `0.010214`，fake `0.011720`，伪造侧略高。
- raw `R_depth_cons` 组内最大值：real `0.128759`，fake `0.184909`。
- raw `R_depth_cons` 视频级 p95 均值：real `0.034957`，fake `0.032257`。
- 这些结果仅显示弱且混合的初步趋势，不能形成正式分类性能结论。

### 当前限制
- 只有 4 个真实和 2 个伪造视频，无法稳定估计真实/伪造分布或最终阈值。
- `real_2` 无目标证据，`real_4` 和 `fake_2` 无可靠 `R_sd` 对，组间比较存在明显缺失偏差。
- 当前每个视频只分析前 20 帧，不代表完整视频的所有时空变化。
- YOLO bbox 面积仍近似 mask area；尚未使用真实实例 mask。
- 单目深度存在相对尺度漂移，不是米制深度。
- tolerance `0.02` 只用于 debug，不能从这 2 个伪造视频反向调成论文阈值。
- 当前未纳入 `R_flow`、`R_track`、`R_occ`、`R_corr`，不能期待覆盖所有伪造类型。

### 下一步计划
- 扩大到更多来源、场景和伪造方法的视频，并保持视频级训练/验证/测试划分。
- 对 `real_2` 检查检测阈值和视频内容；对 `real_3` 人工核查高 `R_sd` 的对象对。
- 对 `book`、`vase` 保持非可靠状态，除非通过尺寸统计与检测质量审核。
- 在独立验证集上选择 tolerance 和融合参数，在独立测试集上一次性报告结果。
- 扩展长视频帧覆盖，并记录抽帧策略对残差分布的影响。

## 2026-07-14 - 独立物理尺度先验审核与严格 R_sd 基线

### 目的
在真实/伪造评估之外独立整理可追溯的物理尺度候选，统一 `R_sd` 的特征线性尺度，
通过确定性规则审核并冻结 `strict_physical_v1`。随后只用冻结先验检查 4 个真实视频和
2 个 AI 伪造视频的 `R_sd` 覆盖率、残差可用性和证据解释，不根据这些视频的投影、
深度、标签、残差或 NaN 反推或放宽先验。

### 新增文件
- `configs/scale_priors_candidates.yaml`：物理候选库；记录统一特征尺度、区间、单位、估计方法、来源、姿态敏感性和多峰风险。
- `configs/scale_priors_strict_v1.yaml`：冻结的 v1 严格物理先验，后续不得覆盖；修订需新建 v2。
- `scripts/audit_physical_scale_priors.py`：生成来源报告并执行可复现的八项可靠性审核。
- `scripts/freeze_strict_scale_priors.py`：从候选和审核结果生成冻结配置，已有目标文件时拒绝覆盖。
- `src/semantic3d/strict_scale_prior.py`：解析 frozen exact/alias/unreliable/missing 先验，只启用 `reliable_single`。
- `src/semantic3d/empirical_pair_prior.py`：未来经验类别对先验的抽象接口；本轮不实例化、不训练、不与 physical prior 混合。
- `scripts/run_strict_rsd_baseline.py`：输出逐 pair 几何量、NaN/skip reason、视频覆盖、用户解释、运行哈希和可视化。
- `tests/test_strict_rsd_baseline.py`：验证严格解析、NaN 语义、聚合、解释、绘图和经验接口隔离。

### 修改文件
- `docs/PROJECT_STRUCTURE.md`：补充候选、审核、冻结、strict baseline 和未来经验接口说明。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件
- 无

### 尺度定义与资料来源
- 全局特征尺度固定为 `sqrt(reference frontal physical height * reference frontal physical width)`，单位为米；球体使用物理直径，因为参考投影的两个维度均为直径。
- `person` 候选来自 NASA 公开人体测量百分位资料，使用站立身高和 frontal inter-scye breadth；由于 YOLO `person` 同时包含儿童和成人，而来源是成人乘员资料，审核为 `conditional_multimodal`。
- `cup` 候选来自 IKEA 与 Hydro Flask 的 4 个公开产品规格，使用直立容器高度和直径；由于 COCO `cup` 合并杯、马克杯、tumbler 和 travel vessel，审核为 `conditional_multimodal`。
- `ball` 候选来自 IFAB Law 2 足球周长标准并换算直径；由于 YOLO 原始类别是广义 `sports ball`，当前映射无法确认足球子类，审核为 `conditional_multimodal`。
- 其余 17 个候选未在本轮收集到满足统一维度和来源要求的正区间，审核为 `unsupported`，没有用常识猜测数值补齐。

### 自动审核规则
- 检查来源 URL、标题和发布者是否可追溯。
- 检查用于区间的来源是否描述同一物理维度。
- 检查单位是否统一换算为米。
- 检查区间是否来自统计、官方标准或足量明确产品规格。
- 检查类别内部严重多峰、姿态/开合敏感性。
- 检查 `max/min <= 2.5`，避免过宽区间失去约束。
- 检查来源数量、发布者多样性和字段完整性。
- 仅 `audit_status=reliable_single` 可进入严格主 `R_sd`。

### 运行参数
- 视频 manifest：`data/manifests/pilot_real_fake.csv`，4 real + 2 fake，全部 `val`。
- observation：读取先导实验已生成并去重关联的前 20 帧结果，不重新训练或下载模型。
- depth：`real_depth_invert` 单目相对深度，不是米制绝对深度；项目约定数值越大表示越远。
- strict config：`configs/scale_priors_strict_v1.yaml`。
- prior source：`physical`。
- empirical pair prior：`false`，未启用。

### 验证命令
```bash
.venv/bin/python scripts/audit_physical_scale_priors.py

.venv/bin/python scripts/freeze_strict_scale_priors.py \
  --output_config configs/scale_priors_strict_v1.yaml \
  --prior_version strict_physical_v1

.venv/bin/python scripts/run_strict_rsd_baseline.py \
  --manifest data/manifests/pilot_real_fake.csv \
  --observation_root outputs/evaluation/pilot_6video \
  --scale_prior_config configs/scale_priors_strict_v1.yaml \
  --evaluation_config configs/structural_residual_eval.yaml \
  --output_dir outputs/evaluation/rsd_strict_baseline \
  --overwrite

.venv/bin/python -m pytest tests/test_strict_rsd_baseline.py -q
.venv/bin/python -m pytest -q
```

### 验证结果
- 来源报告：`outputs/results/scale_prior_source_report.csv`，包含 6 条可追溯来源记录。
- 审核报告：`outputs/results/scale_prior_audit_report.csv`；`conditional_multimodal=3`，`unsupported=17`，`reliable_single=0`。
- 先验版本和评估配置分别以 SHA-256 写入 `outputs/evaluation/rsd_strict_baseline/run_metadata.json`。
- 视频：6/6 完成 strict 诊断，没有程序失败。
- 检测对象：281；候选对象对：278；有效 strict pair：0；严格对象对覆盖率：0%。
- 6/6 视频状态均为 `insufficient_rsd_evidence`，所有 `R_sd` 聚合为 NaN，未写成 0。
- `real_2` 没有检测对象；其余视频的候选 pair 均因至少一侧先验不是 `reliable_single` 被跳过。
- 逐 pair 输出保留 canonical label、先验状态/区间、投影尺度、相对深度、期望区间、NaN、skip reason 和中文解释。
- strict 专项测试：11 passed。
- 全量测试：138 passed，1 个非阻塞 CUDA 驱动警告；本轮 CPU 路径与输出不受影响。

### 初步分布趋势
- 当前没有任何有效 strict `R_sd` pair，因此真实组和伪造组的 `rsd_log_mean` 都是 NaN。
- 本轮无法判断真实与伪造之间是否存在有效残差分布趋势，也不能据此计算 AUC、accuracy 或选择阈值。
- 当前诊断明确属于“严格物理先验覆盖不足”，不是“有效残差没有区分趋势”，也不是“六个视频均正常”。

### 当前限制
- 通用 YOLO 标签往往包含物理多峰子类或显著姿态变化，单一类别区间很难同时满足可追溯、同维度和 `reliable_single`。
- 当前使用 bbox area 近似 silhouette area；空背景、遮挡和视角会进一步削弱物理尺度与投影尺度的对应。
- 单目 depth 只提供相对深度，存在帧间尺度漂移和模型偏差，不是米制测距。
- 严格 v1 零覆盖意味着本轮只能验收“证据管理与解释闭环”，不能验收真实/伪造检测性能。
- 只有 2 个伪造视频，不能训练正式经验类别对先验或夸大 clip 样本量。

### 下一步计划
- 优先在独立 RGB-D/3D 标注或明确子类别数据上收集同维度统计，并生成 `strict_physical_v2`，不覆盖 v1。
- 改善 detector label 粒度和对象状态信息，例如区分成人/儿童、足球/其他球、直立杯/旅行杯，再重新执行同一审核规则。
- 只有在更多独立真实训练视频上积累足量类别对样本后，才实现 `EmpiricalPairPriorResolver`；经验先验必须标记 `prior_source=empirical_pair`，不得与物理先验静默混合。
- 当前尚无有效 strict R_sd，不能据此直接进入主实验融合；`R_depth_cons` 可作为独立时序残差继续验证，但不能由本轮 strict R_sd 结果替代其验收。

## 2026-07-15 - Dimension-Aligned Strict Physical R_sd v2

### 目的
在不修改 strict physical v1 的前提下，解决“不同物理维度无条件与
`sqrt(bbox_area/frame_area)` 比较”的问题。v2 将物理高度、宽度、直径和对角线分别
对齐到对应二维投影量，只允许相同 compatibility group 的对象对计算，并通过不依赖
pilot 六视频调参的轻量几何门控排除明显不完整或不适用的观测。

### 新增文件
- `configs/projected_measurement_rules.yaml`：定义四种投影量、兼容组和通用几何有效性规则。
- `configs/scale_priors_strict_v2.yaml`：冻结的 dimension-aligned v2 物理先验与 observation gate。
- `src/semantic3d/projected_measurement.py`：计算 bbox 高度、宽度、等效直径和对角线投影量。
- `src/semantic3d/physical_prior_gate.py`：执行置信度、边界、尺寸、面积、宽高比、深度和可选轨迹稳定性检查。
- `src/semantic3d/dimension_aligned_scale_depth.py`：解析 v2 先验并计算同维度 ratio/log `R_sd`。
- `scripts/run_strict_rsd_v2.py`：按视频输出 v2 pair、coverage、tier 聚合、解释和元数据。
- `scripts/compare_strict_rsd_versions.py`：生成 v1/v2 pair 与视频覆盖率比较表和三张图。
- `tests/test_projected_measurement.py`：投影量公式及非法 bbox 测试。
- `tests/test_physical_prior_gate.py`：轻量通用门控测试。
- `tests/test_dimension_aligned_rsd.py`：兼容维度、跨轴跳过、conditional 与姿态敏感行为测试。
- `tests/test_strict_rsd_v2.py`：v1 哈希、目录隔离、tier 汇聚、NaN 和版本比较测试。

### 修改文件
- `scripts/run_strict_rsd_baseline.py`：新增 `--video_manifest` 参数别名，并按 prior schema 自动分派 v1/v2；v1 计算路径保持不变。
- `src/semantic3d/__init__.py`：延迟暴露 v2 投影、门控和 dimension-aligned API。
- `docs/PROJECT_STRUCTURE.md`：补充 v2 模块、配置、脚本、测试和输出说明。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件
- 无

### strict v1 冻结核验
- `configs/scale_priors_strict_v1.yaml` 未修改、未覆盖、未删除。
- SHA-256 保持为 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- v1 结果仍保存在 `outputs/evaluation/rsd_strict_baseline/`：0/278 有效 pair，0/6 有效视频。

### 投影测量定义
- `bbox_height_norm = bbox_height / frame_height`，compatibility group 为 `vertical_extent`。
- `bbox_width_norm = bbox_width / frame_width`，compatibility group 为 `horizontal_extent`。
- `equivalent_diameter_norm = sqrt(4*area/pi) / sqrt(frame_width*frame_height)`，compatibility group 为 `radial_extent`；无 mask 时明确标为 bbox-area approximation。
- `bbox_diagonal_norm = sqrt((bbox_width/frame_width)^2 + (bbox_height/frame_height)^2)`，compatibility group 为 `diagonal_extent`。
- 默认只允许同 compatibility group 比较；没有相机内参时禁止跨轴换算。

### v2 物理先验状态
- `soccer_ball`：`strict_high`；物理直径对应 `equivalent_diameter_norm/radial_extent`，来源为 IFAB Law 2。
- `person`：`conditional_physical`；站立身高对应 `bbox_height_norm/vertical_extent`，来源为 NASA 成人乘员人体测量资料。
- `cup`：`conditional_physical`；直立容器高度对应 `bbox_height_norm/vertical_extent`，来源为 IKEA/Hydro Flask 产品规格。
- `bottle`：`conditional_physical`；直立瓶高对应 `bbox_height_norm/vertical_extent`，来源为 Nalgene/Hydro Flask 产品规格。
- `car`：配置车身宽度与 `bbox_width_norm/horizontal_extent`，但 bbox 无法可靠确定正视/后视，保持 `pose_sensitive`。
- table、couch、vase、book、potted_plant 保持 `pose_sensitive`；mouse、remote 保持 `unsupported`。

### 条件门控
- 检查 detection confidence、bbox 边界截断、最小像素尺寸、面积占比、类别宽高比和有限正深度。
- 可选检查最近轨迹的宽高比与投影尺度稳定性；当前冻结配置未强制启用轨迹稳定门控。
- `gate_score` 仅为“通过检查数/检查总数”的 heuristic quality score，不是概率置信度。
- 门控阈值在运行六视频前冻结；配置声明且测试确认未包含 pilot 视频路径或实验结果。

### 证据等级
- 仅当两侧均为 `strict_high` 时，pair tier 为 `strict_high`。
- 只要一侧为 `conditional_physical`，有效 pair 单独记录为 `conditional_physical`。
- 两类证据分别汇聚；`combined_available` 只表示存在任一可用证据，不将 tier 静默混合成论文主分数。

### 运行命令
```bash
.venv/bin/python scripts/run_strict_rsd_baseline.py \
  --video_manifest data/manifests/pilot_real_fake.csv \
  --scale_prior_config configs/scale_priors_strict_v2.yaml \
  --output_dir outputs/evaluation/rsd_strict_v2 \
  --depth_mode real_depth_invert \
  --device cpu \
  --overwrite

.venv/bin/python scripts/compare_strict_rsd_versions.py \
  --v1_dir outputs/evaluation/rsd_strict_baseline \
  --v2_dir outputs/evaluation/rsd_strict_v2 \
  --output_dir outputs/evaluation/rsd_strict_v2

.venv/bin/python -m pytest \
  tests/test_projected_measurement.py \
  tests/test_physical_prior_gate.py \
  tests/test_dimension_aligned_rsd.py \
  tests/test_strict_rsd_v2.py -v

.venv/bin/python -m pytest -q
```

### 验证结果
- v2 prior SHA-256：`3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。
- projected rules SHA-256：`10dce5806f7c126271e9ff97e784ba94a71de8d01a0736f0fd4a9f2f4c4c0bd1`。
- 六视频候选 pair：278；v2 有效 pair：20；pair 覆盖率：7.194%。
- 有效视频：1/6；仅 `real_3` 提供 20 个 person-cup `conditional_physical` pair。
- `strict_high` pair：0；六视频没有形成可用的 soccer_ball-soccer_ball 径向对象对。
- 20 个有效 pair 均使用 `bbox_height_norm + vertical_extent`，物理维度为 person body height 与 cup upright height，满足同轴对齐。
- `real_3` conditional `R_sd_log` mean/max/top-k 分别约为 1.0995/1.1506/1.1441，均为较高偏差；当前不能据此认定伪造，因为该视频标签为 real，且其余视频没有可比 conditional evidence。
- 主要跳过原因：`pose_sensitive_prior_a=102`、`pose_sensitive_prior_b=102`、`unsupported_prior_a=54`；另有 21 个不足两个对象的帧诊断行。
- v1 到 v2：有效 pair 从 0/278 提升到 20/278，有效视频从 0/6 提升到 1/6。
- 专项测试：22 passed。
- 全量测试：160 passed，1 个非阻塞 CUDA 驱动警告；CPU 评估不受影响。

### 当前限制
- 当前只有真实视频 `real_3` 获得有效 conditional evidence，fake 侧为 0 个有效 pair，不能初步比较真实与伪造残差分布。
- person 身高先验来自成人资料；儿童、蹲坐、弯腰仍不适用。cup/bottle 只覆盖所引用的常见直立产品。
- YOLO bbox area 不是实例 mask；ball 等效直径只能标记为 bbox-area approximation，遮挡和背景会带来偏差。
- car 的水平宽度需要正视/后视判断，当前没有视角解析和相机内参，因此保持 pose-sensitive。
- 单目 depth 是相对深度，不是米制绝对距离；项目仍约定 depth 越大表示越远。
- `real_3` 的高 conditional residual 可能来自检测框包含背景、物体姿态、单目深度偏差或物理先验域不匹配，需要独立诊断，不能用本轮视频反调 gate 或区间。

### 下一步计划
- strict `R_sd` 继续作为条件性证据分支，不与 `R_depth_cons` 静默融合。
- 优先接入实例分割以改善 projected area/diameter；对跨轴比较则需要相机内参或可靠标定，不能用当前归一化宽高强行换算。
- 在更大且独立的数据上验证 person/cup/bottle 条件门控，并增加足球等同类径向对象对。
- 经验类别对先验仍只保留接口；只有获得足量独立训练数据后才可实现，且必须标记 `prior_source=empirical_pair`。

## 2026-07-15 - Strict R_sd v2 person–cup Error Attribution

### 目的
对 dimension-aligned strict physical `R_sd` v2 中 `real_3` 的 person-cup 持续高残差进行只读误差归因，区分重叠 clip 重复、公式方向错误、对象区域深度污染、bbox 投影代理失配、姿态门控遗漏和物理先验适用域问题。该诊断不根据 `real_3` 修改物理 min/max，不覆盖冻结配置、正式 observation 或正式 `R_sd`。

### 新增文件
- `src/semantic3d/rsd_v2_error_audit.py`：实现稳定 track/frame pair ID、显式 ratio/log 公式复算、安全 bbox 裁剪、10 种对象区域深度统计、交换不变性和启发式诊断标签。
- `scripts/audit_strict_rsd_v2_errors.py`：生成逐帧、深度策略、投影策略、轨迹对、PNG 和最终 JSON 误差归因结果。
- `tests/test_rsd_v2_error_audit.py`：覆盖去重、公式、NaN、深度策略、图像、报告和冻结哈希等 16 项测试。

### 修改文件
- `src/semantic3d/__init__.py`：延迟暴露误差审计核心数据结构和函数。
- `docs/PROJECT_STRUCTURE.md`：补充误差审计模块、脚本、测试与输出目录说明。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件
- 无

### 主要变化
- 为每个有效对象对生成顺序无关的 `track_pair_id` 和使用视频全局帧号的 `frame_pair_id`；先去重再执行所有诊断策略。
- 原始重叠 clips 中同一全局帧对象对出现 12 次重复，最终正式审计保留 20 个唯一 frame pair，`duplicate_frame_pair_count=0`。
- 20 个 frame pair 全部属于同一个连续 track pair `real_3:trk_000001:trk_000002`，不能解释为 20 个独立场景。
- 同时复算 ratio-space 与 log-space 公式，并检查区间顺序、A/B 交换后的 log 残差不变性和 `real_depth_invert` 深度模式。
- 诊断比较 full bbox、中心 70/50/30%、p25/p50/p75、trimmed mean、前景深度簇和可选 mask median；所有替代值只写入诊断输出。
- 额外比较 `bbox_height_norm`、`sqrt_bbox_area_norm` 和 `bbox_diagonal_norm`，不改变正式 v2 compatibility group。
- 单目深度是相对深度，不是米制绝对深度；工程约定仍为数值越大表示越远。

### 验证命令
```bash
.venv/bin/python scripts/audit_strict_rsd_v2_errors.py \
  --input_dir outputs/evaluation/rsd_strict_v2 \
  --video_dir data/tests_videos/tests_real_videos \
  --video_id real_3 \
  --label_a person \
  --label_b cup \
  --output_dir outputs/evaluation/rsd_strict_v2_error_audit

.venv/bin/python -m pytest tests/test_rsd_v2_error_audit.py -v
.venv/bin/python -m pytest -q
```

### 验证结果
- 唯一 frame pair：20；唯一 track pair：1；最终重复 frame pair：0；原始重叠 clip 额外重复出现：12。
- `formula_error_count=0`、`interval_order_error_count=0`、`swapped_pair_inconsistency_count=0`、`depth_mode_error_count=0`。
- 正式 `mean R_sd_log=1.099453`，20/20 帧均高于 debug threshold 0.1，`persistent_high_residual_ratio=1.0`。
- 最低诊断残差来自 `center_30_bbox_median`，`mean R_sd_log=0.984715`，相对正式结果仅下降约 10.44%，仍为持续高残差。
- 正式 `bbox_height_norm` 的平均残差约 1.0995；诊断性的等效面积与对角线分别约 1.4111 和 1.5436，未显示改用旧面积或对角线能够修复问题。
- person/cup 正式平均相对深度约为 9.2773/4.8290，观测深度比约 1.9216；物理投影关系给出的期望比下界平均约 5.7764，偏差长期稳定。
- person bbox 高度 CV 约 0.0501，cup 高度 CV 约 0.0032，深度比 CV 约 0.0127；结果不是明显的瞬时检测跳变。
- 代表帧显示 person 为坐姿、弯身，bbox 不代表 v2 person 先验要求的完整直立身高；cup 为带盖、带把手的大型杯具。正式门控 20 帧均通过，说明当前通用 aspect/boundary gate 未识别姿态与对象子类型适用域。
- 主要启发式归因：`likely_person_pose_mismatch`；次要归因包括对象区域深度/背景混入与 `likely_prior_domain_mismatch`。这些是诊断标签，不是训练模型预测。
- 实例分割预计有中等帮助，对象区域深度策略优化预计有中等帮助；姿态/子类型门控增强预计帮助较高。不得据此修改冻结 min/max。
- strict v1 SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。
- 专项测试：16 passed；全量测试：176 passed，1 个非阻塞 CUDA 驱动警告，CPU 路径不受影响。

### 当前限制
- 没有真实实例 mask，`mask_median_depth` 保持 NaN；不能直接量化实例分割的真实收益。
- 自动诊断只能从 bbox、深度分布和时间稳定性推测姿态/背景污染，不能替代姿态估计、实例分割或人工帧审查。
- person/cup 只有一个连续轨迹对，因此持续 20 帧不增加独立场景样本量。
- 当前结果只能解释 `real_3` 的误报来源，不能证明其他类别对或其他视频具有相同问题。
- v2 在六视频中仍只有 1/6 视频和 20/278 pair 有效，覆盖率问题尚未解决。

### 下一步计划
- 在独立视频上增加“直立 person / 非直立 person”和普通 cup / 大型带盖杯具的适用性检查，不使用 `real_3` 反调物理区间。
- 优先实现只用于观测质量的姿态/截断 gate，并保留 gate 前后的覆盖与误报对照。
- 评估真实实例 mask 对 cup 投影高度和对象区域相对深度的改善，保持正式 v2 结果只读。
- 在多个独立 track pair 上复现后，再决定门控增强和对象区域深度优化是否进入下一版方法。

## 2026-07-15 - R_sd Physical Prior Applicability Gate

### 目的
根据 strict v2 person-cup 误差归因结果，在不修改冻结物理 min/max 的前提下，判断检测 observation 是否真的能够代表先验定义的特征物理量。重点防止坐姿、弯身、跪姿、截断或关键点不完整的人体 bbox 被当作完整直立身高，并为后续二维/三维点轨迹保留统一关键点数据边界。

### 新增文件
- `src/semantic3d/keypoint_provider.py`：定义 `Keypoint2D`、`KeypointPrediction`、`BaseKeypointProvider`、测试用 mock provider 和 Ultralytics YOLOv8 pose 人体关键点 provider。
- `src/semantic3d/pose_applicability.py`：实现 person 完整直立身高门控、cup upright-height 轻量门控及稳定 skip reason 映射。
- `scripts/run_rsd_applicability_gate_eval.py`：只读加载 frozen strict v2 结果，运行关键点/适用性门控，输出 pair/video/skip/comparison 表和可视化。
- `tests/test_pose_applicability.py`：覆盖直立、坐姿、弯身、缺踝、边界、低置信度、cup 质量、NaN、旧 JSON、真实姿态 smoke 和冻结哈希。

### 修改文件
- `src/semantic3d/dimension_aligned_scale_depth.py`：新增可选 `require_applicability`、person/cup 门控输入、适用性字段和 NaN skip reason；默认关闭以保持旧调用兼容。
- `src/semantic3d/observations.py`：增加可选 `keypoints_2d`、`keypoint_confidences`、`pose_provider`、`keypoint_frame_index` 和 `person_track_id`，旧 JSON 缺少这些字段时仍可读取。
- `src/semantic3d/__init__.py`：延迟暴露关键点和适用性门控公共 API。
- `docs/PROJECT_STRUCTURE.md`：补充新模块、脚本、测试和输出说明。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件
- 无

### 人体关键点 Provider
- 当前真实 provider 使用 `checkpoints/yolov8n-pose.pt` 和 Ultralytics 8.4.90，模型只输出 COCO person 的 17 个二维关键点。
- pose 权重 SHA-256 为 `7f80660bc2f97d664d86fc9f50fd5903af392fe332c0d603fa0dd6c78bf8844c`；权重位于 `.gitignore` 排除的 `checkpoints/`。
- provider 在整帧推理后按 IoU 匹配已有 YOLO person bbox，同一图片结果缓存，非 person 返回 `unsupported_label` 而不报错。
- 当前不实现三维反投影、相机位姿、三维速度、`R_track` 或 `R_reproj`。

### 完整身高适用性门控
- 检查肩/上身、髋、同侧膝和踝的可见性与置信度。
- 检查 bbox 边界截断、躯干相对竖直方向、髋-膝-踝顺序与跨度，以及关键点纵向跨度是否覆盖 bbox。
- 状态包括 `applicable_upright_full_body`、`incomplete_body`、`sitting_or_bending`、`boundary_truncated`、`insufficient_keypoints`、`low_keypoint_confidence` 和 `unresolved_pose`。
- `applicability_score` 是通过检查数占比的 heuristic quality score，不是概率，也不用于识别动作异常。

### Cup 轻量门控
- 检查检测置信度、最小 bbox 宽高、边界、aspect ratio、短轨迹投影高度 CV、bbox 深度 IQR 和有效深度像素比例。
- 状态包括 `applicable`、`too_small`、`boundary_truncated`、`unstable_projection`、`unstable_depth` 和 `pose_unresolved`。
- 当前没有 cup 子类型、开合或姿态分类器；该门控只过滤明显低质量观测。

### 运行命令
```bash
.venv/bin/python scripts/run_rsd_applicability_gate_eval.py \
  --input_dir outputs/evaluation/rsd_strict_v2 \
  --manifest data/manifests/pilot_real_fake.csv \
  --scale_prior_config configs/scale_priors_strict_v2.yaml \
  --pose_model_path checkpoints/yolov8n-pose.pt \
  --device cpu \
  --output_dir outputs/evaluation/rsd_strict_v2_applicability_gate

.venv/bin/python -m pytest tests/test_pose_applicability.py -v
.venv/bin/python -m pytest -q
```

### 门控前后覆盖率
- 六视频候选 pair：278。
- strict v2 门控前有效 pair：20；有效视频：1/6。
- applicability gate 后有效 pair：0；有效视频：0/6。
- person pose/关键点拒绝：20；cup 质量拒绝：0。
- 门控后的 278 个候选 pair 均没有有效 strict physical residual，全部保持 NaN，没有填 0。
- 其余 258 个 pair 原本已经因为 pose-sensitive、unsupported、维度或对象数量等原因无效，适用性门控不覆盖原 skip reason。

### real_3 结果
- 原始 strict v2 person-cup 有效 pair：20；门控后：0。
- 20/20 person 状态均为 `insufficient_keypoints`；20/20 cup 状态为 `applicable`。
- YOLOv8 pose 在代表帧与已有 person bbox 的匹配 IoU 约 0.9202，肩和髋稳定可见，但膝/踝置信度不足或落入错误区域，无法支持完整身高。
- 主要新拒绝原因是 `person_insufficient_keypoints`；门控后 `R_sd_ratio` 和 `R_sd_log` 为 NaN，不是零。
- 这消除了已确认的 person–cup 持续高残差证据。有效 pair 从 20 降为 0 表示提高了物理证据可信度，不代表 real_3 被判为正常，也不应称为算法覆盖退化错误。

### 冻结检查
- strict v1 SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。
- 没有根据 `real_3` 修改 person/cup min-max，也没有覆盖 frozen prior 或正式 v2 输出。
- applicability 专项测试：16 passed，包含真实 `real_3` 姿态推理 smoke test。
- 全量测试：192 passed，1 个非阻塞 CUDA 驱动警告；当前 CPU 推理与结果不受影响。

### 当前限制
- YOLOv8n pose 是通用 COCO 人体关键点模型，没有针对本研究视频的坐姿/遮挡标注做校准；门控状态仍是启发式证据质量判断。
- 对膝/踝完全不可见的 person，系统只能安全报告 `insufficient_keypoints`，不会强行区分坐姿、弯身或遮挡。
- 目前新增门控后六视频没有任何有效 strict `R_sd`，因此不能比较真实/伪造 residual 分布；这再次说明 strict physical `R_sd` 是稀疏条件证据，不是独立完整检测器。
- cup 门控没有真实实例 mask、子类型识别和姿态估计，可能仍接受大型带盖杯具等先验域边缘样本。
- 当前只有一个原本有效的独立 person-cup track pair，不能据此评估门控的普遍召回率。

### 下一步计划
- 在独立、带站立/坐姿/截断标签的数据上验证 person applicability gate 的通过率与拒绝准确性，冻结下一版 gate config。
- 将每帧 `keypoints_2d + person_track_id` 沿现有 object track 串联，先检查二维点轨迹完整性与时间稳定性。
- 在有相机内参、深度和位姿之后再设计三维关键点反投影及 `R_track/R_reproj`；当前不得从二维关键点直接声称三维运动异常。
- 扩大包含直立 person-cup、person-bottle 和同类 ball 对的独立视频，恢复可信 strict physical evidence 覆盖后再进行真实/伪造趋势比较。

## 2026-07-15 - P0 Shared 3D Data Contracts and Depth Semantics

### 目的
在不实现反投影、相机位姿估计或三维残差的前提下，建立后续共享对象中心三维重建所需的数据边界；纠正单目深度语义，隔离几何深度与可视化深度，并为新模块统一使用 `NaN + valid mask` 表示缺失证据。旧二维/2.5D 实验保持兼容，已有实验数值不重算、不改写。

### 新增文件
- `src/semantic3d/geometry/__init__.py`：导出相机数据契约。
- `src/semantic3d/geometry/camera.py`：定义 `CameraObservation`、坐标约定、K/T 验证和显式缺失状态。
- `src/semantic3d/shared_3d_observation.py`：定义二维点、三维点、对象三维观测和共享帧三维观测契约。
- `src/semantic3d/validity.py`：定义 `ResidualEvidence`、稳定缺失原因和 evidence-aware 聚合。
- `tests/test_p0_shared_3d_contracts.py`：覆盖深度语义、相机缺失、三维尺度隔离、NaN 证据、旧接口兼容、元数据复制和冻结哈希。

### 修改文件
- `src/semantic3d/depth_provider.py`：将 `BaseDepthProvider.predict_observation() -> DepthObservation` 设为 canonical 接口；保留旧 `predict_depth()`，增加 legacy adapter，并分离 raw geometry depth、可视化深度和旧 `[1,10]` 归一化深度。
- `src/semantic3d/build_observations.py`：使用 `dataclasses.replace` 替换对象 depth，避免丢失 track、canonical label、关键点、mask、provenance、quality 和 metadata；旧构建流程继续使用 legacy depth 以保持实验兼容。
- `src/semantic3d/observations.py`：为旧 JSON 数据类增加向后兼容的 provenance、quality、metadata 和 depth metadata 字段。
- `src/semantic3d/interfaces.py`：将原 `DepthProvider` 明确标记为 legacy frame-array 接口。
- `src/semantic3d/providers.py`：将基于 label 的 mock depth 明确标记为旧对象测试辅助类。
- `src/semantic3d/real_object_provider.py`：在不改变检测数值的情况下保留对象来源和质量元数据。
- `src/semantic3d/scale_depth.py`：增加 `rsd_2d_coarse` / `rsd_2d_coarse_log` 明确基线名称，旧函数继续可用。
- `src/semantic3d/dimension_aligned_scale_depth.py`：增加 `rsd_2d_dimension_aligned` 兼容名称。
- `src/semantic3d/depth_temporal_consistency.py`：增加 `r_depth_cons_2p5d` 名称和 evidence-aware transition adapter；旧 zero-missing 聚合保持兼容并明确标记 legacy。
- `src/semantic3d/multilevel_residuals.py`：保留旧 zero-missing 对象聚合，新增 `ObjectLevelResidualEvidence` 路径。
- `src/semantic3d/aggregation.py`：为旧 zero-on-empty 默认行为增加明确 legacy 标记和 evidence-aware 使用说明，数值行为不变。
- `src/semantic3d/residual_fusion.py`：保留旧有限值融合，新增严格 evidence-aware 融合。
- `src/semantic3d/__init__.py`：延迟导出 P0 数据契约、明确基线名称和兼容适配器。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件
- 无

### 深度语义与接口
- canonical depth provider 为 `BaseDepthProvider.predict_observation(frame_path, frame_index) -> DepthObservation`。
- `DepthObservation.depth_map` 是供几何使用的深度表示；`raw_model_output` 保留模型原始输出；`visualization_depth` 只用于显示，三者不得互换。
- 当前 Depth Anything 输出按单目相对、inverse-depth-like 语义处理：模型原始值越大通常越近。启用既有 `invert_depth` 语义后，canonical geometry depth 使用倒数转换为“值越大越远”的 `relative_depth`。
- 当前尺度状态为 `relative_per_frame`，不是公制深度，也不声明跨帧共享尺度。
- 旧 `predict_depth()` 仍保留历史逐帧 `[1,10]` 仿射归一化行为，并标记 `legacy_normalized_depth`；canonical 三维接口会拒绝该数据用于几何。

### 相机与共享三维契约
- `CameraObservation` 同时容纳 K、畸变、`T_world_camera` 和 `T_camera_world`，并记录坐标约定、来源、质量和缺失原因。
- 缺少 K/T 时矩阵保持 `None` 且 `valid=false`；禁止用 3x3 单位矩阵冒充有效内参。
- `Shared3DFrameObservation` 统一关联 video/frame、图像尺寸、`DepthObservation`、`CameraObservation`、对象三维观测、有效性、质量、缺失原因和源二维帧。
- `Object3DObservation.observed_scale_3d` 与 `normalized_structure_points` 分开保存；`metric_3d`、`relative_3d` 和 normalized shape 采用独立状态，归一化点不能恢复真实物理尺度。
- 当前无三维重建时使用 invalid observation、`center_3d=None` 和空点集，不填零坐标。

### 残差有效性与兼容边界
- 新 `ResidualEvidence` 允许有效正常残差为 0；无证据时强制 `value=NaN`、`valid=false` 并记录稳定 `missing_reason`。
- 新三维模块只能使用 evidence-aware 聚合，不得调用 legacy zero-missing 路径。
- 为保持旧结果，`depth_temporal_consistency` 的旧 invalid transition/clip 聚合、`multilevel_residuals` 的缺失 flow/track/corr 和旧 `residual_fusion` 仍保留历史行为，并带有 `LEGACY_*` 标记及弃用说明。
- 新增的 depth transition、object residual 和 fusion evidence-aware 路径不会将 missing 转为 0。

### 验证命令
```bash
.venv/bin/python -m pytest tests/test_p0_shared_3d_contracts.py -q
.venv/bin/python -m pytest -q
sha256sum configs/scale_priors_strict_v1.yaml configs/scale_priors_strict_v2.yaml
```

### 验证结果
- P0 专项测试：17 passed。
- 全量测试：209 passed，1 个非阻塞 CUDA 驱动版本警告；CPU 路径不受影响。
- strict v1 SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。
- 没有修改冻结先验，也没有改变或覆盖已有二维/2.5D 实验数值。

### 当前限制
- 尚无真实或标定相机内参、畸变参数和相机位姿；`CameraObservation` 当前只提供数据契约和验证。
- 尚未实现二维点反投影、坐标变换、对象三维中心/边界/结构点重建、公制尺度校准或尺度锚点。
- 当前 Depth Anything 深度是逐帧单目相对深度；未经序列尺度对齐和公制标定，不能与米制尺度先验直接组成正式三维语义尺度残差。
- 尚未实现 `R_semantic_size_3d`、三维点轨迹、遮挡、重投影、相机运动补偿或三维残差融合。
- legacy zero-missing 路径仍存在，仅用于复现实验；新三维实现必须避开这些路径。

### 下一步计划
- P1 从合成已知 K/T/depth 的可验证场景开始实现反投影和坐标变换，不用单位矩阵或默认位姿伪造有效相机参数。
- 增加相机内参来源、序列相对深度尺度对齐和可选公制校准接口，再构建对象三维中心与结构点。
- 先完成反投影/重投影闭环及 NaN/质量掩码测试，再进入三维静态或动态残差。

## 2026-07-16 - P1 Basic 3D Geometry and Sparse Object Reconstruction

### 目的
在 P0 共享数据契约上建立可验证的基础三维几何闭环：统一坐标语义，完成投影、反投影和 camera/world 变换，并根据 canonical Z-depth、对象二维关键点或边界点恢复对象中心稀疏三维结构。本阶段只验证几何和数据共享，不实现三维异常残差、动态轨迹、遮挡、相机位姿估计或真实视频检测指标。

### 新增文件
- `src/semantic3d/geometry/backprojection.py`：实现单点/批量 pinhole 反投影及逐点有效性传播。
- `src/semantic3d/geometry/projection.py`：实现 camera-frame 三维点的单点/批量投影。
- `src/semantic3d/geometry/transforms.py`：实现方向明确的刚体变换、camera/world 转换和世界坐标相机中心。
- `src/semantic3d/reconstruction/__init__.py`：导出 P1 重建公共 API。
- `src/semantic3d/reconstruction/depth_sampling.py`：实现 exact pixel、3x3/5x5 局部中位数、valid mask、IQR 和局部异常值过滤。
- `src/semantic3d/reconstruction/object_3d_reconstructor.py`：实现对象关键点/边界点采样、稀疏反投影、稳健中心、尺度描述、归一化结构和可选 world-frame 输出。
- `src/semantic3d/reconstruction/shared_3d_builder.py`：使用同一 `CameraObservation` 和 `DepthObservation` 构建共享帧三维观测；单对象失败不会中断整帧。
- `tests/synthetic_geometry.py`：生成已知 K、位姿、Z-depth、长方体三维点和二维投影的测试工具，仅用于几何验证。
- `tests/test_camera_geometry.py`：验证坐标语义、单向 pose 求逆、非互逆 pose、approximate K 和约定拒绝。
- `tests/test_backprojection.py`：验证单点/批量投影反投影闭环、非正深度和 valid mask。
- `tests/test_coordinate_transforms.py`：验证单位、旋转平移、互逆变换、相机中心和缺失 pose。
- `tests/test_object_3d_reconstruction.py`：验证局部深度采样、立方体中心/尺度、离群值稳健性、单位隔离、归一化结构和 legacy depth 拒绝。
- `tests/test_shared_3d_builder.py`：验证 camera/depth 实例共享、对象失败隔离和冻结哈希。
- `scripts/demo_synthetic_3d_reconstruction.py`：输出合成投影—重建—重投影闭环结果。

### 修改文件
- `src/semantic3d/geometry/camera.py`：增加 depth、pixel centre 和 transform 约定；增加 canonical `T_world_from_camera/T_camera_from_world` 属性和构造入口；只提供一个方向时自动求逆，非互逆时显式 invalid。
- `src/semantic3d/geometry/__init__.py`：延迟导出 P1 几何 API，避免数据契约循环导入。
- `src/semantic3d/depth_provider.py`：几何入口额外要求 `larger_value_means=farther`，防止未转换 inverse depth 进入 Z-depth 几何。
- `src/semantic3d/shared_3d_observation.py`：扩展 camera/world 中心和点集、尺度单位/方法/质量、尺度描述、重建坐标域和深度尺度域；允许有效共享帧保留无效对象记录。
- `src/semantic3d/__init__.py`：公开 P1 几何与重建接口。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件
- 无

### 坐标与矩阵约定
- camera frame：X 向右、Y 向下、Z 向前。
- pixel frame：u 向右、v 向下；整数图像索引表示 pixel centre，不额外加 0.5 offset。
- depth：沿相机光轴的 Z-depth，不是 Euclidean range。
- 点使用齐次 column vector；变换左乘。
- `T_world_from_camera` 将 camera 点变到 world；`T_camera_from_world` 是其逆。
- 相机中心世界坐标是 `T_world_from_camera[:3, 3]`，等价于从 world-to-camera 外参计算 `-R^T t`。

### 对象稀疏三维恢复
- 默认使用 `local_median_3x3` 采样深度，保存 source pixel、sampled Z、method、local valid ratio、depth IQR 和 point quality。
- 每个无效采样点保留 invalid point 和 missing reason，不填零坐标。
- 对象中心默认采用有效结构点逐轴中位数；可配置 geometric median。
- 尺度描述包括 axis-aligned extent、bounding-box diagonal、PCA principal extents 和 equivalent linear scale；`observed_scale_3d` 当前采用 bounding-box diagonal。
- 归一化点使用 `(point - center) / observed_scale`，只表达 unitless shape，不能恢复物理尺度。
- metric-calibrated depth 输出 `metric_3d + meter`；relative depth 输出 `relative_3d + relative_unit`。
- `relative_per_frame` 对象会拒绝跨帧尺度直接比较；没有调用 person/cup/car 等类别物理 min-max。
- pose 缺失时 camera-frame 对象仍可有效，world-frame 字段为 `None` 并记录 `no_camera_pose`。

### 合成闭环结果
- 输出目录：`outputs/geometry_demo/`。
- `synthetic_points.csv`：逐点 ground truth、投影、重建、重投影及误差。
- `reconstructed_object.json`：对象中心、点集、尺度、单位、质量和 provenance。
- `reprojection.png`：原始投影与重投影对照。
- `geometry_report.json`：闭环误差和坐标/尺度状态。
- mean 3D reconstruction error：`3.1401849173675503e-16`。
- max 3D reconstruction error：`3.1401849173675503e-16`。
- mean/max reprojection error：`0.0 px`。
- center error：`0.0`；scale error：`0.0`。
- 以上是 synthetic metric ground truth 验证，不代表真实视频公制重建精度。

### 验证命令
```bash
.venv/bin/python -m pytest \
  tests/test_camera_geometry.py \
  tests/test_backprojection.py \
  tests/test_coordinate_transforms.py \
  tests/test_object_3d_reconstruction.py \
  tests/test_shared_3d_builder.py -q

.venv/bin/python scripts/demo_synthetic_3d_reconstruction.py
.venv/bin/python -m pytest -q
sha256sum configs/scale_priors_strict_v1.yaml configs/scale_priors_strict_v2.yaml
```

### 验证结果
- P0 与 P1 组合专项：44 passed；对象重建专项：9 passed。
- 全量测试：236 passed，1 个原有非阻塞 CUDA 驱动警告；CPU 路径不受影响。
- strict v1 SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。
- 未修改已有二维/2.5D 计算接口、结果文件或 frozen physical prior。

### 当前限制
- 当前真实视频没有真实标定 K 和正式相机位姿；approximate K 只能标为 `intrinsics_source=approximate` 且 quality < 1。
- 当前真实 Depth Anything 数据为 `relative_per_frame`，因此最多恢复 camera-frame relative 3D；不能声称米制尺度，也不能直接比较跨帧速度或物理尺寸。
- 没有实现相机内参估计、相机位姿估计、序列尺度对齐、公制尺度校准或 distortion correction。
- 没有实现 `R_semantic_size_3d`、深度层次/空间关系残差、三维轨迹、速度、遮挡、跨帧重投影残差或真假分类。
- 合成闭环只证明公式和软件实现一致，不证明单目深度或检测器在真实场景中的精度。

### 下一步计划
- 在进入正式 P2 前，为真实输入明确 approximate/calibrated K 来源，并决定静态分支使用 camera-frame relative 3D 还是经过外部标定的 metric 3D。
- P2 可先实现不依赖公制单位的同帧三维深度层次和空间关系证据；三维语义物理尺度残差必须等待 metric calibration 或有理论依据的相对尺度锚点。
- 所有 P2 残差继续使用 `ResidualEvidence`，缺失证据保持 NaN，不调用 legacy zero-missing 聚合器。

## 2026-07-16 - P1.5 Real-Frame 3D Smoke and P2 Static 3D Evidence Foundation

### 目的
在 P1 合成几何闭环之后，以真实视频单帧验证 YOLO、canonical Depth Anything 深度、二维人体关键点、approximate K 和 `Shared3DFrameBuilder` 的连接，并建立首批 evidence-aware 静态三维质量、深度层次、边界深度、空间相交和语义尺度接口。本阶段只建立 camera-frame relative sparse 3D 与静态证据基础，不实现动态三维轨迹、相机运动补偿、动态遮挡、融合训练或真假分类指标。

### 新增文件
- `scripts/run_real_frame_3d_smoke.py`：解码单个真实视频帧，运行真实对象/深度/可选人体关键点 provider，使用 approximate K 构建共享三维帧，并输出 JSON、CSV、当前帧闭环 QA 图和质量报告。
- `src/semantic3d/static_3d/__init__.py`：导出 evidence-aware 静态三维模块。
- `src/semantic3d/static_3d/residual_types.py`：定义静态证据角色、共享帧只读上下文和 `reconstruction_cycle_error` QA。
- `src/semantic3d/static_3d/reconstruction_quality.py`：实现重建质量证据与后续残差 gate 所需诊断字段。
- `src/semantic3d/static_3d/depth_order.py`：实现对象中心前后关系诊断和有显式二维遮挡方向时的深度一致性残差。
- `src/semantic3d/static_3d/boundary_depth.py`：实现 mask 内外深度跳变一致性；bbox 仅作稀疏、低质量 fallback。
- `src/semantic3d/static_3d/spatial_intersection.py`：实现 sparse point/AABB 空间相交诊断，不将 AABB 重叠解释为确定物理穿插。
- `src/semantic3d/static_3d/semantic_size.py`：实现 metric、独立锚点、leave-one-out 和同帧相对比例四类语义三维尺度证据路径。
- `tests/test_real_frame_3d_smoke.py`：验证视频解码、共享三维输出、approximate K、canonical depth、world 字段缺失和 QA 角色。
- `tests/test_static_3d_reconstruction_quality.py`：验证质量证据与异常分数隔离、缺失 NaN 和闭环 QA。
- `tests/test_depth_order_3d.py`：验证中心前后关系不直接判异常、遮挡方向一致/矛盾和缺失关系 NaN。
- `tests/test_boundary_depth_3d.py`：验证 mask 优先、bbox fallback 降质和缺失边界 NaN。
- `tests/test_spatial_intersection_3d.py`：验证 AABB/sparse point 仅为低权重诊断和点数不足 NaN。
- `tests/test_semantic_size_3d.py`：验证 metric 区间距离、relative/metric 隔离、独立锚点、防自证、leave-one-out 和冻结哈希。

### 修改文件
- `src/semantic3d/reconstruction/object_3d_reconstructor.py`：在对象 provenance 中保留 source bbox/mask，并为每个三维点保留原始二维坐标，使当前帧闭环误差可直接核验。
- `src/semantic3d/__init__.py`：延迟导出静态三维 evidence-aware 公共 API。
- `tests/synthetic_geometry.py`：增加显式 metric/relative 合成对象及共享帧构造器。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件
- 无

### 主要变化
- 所有正式静态三维接口返回 `ResidualEvidence`；有效正常值允许为 0，缺失证据强制为 NaN 并带 `missing_reason`。
- `ReconstructionQualityEvidence` 汇总有效点比例、关键点/边界比例、点质量、局部 depth IQR、approximate K 质量、闭环误差和尺度状态；该值是 heuristic quality，不是概率，也不是伪造异常分数。
- `center_depth_order` 只描述前后顺序；只有具备对象重叠和明确 foreground object 的 `occlusion_depth_consistency` 才形成异常残差。
- `BoundaryDepth3DResidual` 可在相对深度下工作；实例 mask 优先，bbox 只采样稀疏中段边缘且将来源质量降为 0.30。
- `SpatialIntersection3DResidual` 输出 AABB overlap、稀疏点距离和 center-inside 诊断，metadata 明确 `approximation=aabb_or_sparse_points`、`definite_physical_penetration=false`。
- `SemanticSize3DResidual` 仅在 `metric_3d + meter + metric_scale_source` 时直接比较米制类别区间；`relative_per_frame` 禁止 unary metric 比较，只能使用独立锚点、leave-one-out 或同帧比例证据。
- 锚点对象不得评价自身；leave-one-out 明确排除目标对象，metadata 保存 calibration source、anchor ids、excluded ids 和 estimated scene scale。
- 当前帧同点投影闭环命名为 `reconstruction_cycle_error`，角色固定为 QA；它检查 K、坐标约定和点数据，不能作为异常重投影残差。
- 所有静态模块读取同一个 `Shared3DFrameObservation`，不重新调用 depth provider、不重新估计 K，也不复制后修改对象三维观测。

### 真实帧 smoke 结果
- 输入：`data/tests_videos/tests_real_videos/real_1.mp4`，global frame index 0，CPU。
- 输出目录：`outputs/real_frame_3d_smoke/`。
- 检测对象：2 个 `mouse`、1 个 `dining_table`；对象数 3，有效三维对象数 3。
- `Shared3DFrameObservation.valid=true`，frame quality=`0.7159003381374762`。
- 深度：`transformers:depth-anything/Depth-Anything-V2-Small-hf`，`relative_depth + relative_per_frame`，数值越大表示越远；不是米制深度。
- 相机：`intrinsics_source=approximate`、quality=0.5；没有相机位姿，因此 camera-frame 中心有效，全部 world 字段为 `None`。
- 有效重建点 24 个；三对象同点 `reconstruction_cycle_error=0.0 px`，仅表明同一 K/点的投影闭环一致，不证明深度或检测精度正确。
- 输出文件：`shared_3d_frame.json`、`object_3d_observations.csv`、`reconstructed_points_3d.csv`、`current_frame_reprojection.png`、`reconstruction_quality_report.json`，另保留精确输入帧 `source_frame.png`。

### 验证命令
```bash
.venv/bin/python -m pytest \
  tests/test_real_frame_3d_smoke.py \
  tests/test_static_3d_reconstruction_quality.py \
  tests/test_depth_order_3d.py \
  tests/test_boundary_depth_3d.py \
  tests/test_spatial_intersection_3d.py \
  tests/test_semantic_size_3d.py -q

.venv/bin/python scripts/run_real_frame_3d_smoke.py \
  --video_path data/tests_videos/tests_real_videos/real_1.mp4 \
  --frame_index 0 \
  --output_dir outputs/real_frame_3d_smoke \
  --device cpu

.venv/bin/python -m pytest -q
sha256sum configs/scale_priors_strict_v1.yaml configs/scale_priors_strict_v2.yaml
```

### 验证结果
- P1.5/P2 静态专项：21 passed。
- 全量测试：257 passed，1 个原有非阻塞 CUDA 驱动警告；CPU 路径和本轮结果不受影响。
- strict v1 SHA-256 仍为 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 仍为 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。
- 未修改冻结先验、当前六视频阈值或已有二维/2.5D 实验数值。

### 当前限制
- approximate K 可用于软件闭环和粗略 camera-frame 几何，但不能替代真实标定内参；当前无 distortion calibration 和相机位姿。
- Depth Anything 仍是逐帧单目相对深度。不同帧的尺度不保证一致，当前结果不能称为公制三维，也不能直接计算可信物理速度。
- bbox 边界不是真实实例轮廓；没有实例 mask 时 boundary-depth 和对象 extent 的物理解释受限。
- 空间相交仅为低置信 AABB/sparse point 诊断，不能证明真实物体穿插。
- 当前六视频没有 metric-calibrated depth，也没有经独立验证的 scene-scale anchor，因此不会产生 unary metric `SemanticSize3DResidual`；无证据必须保持 NaN。
- 当前帧同点重投影不独立，不能用于伪造异常判断。静态异常重投影仍需留出点、独立 mask/boundary 或拟合结构模型。

### 下一步计划
- 在进入正式动态三维分支前，优先获得序列共享尺度深度或公制标定，并接入可靠相机位姿/运动补偿；否则跨帧 3D 位移和速度不具有稳定尺度语义。
- 在实例 mask 可用后，将 bbox fallback 替换为真实边界，独立验证 boundary-depth 证据质量。
- P3 可以先建立不声称物理速度的 evidence-aware 数据接口和合成闭环测试；真实动态三维异常实验必须等待跨帧尺度与相机运动问题得到解决。
- 保持当前六视频只用于验证管线与覆盖，不用它们回调物理先验或训练锚点分布。

## 2026-07-16 - P3-0 Shared Sequence Geometry and Depth Alignment Foundation

### 目的
在 P0/P1/P2 的单帧共享三维契约和静态证据基础上，建立供后续静态、动态分支共同使用的序列级三维几何表示。本阶段只验证相机位姿契约、动态前景排除、镜头切换边界、背景深度序列对齐和几何质量传播；不实现动态伪造残差、三维速度异常、遮挡异常、真假分类或残差融合训练。

### 新增文件
- `src/semantic3d/sequence_geometry/__init__.py`：统一导出 P3-0 序列几何数据契约、provider、深度对齐和质量 API。
- `src/semantic3d/sequence_geometry/observation.py`：定义 `Shared3DClipObservation`、`SequenceGeometryObservation`、`RelativePoseObservation`、`DepthAlignmentObservation`、`SequenceScaleStatus` 和 `DepthAlignmentMode`。
- `src/semantic3d/sequence_geometry/provider.py`：实现 canonical provider 接口、synthetic/mock/legacy provider、统一 provider 预留接口、前景 mask、背景位姿估计和镜头切换检测。
- `src/semantic3d/sequence_geometry/depth_alignment.py`：实现 `scale_only`、`affine_depth`、`affine_inverse_depth`、`provider_shared_scale` 和 unsupported 路径，以及基于背景对应点的鲁棒拟合。
- `src/semantic3d/sequence_geometry/quality.py`：实现 `SequenceGeometryQuality`；该值是工程几何质量，不是伪造异常分数或概率。
- `scripts/run_real_clip_sequence_geometry_smoke.py`：对真实短片段运行 YOLO、canonical Depth Anything、approximate K、背景 VO 和深度对齐，输出序列契约、逐帧诊断 CSV、质量 JSON 和 PNG。
- `tests/synthetic_sequence_geometry.py`：提供移动相机、移动对象、镜头切换、静态背景和深度尺度漂移的合成序列工具。
- `tests/test_sequence_geometry_observation.py`：验证 pose missing/identity 区分、camera/world 方向、真实世界静态点、移动对象、scale status、provider 契约和冻结哈希。
- `tests/test_sequence_depth_alignment.py`：验证 scale-only、affine depth、affine inverse-depth、鲁棒离群值处理、失败 NaN 和状态提升。
- `tests/test_sequence_foreground_scene_cut.py`：验证 mask/bbox 前景排除、超大 bbox 安全处理、静止背景 identity 证据和 scene cut。
- `tests/test_sequence_geometry_quality.py`：验证序列几何质量字段、计数和“非异常分数”语义。
- `tests/test_real_clip_sequence_geometry_smoke.py`：使用真实视频解码和注入式轻量 provider 验证 smoke 的 9 个输出文件及关键字段。

### 修改文件
- `src/semantic3d/__init__.py`：延迟导出 P3-0 公共接口，保持现有二维、2.5D、P1/P2 API 可用。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件
- 无

### 主要变化
- `Shared3DClipObservation` 统一保存 video/clip 标识、全局帧索引、共享三维帧、参考帧、双向 camera/world 变换、相对位姿、深度对齐、scene cut、背景轨迹、前景对象、provider、质量和缺失原因。
- `RelativePoseObservation` 规定 `X_current = T_current_from_previous X_previous`；`T_world_from_camera` 与 `T_camera_from_world` 必须互逆。参考帧单位矩阵是显式 coordinate gauge，不计入跨帧位姿成功率。
- 有背景支持和估计证据的静止相机可输出有效 identity relative pose；位姿缺失时所有矩阵为 `None`、重投影误差为 NaN、`valid=false`，不能用单位矩阵冒充。
- 相机位姿支持点先排除对象 mask；无 mask 时使用膨胀 bbox 并降低质量。覆盖近乎整帧的 bbox fallback 不会删除全部背景，而是记录 `skipped_oversized_bbox_ids`。
- 背景跟踪使用 Shi-Tomasi、LK forward/backward consistency 和目标帧前景排除；背景支持不足或 essential/recoverPose 内点不足时显式 invalid。
- scene cut 同时支持 histogram/content difference、feature matching collapse 和 provider flag；切换边界截断 clip，不生成跨切换位姿或轨迹。
- 深度对齐仅使用背景稳定对应点和 canonical depth，采用 MAD 鲁棒拟合并记录 support、inlier ratio、fit error 和 missing reason；不调用 person/car/cup 等尺度先验，也不以待检测对象作尺度锚点。
- `relative_per_frame` 只有在所有必要对齐与位姿条件成立时才允许提升；局部对齐成功但位姿链断裂不会错误提升为 `relative_aligned_sequence`。
- 只有 `metric_sequence`、`relative_shared_sequence` 或 `relative_aligned_sequence` 才具备未来动态三维计算的尺度条件；还必须同时具有连续有效位姿、无 scene cut 和兼容的 pose/depth scale。
- `SyntheticSequenceGeometryProvider` 使用已知真值；`MockSequenceGeometryProvider` 不伪造 pose；`LegacyDepthPoseSequenceAdapter` 组合 canonical frame/depth 与外部 pose；`UnifiedSequenceGeometryProvider` 仅作为未来统一 K/pose/depth/correspondence 模型的抽象接口。

### 合成验证结果
- 静态场景、移动相机补偿后对象背景世界中心 mean/max error 均为 `0.0`（测试容差 `1e-9`）。
- 静止相机在有静态背景支持时输出有效 identity；缺失 pose 与 identity 被严格区分。
- 移动前景被 source/target foreground mask 排除，未污染静态背景 identity 估计。
- 移动相机与移动对象场景保留真实 world-frame 对象位移，不把相机位移重复叠加到对象。
- 合成 affine inverse-depth 真值 `scale=1.7`、`shift=0.25`，恢复值分别为 `1.7` 和 `0.2500000000000002`，fitting error=`2.364279820704291e-16`。
- 对齐支持不足时 scale/shift/error 为 NaN 且 invalid；未对齐的 `relative_per_frame` 禁止动态三维计算。

### 真实短片段 smoke
- 输入：`data/tests_videos/tests_real_videos/real_1.mp4`，global frame 0-7，共 8 帧，CPU，无检测到的 scene cut。
- 对象来源：真实 YOLO；深度来源：`transformers:depth-anything/Depth-Anything-V2-Small-hf` canonical depth，语义为单目相对深度，不是米制深度。
- K 来源：`approximate`；pose source：`opencv_background_visual_odometry`；深度对齐模式：`affine_depth`。
- 参考帧 0 为有效 coordinate gauge。7 个跨帧 transition 均未形成连续有效 pose，`valid_pose_ratio=0.0`；主要原因为 `insufficient_pose_inliers` 和后续 `pose_chain_broken`。
- 7 个深度对齐 transition 中 3 个局部拟合有效，`depth_alignment_valid_ratio=0.42857142857142855`；mean fitting error=`0.0014882170988172983`。
- 背景支持比例=`0.999463036545073`，背景内点比例=`0.42900042900042895`；没有有效跨帧位姿时 mean background reprojection error 保持 NaN。
- `sequence_scale_status=relative_per_frame`，没有提升为 `relative_aligned_sequence`；`SequenceGeometryQuality.quality=0.48194361106463196`。
- `allows_dynamic_3d=false`。当前真实结果只证明 smoke 管线、有效性传播和失败诊断工作，不能声称恢复了真实相机轨迹或可计算可信三维速度。
- 输出目录：`outputs/real_clip_sequence_geometry_smoke/`。
- 输出文件：`shared_3d_clip.json`、`camera_trajectory.csv`、`relative_poses.csv`、`depth_alignment.csv`、`background_tracks.csv`、`background_reprojection.csv`、`sequence_geometry_quality.json`、`camera_trajectory.png`、`reprojection_diagnostics.png`。
- 本地已有模型缓存时可使用 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 运行，避免网络元数据请求影响复现。

### 验证命令
```bash
.venv/bin/python -m pytest \
  tests/test_sequence_geometry_observation.py \
  tests/test_sequence_depth_alignment.py \
  tests/test_sequence_foreground_scene_cut.py \
  tests/test_sequence_geometry_quality.py \
  tests/test_real_clip_sequence_geometry_smoke.py -q

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/run_real_clip_sequence_geometry_smoke.py \
  --video_path data/tests_videos/tests_real_videos/real_1.mp4 \
  --start_frame 0 \
  --num_frames 8 \
  --output_dir outputs/real_clip_sequence_geometry_smoke \
  --device cpu \
  --depth_alignment_mode affine_depth

.venv/bin/python -m pytest -q
sha256sum configs/scale_priors_strict_v1.yaml configs/scale_priors_strict_v2.yaml
```

### 验证结果
- P3-0 专项：22 passed。
- 全量测试：279 passed，1 个原有 CUDA 驱动版本警告；CPU 路径与本轮结果不受影响。
- strict v1 SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。
- 未实现或计算任何正式动态异常残差、真假分类、三维速度、遮挡异常或融合分数；没有修改已有二维/2.5D 实验数值和冻结先验。

### 当前限制
- 真实 smoke 使用 approximate K，无 distortion calibration；camera trajectory 即使恢复成功也只能是相对 pose unit，不能称为米制轨迹。
- 当前 OpenCV essential-matrix 背景 VO 对近静止、微小基线或退化场景不稳。本次真实片段没有形成连续有效位姿链。
- Depth Anything 仍输出逐帧单目相对深度。局部 affine 对齐成功不能替代完整序列共享尺度证明；本次仍为 `relative_per_frame`。
- bbox 是无实例 mask 时的低质量前景 fallback；超大检测框跳过前景遮罩是保留背景支持的工程策略，并不等价于精确前景分割。
- `SequenceGeometryQuality` 是启发式工程质量汇总，不是训练概率、异常分数或真假证据；几何估计失败也不等于视频伪造。
- 当前不能声称完成动态三维伪造检测、真实米制相机轨迹、真实三维速度、跨帧异常残差或遮挡异常检测。

### 下一步计划
- 软件契约、合成真值和失败传播已具备进入 P3-A 的接口基础，但本次真实片段尚不满足可信动态三维计算条件。
- 在正式实现三维轨迹与动态重投影前，应先提高真实 pose 连通率：支持静态/微基线退化检测、rotation-only/homography fallback、可靠外部 VO/SLAM pose 或统一几何 provider。
- 需要在独立真实序列上验证 depth 对齐的跨帧闭环和尺度漂移，并仅在完整条件满足后提升为 `relative_aligned_sequence`。
- 后续动态模块必须直接复用 `Shared3DClipObservation`，保持前景 mask、K、pose、depth、对象三维点和质量来源一致，不得另建一套不兼容几何状态。

## 2026-07-16 - P3-0.5 Real Sequence Geometry Stabilization

### 目的
解决 P3-0 真实短片段中低视差被统一判为位姿失败、单条相邻边导致后续 pose chain 断裂，以及逐帧相对深度缺少序列级一致尺度的问题。本阶段只建立相机运动状态分类、分层位姿候选、位姿图、带留出集的深度模型选择、序列级深度对齐和联合有效性门控；不计算动态伪造残差、真假分类、对象速度异常、结构时序异常、遮挡异常或融合分数。

### 新增文件
- `src/semantic3d/sequence_geometry/motion_regime.py`：根据背景支持、流量、视差、homography/essential 支持和图像差异区分静止、旋转主导、一般 SE3、背景不足、低纹理、退化几何和镜头切换。
- `src/semantic3d/sequence_geometry/pose_estimation.py`：实现前景过滤 LK 诊断、静止 identity、rotation/homography、essential SE3、depth-assisted PnP 及 `t-1/t-2/t-4` 自适应候选；旋转模型不冒充完整 SE3。
- `src/semantic3d/sequence_geometry/pose_graph.py`：实现相邻边和跳帧边构图、参考帧传播、连通分量、断开帧和质量统计；缺失边不补单位矩阵。
- `src/semantic3d/sequence_geometry/stabilization.py`：组合位姿图、深度对齐图和联合有效性门控；无效动态三维帧使用 NaN 和具体 missing reason。
- `scripts/run_sequence_geometry_stabilization.py`：对静止、慢移动和明显移动三个真实 8 帧片段执行 YOLO、真实单目相对深度、运动分类、位姿图和深度全局对齐，并输出九类诊断文件。
- `tests/test_motion_regime.py`：验证有证据静止 identity、无证据不生成 identity 以及运动状态分类。
- `tests/test_pose_estimation_fallbacks.py`：验证 reference gauge、rotation-only、general SE3、前景排除和分层回退语义。
- `tests/test_pose_graph.py`：验证跳帧连接、单边失败、scene cut 和不连通帧的 NaN 传播。
- `tests/test_sequence_depth_alignment_global.py`：验证 raw inverse-depth、fit/holdout 隔离、错误模型拒绝、全局对齐和尺度状态升级。
- `tests/test_real_sequence_geometry_stabilization.py`：验证真实三片段 runner 的固定输出、无语义先验、冻结哈希和 NaN 门控。

### 修改文件
- `src/semantic3d/sequence_geometry/observation.py`：为深度对齐观测增加 holdout error/count 和 physical validity，保持旧构造方式兼容。
- `src/semantic3d/sequence_geometry/depth_alignment.py`：增加三种候选模型、确定性 fitting/holdout、物理有效性检查、全局图传播和 reference-connected 全局仿射求解；inverse 模式使用 `raw_model_output`，不读取 visualization depth。
- `src/semantic3d/sequence_geometry/__init__.py`：统一导出 P3-0.5 公共 API。
- `src/semantic3d/__init__.py`：延迟导出新增公共接口，保留旧二维、2.5D、P1、P2 和 P3-0 接口。
- `src/semantic3d/sequence_geometry/pose_estimation.py`：补充 OpenCV LK 前向或反向跟踪返回空值时的显式 `matching` 失败诊断。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件
- 无

### 主要变化
- 低视差不再自动等于失败：背景支持充分且背景流接近零时生成带 `evidence_source` 的 `static_identity`；无支持时保持 missing。
- `reference_gauge`、`static_identity`、`rotation_homography`、`essential_se3`、`depth_pnp` 和 `missing` 具有不同模型及平移尺度状态；rotation-only 的 `translation_valid=false`。
- 对每个目标帧保留所有 `t-1/t-2/t-4` 候选和拒绝原因，图传播优先最近的有效边，并可由跳帧边跨过单个相邻失败，但禁止跨 scene cut。
- 背景诊断记录 foreground mask 前候选数、mask 后点数、匹配数、几何内点数、空间覆盖、象限支持、集中度、失败阶段和 oversized bbox fallback decision。
- 深度对齐比较 `scale_only`、`affine_depth`、`affine_inverse_depth`；选择时优先保证参考帧连通覆盖，再比较留出误差和模型复杂度，避免局部低误差但与参考帧断开的子图获胜。
- 序列全局对齐保存参考帧、逐帧 scale/shift、alignment domain、支持边、全局一致性误差和 drift；`relative_per_frame` 仅在深度图验证连通后升级为 `relative_aligned_sequence`。
- 联合门控同时要求 pose/depth 图连通、无镜头切换、坐标约定一致、pose quality 和 depth holdout quality。旋转-only 虽可形成相机旋转图，但缺少共享平移尺度，动态三维值保持 NaN。
- 几何模块不读取 scale prior YAML、person/car/cup 物理尺度、`R_sd` 或 `R_semantic_size_3d`。Depth Anything 输出在本阶段仍是单目相对深度，不是米制深度。
- `SequenceGeometryQuality` 是启发式工程质量，不是概率、伪造分数或论文最终阈值。

### 验证命令
```bash
.venv/bin/python -m pytest \
  tests/test_motion_regime.py \
  tests/test_pose_estimation_fallbacks.py \
  tests/test_pose_graph.py \
  tests/test_sequence_depth_alignment_global.py \
  tests/test_real_sequence_geometry_stabilization.py \
  tests/test_sequence_geometry_observation.py \
  tests/test_sequence_depth_alignment.py \
  tests/test_sequence_foreground_scene_cut.py \
  tests/test_sequence_geometry_quality.py \
  tests/test_real_clip_sequence_geometry_smoke.py -q

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/run_sequence_geometry_stabilization.py \
  --output_root outputs/sequence_geometry_stabilization \
  --device cpu

.venv/bin/python -m pytest -q
sha256sum configs/scale_priors_strict_v1.yaml configs/scale_priors_strict_v2.yaml
```

### 验证结果
- P3-0/P3-0.5 专项：`42 passed in 2.60s`。
- 全量：`299 passed in 90.58s`；仅有一条既有 CUDA 驱动版本警告，CPU 路径不受影响。
- strict v1 SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。
- 三个真实片段均为 8 帧、无 scene cut；pose graph 和 depth graph 的 connected frame ratio 均为 `1.0`，pose chain length 均为 `7`，sequence scale status 均升级为 `relative_aligned_sequence`。
- `static_camera`（`real_3`）：相邻状态为 6 个 static、1 个 rotation-dominant；7 个 identity 候选、选中 1 条 rotation-only、0 条 full SE3；pose quality=`0.915341`，depth=`affine_depth / geometry_z_depth`，fit=`0.0300853`、holdout=`0.0767013`、global consistency=`0.00862081`；背景二维运动/补偿后重投影为 `0.317024/0.015353 px`，背景深度稳定性代理量由 `0.0667628` 降至 `0.0175772`；sequence quality=`0.821623`。
- `slowly_moving_camera`（`real_2`）：相邻状态为 2 个 static、5 个 rotation-dominant；2 个 identity、5 条 rotation-only、0 条 full SE3；pose quality=`0.914429`，depth=`affine_inverse_depth / raw inverse-depth domain`，fit=`0.0379783`、holdout=`0.115278`、global consistency=`0.00541202`；背景二维运动/补偿后重投影为 `0.433465/0.0658748 px`。背景深度稳定性代理量由 `0.0327663` 变为 `0.0442668`，未改善，作为真实限制保留；sequence quality=`0.878918`。
- `clearly_moving_camera`（`real_1`）：原 7 条相邻位姿失败被重新解释为 3 个有证据 static identity 和 4 个 rotation-dominant rotation-only，真正无证据 invalid 为 0；full SE3 为 0；pose quality=`0.832945`，depth=`affine_inverse_depth / raw inverse-depth domain`，fit=`0.0125741`、holdout=`0.0137055`、global consistency=`0.000764277`；背景二维运动/补偿后重投影为 `0.551394/0.213366 px`，背景深度稳定性代理量由 `0.0197267` 降至 `0.00163657`；sequence quality=`0.884608`。
- 三段的 `dynamic_3d_valid=false`。图连通和深度对齐成功不等于平移尺度已恢复；本次真实边只有 static identity 或 rotation-only，未得到可信 full SE3。
- 每个片段均生成 `motion_regimes.csv`、`pose_candidates.csv`、`pose_graph.json`、`camera_trajectory.csv`、`depth_alignment_candidates.csv`、`depth_alignment_global.csv`、`background_tracks.csv`、`geometry_quality.json` 和 `diagnostics.png`。

### 当前限制
- 三个片段使用 approximate K、无 distortion calibration；不存在可信公制相机轨迹。
- Depth Anything 是逐帧单目相对深度。对齐后得到共享序列相对尺度，不是米制尺度；不同 alignment domain 的 raw fit/holdout 数值不能直接作物理单位比较。
- 真实片段没有恢复出 full SE3 平移；rotation/homography 只能支持旋转补偿，不能支撑可靠三维速度或完整动态重投影。
- foreground 仍以 bbox 为主，静止片段部分帧排除比例约 0.65，且 bbox fallback quality 为 0.35；缺少实例 mask 会影响背景支持的纯度和覆盖。
- `background_depth_stability` 是整幅背景中位数的工程代理量，受视点和可见背景变化影响；`real_2` 对齐后该代理量变差，说明深度稳定性尚未在所有真实片段上成立。
- 当前“global refinement”是带质量权重的全局仿射图最小二乘，并非完整 bundle adjustment 或鲁棒 SLAM 优化。
- 三个 8 帧正常短片段只用于工程 smoke，不能作为论文性能、阈值或泛化结论。

### 下一步计划
- 当前可以进入 P3-A 的接口设计和合成真值实现，但不能直接宣称具备可信真实三维轨迹前提。
- 在真实 P3-A 前应优先获得带平移尺度的可靠 pose：接入标定 K、实例 mask 和可复现的 VO/SLAM/外部 pose，或改进 depth-assisted PnP 的背景三维支持。
- 增加更长、具有明确平移和视差的独立正常序列，验证 full SE3、跳帧图闭环、全局深度对齐和背景三维稳定性，而不是继续在这三个 smoke 片段上调阈值。
- 后续动态模块只能消费联合门控通过的共享三维帧；无平移尺度或无有效证据时必须保持 NaN，不能写成正常残差 0。

## 2026-07-16 - P3-0.6 Dynamic 3D Readiness and P3-A Synthetic Dynamic Evidence

### 目的
在 P3-0.5 图连通和序列深度对齐基础上，增加按几何能力分级的动态三维可用性门控，并建立独立二维点轨迹、共享深度反投影、三帧三维连续性和独立跨帧重投影接口。本阶段只验证几何授权边界、合成真值公式和真实工程 smoke；不执行真实视频真假分类、速度/加速度异常、对象内部结构时序异常、动态遮挡异常、融合训练、性能指标或阈值调节。

### 新增文件
- `configs/dynamic_3d_readiness.yaml`：保存可复现的工程 smoke 门控阈值，并明确不是论文最终阈值、未根据当前六视频调参。
- `src/semantic3d/dynamic_3d/__init__.py`：统一导出动态三维 readiness、cache、track 和 residual API。
- `src/semantic3d/dynamic_3d/readiness.py`：定义 `DynamicGeometryMode`、`Dynamic3DReadinessThresholds`、`Dynamic3DReadiness`、相对改善公式和严格模式授权。
- `src/semantic3d/dynamic_3d/cache.py`：从 P3-0.5 缓存恢复 `Shared3DClipObservation`、对齐深度、K 和 pose，不调用任何模型或几何估计器。
- `src/semantic3d/dynamic_3d/track_observation.py`：定义独立 `PointTrack2DObservation`、`PointTrack3DObservation`、`ObjectTrack3DObservation`、canonical tracker 接口、synthetic/mock/adapter 和共享几何反投影。
- `src/semantic3d/dynamic_3d/track_residual.py`：实现三帧常速度预测、first-order displacement 诊断、second-difference raw residual 和可选对象尺度归一化证据。
- `src/semantic3d/dynamic_3d/reprojection_residual.py`：实现背景重投影 QA、前景相机补偿运动诊断和具有历史运动预测时的正式动态重投影 `ResidualEvidence`。
- `scripts/run_real_dynamic_3d_smoke.py`：只读取 P3-0.5 cache，执行独立 KLT、readiness、三维点/残差接口和八个规定输出，不重新估计 depth/K/pose。
- `tests/synthetic_dynamic_3d.py`：生成静止/移动相机、匀速/跳变对象、同时运动、局部异常、ID 断裂和 scene cut 的可控真值。
- `tests/test_dynamic_3d_readiness.py`：验证模式授权、改善门控、real_2 风格恶化拒绝、NaN 和 identity/missing 区分。
- `tests/test_point_track_3d.py`：验证独立观测、稳定 point ID、shared clip 反投影、relative/metric 状态和 rotation-only 世界坐标限制。
- `tests/test_track_3d_residual.py`：验证匀速、跳变、局部单点、ID 断裂、scene cut、归一化缺失、metric/relative aligned 和 rotation-only 限制。
- `tests/test_dynamic_reprojection_residual.py`：验证移动相机补偿、相机与对象同时运动、QA/异常证据分离、独立观测实际参与、rotation-only 和同源闭环拒绝。
- `tests/test_real_dynamic_3d_smoke.py`：验证 cache-only runner、规定输出、不实例化真实模型、real_2 拒绝和冻结哈希。

### 修改文件
- `scripts/run_sequence_geometry_stabilization.py`：在原九类 P3-0.5 输出之外保存 shared geometry cache；缓存 canonical per-frame depth、aligned depth、valid mask、foreground mask、K、pose graph 和完整来源，不保存 visualization depth。
- `src/semantic3d/__init__.py`：延迟导出新增动态三维公共 API，保留全部旧接口。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件
- 无

### 主要变化
- `DynamicGeometryMode` 严格分为 `unavailable`、`static_camera_3d`、`rotation_compensated` 和 `full_se3_3d`。rotation-only 只允许 bearing/ray 与二维旋转补偿，不允许完整世界坐标位移或三维连续性残差。
- readiness 不以 pose graph connected ratio 或总体 quality 单独判定，必须同时通过共享帧、pose/depth 图、独立轨迹、无 scene cut，以及背景重投影、深度稳定性和背景三维稳定性的改善门控。
- 所有改善量按 `(before-after)/(before+epsilon)` 计算。不可比较输入保持 NaN；对齐恶化不被当成 0 改善。
- 独立 tracker 的当前帧 `pixel_uv` 必须声明 `independent_observation=true` 且 `generated_from_projection=false`；投影点不能冒充观测。
- P3-A 反投影只使用传入 `Shared3DClipObservation` 中的 aligned depth、K 和 pose。真实动态脚本源码不实例化 `RealDepthProvider`、`RealObjectProvider` 或 `stabilize_sequence_geometry`。
- `PointTrack3DObservation` 在 static 模式保存共享 camera-frame 3D，在 rotation 模式只保留用于 ray/reprojection 的 camera point、不生成 world point，在 full SE3 模式才生成 world point。
- 三维连续性至少要求同一 point ID 三帧、统一坐标/尺度、无 scene cut 且非 `relative_per_frame`。对象尺度缺失时 raw residual 保留，normalized residual 为 NaN。
- 背景重投影是 `BackgroundReprojectionQuality`；没有历史对象运动预测的前景误差是 `CameraCompensatedMotionDiagnostic`；只有与独立当前观测和历史运动预测不一致时才形成 `DynamicReprojectionResidual`。
- 几何 provider 失败、背景误差、前景存在运动和对象速度较大均不直接表示伪造。

### 合成真值结果
- 静止相机匀速对象的最大三帧 raw continuity residual=`5.551115123125783e-16`，接近浮点误差。
- 静止相机跳变对象的最大 raw residual=`1.9999999999999996`，明显高于匀速场景。
- 移动相机静止背景不补偿时重投影误差=`3.0 px`，使用已知 full SE3 补偿后=`0.0 px`。
- 相机和对象同时运动且使用独立历史运动真值预测时，动态重投影误差接近 0。
- 局部单点跳变只提高对应 point ID 的 residual；其他点保持接近 0。
- point ID 断裂、scene cut、`relative_per_frame`、rotation-only world continuity 和缺失观测均输出 invalid + NaN，不填 0。
- `relative_aligned_sequence` 允许相对单位 raw 轨迹，`metric_sequence` 允许米制诊断；两者不会静默混合。

### 真实动态 smoke
- 输入复用 P3-0.5 的三个 8 帧正常片段和 shared geometry cache；二维点来源为独立 OpenCV KLT forward/backward tracker，首帧检测和背景三维稳定性计算均复用上游 foreground mask。
- `static_camera`（`real_3`）：mode=`static_camera_3d`，ready=true；141 条独立 point track，mean length=`7.95035`，coverage=`0.993794`；有效 3D observation=`1121/1128`，有效 continuity interface result=840，reprojection QA=980；背景 3D stability=`0.0519657 -> 0.0150080`，improvement=`0.711194`。
- `clearly_moving_camera`（`real_1`）：mode=`rotation_compensated`，ready=true；160 条独立 point track，mean length=`7.98125`，coverage=`0.997656`；有效 camera-ray/depth observation=`1277/1280`，world continuity result=0，rotation reprojection QA=1117；背景 3D stability=`0.0152242 -> 0.00263566`，improvement=`0.826877`。
- `slowly_moving_camera`（`real_2`）：mode=`unavailable`，ready=false；虽然 reprojection improvement=`0.848027`、background 3D improvement=`0.396354`，depth stability improvement=`-0.350984`，因此 missing reason=`depth_stability_improved`，有效 3D、continuity 和 reprojection QA 均为 0。
- 三段 `formal_dynamic_reprojection_evidence_count=0`。当前 KLT 是场景几何 QA 点，尚无可靠对象关联和独立拟合的前景历史运动模型，因此不输出真实伪造异常结论。
- 每段输出 `dynamic_readiness.json`、`point_tracks_2d.csv`、`point_tracks_3d.csv`、`track_residuals.csv`、`reprojection_residuals.csv`、`track_diagnostics.png`、`reprojection_diagnostics.png` 和 `smoke_report.json`；总汇总为 `outputs/real_dynamic_3d_smoke/suite_report.json`。

### 验证命令
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/run_sequence_geometry_stabilization.py \
  --output_root outputs/sequence_geometry_stabilization \
  --device cpu

.venv/bin/python scripts/run_real_dynamic_3d_smoke.py \
  --geometry_root outputs/sequence_geometry_stabilization \
  --output_root outputs/real_dynamic_3d_smoke

.venv/bin/python -m pytest \
  tests/test_dynamic_3d_readiness.py \
  tests/test_point_track_3d.py \
  tests/test_track_3d_residual.py \
  tests/test_dynamic_reprojection_residual.py \
  tests/test_real_dynamic_3d_smoke.py -q

.venv/bin/python -m pytest -q
sha256sum configs/scale_priors_strict_v1.yaml configs/scale_priors_strict_v2.yaml
git diff --check
```

### 验证结果
- P3-0.6/P3-A 专项：`32 passed in 2.56s`。
- 全量：`331 passed in 92.41s`；仅有一条既有 CUDA 驱动版本警告，CPU 路径不受影响。
- strict v1 SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。
- `git diff --check` 通过；`.gitignore` 已确认忽略 `outputs/`、视频和 `checkpoints/`。
- 未修改已有二维、2.5D、P1、P2、P3-0、P3-0.5 实验结论或冻结尺度先验。

### 当前限制
- 真实三片段仍使用 approximate K、无 distortion calibration；没有任何片段恢复 full SE3 translation，因此没有可信真实 world 3D 位移或速度。
- real_3 仅能在静止相机 camera gauge 下验证相对单位轨迹；real_1 仅能做 rotation-compensated bearing/reprojection QA；real_2 被严格门控拒绝。
- KLT 点没有可靠对象实例归属，也不是语义关键点；真实 track continuity CSV 只能作为接口和几何 QA，不能直接作为伪造异常证据。
- 当前没有独立训练/拟合的前景历史运动模型，因此真实 `DynamicReprojectionResidual` 正式证据为 0。
- 单目 aligned depth 仍是序列相对尺度，不是米制尺度；真实三维 residual 的数值不能解释为米或米每秒。
- 背景三维稳定性是 KLT 稀疏点的工程代理量，依赖 bbox foreground mask 和 approximate pose，不是完整 bundle adjustment 结果。
- 三个正常 8 帧片段仅是工程 smoke，不支持真实/伪造性能、阈值、AUC 或泛化结论。

### 下一步计划
- 可以进入 P3-B 的接口与合成真值设计，复用相同 point ID、shared clip、scale status 和 validity mask，验证方向、相对速度与结构时序公式。
- 尚不能进入可信真实 P3-B 性能实验。应先接入对象级长期 point association、实例 mask/语义关键点和独立历史运动模型。
- 完整 world 3D 动态分支仍需标定 K 和带兼容 translation scale 的 full SE3 pose，建议在具有明确平移视差和外部 pose/SLAM 参考的独立序列上验证。
- 后续所有动态 evidence 必须继续区分 QA、motion diagnostic 和 formal anomaly residual；门控失败保持 NaN，不得用 0 表示无证据。
