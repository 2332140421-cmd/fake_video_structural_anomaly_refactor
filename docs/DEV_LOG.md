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

## 2026-07-21 - P3-A.5 Object Point Binding and P3-B Dynamic Structural Residuals

### 目的
在 P3-0.6/P3-A 的共享几何和独立二维点轨迹契约上，建立对象与稳定点身份绑定、只读历史运动预测、固定对象结构图，以及方向、尺度归一化相对速度、内部结构时序和对象运动重投影证据。本阶段仅验证动态结构接口、合成真值和三个真实正常短片段 smoke，不实现遮挡残差、最终融合、真假训练、AUC/F1、公制速度或阈值调节。

### 新增文件
- `src/semantic3d/dynamic_3d/object_track_binding.py`：定义 `ObjectPointBinding`、`ObjectPointTrack3D`、`ObjectDynamicObservation`，按实例 mask、语义关键点、跟踪 mask、收缩 bbox、普通 bbox 的优先级绑定稳定 point ID，并拒绝对象切换、归属丢失、边界点、低质量和过度三维跳变。
- `src/semantic3d/dynamic_3d/structure_graph.py`：定义固定 `ObjectStructureGraph` 和 `StructureEdge`；人体仅连接确定语义关键点，普通对象在首个有效帧建立固定 kNN 图，后续不重建邻接。
- `src/semantic3d/dynamic_3d/motion_model.py`：定义历史只读 `BaseObjectMotionModel`、点常速度模型和 leave-one-point-out 对象中位平移模型；预测不读取目标帧观测。
- `src/semantic3d/dynamic_3d/direction_residual.py`：实现点自身历史、对象中位方向和局部邻点方向的 `1-cos` 残差；近静止位移保持 invalid/NaN。
- `src/semantic3d/dynamic_3d/relative_velocity.py`：实现相对单位位移、`object_scale_per_frame/second` 速度诊断、速度变化和点相对对象中位速度残差；不声称 m/s。
- `src/semantic3d/dynamic_3d/structure_temporal_residual.py`：沿固定边计算原始/尺度归一化边长变化，并保留异常点和边 ID；缺边为 NaN。
- `src/semantic3d/dynamic_3d/object_dynamic_aggregation.py`：以 median、trimmed mean 和 top-k mean 汇聚有效点/边证据，同时保留定位结果，不生成视频真假分数。
- `scripts/run_real_object_dynamic_3d_smoke.py`：只读取 P3-0.5 shared geometry cache、P3-0.6 readiness 和已有全局关联 observation，在对象 bbox 内运行独立 KLT 并输出对象绑定、结构图、历史预测和 P3-B 诊断。
- `tests/test_object_track_binding.py`：验证稳定归属、对象切换拒绝、assignment lost 和显式背景点。
- `tests/test_dynamic_motion_model.py`：验证常速度、leave-one-point-out、目标帧信息隔离和 rotation-only bearing 预测。
- `tests/test_p3b_dynamic_residuals.py`：验证匀速、转向、整体加速、局部点跳变、人体骨段稳定/伸缩、rotation-only、unavailable 和缺边 NaN。
- `tests/test_real_object_dynamic_3d_smoke.py`：验证 cache-only 真实 runner、规定输出、不实例化几何模型和来源复用。

### 修改文件
- `src/semantic3d/dynamic_3d/reprojection_residual.py`：正式前景重投影新增 motion model type、历史帧和支持点 provenance；严格路径拒绝当前帧泄漏，旧 P3-A 外部历史预测调用继续兼容。
- `src/semantic3d/dynamic_3d/__init__.py`：统一导出 P3-A.5/P3-B API。
- `src/semantic3d/__init__.py`：延迟导出新增公共 API，保留旧接口。
- `tests/test_dynamic_reprojection_residual.py`：增加严格历史元数据的当前帧泄漏拒绝测试。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件
- 无

### 主要变化
- 同一 point ID 不能静默从一个 `object_track_id` 切到另一个对象；发生切换时整条绑定 invalid，离开对象区域时该帧为 `assignment_lost`。普通 bbox 边缘只获得低质量边界角色，默认不进入正式内部结构残差。
- 稳定点筛选要求轨迹长度、tracking/depth quality、assignment consistency、遮挡比例和三维跳变均可接受。真实 smoke 当前没有实例 mask，因此来源如实记录为 `shrunk_bbox` 或 `bbox_fallback`。
- 历史预测只消费目标帧之前的样本。点模型使用 `t-2,t-1`，对象模型使用其他支持点的历史中位位移并执行 leave-one-point-out；当前 `observed_uv_t` 只用于最终比较。
- `static_camera_3d` 允许 camera-gauge 共享尺度方向、相对速度、结构和对象运动重投影；`rotation_compensated` 只允许补偿 bearing/angular 与 rotation-only 重投影，不输出完整三维速度和结构；`unavailable` 正式证据均为 NaN。
- 速度单位为 `object_scale_per_frame` 或 `object_scale_per_second`，不是 m/s。速度大本身只是诊断，只有相对历史或对象整体不一致才形成残差。
- 结构边在参考帧固定，后续沿相同 point ID 计算；人体骨段长度与普通刚体固定边分别标记。缺点、缺尺度和缺边不写成 0。
- 动态重投影正式证据严格要求对象归属、至少两个先前历史时刻、历史运动模型、独立当前 tracker 观测和当前相机几何。背景 QA、前景诊断与正式异常证据继续分离。

### 合成真值结果
- 刚性对象匀速平移时方向残差和尺度归一化速度变化接近 0。
- 突然转向时 `1-cos` 方向残差大于 `0.9`。
- 整体加速提高速度变化残差，但固定结构边残差保持接近 0。
- 单点局部跳变提高相邻固定边残差，并定位到对应 point ID 和 edge ID。
- 人体整体关节运动且骨段长度稳定时结构残差接近 0；骨段异常伸缩时残差大于 `0.9`。
- 对象归属切换不会生成连续正式轨迹；点离开 bbox 产生 `assignment_lost`；缺边、rotation-only 完整速度和 unavailable 模式均保持 invalid + NaN。
- 对目标帧观测进行任意跳变不会改变由历史生成的预测，确认当前帧信息未进入预测模型。

### 真实动态 smoke
- `static_camera`（`real_3`）：72 个对象绑定点，14 个点同时通过归属、轨迹和有效三维深度筛选；mask=0、shrunk bbox=61、普通 bbox fallback=11，平均绑定持续 `7.90278` 帧；3 个对象具备至少 3 帧二维归属历史。cup/table 的 46 个长二维轨迹在后续帧缺少有效深度，person 又没有语义关键点，因此没有对象建立可信固定结构图；正式方向=84、速度变化=84、结构帧证据=0、动态重投影=84。
- `clearly_moving_camera`（`real_1`）：50 个对象绑定点，48 个稳定点；mask=0、shrunk bbox=49、普通 bbox fallback=1，平均绑定持续 `8.0` 帧；3 个对象具备至少 3 帧历史；rotation-compensated 正式 bearing direction=280、rotation-only 动态重投影=288，完整三维速度=0、结构残差=0。
- `slowly_moving_camera`（`real_2`）：模式继续为 `unavailable`。现有全局关联 observation 的前 8 帧没有检测对象，因此绑定点和所有正式证据均为 0 条；无证据未解释为低异常分数。
- 相比 P3-A 的真实正式动态重投影证据 0 条，本轮 real_3 和 real_1 分别增加到 84 和 288 条。它们是无阈值对象运动一致性证据，不是伪造分类结论。
- 输出位于 `outputs/real_object_dynamic_3d_smoke/<clip_id>/`，包含对象绑定 CSV、固定结构图 JSON、历史预测、方向、相对速度、结构、动态重投影、对象汇总、诊断图和 smoke report。

### 验证命令
```bash
.venv/bin/python scripts/run_real_object_dynamic_3d_smoke.py

.venv/bin/python -m pytest \
  tests/test_dynamic_3d_readiness.py \
  tests/test_point_track_3d.py \
  tests/test_track_3d_residual.py \
  tests/test_dynamic_reprojection_residual.py \
  tests/test_object_track_binding.py \
  tests/test_dynamic_motion_model.py \
  tests/test_p3b_dynamic_residuals.py \
  tests/test_real_dynamic_3d_smoke.py \
  tests/test_real_object_dynamic_3d_smoke.py -q

.venv/bin/python -m pytest -q
sha256sum configs/scale_priors_strict_v1.yaml configs/scale_priors_strict_v2.yaml
git diff --check
```

### 验证结果
- P3 动态专项：`51 passed in 5.03s`。
- 全量：`350 passed in 96.11s`；仅有一条既有 CUDA 驱动版本警告，CPU 路径不受影响。
- strict v1 SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。
- `.gitignore` 继续忽略 `outputs/`、`checkpoints/` 和 `*.mp4`。本轮未修改二维、2.5D、静态 3D、P3-0 至 P3-A 的既有实验结论。
- `git diff --check` 对本轮新增 Python/Markdown 无格式错误；命令仍报告用户此前修改的两个 manifest CSV 使用 CRLF 行尾，本轮未改动这些文件。

### 当前限制
- 三个真实 smoke 仍是 8 帧正常片段，不支持真假分类、阈值、AUC/F1 或泛化结论。
- 当前真实点归属完全依赖 bbox，mask binding 数量为 0；普通 bbox 无法可靠表达遮挡边界、前后景层次或可见区域，正式点质量仍有限。
- real_3 是静止相机 camera gauge 的序列相对三维，不是米制世界轨迹；相对速度不能解释为 m/s。
- real_1 的 rotation-only 结果是补偿 bearing/angular 与图像重投影，不是完整三维平移、世界位移或速度。
- real_2 同时缺少可用动态几何和前 8 帧对象检测，严格保持拒绝。
- 普通对象结构图使用固定 kNN；人体只有在已有确定语义关键点时才建立语义骨架。当前真实 KLT 人体点不能冒充语义关节。
- 当前对象尺度来自同一共享相对三维点集的参考帧工程尺度，只用于内部归一化，不是公制语义尺度。

### 下一步计划
- 可以进入 P3-C 的数据契约、合成真值和受控 smoke，复用现有 point identity、object binding、fixed edge、visibility 和 validity 结构。
- 尚不能直接形成可信真实遮挡结论。进入真实 P3-C 前应优先接入实例 mask 或可靠 tracked mask，明确前后景层次、边界穿越、显隐状态和重新出现身份一致性。
- P3-C 必须继续保持模式授权和 NaN 规则，遮挡/显隐证据与本轮方向、速度、结构、重投影证据分开，不在当前三个正常片段上调整异常阈值。

## 2026-07-21 - P3-C0 Instance Visibility Layer and P3-C Occlusion Evidence

### 目的
在 P3-A.5/P3-B 的共享对象轨迹和动态几何模式上，建立实例 mask、历史支撑预测、可见性状态、稀疏遮挡关系、遮挡深度顺序、显隐解释、遮挡边界和重新出现的数据契约。本阶段只完成接口、合成真值和真实正常短片段 smoke；不执行真假分类、融合训练、AUC/F1、阈值调节，也不把漏检、bbox 重叠、几何失败或 scene cut 当作伪造证据。

### 新增文件
- `src/semantic3d/occlusion/__init__.py`：统一导出 P3-C0/P3-C 公共接口。
- `src/semantic3d/occlusion/mask_observation.py`：定义可见/非模态 mask、独立边界、历史预测支撑域和 tracked mask 契约；缺失数据使用 `None/NaN + valid=false`。
- `src/semantic3d/occlusion/mask_provider.py`：定义 canonical mask provider，以及 synthetic、mock、已有 artifact 适配器和不下载模型的真实 provider 接口。
- `src/semantic3d/occlusion/mask_tracking.py`：用对象历史预测或独立光流/点 warp 传播 mask，并以当前独立 mask 做 IoU 和边界验证。
- `src/semantic3d/occlusion/support_prediction.py`：只使用目标帧之前的两帧 mask 预测当前支撑；明确限制 static、rotation、full SE3 和 unavailable 模式。
- `src/semantic3d/occlusion/visibility_state.py`：区分 fully/partially occluded、out-of-frame、detector missing、reappeared、scene cut 和 uncertain。
- `src/semantic3d/occlusion/occlusion_graph.py`：只为历史预测支撑发生重叠的对象建立稀疏候选关系，不生成完整 n×n 矩阵。
- `src/semantic3d/occlusion/depth_order_residual.py`：实现前景 Z 小于背景 Z 的顺序证据；对象中心深度明确标记为低质量来源。
- `src/semantic3d/occlusion/visibility_residual.py`：区分可解释遮挡/出画、检测不确定和无解释消失/出现；只有后两者可形成正式候选。
- `src/semantic3d/occlusion/boundary_occlusion_residual.py`：比较历史预测和当前独立观测的对象接触边界；bbox 边界不进入正式证据。
- `src/semantic3d/occlusion/reappearance.py`：以语义、外观、结构、相对深度和运动方向多线索保守验证重现身份，并禁止 scene cut 跨越关联。
- `scripts/run_real_occlusion_observation_smoke.py`：只读取 P3-0.5 shared geometry cache、P3-0.6 readiness 和已有全局对象 observation，输出规定的 mask、状态、关系、残差、诊断图和报告。
- `tests/synthetic_occlusion.py`：登记正常遮挡、深度反转、部分/完全遮挡、出画、漏检、无解释显隐、交叉运动、错误 re-id、rotation-only、scene cut、缺 mask 和 bbox-only 等 A-N 场景。
- `tests/test_occlusion_masks.py`：验证 mask 契约、尺寸、bbox 降级、历史隔离、光流/点 warp、模式限制和完整合成场景目录。
- `tests/test_occlusion_residuals.py`：验证深度顺序、显隐状态、边界、reappearance、scene cut、检测漏失和 unavailable 的 NaN 语义。
- `tests/test_real_occlusion_observation_smoke.py`：验证真实 cache-only runner、不调用几何估计器、bbox 不产生正式证据、real_2 拒绝和冻结配置哈希。

### 修改文件
- `src/semantic3d/__init__.py`：延迟导出 P3-C0/P3-C 公共 API，保留已有接口。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件
- 无

### 主要变化
- `InstanceMaskObservation` 的 visible mask 与 amodal mask 必须显式二选一。当前可见轮廓不会被命名为完整对象轮廓；没有 mask 时不制造高质量 bbox mask。
- `ExistingDetectionMaskAdapter` 可以生成 bbox fallback，但 confidence 上限为 `0.25`，并记录 `legacy_bbox_fallback=true`、`formal_mask_evidence=false`。该路径只用于观察流程是否连通。
- 当前目标帧 mask 只用于 IoU/边界验证，不进入同帧支撑预测。历史预测保存 `current_frame_used_for_prediction=false` 和具体历史帧。
- `full_se3_3d` 没有外部三维投影支撑时直接 missing；`rotation_compensated` 只允许旋转补偿或二维历史运动；`unavailable` 不产生正式支撑或残差。
- 序列起点和历史不足默认是 `uncertain`，不会自动写成 unexplained appearance。合成异常必须显式声明“无历史出现是待检事件”。
- detector missing 不等于 fully occluded；out-of-frame 由预测支撑的画内比例判断；scene cut 会清空状态和 mask 历史。
- 只有历史支撑重叠才建立遮挡候选；bbox overlap 关系保持 invalid。缺 mask、深度不确定、边界不足和 re-id 不确定均使用 NaN，不用 0 表示无证据。
- 遮挡图和残差继续复用 `Shared3DClipObservation` 的 aligned relative depth、K、pose 及 P3 readiness；脚本不重新估计 depth/K/pose，也不下载或训练实例分割模型。

### 合成验证结果
- 正常前景遮挡的深度顺序残差为 `0`；交换前后深度后残差明显大于 `0.5`。
- partial 与 full occlusion 状态可区分；正常出画和低置信检测漏失不生成正式异常残差。
- 显式无解释消失/出现产生正式高残差；序列起点默认不产生无解释出现。
- 正常多线索 reappearance 残差低；错误语义身份、scene cut 和低多线索质量均被拒绝并保持 NaN。
- bbox-only 关系无法形成正式遮挡边、深度顺序或边界残差；unavailable 模式即使有当前 mask 也保持 invalid + NaN。
- 目标帧 mask 改变只影响最终验证 IoU，不改变历史生成的 predicted support，确认无当前帧 mask 泄漏。

### 真实 mask smoke
- `static_camera`（`real_3`）：24 个对象 mask 记录全部来自 bbox fallback；诊断 mask 有效率 `100%`、正式 mask 有效率 `0%`。18 个历史验证结果的 mean IoU=`0.989606`，mean boundary distance=`3.91963 px`；状态为 scene_cut=3、uncertain=3、fully_visible=18；候选重叠关系=6，全部因 bbox-only 无效；正式 depth-order/visibility/boundary/reappearance 均为 0 条。
- `clearly_moving_camera`（`real_1`）：24 个对象 mask 记录全部来自 bbox fallback；诊断 mask 有效率 `100%`、正式 mask 有效率 `0%`。18 个历史验证结果的 mean IoU=`0.989749`，mean boundary distance=`1.32760 px`；状态为 scene_cut=3、uncertain=3、fully_visible=18；候选重叠关系=12，全部因 bbox-only 无效；正式 depth-order/visibility/boundary/reappearance 均为 0 条。
- `slowly_moving_camera`（`real_2`）：readiness=`unavailable`，对应 8 帧没有检测对象；mask 记录=0、候选关系=0、所有正式证据=0 条。无证据未解释为正常残差 0。
- 三段均未发现可声明的 partial/full occlusion 或 reappearance。当前输出验证了接口和拒绝逻辑，不构成真实遮挡检测性能结论。
- 输出位于 `outputs/real_occlusion_observation_smoke/<clip_id>/`，每段包含 `instance_masks.json`、`tracked_masks.json`、`visibility_states.csv`、`predicted_support_masks.json`、四类关系/残差 CSV、reappearance CSV、诊断图和 smoke report。

### 验证命令
```bash
.venv/bin/python scripts/run_real_occlusion_observation_smoke.py

.venv/bin/python -m pytest \
  tests/test_occlusion_masks.py \
  tests/test_occlusion_residuals.py \
  tests/test_real_occlusion_observation_smoke.py \
  tests/test_object_track_binding.py \
  tests/test_dynamic_motion_model.py \
  tests/test_p3b_dynamic_residuals.py \
  tests/test_dynamic_reprojection_residual.py \
  tests/test_real_object_dynamic_3d_smoke.py -q

.venv/bin/python -m pytest -q
sha256sum configs/scale_priors_strict_v1.yaml configs/scale_priors_strict_v2.yaml
git diff --check -- src/semantic3d/occlusion scripts/run_real_occlusion_observation_smoke.py tests/test_occlusion_masks.py tests/test_occlusion_residuals.py tests/test_real_occlusion_observation_smoke.py
```

### 验证结果
- P3-C 与前序动态模块联合专项：`52 passed in 5.08s`。
- P3-C 专项：`26 passed in 3.10s`。
- 全量：`376 passed in 105.40s`；仅有一条既有 CUDA 驱动版本警告，CPU 路径不受影响。
- real_1/real_3 的 invalid depth-order、visibility 和 boundary CSV 行全部保持 NaN；real_2 空证据 CSV 只保留表头。
- strict v1 SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。
- `.gitignore` 继续保护 `outputs/`、`checkpoints/` 和视频/模型权重；未修改任何冻结先验或已有实验数值。

### 当前限制
- 真实 observation 尚无实例分割 artifact；real_1/real_3 的 100%“mask 有效”只指 bbox 诊断可用，正式实例 mask 有效率实际为 0%。IoU 和边界距离因此不能解释为真实轮廓跟踪性能。
- 没有冻结的真实实例分割模型、amodal mask 或跨遮挡 mask propagation 模型；`RealInstanceMaskProvider` 当前只是外部 callback 接口，不会自动下载模型。
- 真实候选关系完全来自 bbox 历史支撑重叠，均被正确拒绝。当前无法据此判断真实遮挡深度顺序、边界异常或显隐伪造。
- 深度顺序目前使用共享相对深度的对象区域中位数，明确是低质量中心深度；尚未实现遮挡边界两侧的高质量局部深度采样。
- full SE3 mask 投影只有接口，没有真实三维对象体积；real_1 仍是 rotation-only，real_2 仍 unavailable。
- reappearance 需要可靠外观/结构/深度/运动多线索；当前真实 smoke 不提供这些输入，因此没有正式 re-id 证据。
- 三段 8 帧正常片段不支持真假分类、阈值、AUC/F1 或泛化结论。

### 下一步计划
- 在不使用真假标签的前提下，接入冻结且版本可追溯的真实实例分割 provider，保存真实 visible mask、模型版本、置信度和检测来源，并重新统计正式 mask 覆盖率。
- 使用独立点轨迹/光流传播历史 mask，验证当前 segmentation 仅用于同帧校验；增加明确包含 partial/full occlusion 和重现事件的独立真实片段。
- 在真实高质量 mask 和边界局部深度就绪前，不进入正式三维遮挡片段级聚合或定位性能实验。当前仅可进入该聚合器的接口与合成真值设计。
- 后续仍需保持 static/rotation/full-SE3/unavailable 分级、scene-cut 隔离和 NaN 规则，且不得在当前六视频上调遮挡阈值。

## 2026-07-21 - P3-D Real Segmentation, Human Keypoints, and Evidence Coverage

### 目的
在 P3-C 遮挡契约基础上，接入本地冻结实例分割 provider、分割实例与既有对象轨迹的一对一关联、真实人体语义关键点与共享深度三维反投影、固定人体骨架，以及不使用真假标签和残差大小的真实证据片段筛选。本阶段只统计观测与结构证据覆盖，不进行融合、真假分类、AUC/F1 或阈值调节；缺少实例分割权重时不联网、不使用 bbox 冒充正式 mask。

### 新增文件
- `src/semantic3d/occlusion/mask_object_association.py`：定义分割候选与 `ObjectObservationJSON/object_track_id` 的一对一关联，综合同源 detection ID、语义类别、mask/bbox IoU、中心距离、containment 和置信度，并保留所有候选及拒绝原因。
- `src/semantic3d/occlusion/scene_cut_statistics.py`：提供 `scene_cut_statistics_v2`，分离真实 clip cut、对象级 scene-cut marker、首帧 track 初始化和状态机边界 marker。
- `src/semantic3d/dynamic_3d/person_keypoint_binding.py`：将标准 COCO 人体关键点绑定到稳定 person track，以 `track_id:keypoint:name` 保持左右语义身份，逐点保存 invalid，并复用 shared depth/K/pose 生成三维关键点。
- `scripts/find_real_3d_evidence_clips.py`：仅按观测可用性和事件类型筛选 person structure、mask overlap、partial/full occlusion/reappearance、out-of-frame 和 stable multi-object 候选；拒绝带 truth label 或 residual 字段的输入。
- `scripts/run_real_3d_evidence_coverage.py`：运行真实 mask、关联、人体关键点、固定结构图、结构时序、可见性和遮挡 coverage，并生成规定 CSV、JSON 和诊断图。
- `tests/test_real_instance_mask_provider.py`：验证缺失权重错误、真实可见 mask、非 bbox 实心区域和 visible/amodal 标志。
- `tests/test_mask_object_association.py`：验证一对一关联、类别冲突拒绝、同源 detection ID 优先和候选拒绝原因。
- `tests/test_real_mask_tracking.py`：验证 observed/predicted mask 分离、面积变化、assignment consistency 和 scene-cut 禁止传播。
- `tests/test_person_keypoint_object_binding.py`：验证关键点绑定 person track、稳定左右身份和低置信点逐点 invalid。
- `tests/test_person_structure_graph_real_path.py`：验证固定人体骨架只使用持续有效的语义关节点，缺失关节只移除相关边。
- `tests/test_real_evidence_clip_finder.py`：验证无事件返回空候选，且筛选器拒绝 truth label/residual 输入。
- `tests/test_scene_cut_statistics.py`：验证真实 cut 按帧边界唯一计数，首帧初始化不计为真实 cut。
- `tests/test_real_3d_evidence_coverage.py`：验证真实 coverage 输出、缺失分割权重语义、无标签筛选和冻结先验哈希。

### 修改文件
- `src/semantic3d/occlusion/mask_provider.py`：实现本地冻结 Ultralytics 实例分割路径和 `InstanceMaskCandidate`；仅加载显式权重路径，不静默下载，缺失时返回明确 missing reason。
- `src/semantic3d/occlusion/mask_observation.py`：为 tracked mask 增加 independent observed/history predicted 别名、area change ratio 和 assignment consistency。
- `src/semantic3d/occlusion/mask_tracking.py`：继续保证当前 mask 只参与验证，并保留独立 warp、对象动态、相机运动和历史模型 provenance。
- `src/semantic3d/occlusion/visibility_state.py`：无 mask、scene cut、unavailable、历史不足和 detector missing 的不可观测面积/比例改为 NaN，不再用 0 占位。
- `src/semantic3d/occlusion/visibility_residual.py`：没有有效 visibility observation 时 diagnostic/formal evidence 均为 invalid + NaN；legacy bbox 诊断仍保留但不能升级为正式证据。
- `src/semantic3d/occlusion/occlusion_graph.py`：明确区分 `no_occlusion_event` 与 `occlusion_observation_missing`。
- `src/semantic3d/occlusion/__init__.py`、`src/semantic3d/dynamic_3d/__init__.py`、`src/semantic3d/__init__.py`：导出 P3-D 公共 API。
- `scripts/run_real_occlusion_observation_smoke.py`：允许注入真实 mask provider；新增 versioned scene-cut 统计并保留 legacy 状态计数。
- `tests/test_occlusion_residuals.py`：增加缺失 visibility 的 NaN diagnostic 检查。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件
- 无

### 主要变化
- `RealInstanceMaskProvider` 默认只查找 `checkpoints/yolov8n-seg.pt`。本地文件不存在时 `available=false`，missing reason 为 `real_instance_segmentation_weights_missing`；不会将 `yolov8n.pt` 检测权重误用为实例分割，也不会触发网络下载。
- 模型可用时先生成未关联 visible mask candidate，记录 model name/version、class id/name、confidence、source detection ID、device 和 preprocessing metadata；模型没有 amodal 输出，因此 `is_visible_mask=true`、`is_amodal_mask=false`。
- mask-object 关联先执行类别兼容检查，再做一对一最高质量分配；同源 detection ID 在兼容候选中优先。失败对象保持 invalid，不生成 bbox mask。
- 当前帧独立 segmentation、历史 predicted support 和可选 flow/point warp 保持三个来源；当前 observed mask 只用于 IoU、边界和面积变化验证，不参与同帧预测。scene cut 清空历史。
- `RealHumanKeypointProvider` 使用本地 `checkpoints/yolov8n-pose.pt`。17 个 COCO 关键点逐点门控，左右语义名称固定；坐姿、弯身或下肢低置信只影响对应点和骨架边，不自动形成异常。
- 人体关键点通过既有 `reconstruct_point_tracks_3d` 复用 shared aligned relative depth、K 和 readiness；不重新估计深度、内参或位姿。静止相机模式可建立固定三维骨架，rotation-only 不输出完整三维结构图。
- 普通对象只有真实 instance mask 内部、远离边界且具有稳定 point ID 的点可以建立正式固定结构图。bbox 内部点只保留对照统计；本轮因真实 mask 为 0，普通对象正式结构图为 0。
- clip finder 不接收真假标签或 residual 字段，只依据关键点/mask 覆盖、遮挡状态、重叠持续时间、深度顺序质量、scene cut 和 geometry mode。无真实事件时返回空候选。
- `scene_cut_statistics_v2` 显示 real_1/real_3 的真实 clip cut 均为 0。旧 `scene_cut=3` 是首帧三个对象被标成状态机边界的 legacy 计数，不是三次镜头切换；新字段记录 `track_initialization_markers=3`、`state_machine_boundary_markers=3`、`object_visibility_scene_cut_markers=0`。
- 无 mask、无关键点、无结构边、无事件、事件观测失败和 geometry unavailable 均通过 valid/quality/missing_reason/NaN 区分，不使用 0 表示缺失证据。

### 真实 coverage 结果
- 本地实例分割权重不存在：`RealInstanceMaskProvider available=false`，明确要求将冻结权重放到 `checkpoints/yolov8n-seg.pt`。三个片段正式 mask 总数为 0。
- P3-D 正式路径禁用了 bbox fallback，因此 real_1/real_3 的 bbox fallback ratio 为 0；这表示没有使用 fallback，不表示真实 mask 覆盖提升。正式 mask ratio 和 mask-object association success ratio仍为 0。
- `real_3/static_camera`：8 帧均检测 person，真实 pose 每帧有效 12-13/17 个关键点，平均 keypoint valid ratio=`0.742647`，有效关键点样本总数=101；形成 1 个正式 `semantic_human_skeleton`，包含 12 个持续三维点和 8 条固定边，graph quality=`0.697688`，产生 7 帧正式结构时序证据。
- real_3 的低置信膝/踝等点逐点 invalid；可用肩、肘、腕、髋等边仍形成骨架。人体坐姿没有被自动视为异常，本轮只统计覆盖。
- `real_1/rotation_compensated` 的对象为 mouse、remote、dining_table，没有 person；完整人体结构图为 0。`real_2/unavailable` 前 8 帧无检测对象，所有覆盖为 NaN 或 0 个事件，而非低异常分数。
- 普通对象真实 mask 内部结构图=0，mask internal point=0；原因是缺少真实实例分割权重，不是结构残差为零。
- partial occlusion 候选=0、full occlusion/reappearance 候选=0、正式遮挡证据=0、正式 reappearance=0。没有为产生分数降低 mask 门控。
- observation-only finder 找到 1 个候选：real_3 frame 0-7 的 `person_structure`；没有找到 mask overlap、partial/full occlusion、out-of-frame 或 stable multi-object mask 片段。
- 输出位于 `outputs/real_3d_evidence_coverage/`，包括 `mask_coverage.csv`、`mask_object_association.csv`、`keypoint_coverage.csv`、`structure_graph_coverage.csv`、`visibility_event_coverage.csv`、`occlusion_evidence_coverage.csv`、`per_video_summary.csv`、`global_summary.json`、候选 clip CSV、frame availability JSON 和诊断图。

### 验证命令
```bash
.venv/bin/python scripts/run_real_3d_evidence_coverage.py

.venv/bin/python -m pytest \
  tests/test_real_instance_mask_provider.py \
  tests/test_mask_object_association.py \
  tests/test_real_mask_tracking.py \
  tests/test_person_keypoint_object_binding.py \
  tests/test_person_structure_graph_real_path.py \
  tests/test_real_evidence_clip_finder.py \
  tests/test_scene_cut_statistics.py \
  tests/test_real_3d_evidence_coverage.py -q

.venv/bin/python -m pytest -q
sha256sum configs/scale_priors_strict_v1.yaml configs/scale_priors_strict_v2.yaml
git diff --check -- src/semantic3d/occlusion src/semantic3d/dynamic_3d/person_keypoint_binding.py scripts/find_real_3d_evidence_clips.py scripts/run_real_3d_evidence_coverage.py tests/test_real_instance_mask_provider.py tests/test_mask_object_association.py tests/test_real_mask_tracking.py tests/test_person_keypoint_object_binding.py tests/test_person_structure_graph_real_path.py tests/test_real_evidence_clip_finder.py tests/test_scene_cut_statistics.py tests/test_real_3d_evidence_coverage.py docs/DEV_LOG.md
```

### 验证结果
- 初始 P3-D 八组专项：`14 passed in 7.25s`；补充同源 detection ID 优先测试后新增测试总数为 15。
- P3-D/P3-C/P3-B 联合专项：`48 passed in 6.80s`。
- 全量最终结果：`391 passed in 99.15s`；仅有一条既有 CUDA 驱动版本警告，CPU 推理路径不受影响。
- strict v1 SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。
- P3-D 输出中 invalid 遮挡 residual 全部为 NaN；`.gitignore` 继续保护 `outputs/`、`checkpoints/`、视频和模型权重。

### 当前限制
- `checkpoints/yolov8n-seg.pt` 尚不存在，因此真实实例 mask 有效率和 mask-object 关联成功率仍为 0；不能评价真实分割质量或遮挡性能。
- bbox fallback ratio 从 P3-C 的 100% 变为 P3-D 的 0% 是因为正式路径完全禁用了 fallback，不是因为真实分割已经替换成功。
- real_3 只有一个坐姿 person track。其上半身骨架覆盖验证了工程链路，但不能代表不同姿态、多人交互、遮挡或快速运动场景。
- 人体关键点深度是逐点 shared relative depth 采样，不是公制三维关节；相机仍使用 approximate K。
- 普通对象尚无 mask internal stable points、固定结构图或结构时序证据。
- 当前三个短片段没有真实 partial/full occlusion、out-of-frame 或 reappearance 候选，无法验证正式遮挡残差在真实事件上的行为。
- `RealInstanceMaskProvider` 已支持本地 Ultralytics segmentation 权重，但没有在本轮自动下载；模型版本、许可、来源和冻结哈希需在放入权重后单独记录。
- 真实 evidence coverage 只覆盖 real_1/real_2/real_3 的现有 8 帧 P3 shared clips，不是完整数据集性能实验。

### 下一步计划
- 准备来源和版本可追溯的本地 `yolov8n-seg.pt` 或等价冻结实例分割权重，放入 `checkpoints/` 后重新运行 coverage，首先验证 mask 非 bbox、关联一对一、正式 mask 覆盖和 tracked mask 稳定性。
- 采集或筛选不依赖真假标签的真实多人交互/遮挡片段，覆盖 partial、full occlusion、out-of-frame 和 reappearance；仍只按事件与观测质量筛选。
- 实例 mask 到位后，为 cup/table/car 等普通对象统计 mask internal/boundary/bbox point 来源，并验证固定结构图和缺边 NaN。
- 当前仅 real_3 人体结构分支具备进入质量感知聚合接口设计的观测基础；整个 P4 真实多分支聚合仍不具备条件，至少需补齐正式 mask、普通对象结构和真实遮挡事件覆盖。
## 2026-07-21 - P3-D.1 Full-Video Evidence Coverage and P4-A Contracts

### 目的

在 P3-D 三片段 smoke 之后启用可审计的冻结实例分割路径，扫描六个完整测试视频，并建立只表示观测覆盖率的 `CoverageReadiness`。同时建立 P4-A 分层证据聚合数据契约和合成验证，不进行真假分类、AUC/F1、阈值优化或真实视频异常性能评价。

### 新增文件

- `src/semantic3d/occlusion/mask_structure_points.py`：只在正式可见实例 mask 的收缩内部选择和跟踪普通对象稳定点；bbox fallback 明确返回空正式点集。
- `src/semantic3d/occlusion/event_validation.py`：正式 partial/full occlusion 事件门控，要求 mask、历史预测、可见面积变化、遮挡对象、深度次序、非 scene cut 和 tracking 质量同时成立。
- `src/semantic3d/coverage_readiness.py`：定义 coverage-only readiness，并区分 `not_applicable` 与 `observation_missing`。
- `src/semantic3d/aggregation_v2/contracts.py`：定义 point、edge、object、frame、clip 五级 evidence aggregate。
- `src/semantic3d/aggregation_v2/aggregation.py`：实现 quality-aware top-k、median 和 trimmed mean 聚合，保留 NaN、valid、quality、coverage 和来源。
- `src/semantic3d/aggregation_v2/__init__.py`：导出 P4-A 数据接口。
- `scripts/run_real_3d_evidence_coverage_v2.py`：六视频全帧实例 mask、关联、mask tracking、结构覆盖、事件筛选和 readiness 执行器。
- `tests/test_mask_internal_structure_points.py`：验证正式 mask 内点与当前帧 mask 无预测泄漏。
- `tests/test_coverage_readiness.py`：验证无事件和观测失败语义隔离。
- `tests/test_aggregation_v2.py`：验证 P4-A 的 NaN、quality、top-k、median、trimmed mean 与可追溯性。
- `tests/test_real_3d_evidence_coverage_v2.py`：验证缺失权重时仍生成完整报告、不得联网或 bbox 降级，以及 strict prior 哈希冻结。

### 修改文件

- `src/semantic3d/occlusion/mask_provider.py`：增加权重可读性、文件大小、SHA-256、模型任务、Ultralytics 版本和真实 mask 输出校验；检测模型权重不能进入正式分割路径。
- `src/semantic3d/occlusion/mask_object_association.py`：将分割权重哈希传播到正式 mask provenance。
- `src/semantic3d/occlusion/__init__.py`、`src/semantic3d/__init__.py`：导出新增 provider metadata、mask 内点、readiness 和 aggregation API。
- `scripts/find_real_3d_evidence_clips.py`：扩展 full-video 事件类型，并拒绝 truth label、异常分数和残差字段作为筛选输入。
- `tests/test_real_instance_mask_provider.py`：增加权重哈希、mask 输出和 detection-task 拒绝测试。
- `tests/test_real_evidence_clip_finder.py`：增加异常分数字段拒绝和全视频观测事件筛选测试。

### 删除文件

- 无。

### 主要变化

- 实例分割运行时不会自动联网下载；缺少权重时统一记录 `instance_segmentation_weights_missing`。
- 仅 `task=segment` 且实际推理结果包含 mask 的本地模型可以产生正式 mask；检测输出和 bbox 实心区域不能进入正式路径。
- 全视频扫描覆盖 `real_1` 至 `real_4`、`fake_1` 和 `fake_2`，合计 984 帧，不读取真假标签或异常残差决定证据片段。
- 普通对象可以准备 mask 内稳定 2D 点，但在没有共享三维重建时不会计为正式三维结构图或结构残差。
- `CoverageReadiness` 不是异常分数；缺少事件是 `not_applicable`，观测链失败是 `observation_missing`。
- 遮挡事件缺少物理事件条件时为 `not_applicable`，缺少正式 mask、历史预测或 tracking 质量时为 `observation_missing`，不会产生残差 0。
- P4-A 聚合保留 value、valid、quality、coverage、missing reason、source IDs 和 branch names；全缺失输入输出 NaN，不输出真假结论。

### 权重状态

- 预期路径：`checkpoints/yolov8n-seg.pt`。
- 本次启动时文件不存在；从 Ultralytics 官方 GitHub release 的显式下载尝试被当前 WSL 到 `github.com:443` 的网络连接阻断。
- 未使用第三方不可信 PyTorch pickle 镜像，未生成残缺权重文件，也未将 `yolov8n.pt` 检测模型当作分割模型。
- `model_metadata.json` 当前记录：`weights_readable=false`、`weight_sha256=""`、`output_contains_instance_masks=false`、`available=false`。

### 六视频验证结果

- 输出目录：`outputs/real_3d_evidence_coverage_v2/`，旧 `outputs/real_3d_evidence_coverage/` 未覆盖。
- 视频数量：6；完整视频总帧数：984。
- 正式实例 mask：0；稳定 mask track：0；普通对象正式三维结构图：0。
- 现有共享三维 smoke 仍提供 `real_3` 的 1 条 person structure track 和 7 条正式结构时序证据。
- partial/full occlusion、正式 depth-order、boundary occlusion 和 reappearance 证据均为 0。
- `real_3` 因独立 person structure 分支达到 `ready_for_partial_p4=true`；其余视频为 false。
- 六个视频均为 `ready_for_full_p4=false`。
- 证据筛选器保留 1 个 `real_3` 帧 0-7 的 `person_structure` 候选；未找到 mask/遮挡/显隐候选。
- 无真实事件候选时输出空 candidate CSV；没有降低标准制造事件。

### 验证命令

```bash
.venv/bin/python -m pytest tests/test_real_instance_mask_provider.py tests/test_mask_object_association.py tests/test_real_mask_tracking.py tests/test_mask_internal_structure_points.py tests/test_person_keypoint_object_binding.py tests/test_person_structure_graph_real_path.py tests/test_real_evidence_clip_finder.py tests/test_scene_cut_statistics.py tests/test_coverage_readiness.py tests/test_aggregation_v2.py tests/test_real_3d_evidence_coverage_v2.py -q
.venv/bin/python scripts/run_real_3d_evidence_coverage_v2.py
.venv/bin/python -m pytest -q
```

### 验证结果

- P3-D.1/P4-A 专项：`26 passed`。
- 全量：`405 passed in 97.72s`。
- 仅一条既有 CUDA 驱动版本警告；CPU 路径可用。
- `configs/scale_priors_strict_v1.yaml` SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- `configs/scale_priors_strict_v2.yaml` SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。

### 当前限制

- `yolov8n-seg.pt` 尚未到位，因此正式 mask、mask-object association、mask tracking、普通对象结构和遮挡分支尚未获得真实覆盖数据。
- 六视频全扫描目前完成的是输入帧范围和缺失前置条件审计；不能把 0 个正式 mask 解释为场景没有对象或没有异常。
- full-video 共享深度、相机几何和三维点重建尚未覆盖全部 984 帧；现有正式结构证据仍来自冻结的短片段 cache。
- 当前没有可验证的 partial/full occlusion 或 reappearance 真实事件，不能进入 full P4。

### 下一步计划

- 通过可信渠道将官方冻结 `yolov8n-seg.pt` 放入 `checkpoints/`，先核验 SHA-256、模型 task 和真实 mask 输出，再重跑相同 coverage v2 命令。
- 对筛选出的 mask tracking、普通对象结构和遮挡候选片段建立共享深度/K/pose cache，再运行 P3-B/P3-C 正式残差。
- 只有 coverage readiness 达标后才将真实证据接入 P4-A；在此之前不做真假性能评价或阈值调优。

## 2026-07-22 - P3-D.1 Local Segmentation Weight Revalidation

### 目的

在用户将 `yolov8n-seg.pt` 从 Windows 手动复制到 WSL 后，重新验证冻结实例分割权重、真实像素 mask 输出、六视频全帧覆盖、对象关联、mask tracking、结构证据、可见性事件和 P4 coverage readiness。本次仍不进行真实/伪造性能评价。

### 新增文件

- 无源码文件；本次生成的验证产物位于被 Git 忽略的 `outputs/real_3d_evidence_coverage_v2/`。

### 修改文件

- `docs/DEV_LOG.md`：追加本地实例分割权重到位后的可复现验证结果。

### 删除文件

- 无。

### 权重验证

- 本地路径：`checkpoints/yolov8n-seg.pt`；文件可读，大小 `7,071,756` bytes。
- SHA-256：`a7cd8f929e1903d78a12a48efecab430209f18dc46cb96c3599a5980c63c423c`。
- 加载结果：`model_name=yolov8n-seg`、`model_task=segment`、`ultralytics=8.4.90`、`device=cpu`。
- `fake_1` 首帧实际输出 cup、couch 和 dining table 三个实例 mask；mask/enclosing-bbox 填充率分别约为 `0.8944`、`0.5988` 和 `0.6203`，证明正式输出不是 bbox 实心区域。
- 推理后 `output_contains_instance_masks=true`；没有自动下载，`bbox_used_as_formal_mask=false`。
- 权重由用户手动复制到本地；当前哈希可固定实验文件，但其下载来源与许可仍需在论文复现材料中独立登记。

### 六视频验证结果

- 扫描 6 个视频、共 984 帧；对象观测 2,455 个，正式可见实例 mask 1,823 个。
- 全局 mask-object association 成功率为 `0.742566`；bbox fallback 比例为 `0`，所有正式 mask 均为 visible mask，未推断 amodal mask。
- 各视频正式 mask 有效率：fake_1=`0.838806`、fake_2=`0.756048`、real_1=`0.159468`、real_2=`1.000000`（仅 3 个对象观测）、real_3=`0.667219`、real_4=`0.995810`。
- 稳定 mask track 共 11 条；candidate clip 共 47 个，包括 detector-missing control 25 个、stable multi-object 14 个、stable mask tracking 7 个和 person structure 1 个。
- 人体正式结构 track 仍为 1 条，普通对象正式三维结构 track 仍为 0，正式 structure temporal residual 共 7 条。
- 没有找到满足严格条件的 partial/full occlusion 或 reappearance；正式 depth-order 和 boundary-occlusion evidence 仍为 0。无事件被记录为 `not_applicable`，观测链失败记录为 `observation_missing`，均未写成正常残差 0。
- 只有 `real_3` 达到 `ready_for_partial_p4=true`；没有视频达到 `ready_for_full_p4=true`。

### 验证命令

```bash
sha256sum checkpoints/yolov8n-seg.pt

.venv/bin/python -m pytest \
  tests/test_real_instance_mask_provider.py \
  tests/test_mask_object_association.py \
  tests/test_real_mask_tracking.py \
  tests/test_mask_internal_structure_points.py \
  tests/test_coverage_readiness.py \
  tests/test_aggregation_v2.py \
  tests/test_real_3d_evidence_coverage_v2.py -q

.venv/bin/python scripts/run_real_3d_evidence_coverage_v2.py \
  --video_root data/tests_videos \
  --output_root outputs/real_3d_evidence_coverage_v2 \
  --mask_model_path checkpoints/yolov8n-seg.pt \
  --device cpu

.venv/bin/python -m pytest -q
sha256sum configs/scale_priors_strict_v1.yaml configs/scale_priors_strict_v2.yaml
```

### 验证结果

- P3-D.1 实例分割/关联/tracking/readiness/P4-A 专项：`20 passed in 2.90s`。
- 全量：`405 passed in 112.92s`。
- 唯一警告为本机 NVIDIA 驱动与当前 PyTorch CUDA 构建不匹配；本次显式使用 CPU，结果不受该警告影响。
- strict v1 SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。

### 当前限制

- 实例分割已真实启用，但不同视频的关联覆盖差异很大，特别是 real_1 仅约 15.95%；这需要单独诊断检测/分割类别与框的一致性，不能当作低异常。
- full-video mask tracking 已有覆盖，但普通对象 mask 内 2D 点尚未接入 full-video shared depth/K/pose 三维重建，因此普通对象正式三维结构图仍为 0。
- 当前六视频没有被严格规则确认的 partial/full occlusion 或 reappearance，不能验证真实遮挡残差，也不能进入 full P4。
- `real_3` 的 7 条正式结构残差仍来自已有短片段共享三维路径；不能据此代表六视频动态结构覆盖。
- 当前结果是观测覆盖与工程闭环验证，不是伪造检测性能结论。

### 下一步计划

- 优先诊断 real_1 的低 mask-object association 覆盖，并保留所有候选和拒绝原因，不降低一对一匹配标准。
- 将筛选出的稳定 mask tracks 接入 full-video shared depth、相机几何与对象三维重建，生成普通对象固定结构图和正式结构时序证据。
- 增补或独立采集包含明确 partial/full occlusion、out-of-frame 和 reappearance 的观测覆盖视频；仅按事件与观测质量筛选。
- 在普通对象结构与真实遮挡事件达到覆盖要求前，只能进入 partial P4 接口验证，不能开展 full P4 真实真假评价。

## 2026-07-22 - P4-A.1 Partial Aggregation and P3-D.2 Coverage Funnel Audit

### 目的

在不进行真假分类、AUC/F1、标签驱动权重学习或阈值调优的前提下，实现部分可用正式分支的质量感知五级聚合与时空定位；同时对 11 条稳定正式 mask track 逐条审计普通对象结构覆盖漏斗，并以 8/16/24/32 帧窗口诊断真实遮挡事件形成条件。

### 新增文件

- `configs/partial_p4_aggregation.yaml`：显式 debug 聚合与时间定位参数；阈值不来自六视频真假结果。
- `src/semantic3d/aggregation_v2/applicability.py`：定义五种 `EvidenceApplicability` 和完整局部证据记录。
- `src/semantic3d/aggregation_v2/evidence_registry.py`：注册 13 个正式静态/动态分支及其几何模式、级别、事件条件、质量门槛、归一化与定位目标。
- `src/semantic3d/aggregation_v2/hierarchy.py`：实现 point、edge、object、frame、clip 五级可追溯聚合。
- `src/semantic3d/aggregation_v2/temporal_localization.py`：实现 causal moving median、连续高证据分组、小间隔合并和最短时长过滤。
- `scripts/run_partial_p4_aggregation_smoke.py`：只对 `CoverageReadiness` 放行的视频运行 partial P4 聚合。
- `scripts/audit_ordinary_object_structure_coverage.py`：逐条输出稳定 mask track 的正式结构覆盖漏斗。
- `scripts/audit_occlusion_event_coverage.py`：从正式 mask/tracking 记录生成逐帧信号并运行多窗口遮挡诊断。
- `tests/test_aggregation_v2_multilevel.py`：验证 applicability、六种稳健算子、五级追溯、空间定位和时间区间。
- `tests/test_partial_p4_aggregation_smoke.py`：验证只运行 ready 输入且不消费真假标签。
- `tests/test_structure_and_occlusion_coverage_audit.py`：验证自适应 erosion、stable track 不丢失和遮挡拒绝原因。

### 修改文件

- `src/semantic3d/aggregation_v2/contracts.py`：为五级 aggregate 增加 applicability 计数、top contributors、四维 coverage、对象 mask 定位和 clip 时间定位字段。
- `src/semantic3d/aggregation_v2/aggregation.py`：新增 `median`、`trimmed_mean`、`top_k_mean`、`quality_weighted_mean`、`quality_weighted_top_k` 和 `hybrid_median_top_k`；保留 legacy `topk_mean` 兼容。
- `src/semantic3d/aggregation_v2/__init__.py`、`src/semantic3d/__init__.py`：导出新增 P4-A.1 API。
- `src/semantic3d/occlusion/mask_structure_points.py`：增加仅由 mask 像素尺寸决定的自适应 erosion，并支持灰度帧的低内存点跟踪审计。
- `src/semantic3d/occlusion/__init__.py`：导出自适应 erosion API。
- `scripts/find_real_3d_evidence_clips.py`：增加多窗口遮挡信号形成与拒绝原因输出，继续拒绝 truth label 和 anomaly score 字段。
- `scripts/run_real_3d_evidence_coverage.py`：持久化已计算的人体对象级和固定边 structure temporal residual，避免 P4 绘图阶段重建数值。

### 删除文件

- 无。

### 主要变化

- 只有 `applicable + valid + finite + quality >= floor` 的证据进入异常值聚合；`not_applicable` 不降低覆盖，`observation_missing` 降低覆盖，`invalid_geometry` 和 `unsupported_mode` 单独计数。
- 缺失或无事件继续使用 NaN，不写成 0。quality 是工程质量分数，不是训练概率；anomaly、quality 和 branch/object/temporal/spatial coverage 分离保存。
- 注册表包含 13 个正式分支；diagnostic 分支必须显式 opt-in，不能自动成为正式异常证据。
- real_3 是唯一进入 partial P4 的视频；其余五个视频均输出 `not_ready` 和 `no_formal_structure_or_occlusion_residual_branch`，没有按真假标签区别处理。
- real_3 实际聚合 `direction_consistency`、`relative_velocity_change`、`dynamic_reprojection` 和 `structure_temporal` 四个正式分支，13 分支注册表覆盖率为 `4/13=0.307692`。
- P3-D 的 7 个 object structure transitions 已持久化为 7 条对象 residual 和 56 条固定边 residual；P4 使用这 56 条真实 edge evidence，没有构造替代值。
- real_3 生成 98 个 point aggregate、56 个 edge aggregate、7 个 object-frame aggregate、7 个 frame aggregate、1 个 clip aggregate；所有 7 个对象定位均引用正式 visible-mask 文件。
- 显式 debug threshold `0.50` 下形成 1 个时间候选区间（global frame 3-7）；该区间只表示聚合 smoke 的高证据区间，不是真假判定或论文阈值。
- 11 条稳定 mask track 全部进入漏斗且无静默丢失：11 条均足够长、均有 mask 内候选点和稳定 ID；只有 real_3 的 person/cup 两条处于 static-camera 几何模式，但 shared depth 仅覆盖约 `8/201=0.0398`，没有任何 track 达到 full-video depth 覆盖门槛。
- 普通对象正式三维结构 graph 和 structure residual 仍为 0。其余稳定 track 的主要停止原因是 unsupported geometry mode；real_3 cup 停在 insufficient shared depth coverage；real_3 person 进入独立人体结构分支，不计作普通对象。
- 遮挡审计生成 798 条逐帧信号并测试 410 个窗口：261 个为 `no_observable_occlusion_event`，149 个为有接触/显隐信号但被正式门控拒绝的预候选；全部窗口缺少同步正式 depth-order，正式遮挡候选仍为 0。

### 验证命令

```bash
.venv/bin/python scripts/run_real_3d_evidence_coverage.py \
  --output_root outputs/real_3d_evidence_coverage_v2/shared_3d_smoke \
  --mask_model_path checkpoints/yolov8n-seg.pt \
  --pose_model_path checkpoints/yolov8n-pose.pt \
  --device cpu

.venv/bin/python scripts/run_partial_p4_aggregation_smoke.py
.venv/bin/python scripts/audit_ordinary_object_structure_coverage.py
.venv/bin/python scripts/audit_occlusion_event_coverage.py

.venv/bin/python -m pytest \
  tests/test_aggregation_v2.py \
  tests/test_aggregation_v2_multilevel.py \
  tests/test_partial_p4_aggregation_smoke.py \
  tests/test_structure_and_occlusion_coverage_audit.py \
  tests/test_real_3d_evidence_coverage.py \
  tests/test_real_3d_evidence_coverage_v2.py -q

.venv/bin/python -m pytest -q
sha256sum configs/scale_priors_strict_v1.yaml configs/scale_priors_strict_v2.yaml
```

### 验证结果

- 聚合/覆盖专项联合回归：`46 passed in 1.76s`。
- 最终关键专项：`19 passed in 5.24s`，仅有既有 CUDA 驱动警告。
- 最终全量：`416 passed in 90.24s`，仅有同一条 CUDA 驱动警告；CPU 路径正常。
- strict v1 SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。

### 当前限制

- real_3 的动态 CSV 没有持久化原始 evidence quality；方向、相对速度和动态重投影暂用明确标注的保守 engineering quality=`0.50`，不是概率。structure edge 使用源码实际计算质量。
- partial P4 只有 1 个视频、1 条 person track 和 7 个有效帧，不能用于性能评价、阈值选择、权重学习或消融结论。
- 普通对象虽已有正式 mask 内稳定 2D 点，但 full-video shared depth、K、pose 和 3D reconstruction 尚未接入，不能形成正式普通对象三维结构图。
- 149 个遮挡预候选缺少正式 depth order；mask 接触和 detector-missing 信号本身不足以证明遮挡。
- 当前没有任何通过严格门控的 partial/full occlusion 或 reappearance，不能进入 full P4。

### 下一步计划

- 可以开始 P4-B 离线结构增强数据集的 schema、采集和缓存构建，但不能开始标签驱动聚合训练或真实性性能结论。
- 优先把 11 条稳定 mask track 接入 full-video shared relative depth、相机几何与三维点重建，先让 real_3 cup 获得普通对象固定结构图。
- 为遮挡窗口同步生成正式深度次序与历史支撑证据，并补充独立的几何验证视频，覆盖 partial/full occlusion、reappearance 和 out-of-frame。
- 在多视频、多对象、多事件覆盖到位后，再讨论 P4-B 分支标准化、无标签权重设定与冻结实验协议。

## 2026-07-22 - P4-B Versioned Structural Enhancement Dataset

### 目的

建立版本化、可复现、可断点续跑且与真假标签隔离的离线结构增强数据集。将 P3-D.1 的六视频全帧正式实例 mask、real_3 全视频真实单目相对深度、clip-local shared 3D，以及 P4-A.1 已有正式动态证据迁入统一 Parquet/array schema。本阶段不训练分类器，不计算 AUC/F1/准确率，不使用真假标签选择 clip、拟合归一化统计或学习聚合权重。

### 新增文件

- `configs/p4b_six_video_smoke.yaml`：六视频 P4-B 数据源、scene-aware clip、provider、阶段与标签隔离配置。
- `src/semantic3d/dataset_builder/schema.py`：schema 版本、dataset manifest、evidence applicability 和 NaN/valid 契约。
- `src/semantic3d/dataset_builder/ids.py`：dataset/video/clip/frame/object/point/edge/evidence/coordinate system 稳定 ID。
- `src/semantic3d/dataset_builder/manifest.py`：视频探测、全帧 scene signature、scene segment、重叠 clip 和唯一 owner 分配。
- `src/semantic3d/dataset_builder/cache.py`：内容寻址阶段缓存、artifact hash、损坏检测和 atomic completion record。
- `src/semantic3d/dataset_builder/stages.py`：01–13 阶段依赖图与下游失效传播。
- `src/semantic3d/dataset_builder/clip_alignment.py`：跨 clip 相似变换契约；无可靠 overlap alignment 时禁止三维坐标相减。
- `src/semantic3d/dataset_builder/writer.py`、`reader.py`：真实 Apache Parquet、压缩数组引用、SHA-256、atomic write 和显式标签读取拒绝。
- `src/semantic3d/dataset_builder/validation.py`：ID、外键、owner、scene、coordinate、NaN、array hash、正式 mask、diagnostic 和标签隔离验证。
- `src/semantic3d/dataset_builder/pipeline.py`：标签隔离的 13 阶段离线构建器、逐帧深度 resume、正式证据迁移和 applicability coverage。
- `scripts/build_structural_enhancement_dataset.py`：支持 `--config`、`--video-id`、`--stage`、`--resume`、`--force-stage`、`--device`、`--num-workers` 和 `--dry-run` 的统一 runner。
- `scripts/validate_structural_enhancement_dataset.py`：数据集完整性和覆盖验证入口。
- `scripts/build_labels_manifest.py`：结构构建完成后独立生成标签表；不属于 13 阶段且不会被结构 builder 读取。
- `tests/test_structural_enhancement_dataset.py`：稳定 ID、scene/owner、坐标隔离、NaN、Parquet、标签隔离和确定性 resume 测试。
- `tests/test_dataset_builder_cache.py`：cache key、权重/配置失效、损坏缓存、临时文件和 atomic write 测试。

### 修改文件

- `pyproject.toml`：增加 `pyarrow>=15.0`，用于真实 Parquet 输出。
- `docs/DEV_LOG.md`：追加本次 P4-B 设计、真实运行结果和限制。

### 删除文件

- 无。

### 主要变化

- schema 版本为 `semantic3d_structural_enhancement_v1`，pipeline 版本为 `p4b_v1`；当前 dataset ID 为 `dataset_bf697e96fded98a1838f`。
- 六视频全部完成索引：984 个唯一源帧、59 个 scene-aware clip、2,308 条 clip-frame membership；984 个可评分帧各有且只有一个 owner，owner 覆盖率 100%。其余 1,324 条是 context membership，不写重复正式证据。
- dataset manifest 保存 Git commit、完整 config SHA-256、provider metadata、权重哈希、随机种子、Python/PyTorch/OpenCV/Ultralytics/PyArrow/OS/device/package-lock hash。结构构建器没有 label 参数，不读取真假或篡改标签；独立脚本在结构阶段完成后通过 video_id 写入 6 条 `labels_manifest.parquet`，该表不进入结构 cache graph。
- 正式对象观测 2,455 条；正式 visible instance mask 1,823 条，有效率 `0.742566`；bbox fallback 为 0，未推断 amodal mask。所有迁入 mask 已复制到 dataset arrays 并记录 shape/dtype/SHA-256。
- 使用本地缓存的 Depth Anything V2 Small 对 real_3 全部 201 帧执行真实单目推理。保存 reciprocal 转换后的 `relative_depth`，项目约定值越大越远；该深度不是米制深度。其他五个视频当前没有 P4-B full-video depth，均保持 invalid/NaN。
- real_3 通过 clip 内背景中位数尺度对齐获得 201/201 个 owned shared-3D frame，显著高于旧 8 帧 smoke；全数据集 shared-3D 覆盖为 `201/984=0.204268`。12 个 real_3 clip 为 `static_camera_3d`，其余 47 个 clip 为 unavailable。不同 clip 使用不同 coordinate system，未做未经验证的跨 clip 三维减法。
- 关键点表迁入 8 条可用 real_3 person frame coverage 记录；坐标本身没有伪造补全，因此不能声称全视频关键点覆盖。
- 全视频普通对象漏斗发现 2 条 track 同时具有正式 mask 与 shared depth，但稳定 mask 内三维点 correspondence 尚未实现，普通对象正式结构图仍为 0，普通对象 structure transition 仍为 0。已有 person structure track 为 1，已验证正式 structure transition 仍为 7。
- 已迁入 98 条 point、56 条 edge、25 条分支化 object、7 条 frame 和 1 条 clip 有效证据。P4-A.1 的四个正式分支保持为 direction、relative velocity、dynamic reprojection 和 person structure；没有从输出异常值学习阈值或权重。
- real_3 的 registry branch coverage 为 `4/13=0.307692`；当前理论适用分支覆盖为 `4/9=0.444444`。`statistically_normalized_value` 全部为 NaN，`normalization_fit_source=none`。
- 同步 depth order 已从旧扫描的 0 帧增加到 real_3 的 201 帧；仅 1 帧同时存在正式 mask overlap 和 depth order。没有片段满足完整可见性变化/历史支撑门控，正式 partial/full occlusion 和 reappearance 仍为 0；分支记录为 `not_applicable`，不是残差 0。
- 阶段输出使用 content-addressed cache、临时文件和 atomic rename。逐帧 depth artifact 可在中断后复用；损坏 array/cache hash 会导致 miss。最终同配置 resume 为 13/13 cache hit。
- dataset validator 检查 16 张必需表，结果为 0 errors、0 warnings、0 array hash errors。报告位于 `outputs/structural_enhancement_dataset/p4b_six_video_smoke/reports/`；大数组和 outputs 保持 Git 忽略。

### 验证命令

```bash
.venv/bin/python scripts/build_structural_enhancement_dataset.py \
  --config configs/p4b_six_video_smoke.yaml \
  --dry-run

.venv/bin/python scripts/build_structural_enhancement_dataset.py \
  --config configs/p4b_six_video_smoke.yaml \
  --resume \
  --device cpu

.venv/bin/python scripts/validate_structural_enhancement_dataset.py \
  --dataset-root outputs/structural_enhancement_dataset/p4b_six_video_smoke

.venv/bin/python scripts/build_labels_manifest.py \
  --input-manifest data/manifests/pilot_real_fake.csv \
  --dataset-root outputs/structural_enhancement_dataset/p4b_six_video_smoke

.venv/bin/python -m pytest \
  tests/test_structural_enhancement_dataset.py \
  tests/test_dataset_builder_cache.py -q

.venv/bin/python -m pytest -q
sha256sum configs/scale_priors_strict_v1.yaml configs/scale_priors_strict_v2.yaml
```

### 验证结果

- P4-B 专项：`14 passed in 1.99s`。
- 全量：`430 passed in 108.69s`；唯一警告仍为本机 NVIDIA 驱动与 PyTorch CUDA 构建不匹配，正式构建显式使用 CPU。
- 六视频索引：984/984 唯一帧，59 clips，984/984 owner frames。
- dataset validation：16/16 必需表存在，`valid=true`，0 integrity errors，0 warnings。
- 最终同配置 resume：13/13 阶段 cache hit；ID、manifest 行数、owner 分配和 applicability 状态保持一致。
- strict v1 SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。

### 当前限制

- 只有 real_3 完成 full-video depth/shared 3D；其余五个视频是完整索引与正式 mask 数据，但没有 full-video geometry，不能用于六视频 P4-C 统计建模。
- real_3 的 full-video shared 3D 当前由 approximate K、static-camera contract 和 clip-local relative-depth alignment 构成，不是公制重建；clip 间没有可靠 alignment，坐标保持隔离。
- keypoint/point-track/dynamic residual 仍只覆盖原 8 帧附近，dynamic-readiness 不是 201 帧全覆盖。
- 普通对象有 2 条可进入下一漏斗的 track，但稳定三维点 correspondence 未实现，所以正式普通对象结构图没有从 0 增加。
- depth order 同步覆盖已改善，但严格遮挡事件仍为 0；不能通过降低事件标准制造遮挡证据。
- 当前标签隔离 schema 和缓存基础已经具备，但样本、几何模式、对象结构和事件覆盖不足，尚不具备进入 P4-C 训练集统计建模或真假性能评价的条件。

### 下一步计划

- 将 full-video depth/shared 3D 扩展到其余五个视频，并独立验证每个 geometry mode 与 clip alignment 质量。
- 在正式 mask 内建立跨帧稳定 point correspondence，优先让 real_3 cup 的 2D 候选点形成 clip-local 固定三维结构图与 transition。
- 对 full-video person keypoints 和 KLT point tracks 做同一 owner/coordinate contract 下的离线提取，避免只迁移 8 帧 smoke。
- 增加独立遮挡事件覆盖视频；在不读取真假标签的前提下验证 partial/full occlusion、out-of-frame 和 reappearance。
- 只有在多视频、多个对象/事件和正式分支覆盖到位后，再冻结无标签归一化拟合数据划分并进入 P4-C；当前六视频不得用于学习正式统计参数。

## 2026-07-22 - P4-B.5 Six-Video Full Structural Observation Coverage

### 目的

在不读取真假标签、不训练分类器、不拟合统计参数和不降低事件门控的前提下，将 P4-B 从 real_3 局部深度与 8 帧动态 smoke 扩展为六视频 984 帧 canonical depth、全 59 clip 二维点轨迹与序列几何诊断。补齐普通对象正式 mask 内固定结构图、owned-frame 动态证据和遮挡候选同步深度次序，并保持 frame-level relative 3D、sequence-aligned geometry 与 dynamic-ready 三种语义严格分离。

### 新增文件

- `configs/p4b5_six_video_full_observation.yaml`：P4-B.5 六视频全观测配置；输出到独立版本目录并声明 P4-B 增量来源。
- `src/semantic3d/dataset_builder/p4b5_contracts.py`：五类覆盖指标、clip geometry decision、跨 clip 身份交接和同步深度次序契约。
- `src/semantic3d/dataset_builder/p4b5_pipeline.py`：全帧 depth/keypoint、全 clip 点跟踪/几何、结构图、动态证据和遮挡重评分的 13 阶段实现。
- `tests/test_p4b5_full_observation.py`：P4-B.5 覆盖语义、frame/sequence 隔离、点 ID、固定图、handoff、深度次序、增量写入和标签隔离测试。

### 修改文件

- `scripts/build_structural_enhancement_dataset.py`：根据配置选择 legacy P4-B 或 P4-B.5 builder，旧入口继续兼容。
- `src/semantic3d/dataset_builder/schema.py`：增加 P4-B.5 pipeline 版本和条件化表主键。
- `src/semantic3d/dataset_builder/__init__.py`：导出 P4-B.5 数据契约和构建器。
- `src/semantic3d/dataset_builder/validation.py`：增加 raw/canonical depth、frame/sequence 语义隔离、owned/context 去重、点 ID、跨 clip 3D、正式 mask 结构点、证据追溯和冻结先验哈希验证。
- `docs/DEV_LOG.md`：追加本次真实构建、验证结果和限制。

### 删除文件

- 无。

### 主要变化

- 新数据集位于 `outputs/structural_enhancement_dataset/p4b5_six_video_full_observation/`，dataset ID 为 `dataset_1ae499fd7cdd9d8501dc`；旧 `p4b_six_video_smoke` 未覆盖。
- 六视频 canonical relative depth 覆盖 `984/984=100%`。保存 raw model output 与 canonical relative depth；`visualization_depth_used=false`。深度仍为 Depth Anything 单目相对深度，不是米制深度。
- frame-level shared relative 3D 覆盖 `656/984=66.67%`。其坐标为逐帧 camera frame，world 字段缺失，禁止跨帧直接相减。
- 59/59 clip 均完成 sequence depth alignment 诊断；只有 2/59 clip 被保守判定为 `static_camera_3d` 并进入 dynamic-ready，57 个为 `unavailable`。其中 26 个缺少可用 full-SE3，26 个只有未物化旋转变换的诊断证据，5 个缺少足够独立点跟踪；没有把 homography 诊断伪装成 rotation-compensated 轨迹。
- 全视频 342 条 person observation 中 209 条获得有效人体关键点，133 条为 `no_pose_detection`；共保存 3,553 个关键点槽位，其中 2,589 个有效。形成 1 条 person 固定结构 track。
- 全 59 clip 保存 83,689 条二维点观测，其中 80,836 条有效，对应 3,411 条独立 clip point track；owned/context 标记与唯一 owner 用于避免重叠 clip 重复写正式证据。
- 保存 83,689 条三维点观测记录；受严格 sequence geometry 限制，只有 4,031 条有效，其余保持 invalid/NaN，不跨未对齐 coordinate system 计算差值。
- 正式动态证据新增 5,364 条有效 point evidence：`track_3d_continuity=1361`、`direction_consistency=1336`、`relative_velocity_change=1336`、`dynamic_reprojection=1331`；另有 64 条有效 `structure_temporal` edge evidence。coverage 报告按 owned point-transition 候选口径记录 `1425/35984=3.96%`，与 evidence 表行数是不同统计口径。
- 旧 P4-B 两条普通对象候选按视频、类别和帧范围映射：real_3 `cup` 从正式 mask、稳定 2D point ID、有效 clip-local 3D 进入 1 个正式固定结构图和 15 个有效 transition；real_3 单帧 `chair` 只有 frame 0 的 1 次观测，因不满足长轨迹条件而未建立点轨迹、结构图或 transition。
- 普通对象正式结构图从 0 增加到 6，分布为 fake_1 的 5 条和 real_3 cup 的 1 条；4 条普通对象 clip-track 共产生 49 条有效 transition。加上 person 的 15 条，正式 structure transition 总数为 64。bbox graph 正式证据数保持 0，跨未对齐 clip transition 保持 0。
- 对旧 149 个遮挡预候选全部补齐同步 depth-order 诊断；全量扫描另产生 367 条具有正式 mask overlap 和有效深度次序的记录。由于 431 条深度关系仍不确定且没有候选同时通过完整事件门控，正式 partial/full occlusion 事件仍为 0；无事件不写成残差 0。
- 五类覆盖率分别持久化为 `frame_depth_coverage`、`frame_shared_3d_coverage`、`sequence_depth_aligned_coverage`、`dynamic_3d_ready_coverage` 和 `formal_dynamic_evidence_coverage`，不再用单一 shared-3D 指标混淆观测与动态可用性。
- 数据集记录 `truth_labels_used=false`、`classification_output=false`，未产生真假分类、AUC/F1/准确率、阈值拟合或标签驱动片段选择。

### 验证命令

```bash
.venv/bin/python scripts/build_structural_enhancement_dataset.py \
  --config configs/p4b5_six_video_full_observation.yaml \
  --resume --device cpu

.venv/bin/python scripts/validate_structural_enhancement_dataset.py \
  --dataset-root outputs/structural_enhancement_dataset/p4b5_six_video_full_observation

.venv/bin/python -m pytest tests/test_p4b5_full_observation.py -q
.venv/bin/python -m pytest -q \
  --junitxml=/tmp/p4b5_full_suite.xml

sha256sum configs/scale_priors_strict_v1.yaml \
  configs/scale_priors_strict_v2.yaml
```

### 验证结果

- 六视频真实离线构建完成，13 个阶段均写入完成记录；stage 01-06 复用可验证缓存/旧数组，stage 07-13 完成全 clip 重建与派生证据。
- 完整构建后以同一配置再次执行 `--resume`，13/13 阶段全部 `cache_hit=true`，没有重新运行模型推理。
- P4-B.5 专项：`15 passed`。
- 全量：`445 tests`，`0 failures`，`0 errors`，`5 skipped`，测试时间约 `94.39s`。
- dataset validation：`valid=true`，25 张表通过检查，`0 errors`、`0 warnings`、`0 array hash errors`；984 个唯一帧、59 个 clip、984 个 owned frame。
- strict v1 SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。

### 当前限制

- 六视频有完整逐帧单目相对深度，但没有公制尺度；frame camera coordinates 不能被解释为世界坐标或跨帧公制轨迹。
- 59 个 clip 中仅 2 个通过严格 dynamic-ready，动态证据集中在少量 static-camera clip；57 个 unavailable clip 的 frame-level 3D 不能替代序列三维证据。
- 当前没有物化的 rotation-only 变换，也没有观测充分的 full-SE3；因此 rotation/full-SE3 正式轨迹仍为 0。
- 普通对象结构图已从 0 增加到 6，但有效 transition 仍集中在 fake_1 与 real_3，不能代表六视频广泛对象覆盖。
- 149 个旧遮挡预候选已获得同步 depth order，但没有候选通过全部遮挡事件门控；正式遮挡事件仍为 0，不能训练或评价遮挡分支。
- 动态 evidence 行数与 formal coverage ratio 使用不同分母：前者是各分支可追溯证据行，后者是 owned point-transition 候选覆盖，报告时不得直接互换。
- 六视频仍是工程覆盖集，不是统计训练集；不能用于学习归一化、聚合权重、真假阈值或报告性能指标。

### 下一步计划

- 当前可以进入 P4-C0 正式数据集规划：冻结 schema、定义采集协议、规划 camera-motion/occlusion/reappearance/ordinary-structure 覆盖和 train/val/test 隔离。
- 当前不能进入 P4-C 统计建模。需要先补充可观测 rotation/full-SE3 几何、独立遮挡事件视频、更多普通对象结构 track，以及足够规模且标签严格隔离的数据。
- 后续若实现跨 clip 3D 连续性，必须先生成并验证 `ClipAlignmentObservation`；身份 handoff 本身不得授权三维坐标相减。

## 2026-07-22 - P4-C0 Experiment Protocol Planning

### 目的

在 P4-B.5 六视频结构观测完成后，建立正式实验所需的数据集目录清单、稳定 source group、group-aware split、独立标签协议、分支能力分层、重复/泄漏审计、missingness shortcut 诊断、覆盖目标和存储规划。本阶段只进行协议 smoke，不读取残差大小决定划分，不拟合均值/方差/MAD/分位数，不学习聚合权重，不选择阈值，也不报告真假性能指标。

### 新增文件

- `configs/p4c0_experiment_protocol_v1.yaml`：P4-C0 输入、独立标签、split、重复阈值、coverage 和 `/mnt/e` 存储规划配置。
- `src/semantic3d/experiment_protocol/schema.py`：DatasetCatalog、LabelRecord、SourceGroupRecord、VideoInventoryRecord、SplitAssignment、BranchEligibilityRecord 和验证结果契约。
- `src/semantic3d/experiment_protocol/inventory.py`：读取结构 metadata 与独立标签 manifest，建立目录和任务资格；禁止从文件名推断标签。
- `src/semantic3d/experiment_protocol/source_grouping.py`：基于明确 original identity 或保守 file-hash fallback 生成稳定 source-group ID。
- `src/semantic3d/experiment_protocol/split_planner.py`：官方 split 优先、source-group 不拆分、标签/类型/来源/分辨率/时长/能力分层平衡的确定性规划器。
- `src/semantic3d/experiment_protocol/duplicate_audit.py`：文件 SHA-256、视频 metadata、抽帧 dHash 和可选 embedding provider 接口。
- `src/semantic3d/experiment_protocol/leakage_audit.py`：source group、exact/near duplicate、官方 split、normalization 和 threshold source 泄漏审计。
- `src/semantic3d/experiment_protocol/branch_eligibility.py`：Tier S/D1/D2/D3/O 与 13 个正式 residual branch 的观察资格和缺失原因。
- `src/semantic3d/experiment_protocol/coverage_planning.py`：分支/事件/标注目标缺口和按 split/class 的 missingness shortcut 风险。
- `src/semantic3d/experiment_protocol/storage_planning.py`：按 P4-B.5 实际文件构成外推磁盘、峰值空间与 CPU 时间范围。
- `src/semantic3d/experiment_protocol/validation.py`：16 类协议一致性和严重泄漏验证。
- `src/semantic3d/experiment_protocol/__init__.py`：导出 P4-C0 公共 API。
- `scripts/plan_p4c_experiment_protocol.py`：只读取 metadata、独立 label/source/split 和 coverage 的统一规划 runner。
- `scripts/validate_p4c_experiment_protocol.py`：独立协议验证入口。
- `tests/test_p4c_experiment_protocol.py`：稳定 group、同源不跨 split、官方 split、重复审计、标签隔离、分支能力、missingness、coverage、storage、test 隔离和冻结哈希测试。

### 修改文件

- `docs/DEV_LOG.md`：追加 P4-C0 六视频 protocol smoke 的真实结果。

### 删除文件

- 无。

### 主要变化

- 生成 `data/experiment_protocol_v1/` 下的 catalog、source groups、inventory、split、label schema、annotation availability、branch eligibility、duplicate/leakage、missingness、coverage、storage、protocol 和 validation 共 15 类输出；没有复制视频或结构大数组。
- 当前 catalog 仅纳入 `local_six_video_protocol_smoke`，角色为 `geometry_validation`，规模 4 real + 2 fake。使用权记录为用户本地提供，发布和再分发权限仍需人工确认；没有凭空登记尚未提供来源与许可信息的外部正式数据集。
- P4-B.5 内部 `labels_manifest.parquet` 为空；P4-C0 显式读取独立 `data/manifests/pilot_real_fake.csv` 并按 source name 映射。所有二元标签均来自该 manifest，没有从 `real_*`/`fake_*` 文件名推断。
- 六个视频都缺少 original-source identity，因此生成 6 个稳定 file-hash fallback group，并全部标记 `source_group_review_required=true`。不能断言真实原片与潜在伪造衍生版已经正确合并。
- 六视频 manifest 原声明全部为 val；协议 smoke 保守地规范化为 6 个 validation、0 train、0 test，避免在 source relation 未确认时制造跨 split。正式 group-aware 算法已实现并通过合成同源 real/fake 和官方 split 测试。
- 15 个视频对均完成 exact/near duplicate 审计：exact duplicate 为 0，最高 near score 约 `0.6347`，低于冻结候选阈值 `0.92`；当前没有重复跨 split 冲突。但 source identity 未知仍是独立风险，不能被低感知相似度替代。
- Tier 覆盖：6/6 视频、45/59 clip 具备 Tier S；2/6 视频、2/59 clip 具备 Tier D1；Tier D2=0，Tier D3=0。Tier O 没有 applicable 事件：53 clip 为 `not_applicable`，6 clip 为 `observation_missing`，两者保持区分。
- 每个 video/clip 同时保存 13 个正式 evidence branch 的 geometry、mask、keypoint、event 要求。单目相对深度没有公制/尺度锚点，因此 `semantic_size_3d` 不因静态几何可用而被误标为正式可用。
- 六视频均为 `geometry_validation_only`。正式 detection training、video classification、temporal/spatial/object localization 和 occlusion validation 资格当前均为 0；二元标签存在不等于六视频可作为正式训练/评价集。
- normal-reference 路线冻结为只使用 train split 的真实视频，并按 branch/geometry/evidence level 拟合；supervised 路线冻结为 train 标签训练、validation 调参、test 仅最终评价。当前 fit/training video ID 列表均为空，未拟合任何参数。
- missingness 审计不会将无 person 的 keypoint 分支填成 0，而是记为 not applicable。当前唯一 shortcut 风险是 validation 内 real/fake frame shared-3D coverage 平均差 `0.4382`；未来需要分层平衡并做 coverage-feature 消融，provider failure 默认不得成为异常证据。
- coverage planning 的 minimum/desired gap 已独立保存：D1 当前 2，D2/D3/partial/full occlusion/reappearance 均为 0；person structure=1，ordinary structure=6，时间和空间定位标注均为 0。
- P4-B.5 实测 984 帧占约 `2,367,328,398` bytes。按 100,000 帧线性规划，dataset output 约 `240.58 GB`，source video 约 `1.69 GB`，cache 约 `0.10 GB`，临时峰值约 `302.97 GB`；CPU 规划范围约 `27.8-138.9` 小时，正式采集前必须实测 benchmark。
- 规划路径为 `/mnt/e/fake_video_structural_data/{source,datasets,cache,archive}`，本轮没有创建或复制正式数据。当前 `/mnt/e` 仅约 `119.19 GB` 可用，无法满足约 `302.97 GB` 峰值，因此 validator 输出容量警告；NTFS 大量小文件也需使用 Parquet/sharded arrays 和 Linux-local active cache 缓解。

### 验证命令

```bash
.venv/bin/python scripts/plan_p4c_experiment_protocol.py \
  --config configs/p4c0_experiment_protocol_v1.yaml

.venv/bin/python scripts/validate_p4c_experiment_protocol.py \
  --protocol-root data/experiment_protocol_v1 \
  --structural-dataset-root \
  outputs/structural_enhancement_dataset/p4b5_six_video_full_observation

.venv/bin/python -m pytest tests/test_p4c_experiment_protocol.py -v
.venv/bin/python -m pytest -q --junitxml=/tmp/p4c0_full_suite.xml

sha256sum configs/scale_priors_strict_v1.yaml \
  configs/scale_priors_strict_v2.yaml
```

### 验证结果

- P4-C0 专项：`13 passed`。
- 最终全量：总计 458 项，`453 passed`、`5 skipped`、`0 failed`，耗时约 `97.37s`。唯一警告为本机 NVIDIA 驱动与 PyTorch CUDA 构建不匹配；CPU 路径正常。
- protocol validation：`valid=true`，16 类检查全部满足，0 error；3 个明确 warning 分别为 `/mnt/e` 峰值空间不足、6 个 source group 待复核、六视频仅为 protocol smoke。
- leakage conflict=0；test 不属于 normalization fit 或 threshold tuning；59 个 clip 均继承视频级 validation split；split decision inputs 不包含 residual 或 anomaly score。
- strict v1 SHA-256 保持 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2 SHA-256 保持 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。

### 当前限制

- 当前只 catalog 了 6 个本地工程视频，尚未提供正式训练、validation、final-test、cross-dataset、occlusion 或 localization 数据集的来源、许可和官方 split。
- 六视频 source identity 全部待复核；0 个已发现重复冲突不等于已排除原始视频与衍生伪造之间的来源泄漏。
- 没有 train/test 样本；所有正式训练与评价任务资格为 0，因此不能进入统计拟合、监督训练或性能评价。
- D2、D3、partial/full occlusion、reappearance 和时间/空间定位标注覆盖均为 0。
- 当前 `/mnt/e` 可用空间不足以承载 100,000 帧计划的临时峰值；需要扩容、降低批次规模或把 active cache 放到更快且容量足够的 Linux 磁盘。
- 100,000 帧存储与时间是线性工程估计，不是正式 benchmark；大规模构建前必须用目标分辨率和 provider 组合重新测量。

### 下一步计划

- 先选择并完成正式数据集的许可、官方 split、original-source identity、manipulation type、时间/空间标注盘点，再写入 catalog；不得根据残差表现选数据。
- 人工确认真实原片、伪造衍生版、压缩/裁剪/转码关系，冻结 source-group manifest 后再执行正式 group-aware split。
- 扩充 D2/D3、普通/人体结构、partial/full occlusion、reappearance 和 localization annotation 覆盖，满足 minimum engineering target 后再讨论 P4-B-scale 构建。
- 为 `/mnt/e` 准备至少约 303 GB 峰值空间并预留安全余量，或改用分批构建和独立 archive/cache 磁盘。
- 当前 `ready_for_p4b_scale_formal_build=false`；只有 source、license、split、storage 和 branch/event coverage 缺口关闭后才能转为 true。

## 2026-07-22 - P4-C1 Frozen Experiment Manifest And Data Integrity Audit

### 目的

在不改变 P4-C0 协议语义、不重新规划 split、不读取异常残差、不训练模型、不拟合统计分布和不选择阈值的前提下，将 P4-B.5 六视频的 59 个重叠 clip 固化为确定性实验清单。逐样本记录视频、帧区间、标签来源、scene、缺失 camera identity、对象/深度/相机/轨迹/semantic3d 表路径、技术可用性和排除原因，并重新审计 source video、source group、原始/派生身份、文件哈希和近重复跨 split 泄漏。

### 新增文件

- `configs/p4c1_experiment_manifest_v1.yaml`：P4-C1 输入、核心/可选模态、完整性检查、泄漏审计和禁止操作配置。
- `src/semantic3d/experiment_protocol/manifest_schema.py`：稳定 sample ID、clip manifest、可用性、泄漏 finding 和构建结果数据契约。
- `src/semantic3d/experiment_protocol/data_inventory.py`：读取 P4-C0/P4-B.5 Parquet 索引并统计 clip 级对象、深度、camera、pose、轨迹和 shared-3D 可用性。
- `src/semantic3d/experiment_protocol/exclusion_policy.py`：只基于数据完整性的显式排除规则；可选动态分支缺失不会被误判为全局不可用。
- `src/semantic3d/experiment_protocol/manifest_builder.py`：继承冻结 split、构建 canonical JSONL、计算哈希并执行独立复现验证。
- `src/semantic3d/experiment_protocol/manifest_report.py`：确定性 CSV/JSONL、排除/可用性/split 汇总和 Markdown 报告。
- `scripts/build_p4c1_experiment_manifest.py`：两次内存重建、写出和验证的统一入口。
- `scripts/validate_p4c1_experiment_manifest.py`：从现有输入重新构建并核验 manifest 哈希、排序、schema 与 split 隔离。
- `tests/test_p4c1_experiment_manifest.py`：稳定 ID、排除策略、全 clip 覆盖、路径、相机身份、哈希和禁止操作测试。
- `tests/test_p4c1_leakage_audit.py`：source video/group、原始来源、相同哈希、近重复和 unresolved provenance 测试。

### 修改文件

- `src/semantic3d/experiment_protocol/leakage_audit.py`：向后兼容保留 P4-C0 `audit_leakage`，新增 clip 去重后的视频级 P4-C1 泄漏审计。
- `src/semantic3d/experiment_protocol/__init__.py`：导出 P4-C1 数据类、构建器、哈希和验证 API。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件

- 无。

### 主要变化

- 输出目录为 `outputs/p4c1_experiment_manifest/`，包含 canonical JSONL、CSV、排除清单、split 汇总、可用性统计、泄漏 finding/summary、元数据、验证 JSON 和 Markdown 报告。
- 59/59 个现有 clip 均进入清单，没有静默跳过；全部继承 P4-C0 的 `validation` split。清单按 sample ID 排序，sample ID 由 dataset、稳定 P4-B video ID、clip ID 和帧区间生成。
- 50 个 clip 技术可用，9 个因 `required_objects_unavailable` 被明确排除。对象缺失样本仍保留在主清单和独立排除清单。
- 视频、完整帧、深度和 camera observation 均覆盖 59/59；对象覆盖 50/59；二维点轨迹覆盖 44/59；有效 semantic3d 覆盖 45/59；sequence pose 和可信物理 camera identity 均为 0/59。后两者作为可选分支缺失记录，不伪造 ID，也不自动排除仍可用于静态审计的样本。
- 泄漏审计先把重叠 clip 按 source video 去重。当前 cross-split error 为 0，因为 6 个视频都在 validation；但 6 个 original/derivative identity 全部未知，分别保存为 `unresolved_original_derivative_identity` warning，不能据此声明数据来源独立。
- 清单没有读取 residual/score，不计算真实性性能；metadata 明确保存 training、fitting、threshold selection 和 classification performance 均为 false。
- protocol SHA-256 为 `004fb982597091e038bbb833370169626326f0ad5ecd0998fd61132ed1dffc12`；manifest SHA-256 为 `3bfa3cc030adc29d1bfd6374d36ee17ad9f4a6fedb48f7b882549bcba4ad57a3`。
- 连续两次完整运行后，输出目录全部文件的 SHA-256 清单完全一致；独立 validator 从 P4-C0/P4-B.5 输入重建得到相同 manifest SHA-256。

### 验证命令

```bash
.venv/bin/python scripts/build_p4c1_experiment_manifest.py \
  --config configs/p4c1_experiment_manifest_v1.yaml

.venv/bin/python scripts/validate_p4c1_experiment_manifest.py \
  --config configs/p4c1_experiment_manifest_v1.yaml \
  --manifest-root outputs/p4c1_experiment_manifest

.venv/bin/python -m pytest \
  tests/test_p4c1_experiment_manifest.py \
  tests/test_p4c1_leakage_audit.py -v

.venv/bin/python -m pytest tests/test_p4c_experiment_protocol.py -q
.venv/bin/python -m pytest -q
```

### 验证结果

- P4-C1 专项：`15 passed`。
- P4-C0 回归：`13 passed`。
- P4-C1 validator：`valid=true`，13 项检查全部通过，0 error。
- 两次完整构建的所有输出文件 SHA-256 完全一致。
- 全量：`468 passed`、`5 skipped`、`0 failed`，耗时约 `96.76s`。唯一 warning 为本机 NVIDIA 驱动版本与当前 PyTorch CUDA 构建不匹配；CPU 路径正常。

### 当前限制

- 6 个视频仍全部属于 validation-only protocol smoke，没有 train/test，不能训练、调参或报告性能。
- 真实原片与潜在伪造衍生版关系没有独立元数据；当前按文件哈希兜底的 6 个 source group 必须人工补充 provenance 后才能固化正式 split。
- 没有可信 camera identity；`coordinate_system_id` 没有被冒充为物理 camera ID。
- 当前 sequence pose 可用性为 0；轨迹和 semantic3d 也不是所有 clip 都可用。它们保持分支级缺失，不写成异常 0。
- 9 个无对象 clip 被保留为 excluded，不能在后续流水线中静默丢弃或当作低异常样本。

### 下一步计划

- 在进入 P4-C2 前，先人工补齐 dataset/source provenance、真实原片与派生伪造关系、camera/scene 元数据和正式 train/validation/test 数据来源。
- 当前阶段不进入 P4-C2，不训练模型，不拟合归一化或先验分布，不选择阈值。

## 2026-07-22 - P4-C2 Formal Data Onboarding And Build Readiness Audit

### 目的

在不下载正式数据、不执行模型推理、不提取大规模特征、不训练、不拟合统计分布、不选择阈值且不读取测试性能的前提下，为正式数据接入建立可复现的 dataset registry、license registry、source lineage、正式 split、逐任务资格、轻量 inventory、存储批次和 missingness 风险审计。当前六视频继续仅作为 `geometry_validation_smoke`，不因已有结构观测而获得正式训练或评价资格。

### 新增文件

- `configs/p4c2_formal_data_readiness_v1.yaml`：P4-C2 输入、六视频 smoke 角色、正式 split 规则、冻结存储快照、missingness 计划、兼容哈希和禁止操作。
- `src/semantic3d/experiment_protocol/formal_schema.py`：正式数据集、source lineage、formal split、任务资格和构建结果数据契约。
- `src/semantic3d/experiment_protocol/formal_registry.py`：配置驱动的 dataset/license registry 与轻量 local-root 检查。
- `src/semantic3d/experiment_protocol/source_lineage.py`：lineage 记录、官方 split 优先、verified origin group-aware fallback 与泄漏审计。
- `src/semantic3d/experiment_protocol/task_eligibility.py`：每视频十任务的 `eligible/ineligible/unknown/not_applicable/provider_failed` 矩阵。
- `src/semantic3d/experiment_protocol/storage_readiness.py`：基于冻结磁盘快照的分批空间、峰值、保留量、可删缓存、校验和失败恢复规划。
- `src/semantic3d/experiment_protocol/p4c2_builder.py`：只读 P4-C0/P4-C1 元数据并构建完整 P4-C2 readiness。
- `src/semantic3d/experiment_protocol/p4c2_report.py`：确定性 JSON/JSONL/CSV/YAML、哈希清单和 Markdown 报告。
- `src/semantic3d/experiment_protocol/p4c2_validation.py`：独立重建、hash、lineage、split、任务状态、stage boundary 和 strict prior 验证。
- `scripts/build_p4c2_formal_data_readiness.py`：连续两次构建、写出和验证入口。
- `scripts/validate_p4c2_formal_data_readiness.py`：已有 P4-C2 artifact 的独立验证入口。
- `tests/test_p4c2_formal_data_readiness.py`：当前 smoke 角色、lineage、资格矩阵、inventory、存储、missingness、哈希和兼容性测试。
- `tests/test_p4c2_source_lineage.py`：官方 split、group-aware fallback、未验证身份阻断、官方冲突和跨 split 泄漏测试。

### 修改文件

- `src/semantic3d/experiment_protocol/__init__.py`：向后兼容导出 P4-C2 数据类、构建、split、泄漏和验证 API。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件

- 无。

### 主要变化

- 输出目录为 `outputs/p4c2_formal_data_readiness/`，包含 `formal_dataset_registry.json`、`dataset_license_registry.json`、`source_lineage.jsonl`、`formal_split_plan.jsonl`、`task_eligibility_matrix.csv`、`data_inventory.csv`、`storage_batch_build_plan.json`、`missingness_audit_plan.yaml`、lineage 泄漏审计、readiness 报告、artifact hashes 和 validation report。
- registry 当前只有 `local_six_video_protocol_smoke`。其 source 是用户本地文件，许可和再分发权待确认；正式训练、模型选择、阈值选择和最终评价资格全部为 false。仓库没有声明可验证来源、许可和 official split 的外部正式数据集，因此没有虚构候选数据源。
- 六视频 source lineage 共 6 条，`original_source_id` 均缺失，身份依据仅为文件 SHA-256，不足以证明原片及派生关系；6 条全部为 `unverified`，formal split 全部留空并记录 `unverified_source_lineage`。正式 train/validation/test 均为 0；P4-C1 的 59 个 validation clip 不被提升为正式 validation 数据。
- 任务资格矩阵为 6 视频 x 10 任务，共 60 行：`eligible=14`、`ineligible=24`、`not_applicable=5`、`provider_failed=17`、`unknown=0`。eligible 只表示当前 smoke 声明范围可用，60 行的 `eligible_for_formal_experiment` 均为 false。D2/D3 缺失为 provider failure；没有观测到遮挡事件与 provider failure 分开记录。
- 轻量 inventory 覆盖 6/6 原视频并校验现有 SHA-256，不解码视频、不执行 provider 或模型推理。P4-C1 的对象、深度、位姿、轨迹和 semantic3d 计数只作为 availability metadata 使用，未读取 residual 或真实性性能。
- missingness 计划按 authenticity、dataset、original source、source group 和 provider 分层，状态保留 available、unknown、not_applicable、provider_failed；明确 provider failure 不是异常证据，缺失不得填 0。
- `/mnt/e` 使用 2026-07-22 冻结快照：可用 `55,817,076,736` bytes，安全余量 `16,106,127,360` bytes。既有 100,000 帧规划的总临时峰值加安全余量为 `319,072,879,645` bytes；10 个 10,000 帧批次中只有首批在不归档既有永久输出的条件下可容纳，正式全量构建被阻止。
- `ready_for_formal_batch_build=false`。阻塞包括：无 verified formal dataset、无 verified original lineage、无正式三 split、E 盘峰值空间不足、D2/D3、正式遮挡/重现以及时间/空间定位标注覆盖不足。
- P4-C2 readiness manifest SHA-256 为 `2094a721bc58c7cae1d567a41d9fd31542990a95f87f0927b87c31119280e981`；连续完整构建的所有 primary artifact SHA-256 完全一致。
- P4-C0 config、P4-C1 config 与 strict v1/v2 均未修改；strict v1/v2 哈希仍为 `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b` 和 `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。

### 验证命令

```bash
.venv/bin/python scripts/build_p4c2_formal_data_readiness.py \
  --config configs/p4c2_formal_data_readiness_v1.yaml

.venv/bin/python scripts/validate_p4c2_formal_data_readiness.py \
  --config configs/p4c2_formal_data_readiness_v1.yaml \
  --output-root outputs/p4c2_formal_data_readiness

.venv/bin/python -m pytest \
  tests/test_p4c2_formal_data_readiness.py \
  tests/test_p4c2_source_lineage.py -v

.venv/bin/python -m pytest \
  tests/test_p4c_experiment_protocol.py \
  tests/test_p4c1_experiment_manifest.py \
  tests/test_p4c1_leakage_audit.py -q

.venv/bin/python -m pytest -q
```

### 验证结果

- P4-C2 专项：`14 passed`。
- P4-C0/P4-C1 回归：`28 passed`。
- P4-C2 validator：18 项检查全部通过，`valid=true`，0 error。
- 最终全量：`482 passed`、`5 skipped`、`0 failed`，耗时约 `94.03s`。唯一 warning 为本机 NVIDIA 驱动与 PyTorch CUDA 构建不匹配；CPU 路径正常。

### 当前限制

- 当前没有经过来源、许可、citation、official split 和 checksum policy 独立确认的正式数据集条目。
- 六视频真实性标签仍只能服务 geometry smoke；缺少可信原片身份，不能证明真实原片与伪造派生视频之间不存在跨 split 泄漏。
- 正式训练、validation 和 test 都为空；本阶段不能启动后续构建、训练、调参、阈值选择或真假性能评价。
- sequence pose、D2、D3、正式 temporal/spatial/object localization 和 occlusion/reappearance 的正式任务覆盖尚未满足。
- 当前 `/mnt/e` 不足以容纳规划的完整临时峰值和安全余量；不能因为单个 10k-frame batch 暂时可放入就启动无法完成的全量构建。

### 下一步计划

- 由用户选定正式候选数据集后，先独立核验官方来源、许可、citation、下载方式、官方 split、标注类型、预计大小和 checksum，再新增版本化 registry 条目。
- 获取 official metadata 或可信 derivation manifest，补齐 original source 与所有派生视频 lineage；未验证记录继续禁止进入正式 split。
- 根据目标任务冻结 annotation/provider coverage 要求，并扩充 D2/D3、定位和遮挡事件数据，先审计 missingness shortcut 风险。
- 扩充或重新规划存储，使完整峰值加安全余量可容纳，再运行 P4-C2 validator。只有 `ready_for_formal_batch_build=true` 后，才可单独批准下一阶段正式分批构建。

## 2026-07-22 - P4-C3A Server Migration And Formal Batch Execution Preparation

### 目的

在尚未迁移服务器、尚无正式数据且禁止正式下载、批量 provider 推理、训练、分布拟合、阈值选择和测试性能读取的条件下，补强 P4-C2 资格语义，建立机器无关 runtime 配置、服务器环境预检、确定性批次规划、原子状态机、断点恢复与 conservative cleanup 框架，并生成可校验的服务器迁移清单和 smoke 验收协议。

### 新增文件

- `configs/runtime/local_wsl.yaml`：本地 WSL 相对路径、CPU、worker 和存储边界配置。
- `configs/runtime/server_template.yaml`：由环境变量解析的服务器路径、CUDA 和执行资源模板。
- `configs/p4c3a_server_smoke_v1.yaml`：服务器迁移后固定的 11 步 smoke 顺序和 P4-C0/C1/C2/strict 哈希。
- `src/semantic3d/runtime/`：runtime 路径解析、canonical path-independent hash、依赖/CUDA/存储/Git/冻结哈希预检、迁移清单和报告模块。
- `src/semantic3d/batch_execution/`：批次 schema、确定性 planner、原子 state store、独占 lock、artifact registry、checkpoint/retry/cleanup policy、validator 和 report。
- `scripts/preflight_p4_server_environment.py`：服务器环境非破坏性预检入口。
- `scripts/plan_p4_server_batches.py`：只规划、不执行的 source-group-preserving batch 入口。
- `scripts/validate_p4_server_batch_plan.py`：批次计划独立验证入口。
- `scripts/inspect_p4_batch_state.py`：只读检查批次状态和恢复建议。
- `scripts/build_p4_server_migration_manifest.py`：生成迁移文件、排除清单、环境重建和 bootstrap 报告。
- `tests/test_p4c3a_runtime_config.py`：环境变量、跨路径 hash、未解析变量、smoke 资格、provider failure guard 和正/负 readiness 测试。
- `tests/test_p4c3a_server_preflight.py`：存储不足和 CUDA 不可用阻断测试。
- `tests/test_p4c3a_batch_state.py`：确定性 batch ID、合法/非法状态迁移、失败历史、cleanup 和并发锁测试。
- `tests/test_p4c3a_migration_manifest.py`：`.venv`、缓存、secret 排除、迁移 hash 和冻结 hash 测试。

### 修改文件

- `src/semantic3d/experiment_protocol/evidence_eligibility_policy.py`：新增 provider failure 的代码级用途限制。
- `src/semantic3d/experiment_protocol/p4c2_builder.py`：公开可合成测试的正向 readiness 判定，保留旧私有别名。
- `src/semantic3d/experiment_protocol/p4c2_report.py`：补充 smoke/formal 显式字段、dataset 统计和视频级资格 scope。
- `src/semantic3d/experiment_protocol/p4c2_validation.py`：验证显式 smoke 禁用字段、任务 scope 和 readiness 统计。
- `src/semantic3d/experiment_protocol/__init__.py`：导出 P4-C2 补强 API。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件

- 无。

### 主要变化

- P4-C2 registry 现在显式输出 `is_formal_dataset=false`、`eligible_for_formal_split=false` 和四项正式用途 false；readiness metadata 分别输出 entries=1、formal=0、smoke=1。任务矩阵明确 `eligibility_scope=video`。
- `provider_failed` 只能进入 quality control 和 missingness audit。转换为 anomaly score、fake label、normal reference fit 或 supervised aggregation 会在代码层抛出异常。
- 新增完全合成的正向 readiness fixture；当 verified formal dataset、verified lineage、三 split、D2/D3、定位、遮挡和存储全部满足时，`ready_for_formal_batch_build=true`。当前真实 P4-C2 仍保持 false。
- runtime 路径支持 `PROJECT_ROOT`、`DATA_ROOT`、`DOWNLOAD_ROOT`、`TEMPORARY_ROOT`、`CACHE_ROOT`、`MODEL_ROOT`、`OUTPUT_ROOT`、`LOG_ROOT` 覆盖。canonical hash 将绝对根替换为稳定 token，并排除 hostname、username、mtime 和 run time；location metadata 单独保存。
- 当前本机预检：Python 依赖均可导入；`ffmpeg/ffprobe` 缺失；Torch 为 `2.12.1+cu130`、compiled CUDA `13.0`、device_count=1，但驱动不兼容导致 `cuda_available=false`，最小 CUDA tensor/matmul 未通过。SHA 和原子 rename 通过，目录可访问，冻结哈希匹配。
- 当前 readiness：`ready_for_server_migration=true`；因 ffmpeg/ffprobe 缺失，`ready_for_server_smoke=false`；当前不是服务器 profile，因此 `ready_for_formal_dataset_registration=false`；CUDA、正式数据、正式 split 和 P4-C2 coverage 均未解除，`ready_for_formal_batch_execution=false`。
- batch 状态支持 planned、downloading、downloaded、processing、validating、completed、failed、cleaned。completed 必须先有 validated artifacts；cleaned 只能从 completed 进入；failed 保留原因、产物和 attempt，retry 必须显式迁移；cleanup 默认 dry-run。
- 当前 formal split 为空，batch plan 为 0 个批次，SHA-256 为 `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`。空计划是正式数据仍被阻断的正确结果，没有启动执行。
- 迁移清单包含源码、脚本、配置、测试、文档、六视频、小型协议索引、P4-C1/C2 冻结产物、环境定义和三个小型 YOLO 权重；排除 `.venv`、`.git`、Python/pytest cache、大型可重建结构数据、临时状态和敏感文件。服务器必须重建 Python 环境。
- P4-C0 protocol SHA 保持 `004fb982597091e038bbb833370169626326f0ad5ecd0998fd61132ed1dffc12`；P4-C1 manifest SHA 保持 `3bfa3cc030adc29d1bfd6374d36ee17ad9f4a6fedb48f7b882549bcba4ad57a3`；P4-C2 semantic manifest SHA 保持 `2094a721bc58c7cae1d567a41d9fd31542990a95f87f0927b87c31119280e981`；strict v1/v2 哈希不变。

### 验证命令

```bash
.venv/bin/python scripts/build_p4c2_formal_data_readiness.py

.venv/bin/python scripts/preflight_p4_server_environment.py \
  --runtime-config configs/runtime/local_wsl.yaml

.venv/bin/python scripts/plan_p4_server_batches.py \
  --runtime-config configs/runtime/local_wsl.yaml

.venv/bin/python scripts/validate_p4_server_batch_plan.py

.venv/bin/python scripts/build_p4_server_migration_manifest.py \
  --runtime-config configs/runtime/local_wsl.yaml

.venv/bin/python -m pytest \
  tests/test_p4c3a_runtime_config.py \
  tests/test_p4c3a_server_preflight.py \
  tests/test_p4c3a_batch_state.py \
  tests/test_p4c3a_migration_manifest.py -v

.venv/bin/python -m pytest -q -rs
```

### 验证结果

- P4-C3A 专项：`21 passed`。
- P4-C0/C1/C2/C3A 联合专项：`63 passed`。
- 全量：`503 passed`、`5 skipped`、`0 failed`，耗时约 `99.03s`。
- skip 原因：1 项缺本地 pose weight 或 real_3 observation；1 项缺 P3 real shared cache；2 项缺已有 P3 shared-geometry smoke artifacts；1 项因六视频 strict-v2 输出不是提交 fixture。均为显式外部资源或派生产物缺失，不是隐藏失败。
- 唯一 warning 为本机 NVIDIA driver 与当前 PyTorch CUDA build 不兼容；预检已将其转化为正式 GPU batch blocker，没有伪造 GPU 可用。

### 当前限制

- 当前机器缺少 ffmpeg/ffprobe，不能通过完整 server smoke；CUDA 驱动与 PyTorch build 不兼容，不能执行正式 GPU batch。
- 服务器尚未迁移和实测；server template 中的环境变量必须在目标机器解析。
- P4-C2 仍没有 verified formal dataset、verified original-source lineage 或 formal train/validation/test。
- D2、D3、时间/空间定位和遮挡/重现的正式数据覆盖不足。
- 当前 Git worktree 非 clean；迁移清单可校验所有实际文件，但提交到远端前仍应创建新的 Git commit/tag。
- 当前 batch plan 为空；不得把框架存在误述为正式批处理已经开始。

### 下一步计划

- 在服务器准备兼容的 NVIDIA driver/CUDA/PyTorch 组合并安装 ffmpeg/ffprobe，解析 server runtime 环境变量后重新运行 preflight。
- 使用迁移文件 SHA-256 校验代码、六视频、P4-C0/C1/C2 和必要权重；服务器重新创建 `.venv` 并运行全量测试和六视频 geometry smoke。
- 只有服务器 smoke、冻结哈希和 Git commit/tag 全部确认后，才允许登记正式数据集。
- 正式数据 registry、lineage、license、split 和 storage readiness 全部通过后，再单独批准 P4-C3B 或正式 batch execution。

## 2026-07-22 - P4-C3A-G Git Release And Server Checkout Preparation

### 目的

将 P4-C3A 的部署方式从物理迁移包复制调整为远程 Git 发布后由服务器
`git clone`/`git pull` 获取源码，并在不执行 Git 提交、推送、正式数据下载、
模型训练或正式批处理的前提下建立发布审计、服务器环境重建和资产校验机制。

### 新增文件

- `src/semantic3d/git_release/`：Git 内容分类、敏感与绝对路径审计、模型注册表校验和发布 readiness。
- `configs/model_registry/yolo_weights_v1.yaml`：三个 YOLO 权重的来源、用途、版本、大小、SHA-256 和再分发门控。
- `configs/data_registry/`：正式数据集注册 schema 和六视频 geometry smoke 的非正式、不可再分发登记。
- `requirements-lock.txt`：不绑定 PyTorch/CUDA 的基础与测试依赖版本。
- `requirements-inference-lock.txt`：当前验证过的顶层推理依赖版本；PyTorch 由服务器按驱动单独安装。
- `scripts/bootstrap_p4_server.sh`：可重复运行的服务器环境重建、preflight、测试和 checkout 验收入口。
- `scripts/verify_p4_git_checkout.py`：服务器 clone 后的 Git、冻结哈希、权重、视频和 smoke readiness 校验。
- `scripts/fetch_registered_models.py`：默认 dry-run 的权重获取与 SHA-256 校验工具。
- `scripts/validate_git_release.py`：确定性生成 Git 发布清单和 readiness 报告。
- `docs/SERVER_ENVIRONMENT.md`：服务器环境变量、系统依赖和 PyTorch/CUDA 安装边界。
- `docs/SERVER_DATA_SETUP.md`：正式数据注册条件及六视频人工放置和校验规则。
- `tests/test_p4c3a_git_release.py`：发布、路径、敏感信息、权重许可、dry-run 和冻结哈希专项测试。

### 修改文件

- `.gitignore`：排除环境、缓存、敏感信息、媒体、权重和大输出；只对白名单中的 P4-C1/C2 及空 P4-C3A 批次计划开放小型冻结产物。
- `pyproject.toml`：Python 范围明确为 3.11-3.13，并记录已验证的 test/YOLO/depth 顶层版本。
- `configs/runtime/server_template.yaml`：临时目录由 `${CACHE_ROOT}/tmp` 派生，不再要求额外本机路径。
- `src/semantic3d/experiment_protocol/storage_planning.py`：默认存储路径改为环境变量/相对路径，冻结 v1/v2 与 P4-C0/C1/C2 语义内容未修改。
- `README.md`：移除本机绝对路径与 Conda base 用法，改为项目本地 `.venv` 和服务器文档入口。

### 删除文件

- 无。

### 主要变化

- 当前远程为 `origin`，分支为 `main`，基准 commit 为 `46d0b8cced5036052a2f55a0f8d421616fe49e3b`；本轮未执行 `git add`、commit、tag 或 push。
- 三个 YOLO 权重本地大小和 SHA-256 均匹配注册表，但再分发许可未独立确认，因此不提交普通 Git、不启用 Git LFS，推荐服务器按官方来源获取或人工复制后校验。
- 六视频许可和 original source identity 未确认，保持 `geometry_validation_smoke`，六项正式资格均为 false，视频本体不上传 Git。
- P4-C1/C2 小型冻结产物和空 P4-C3A batch plan 通过 `.gitignore` 白名单保留，preflight 和大规模实验输出继续忽略。
- location-specific 历史快照与文档被单独标记；核心源码和 server runtime template 没有未解析本机绝对路径。
- canonical semantic hash 继续排除绝对路径、hostname、username 和时间字段；不同 checkout 路径测试一致。

### 验证命令

```bash
.venv/bin/python -m pytest tests/test_p4c3a_git_release.py -q -rs
.venv/bin/python -m pytest tests/test_p4c_experiment_protocol.py tests/test_p4c1_experiment_manifest.py tests/test_p4c1_leakage_audit.py tests/test_p4c2_formal_data_readiness.py tests/test_p4c2_source_lineage.py tests/test_p4c3a_runtime_config.py tests/test_p4c3a_server_preflight.py tests/test_p4c3a_batch_state.py tests/test_p4c3a_migration_manifest.py tests/test_p4c3a_git_release.py -q -rs
.venv/bin/python -m pytest -q -rs
.venv/bin/python scripts/fetch_registered_models.py
.venv/bin/python scripts/validate_git_release.py
bash -n scripts/bootstrap_p4_server.sh
```

### 验证结果

- P4-C3A-G 专项：11 passed。
- P4-C0/C1/C2/C3A 联合回归：74 passed，无 skip。
- 全量测试：514 passed、5 skipped、1 warning；5 个 skip 均为可选本地权重/缓存/旧 smoke 派生输出缺失，P4-C0/C1/C2/C3A 关键测试无 skip。
- 敏感提交候选为 0，核心/正式 runtime 绝对路径 blocker 为 0，普通 Git 大文件提交候选为 0。
- 权重 dry-run 未执行下载；bootstrap shell 语法通过；本地 checkout 逻辑校验通过，但本机 preflight 因 CUDA/系统依赖状态未达到 server smoke。
- `ready_for_git_commit=true`；工作区尚未提交，因此 `ready_for_git_push=false`、`ready_for_server_clone=false`；服务器尚未实际执行，因此 `ready_for_server_smoke=false`；`ready_for_formal_batch_execution=false`。

### 冻结哈希

- P4-C0 semantic protocol：`004fb982597091e038bbb833370169626326f0ad5ecd0998fd61132ed1dffc12`。
- P4-C1 manifest：`3bfa3cc030adc29d1bfd6374d36ee17ad9f4a6fedb48f7b882549bcba4ad57a3`。
- P4-C2 semantic manifest：`2094a721bc58c7cae1d567a41d9fd31542990a95f87f0927b87c31119280e981`。
- strict v1：`e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict v2：`3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。
- P4-C3A server smoke config：`4742f9dc910e154bab03196568c4d33e8a6596a81874b87adc1d4553451b688d`。

### 当前限制

- 工作区包含此前 P4 阶段的大量未跟踪/修改文件，用户需要人工复核后一次性提交；提交前不能声称服务器 clone 可复现当前工作树。
- 当前本机 Torch 2.12.1 CUDA 13.0 与驱动不兼容，该版本未锁定到服务器；服务器必须选择与实际驱动兼容的 PyTorch build。
- `ffmpeg`/`ffprobe` 为服务器系统依赖，不能由 Python lock 文件替代。
- 权重与六视频均不随 Git checkout 获取，必须在服务器按注册表人工准备并校验。
- 当前没有正式数据集，P4-C2 仍阻止正式批次构建。

### 下一步计划

- 用户人工检查 `outputs/p4c3a_git_release/` 后执行 Git add/commit/push；本项目不会自动操作 Git 历史。
- 服务器 clone 后设置 runtime 环境变量，准备已登记权重与 smoke 视频，再运行 `scripts/bootstrap_p4_server.sh`。
- 只有 server preflight、checkout、全量测试、冻结哈希和 smoke 资产校验全部通过后，才讨论后续阶段；本轮不启动 P4-C3B。

## 2026-07-22 - P4-C3A-MD2 Metric Primary Scale Geometry Refactor

### 目的

将对象语义尺度几何从“同帧对象对 R_sd 为核心”重构为固定的三层优先级：
单对象米制绝对尺度为主分支、同对象跨帧尺度稳定性为时序支持、冻结
strict-v2 对象对尺度—深度残差为 fallback/audit。该阶段只完成代码、合成
几何验证和六视频只读资格审计，不运行大型米制深度模型，也不评价真假性能。

### 新增文件

- `configs/p4c3a_scale_geometry_priority_v1.yaml`：冻结三分支优先级、质量门控、默认 `fallback_only` 和默认关闭的米制 provider 配置。
- `src/semantic3d/method_completion/scale_evidence.py`：统一 `ScaleGeometryEvidence`、证据角色、provider 状态和路由结果契约。
- `src/semantic3d/method_completion/multi_interval_prior.py`：按维度保存多个不连续米制区间，并计算到区间并集的最近对数距离。
- `src/semantic3d/method_completion/metric_scale.py`：米制深度门控、投影尺度恢复、ray-distance 到 z-depth 转换、Mask 点云稳健范围和单对象物理尺度残差。
- `src/semantic3d/method_completion/temporal_scale.py`：metric/relative-local 同对象跨帧尺度历史、参考值和隔离门控。
- `src/semantic3d/method_completion/scale_router.py`：固定三分支优先级、对象对策略和 strict-v2 行适配器。
- `src/semantic3d/method_completion/metric_depth_adapters.py`：UniDepthV2、Depth Pro、Metric3Dv2 和 Depth Anything V2 metric 的懒导入、无自动下载接口骨架。
- `src/semantic3d/method_completion/md2_audit.py`：合成验证、六视频只读 routing/funnel 和冻结回归审计。
- `scripts/run_p4c3a_metric_primary_refactor.py`：生成 MD2 的16项审计和报告产物。
- `tests/test_p4c3a_metric_single_object_primary.py`：针孔尺寸、resize/crop、ray depth、门控和维度独立可观测性测试。
- `tests/test_p4c3a_temporal_same_object_scale.py`：时序恒定、突变、稳健参考、ID/provider/depth定义隔离测试。
- `tests/test_p4c3a_scale_evidence_router.py`：主分支选择、惰性 pair fallback、静态 clip 和关系边定位测试。
- `tests/test_p4c3a_multi_interval_size_prior.py`：非连续区间、alias 和最近区间对数距离测试。
- `tests/test_p4c3a_robust_object_extent.py`：Mask 点云分位数范围与离群深度稳健性测试。
- `tests/test_p4c3a_pair_fallback_policy.py`：disabled/audit/fallback 角色和 strict-v1/v2 冻结哈希测试。

### 修改文件

- `src/semantic3d/method_completion/__init__.py`：公开 MD2 契约、分支、路由、provider adapter 和兼容别名。
- `src/semantic3d/method_completion/localization.py`：对象对残差映射到两个 Mask 与关系边，并提供无阈值对象证据排名。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件

- 无。

### 主要变化

- 正式优先级固定为 `metric_single_object_scale`、`temporal_same_object_scale`、`relative_pair_scale_depth`；对象对默认策略为 `fallback_only`。
- 投影模式使用 `H_hat=h_px*Z/fy` 和 `W_hat=w_px*Z/fx`；Mask 点云模式将有效米制 z-depth 反投影到 OpenCV 相机坐标，并使用 `[0.05, 0.95]` 分位数范围，未用裸 min/max 作为唯一估计。
- 米制主分支只接受 metric/sensor_metric、meter、已知 depth definition、有效 K 和物理先验。相对深度、未知定义、近似内参（默认）、provider failure、严重截断及低质量区域均输出 `NaN + failure_reason`。
- 高度、宽度和深度范围分别判断可观测性；一个维度失败不会自动使其他维度失效。
- 多区间残差为 `min_j distance(log(S_hat), [log(S_min_j), log(S_max_j)])`，落入任一区间为0，且不把明显多峰子类强行扩成一个连续区间。
- 时序分支支持 previous-valid、rolling median、track median 和 robust window；禁止跨视频、跨 clip、跨 track、跨 provider、跨 depth definition 或跨内参历史混用。relative-local 只输出 `relative_local_unit`。
- strict-v2 只读取既有行并适配统一 evidence，不重算或改变旧公式；audit evidence 默认不能进入主聚合。
- 静态/低运动 clip 仍可使用静态尺度分支；动态 D1/D2/D3 不适用不等于 provider failure。
- 四个米制 provider 当前均为 `interface_only/not_executed` 骨架，不自动安装依赖、不下载权重、不声称传感器真值。

### 六视频只读审计

- 6 个视频、59 个 clip、2455 条对象观测。
- `metric_single_object_scale` 实际执行数为0：现有深度全部是单目 `relative_per_frame`，不是米制深度。
- `temporal_same_object_scale` 实际执行数为0：虽有可复用 track，但没有物化按维度且局部尺度已对齐的尺度历史，未伪造执行结果。
- 历史 strict-v2 共299条记录，其中278个实际对象对、20个有效 pair；其余包含21条 `insufficient_objects` 帧级记录。strict-v2 文件只读且前后 SHA-256 一致。
- 当前审计证明代码路径、门控和 fallback 语义，不证明真假检测有效性。

### 验证命令

```bash
.venv/bin/python scripts/run_p4c3a_metric_primary_refactor.py

.venv/bin/python -m pytest \
  tests/test_p4c3a_metric_single_object_primary.py \
  tests/test_p4c3a_temporal_same_object_scale.py \
  tests/test_p4c3a_scale_evidence_router.py \
  tests/test_p4c3a_multi_interval_size_prior.py \
  tests/test_p4c3a_robust_object_extent.py \
  tests/test_p4c3a_pair_fallback_policy.py -v

.venv/bin/python -m pytest -q -rs
```

### 验证结果

- MD2 六个专项文件：35 passed。
- MD2 与原 P4-C3A-M 联合专项：49 passed。
- 全量回归：567 passed、5 skipped、0 failed，耗时约115.60秒。
- 5个 skip 均为既有外部资源条件：pose weight/real_3 observation 缺失、P3 shared cache 缺失、2项旧 shared-geometry smoke 产物缺失、strict-v2 六视频输出不是提交 fixture。
- 唯一 warning 为当前 NVIDIA driver 与已装 PyTorch CUDA build 不兼容；本轮未运行 CUDA 或大型模型。
- strict-v1/v2、P4-C0/C1/C2、历史 strict-v2 CSV、D1 evidence table 和 P4-C3A-M D2 合成产物均未写入；冻结复核通过。

### 冻结哈希

- strict-v1：`e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict-v2：`3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。
- P4-C0 config：`8a4a8f5d6ac795646876042a84c9b0a4fdb1d06bec31045b734c3dfb64f8a304`。
- P4-C1 config：`ec48e26da4f434a1356959997b546ac30dc9e439281b2e09174f7c86a35ce086`；manifest：`3bfa3cc030adc29d1bfd6374d36ee17ad9f4a6fedb48f7b882549bcba4ad57a3`。
- P4-C2 config：`fe8c3cda137337330209528f4025d0593fefa42886be89eb214bfd58d38a8d89`；readiness manifest：`2094a721bc58c7cae1d567a41d9fd31542990a95f87f0927b87c31119280e981`。
- strict-v2 历史 pair CSV：`057885df2d5a3adfa3d7bbc76d564091ecf1e3d9bada721e2e92e3fd55733a44`。
- 既有 D1 point evidence table：`693f9c61ae3b16680373e3eef8f8e58af259544c9e61c2ce5e4dc072ea79797b`。
- 既有 P4-C3A-M D2 合成验证：`ad3976d2ae31815d9bbd3da54ce9f0f39ccdcd8c9f5653260084fe89a417bc37`。

### 当前限制

- 当前四个米制深度 adapter 只定义接口和运行前检查，尚未实现/执行具体大型模型推理；`ready_for_server_metric_smoke` 表示几何、路由和审计已就绪，不表示 provider 已跑通。
- 当前 K 主要来自近似焦距；默认 metric gate 不接受 approximate intrinsics。服务器 smoke 仍需可信内参或单独批准其不确定性路径。
- 现有六视频没有米制深度，因此不能生成真实 `R_metric_abs`；relative-local 时序尺寸历史也尚未物化。
- strict-v2 20个有效 pair 仅用于冻结 fallback 回归，不代表新主分支有效性或真假性能。
- 未训练、未拟合、未选阈值、未计算 AUC/F1/accuracy，`method_effectiveness_established=false`。

### 下一步计划

- 在服务器选择一个明确输出单位和 depth definition 的米制 provider，准备本地依赖和权重后单独实现并运行小规模 metric smoke。
- 为 smoke 帧提供可信 K 或记录经过审核的内参来源，先验证米制尺寸闭环和 provider 间尺度偏差，不直接评价真假。
- 将通过门控的按维度尺度观测物化为 track history，再验证 metric 与 relative-local 的时序路径。
- metric smoke 通过且冻结回归保持不变后，再决定是否进入正式批量特征构建；本阶段不启动训练或性能评估。

## 2026-07-24 - P4-C3B-M1 Real Metric Depth Provider and Camera Smoke

### 目的

将 P4-C3A-MD2 中 `interface_only` 的 UniDepthV2 米制深度接口升级为真实、
离线、可审计的 provider，并在一个真实视频和一个 AI 生成视频的少量帧上验证
模型预测米制深度、深度定义、模型预测内参及上游质量统计。该 smoke 不使用真假
标签计算性能，也不代表传感器深度真值。

### 新增文件

- `configs/model_registry/unidepth_v2_vits14_v1.yaml`：登记官方源码/模型 revision、CC-BY-NC-4.0 许可状态、本地权重路径、大小和 SHA-256。
- `configs/p4c3b_metric_provider_smoke_v1.yaml`：固定 CPU/FP32、resolution level 0、`real_1` 第30-32帧和 `fake_1` 第0-2帧的无性能评价 smoke 范围。
- `src/semantic3d/method_completion/metric_provider_smoke.py`：确定性抽帧、真实推理、DepthObservation/CameraObservation 落盘、正式 Mask 区域、相对深度秩一致性、边界质量、时序尺度漂移和内参漂移审计。
- `scripts/run_p4c3b_metric_provider_smoke.py`：P4-C3B-M1 命令行入口。
- `tests/test_p4c3b_metric_provider.py`：离线加载、权重 SHA、深度定义、模型预测 K、失败路径、重复推理和 smoke 产物测试。

### 修改文件

- `src/semantic3d/method_completion/metric_depth_adapters.py`：实现真实 `UniDepthV2Adapter`，增加统一帧结果、lazy import、强制离线、本地权重 SHA 门控、CPU/CUDA 与 FP32/FP16 检查、运行时间/显存、confidence/uncertainty 语义和 provider provenance；其余三个 adapter 保持 `interface_only`。
- `src/semantic3d/method_completion/__init__.py`：公开新的 adapter 结果、错误和 SHA 工具。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件

- 无。

### 主要变化

- UniDepthV2 源码固定在 revision `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`，ViT-S14 模型固定在 revision `038c238f06c87b6c2f5b3749fd51fbf442b1f218`。
- 本地 `model.safetensors` 大小为136777580字节，SHA-256 为 `93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648`；adapter 在缺少 expected SHA 或摘要不匹配时阻断推理，不自动联网。
- 模型内部原始射线半径以 `ray_distance` 保存；公开推理输出的三维点 Z 分量作为标准 `z_depth` 进入 `DepthObservation`。单位声明为米，尺度状态为 `model_predicted`，明确不是传感器真值。
- UniDepth 名为 confidence 的输出按官方定义作为帧内相对 scale-invariant log error 保存为 uncertainty；另生成 `[0,1]` 反向秩质量，仅用于门控，不是校准概率。
- 每帧保存米制 Z depth、valid mask、rank quality、raw uncertainty、raw radius 和模型预测 K。K 的来源为 `model_predicted`，不得解释为 calibrated/metadata intrinsics；provider 未输出内参置信度，因此明确保存为 NaN/`model_predicted_unscored`，位姿仍不可用。
- 正式实例 Mask 直接复用 P4-B.5 只读产物。6帧全部生成有效深度；`fake_1` 8个对象区域、`real_1` 3个对象区域均完成 Mask 内深度审计，Mask 内有效比例均为1.0。
- 6帧深度均无 NaN/Inf，边界有效比例为1.0；米制深度与既有单目相对深度的帧内秩相关为约0.817-0.978，仅作为上游一致性诊断。
- 连续三帧全局深度中位数最大 absolute-log drift：`fake_1=0.02379`、`real_1=0.01650`（最终报告以 JSON 为准）；模型预测内参按帧审计，不视为静态标定值。
- CPU/FP32 实际推理约0.93-1.04秒/帧；同帧重复推理最大绝对差为0。GPU未执行，`peak_gpu_memory=0`。
- `metric_provider_adapter_complete=true`、`metric_provider_real_inference_executed=true`、`metric_depth_output_verified=true`、`ready_for_metric_scene3d_build=true`、`ready_for_full_984_frame_build=false`、`method_effectiveness_established=false`。

### 验证命令

```bash
.venv/bin/python -m pytest tests/test_p4c3b_metric_provider.py -q -rs
.venv/bin/python scripts/run_p4c3b_metric_provider_smoke.py
.venv/bin/python -m pytest \
  tests/test_p4c3a_metric_single_object_primary.py \
  tests/test_p4c3a_robust_object_extent.py \
  tests/test_p4c3a_scale_evidence_router.py \
  tests/test_p4c3b_metric_provider.py -q -rs
.venv/bin/python -m pytest -q -rs
```

### 验证结果

- P4-C3B-M1 专项：9 passed、0 skipped、1 warning。
- MD2 + M1 联合专项：27 passed、0 skipped、1 warning。
- 全量回归：576 passed、5 skipped、1 warning，最终耗时131.38秒。
- 5个 skip 均为既有可选本地资源：pose weight/real_3 observation、P3 shared cache、2项旧 shared-geometry smoke 产物和未提交的 strict-v2 六视频 fixture；P4-C3B-M1 测试无 skip。
- warning 为已安装 PyTorch CUDA build 与当前 NVIDIA driver 不兼容；本次真实推理按配置使用 CPU/FP32 成功完成。
- smoke 请求6帧、有效6帧、provider failure为0；全部必要报告已写入 `outputs/p4c3b_metric_provider_smoke/`。

### 冻结检查

- strict-v1：`e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`，未改变。
- strict-v2：`3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`，未改变。
- P4-C0 config：`8a4a8f5d6ac795646876042a84c9b0a4fdb1d06bec31045b734c3dfb64f8a304`，未改变。
- P4-C1 config：`ec48e26da4f434a1356959997b546ac30dc9e439281b2e09174f7c86a35ce086`，未改变。
- P4-C2 config：`fe8c3cda137337330209528f4025d0593fefa42886be89eb214bfd58d38a8d89`，未改变。
- P4-C3A-M D2 合成验证：`ad3976d2ae31815d9bbd3da54ce9f0f39ccdcd8c9f5653260084fe89a417bc37`，未改变。
- 本轮没有写入 strict-v2 历史结果、D1 evidence 或 P4-C3A-MD2 六视频只读结果目录。

### 当前限制

- 米制值是单目模型预测，未与传感器真值或标定场景比较，绝对尺度误差尚未知。
- K 为逐帧模型预测，短片段中仍有漂移；不能当作 calibrated 相机内参，也没有相机位姿。
- confidence 是帧内相对质量排序，不可跨视频解释为统一概率。
- xFormers、CUDA KNN 和可选 CUDA patch op 未安装；本次 V2 推理可在 CPU 正常运行，但全984帧成本和服务器 GPU兼容性尚未审计。
- Depth Pro、Metric3Dv2、Depth Anything V2 metric 继续为 `interface_only`。
- 只验证2段视频共6帧的上游可用性，没有训练、分布拟合、阈值选择或真假性能评价。

### 下一步计划

- 在进入全984帧前，先用当前 adapter 构建少量 `MetricObjectRegion`，真实执行单对象米制尺度分支并审计对象截断、Mask、K和深度不确定性门控。
- 在服务器修复 PyTorch/CUDA 驱动兼容后重复同一 smoke，比较 CPU/GPU 数值一致性、显存和吞吐。
- 若短片段 K 与绝对尺度稳定性满足门控，再制定分批全帧 scene3d 构建方案；仍不直接进入真假性能评价。
## 2026-07-24 - P4-C3B-M2 Single-Frame Metric Visible-Surface Scene3D

### 目的

在 P4-C3B-M1 已保存的 UniDepthV2 模型估计米制深度、预测相机内参和
P4-B.5 正式可见实例 mask 基础上，建立可审计的单帧米制三维结构。
本阶段只构建“单帧相机坐标系下的米制可见表面 2.5D 结构”，不执行
跨帧世界坐标融合，不把模型估计米制深度描述为传感器真值，也不进行
训练、阈值选择或真假性能评价。

### 新增文件

- `src/semantic3d/metric_scene3d/contracts.py`：定义点类型、米制表面点、
  三维边界点、对象点云、单帧结构边和结构图契约。
- `src/semantic3d/metric_scene3d/image_geometry.py`：显式处理 resize、crop、
  padding/letterbox、内参变换和 mask/depth 异分辨率对齐。
- `src/semantic3d/metric_scene3d/reconstruction.py`：实现 z-depth/ray-distance
  统一、批量反投影、场景可见表面点、对象点云、稳健 extent 和共享
  `Object3DObservation` 适配。
- `src/semantic3d/metric_scene3d/boundary.py`：轮廓简化、均匀弧长采样、
  前景/背景侧深度、深度跳变和三维边界重建。
- `src/semantic3d/metric_scene3d/structure_points.py`：正式 mask 内部的纹理、
  深度平滑和空间分散几何跟踪点候选选择。
- `src/semantic3d/metric_scene3d/structure_graph.py`：单帧边界相邻、kNN 和
  可选半径边结构图。
- `src/semantic3d/metric_scene3d/__init__.py`：导出 M2 公共接口。
- `scripts/run_p4c3b_metric_scene3d_smoke.py`：只读消费 M1/P4-B.5 产物并生成
  M2 smoke 审计结果。
- `configs/p4c3b_metric_scene3d_v1.yaml`：固定 6 个 M1 smoke 帧及重建参数。
- `tests/test_p4c3b_metric_scene3d.py`：覆盖投影闭环、图像变换、边界深度、
  稳健 extent、点语义、NaN 和共享契约。

### 修改文件

- `src/semantic3d/shared_3d_observation.py`：新增明确的 `CoordinateFrame`，
  同时兼容旧 `camera/world` 字符串。
- `src/semantic3d/__init__.py`：通过现有 lazy API 导出 `CoordinateFrame`。
- `src/semantic3d/geometry/backprojection.py`：允许调用方显式指定坐标系，
  默认值仍保持旧接口兼容。
- `src/semantic3d/geometry/projection.py`：接受旧 camera frame 和新的
  `camera_frame_metric/camera_frame_relative`。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件

- 无。

### 主要变化

- M2 真实输出仅使用 `camera_frame_metric`；审计中 world-frame 点为 0。
- 像素反投影使用 `X=(u-cx)Z/fx`、`Y=(v-cy)Z/fy`、`Z=z_depth`。
- 场景点仅表示单帧可见表面，不声称是完整三维场景。
- 对象 extent 使用每轴 5%--95% 稳健分位数，不使用原始 min/max。
- 边界点保存前景深度、背景深度和边界深度跳变；背景缺失降低质量但不
  伪造数值。
- 通用内部点明确标记为 `geometric_track_point`，不是 semantic keypoint；
  跨帧 trackability 当前为未验证。
- 单帧结构图保存米制边长、相对深度、方向、边类型和质量；未计算 D3。
- 缺失坐标保持 NaN/invalid，不使用零坐标；provider/pose/模态缺失不进入
  异常证据。

### 真实 Smoke 结果

- 输入：`fake_1` 帧 0--2、`real_1` 帧 30--32，共 6 帧。
- 场景可见表面：6/6 帧有效，每帧下采样 3600 点。
- 对象点云：11/11 有效。
- 三维边界点：352/352 有效。
- 几何跟踪点候选：220 个有效。
- 单帧结构图：11 个有效，共 1158 条有效边。
- 具有至少两个有效帧的对象轨迹：4 条，因此可进入后续时序尺度
  materialization；本阶段未执行时序残差。
- semantic keypoint：未执行类别专用 provider，状态为不可用。
- world-frame reconstruction：false。
- method effectiveness established：false。

### 验证命令

```bash
.venv/bin/python -m pytest tests/test_p4c3b_metric_scene3d.py -q
.venv/bin/python -m pytest \
  tests/test_backprojection.py \
  tests/test_p0_shared_3d_contracts.py \
  tests/test_object_3d_reconstruction.py \
  tests/test_p4c3a_metric_single_object_primary.py -q
.venv/bin/python scripts/run_p4c3b_metric_scene3d_smoke.py
.venv/bin/python -m pytest -q -rs
sha256sum \
  configs/scale_priors_strict_v1.yaml \
  configs/scale_priors_strict_v2.yaml \
  configs/p4c0_experiment_protocol_v1.yaml \
  configs/p4c1_experiment_manifest_v1.yaml \
  configs/p4c2_formal_data_readiness_v1.yaml \
  outputs/p4c3a_method_completion/d2_synthetic_validation.json
```

### 验证结果

- M2 专项：`13 passed`。
- 旧共享三维/几何/MD2 兼容回归：`41 passed`。
- 全量回归：`589 passed, 5 skipped, 1 warning`。
- 5 个 skip 均为既有可选本地资源/旧派生产物条件：pose weight 或 real_3
  observation、P3 shared cache、两项旧 shared-geometry smoke 产物，以及
  未提交的 strict-v2 六视频 fixture。M2 专项无 skip。
- warning 为本机 NVIDIA driver 与当前 PyTorch CUDA build 不匹配；CPU
  路径可用，M2 smoke 不依赖新模型推理。
- strict-v1：`e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`。
- strict-v2：`3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。
- P4-C0 config：`8a4a8f5d6ac795646876042a84c9b0a4fdb1d06bec31045b734c3dfb64f8a304`。
- P4-C1 config：`ec48e26da4f434a1356959997b546ac30dc9e439281b2e09174f7c86a35ce086`。
- P4-C2 config：`fe8c3cda137337330209528f4025d0593fefa42886be89eb214bfd58d38a8d89`。
- P4-C3A-M D2 synthetic：`ad3976d2ae31815d9bbd3da54ce9f0f39ccdcd8c9f5653260084fe89a417bc37`。

### 当前限制

- UniDepthV2 输出为模型估计米制深度，不是传感器真值。
- 相机内参为 `model_predicted` 且暂无可校准置信度，不能等同 calibrated。
- 实例 mask 只表示可见区域，不提供 amodal 轮廓。
- 内部几何点仅在单帧通过纹理和深度质量筛选，尚未验证跨帧可跟踪性。
- 当前没有相机位姿、世界坐标融合、语义关键点或 D3 时间残差。
- smoke 只有 6 帧，不能建立方法有效性或真假性能结论。

### 下一步计划

- 进入 view observability 和 temporal size materialization 前，先沿现有
  track 验证 geometric track points 的跨帧稳定性与 M1 预测内参漂移。
- 后续单独引入可审计相机位姿后，才能把 camera-frame 点变换到 clip-local
  或 world frame；在此之前不得执行或声称世界坐标融合。

## 2026-07-24 - P4-C3B-M2 Git And Server Conversation Handoff

### 目的

在通过 Git 将项目迁移到无本对话历史的服务器前，把项目语义、冻结状态、
非 Git 依赖、验证命令和下一阶段授权边界固化到仓库，避免新会话只看到源码
文件名便误判实现状态、静默下载依赖、把缺失产物当成算法失败或越过用户授权
直接进入训练和性能评价。

### 新增文件

- `AGENTS.md`：新编码代理自动读取的项目级长期约束和阶段事实。
- `configs/handoff/p4c3b_m2_server_handoff_v1.yaml`：机器可读的阶段状态、
  必需源码、冻结哈希、模型/视频注册表、非 Git 产物和继续阶段策略。
- `docs/SERVER_HANDOFF_P4C3B_M2.md`：Git 与非 Git 文件边界、服务器环境、
  UniDepth 固定 revision、数据/权重迁移、验收顺序和首条服务器提示词。
- `docs/GIT_UPLOAD_CHECKLIST.md`：上传范围、检查命令和建议 commit summary。
- `scripts/verify_p4c3b_server_handoff.py`：只读验证源码、冻结哈希、模型、
  六视频和 M1/P4-B.5/M2 本地产物，分别输出 reproduction readiness。
- `tests/test_p4c3b_server_handoff.py`：验证 handoff 完整性、冻结哈希、
  非 Git 产物规则和六视频 smoke 资格。

### 修改文件

- `docs/SERVER_ENVIRONMENT.md`：补充 UniDepth 外部固定源码和 source-only
  handoff 验证入口。
- `docs/SERVER_DATA_SETUP.md`：补充 M1/P4-B.5 非 Git 产物迁移与
  `blocked_by_input` 语义。
- `scripts/bootstrap_p4_server.sh`：增加 M1/M2/handoff 专项测试和 source-only
  验证，不自动下载 UniDepth 或运行模型。
- `src/semantic3d/git_release/validation.py`：将 handoff、M1/M2 配置和
  UniDepth registry 加入 Git release 必需路径。
- `scripts/run_p4c3a_function_audit.py`、`src/semantic3d/function_audit.py`：
  历史归档默认改为项目相对路径，并支持 `SEMANTIC3D_ARCHIVE_ROOT`。
- `scripts/run_p4c3a_metric_primary_refactor.py`、
  `src/semantic3d/method_completion/audit.py` 和
  `src/semantic3d/method_completion/md2_audit.py`：移除 WSL `/mnt/e` 默认值，
  支持 `SEMANTIC3D_STRICT_V2_RESULT` 显式指定历史 strict-v2 文件。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件

- 无。

### 主要变化

- Git 继续排除 `.venv`、权重、六视频、外部 provider 源码和生成 outputs。
- readiness 被拆分为 source、models、videos、M1、P4-B.5、M1 reproduction
  和 M2 reproduction，避免“源码测试通过”等价于“真实输入齐全”。
- 即使所有本地依赖均通过，`ready_for_next_stage` 仍固定为 false；必须收到
  用户新的阶段提示词。
- 服务器首条提示词要求先只读验收并等待，不得自动开始新算法阶段。
- Git release 绝对路径审计不再被旧 `/mnt/e` 默认值阻断；历史归档仍可通过
  命令行参数或环境变量显式提供。
- 本机完整验证结果：source、3个YOLO权重、UniDepth权重/config、6个视频、
  M1产物和P4-B.5产物均有效；M1/M2均可复现。
- 当前非 Git 依赖约为：权重150MB、六视频16MB、M1产物137MB、
  P4-B.5正式观测2.3GB、M2历史审计2.3MB。

### 验证命令

```bash
.venv/bin/python -m pytest tests/test_p4c3b_server_handoff.py -q
.venv/bin/python scripts/verify_p4c3b_server_handoff.py --source-only
.venv/bin/python scripts/verify_p4c3b_server_handoff.py \
  --require-models --require-videos --require-m2-inputs
bash -n scripts/bootstrap_p4_server.sh
.venv/bin/python -m pytest \
  tests/test_p4c3b_metric_provider.py \
  tests/test_p4c3b_metric_scene3d.py \
  tests/test_p4c3b_server_handoff.py -q -rs
.venv/bin/python -m pytest -q -rs
```

### 验证结果

- handoff 专项：`4 passed`。
- M1/M2/handoff 联合专项：`26 passed, 1 warning`。
- 全量回归：`593 passed, 5 skipped, 1 warning`。
- 5个skip与上一阶段一致，均为可选旧权重/缓存/派生产物，不涉及handoff。
- warning仍为本机NVIDIA驱动与PyTorch CUDA build不匹配；CPU测试正常。
- source-only和本机完整依赖验证均通过；未执行模型推理、下载、训练、
  Git提交、tag或push。

### 当前限制

- Git checkout 不包含模型、视频、M1/P4-B.5/M2 outputs 或外部 UniDepth
  源码；服务器必须单独转移或重建并校验。
- UniDepth 完整安装依赖需要按服务器 GPU 驱动和 CUDA 版本重新审核，不能
  复制 WSL `.venv`。
- 当前 Git 工作区仍包含多个阶段的未提交变更，提交前必须人工审核
  `git status` 中每个文件。

### 下一步计划

- 用户在 GitHub Desktop 审核并提交源码、配置、测试和文档，不提交被
  `.gitignore` 排除的模型、视频与 outputs。
- 服务器 checkout 固定 commit 后，先运行 source-only verifier 和专项测试，
  再分渠道放置模型、视频和大型产物。
- 只有新的用户阶段提示词到达后，才讨论 view observability 或 temporal
  size materialization，不由代理自动开始。

## 2026-07-24 - P4-C3B-M3 Camera View And Scale History

### 目的

在 P4-C3B-M1 模型估计米制深度和 M2 单帧相机坐标系可见表面 2.5D
结构之上，显式建模相机视场、对象视角、姿态风险和逐维可观测性，并物化
同一对象轨迹的米制尺度历史。该阶段只验证现有数据链路与残差契约，不训练、
不选择阈值、不比较真假性能。

### 新增文件

- `src/semantic3d/method_completion/view_observability.py`：定义
  `CameraViewObservation`、`ObjectViewObservation`、视角枚举和逐维门控。
- `src/semantic3d/method_completion/view_scale_smoke.py`：复用 M1、M2 和
  P4-B.5 已保存产物，离线执行视角、单对象米制尺度和同轨迹尺度历史审计。
- `configs/p4c3b_view_scale_history_v1.yaml`：冻结 M3 输入、质量门控、
  时序参考与禁止事项。
- `scripts/run_p4c3b_view_scale_history_smoke.py`：M3 离线 smoke 命令入口。
- `tests/test_p4c3b_view_scale_history.py`：覆盖视角、独立维度、遮挡/触边、
  稳定与跳变尺度、三种历史参考、ID switch、内参变化、静态路由及 NaN。

### 修改文件

- `src/semantic3d/method_completion/metric_scale.py`：向后兼容地增加独立
  `length_observable`，避免把相机 Z 可见范围直接当作 canonical length。
- `src/semantic3d/method_completion/temporal_scale.py`：增加
  `robust_track_median`、无效 NaN 历史、逐维可观测性、轨迹间断、scene cut、
  质量门控和受限内参漂移兼容。
- `src/semantic3d/method_completion/__init__.py`：导出 M3 公共契约。
- `AGENTS.md`：更新当前真实完成阶段、证据覆盖和禁止误解事项。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件

- 无。

### 主要变化

- 相机 FOV 由模型预测 K 显式计算，内参状态记录为
  `model_predicted_unscored`，不称为 calibrated。
- 没有可靠姿态输入时，视角保持 `unknown`；高度、宽度、长度和可见深度
  范围逐维门控，失败维度保持 NaN。
- canonical 物理维度和 `camera_x/y_visible_extent`、
  `camera_z_visible_range` 分开。后者只支持同轨迹稳定性，不冒充对象真实
  height/width/length。
- 单对象米制主分支实际尝试 11 个对象，但有效物理残差为 0：
  couch 缺少冻结物理先验，cup/bottle 未通过既有深度质量门控，另有一个
  couch 检测置信度过低。没有降低门控制造覆盖。
- 物化 77 条 track size history；三种参考方法合计尝试 231 次，主 couch
  track 的相机轴可见 extent 产生 12 条有效时序残差。
- 无效单对象和时序证据均为 NaN，未出现 missing-as-zero；provider failure
  与 motion_unreliable 均不转为异常证据。
- 当前 6 个持久化 smoke 帧均为 `motion_unreliable`，没有真实 static 或
  low-motion 正例；静态路由由契约测试验证，但不虚报真实静态 smoke。

### 验证命令

```bash
.venv/bin/python scripts/run_p4c3b_view_scale_history_smoke.py
.venv/bin/python -m pytest tests/test_p4c3b_view_scale_history.py -v
.venv/bin/python -m pytest \
  tests/test_p4c3a_metric_single_object_primary.py \
  tests/test_p4c3a_temporal_same_object_scale.py \
  tests/test_p4c3a_method_completion.py \
  tests/test_p4c3b_metric_scene3d.py \
  tests/test_p4c3b_metric_provider.py -q -rs
.venv/bin/python -m pytest -q -rs
```

### 验证结果

- M3 专项：`10 passed`。
- 旧 MD2、时序、方法补全、M1/M2 回归：`56 passed, 1 warning`。
- 全量回归：`603 passed, 5 skipped, 1 warning`。
- 5 个 skip 均为既有可选姿态权重/real_3 observation、P3 shared cache、
  两项旧 shared-geometry smoke 产物和未提交 strict-v2 六视频 fixture。
- warning 为本机 NVIDIA driver 与 PyTorch CUDA build 不匹配；M3 未运行
  新模型推理，不依赖 CUDA。
- strict-v1、strict-v2 和 P4-C0/C1/C2 配置 SHA-256 均保持不变。

### 当前限制

- 当前视角状态没有通用类别姿态 provider，11 个真实对象视角均为 unknown，
  canonical 可观测维度为 0。
- 模型预测米制深度与模型预测内参不是传感器真值；时序内参只允许同 provider
  且相对变化不超过显式阈值的比较。
- 正式 mask 仅为 visible mask，不是 amodal mask；遮挡下不能声称完整尺度。
- 当前有效时序结果来自相机轴可见 extent，不是严格语义物理尺度残差。
- 6 帧 smoke 不能建立方法有效性或真假性能结论。

### 下一步计划

- 接入可审计类别姿态/人体关键点来源，先验证 viewpoint 与完整维度的真实
  覆盖，再允许 canonical metric residual 进入主证据。
- 增加至少一个已有米制深度的 static/low-motion 短片段，验证静态真实路由。
- 在不降低质量门控的前提下扩大米制帧覆盖，审计 provider 内参和尺度漂移。

## 2026-07-24 - P4-C3B-M4 Continuous Pose And Real D2 Smoke

### 目的

在既有 D2 合成数学闭环和 M1/M2 持久化米制短片段之上，建立具有明确失败
状态的真实短基线相机位姿 Provider、多证据静态确认、前景排除、短 clip
局部坐标对齐，以及带可见性判定的真实 D2 几何 smoke。本阶段只审核几何
链路，不训练、不选阈值、不使用真假标签评价性能。

### 新增文件

- `src/semantic3d/pose_d2/contracts.py`：统一位姿状态、静态验证和
  validity-aware D2 证据契约。
- `src/semantic3d/pose_d2/provider.py`：实现前景排除 LK、单应/本质矩阵
  诊断和源帧模型预测米制深度辅助 PnP。
- `src/semantic3d/pose_d2/alignment.py`：把相邻相机坐标组合到短片段
  `clip_local_aligned` 参考规范，不声称 world frame。
- `src/semantic3d/pose_d2/residuals.py`：实现点、边界、深度和对象 D2，
  并区分 visible、out_of_frame、occluded、depth_conflict 和
  no_correspondence。
- `src/semantic3d/pose_d2/smoke.py`：只读取 M1/M2/P4-B.5 持久化产物运行
  离线真实 smoke 和审计报告。
- `src/semantic3d/pose_d2/__init__.py`：导出 M4 公共接口。
- `configs/p4c3b_pose_d2_smoke_v1.yaml`：冻结输入片段、质量阈值和禁止事项。
- `scripts/run_p4c3b_pose_d2_smoke.py`：M4 离线运行入口。
- `tests/test_p4c3b_pose_d2.py`：覆盖已知旋转平移、位姿方向、身份变换守卫、
  前景排除、低纹理退化、局部对齐、出画、遮挡和深度冲突。

### 修改文件

- `AGENTS.md`：将当前真实完成阶段、M4 坐标与证据语义写入服务器续接约束。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件

- 无。

### 主要变化

- 位姿状态严格区分 `verified_static`、`estimated_valid`、
  `estimated_low_confidence`、`provider_failed`、
  `blocked_by_intrinsics`、`blocked_by_correspondence`、
  `not_applicable` 和 `unknown`。
- 位姿方向固定为
  `X_target_camera = T_target_from_source @ X_source_camera`。源帧三维点使用
  源帧 K 反投影，目标像素使用目标帧 K 进行 PnP 和重投影。
- 身份变换只在全局流、背景流、单应旋转、本质矩阵运动支持、视差和图像差异
  的多证据审核达到要求后使用；provider 失败不回退为单位矩阵。
- 正式 visible instance mask 内点在相机位姿跟踪前被排除；记录候选背景点、
  被排除前景点、几何内点和退化状态。
- `fake_1` 两个相邻帧对被验证为静态；`real_1` 两个相邻帧对得到有效
  模型尺度 PnP 位姿，平移范数约为 0.0198 m 和 0.0112 m；
  `fake_2` 缺少持久化米制 K/depth，保持 `blocked_by_intrinsics`。
- 真实 D2 smoke 产生 428 个有效点、204 个有效边界和 7 个有效对象聚合；
  另有 34 个 depth conflict、35 个 occluded、3 个 out-of-frame 和 1 个
  invalid-input 记录，全部保持 NaN，不制造高残差。
- 不同相机帧仅对齐到短片段参考规范 `clip_local_aligned`。参考帧单位阵是
  坐标规范定义，不是静态判定，也不构成长期 world map。
- 深度与内参仍为模型预测，不是传感器真值或相机标定真值；D2 是几何质量
  诊断，不是直接的真假判断。

### 验证命令

```bash
.venv/bin/python scripts/run_p4c3b_pose_d2_smoke.py
.venv/bin/python -m pytest tests/test_p4c3b_pose_d2.py -v
.venv/bin/python -m pytest \
  tests/test_p4c3b_pose_d2.py \
  tests/test_pose_estimation_fallbacks.py \
  tests/test_pose_graph.py \
  tests/test_sequence_geometry_observation.py \
  tests/test_dynamic_reprojection_residual.py \
  tests/test_backprojection.py \
  tests/test_coordinate_transforms.py \
  tests/test_p4c3a_method_completion.py \
  tests/test_p4c3b_metric_provider.py \
  tests/test_p4c3b_metric_scene3d.py \
  tests/test_p4c3b_view_scale_history.py -q -rs
.venv/bin/python -m pytest -q -rs
```

### 验证结果

- M4 专项：`11 passed`。
- 相关位姿、几何、D2、M1/M2/M3 回归：`91 passed, 1 warning`。
- 全量回归：`614 passed, 5 skipped, 1 warning`。
- 5 个 skip 均为既有可选人体姿态/real_3 observation、P3 shared cache、
  两项旧 shared-geometry smoke 产物和未提交 strict-v2 六视频 fixture。
- warning 为本机 NVIDIA driver 与 PyTorch CUDA build 不匹配；M4 使用
  CPU OpenCV，未运行新模型推理。
- strict-v1、strict-v2 和 P4-C0/C1/C2 配置 SHA-256 保持不变。

### 当前限制

- 位姿平移尺度来自模型预测米制深度，因此不是传感器真值，跨长序列可能累积
  尺度和位姿漂移。
- 当前只验证相邻帧和 2–3 帧短片段，没有回环、全局优化或长期 world map。
- 目标边界对应使用同 track 可见边界的最近邻诊断；正式 amodal 边界和遮挡后
  完整轮廓不可用。
- motion-unreliable 控制项因缺少同一批持久化米制输入而被正确阻断，尚未覆盖
  “有输入但几何退化”的真实视频样例。
- 当前 smoke 不建立 D2 对真假视频的检测有效性。

### 下一步计划

- 扩展同一 provider 的短片段覆盖，量化连续复合位姿漂移和内参漂移影响。
- 增加真实低纹理、动态前景占比高和明显遮挡片段，完善失败/不可见覆盖。
- 进入 D3 前先冻结 D2 质量门控与坐标约定；D3 只复用 valid D2 和
  `clip_local_aligned` 数据，不把 blocked/provider failure 当异常证据。

## 2026-07-24 - P4-C3B-M5 D3 Higher-Order Relations, Occlusion, and Reappearance

### 目的

在不训练、不选择阈值和不评价真假性能的前提下，将 P4-C3A-M 的 D3
公式接口落实为姿态补偿后的三维结构图残差执行器，并建立遮挡、消失、
出画、漏检、轨迹失败和重现的显式事件契约。

### 新增文件

- `src/semantic3d/d3/contracts.py`：定义 D3 节点、关系、帧图、transition
  门控、残差及定位引用契约。
- `src/semantic3d/d3/graph.py`：构建对象、边界、几何轨迹点和可选语义点
  图，以及对象距离、深度次序、边长、刚性、方位、重叠和可靠接触关系。
- `src/semantic3d/d3/residuals.py`：实现五类核心 D3 残差及额外关系诊断。
- `src/semantic3d/d3/events.py`：实现事件分类和六线索重现残差。
- `src/semantic3d/d3/smoke.py`：复用 M2/M4 持久化产物并运行合成控制验证。
- `src/semantic3d/d3/__init__.py`：导出 M5 公共接口。
- `configs/p4c3b_d3_occlusion_v1.yaml`：固定 M5 输入、图规模和事件门控配置。
- `scripts/run_p4c3b_d3_occlusion_smoke.py`：M5 离线 smoke 入口。
- `tests/test_p4c3b_d3_occlusion.py`：覆盖 D3 公式、身份/姿态门控、事件类别、
  重现、多源定位和 NaN 语义。

### 修改文件

- `src/semantic3d/method_completion/d3_relations.py`：向后兼容扩展 D3 关系枚举
  和公式定义；旧公式级接口继续可用。
- `docs/DEV_LOG.md`：追加本记录。

### 删除文件

- 无。

### 主要变化

- 所有动态 D3 关系只能在 M4 位姿可用、track/point 身份可靠且两帧都处于
  `clip_local_aligned` 时计算；否则残差为 NaN，原因记录为
  `blocked_by_pose_or_correspondence` 或更具体的缺失原因。
- `R_relative_distance`、`R_depth_order`、`R_edge_length`、
  `R_local_rigidity` 和 `R_relative_orientation` 已通过合成刚体/扰动对照。
  对象方位和对象自身相对朝向已分开，未知真实朝向不做猜测。
- 真实 M2/M4 短片段生成 6 个 D3 帧图、19 条有效对象或边界关系残差。
  M2 几何点尚无经验证的跨帧 point identity，因此 16 条显式
  edge/rigidity 诊断被阻断，另外 7 条关系因两帧未同时观测而无效。
- 合成专项集覆盖 partial/full occlusion、out-of-frame、detector miss、
  true disappearance、reappearance、ID switch/track failure 和 no-event。
- 持久化真实对象轨迹产生 7 个 `not_applicable_no_event` 和 1 个
  `blocked_by_input`；没有正式 partial/full occlusion 事件，不降低标准制造
  事件。
- 重现残差同时检查身份、预测位置、深度、物理尺度、三维结构和运动趋势；
  无重现事件时 combined residual 保持 NaN。
- 全部图、残差和事件 ID 唯一；invalid residual 没有任何有限值，valid
  residual 没有 NaN/Inf。所有证据保存 source node/edge、confidence、
  coordinate frame 和 localization reference。

### 验证命令

```bash
.venv/bin/python scripts/run_p4c3b_d3_occlusion_smoke.py
.venv/bin/python -m pytest tests/test_p4c3b_d3_occlusion.py -v
.venv/bin/python -m pytest \
  tests/test_p4c3a_method_completion.py \
  tests/test_p4c3b_metric_scene3d.py \
  tests/test_p4c3b_pose_d2.py \
  tests/test_p4c3b_d3_occlusion.py -v
.venv/bin/python -m pytest -q -rs
```

### 验证结果

- M5 专项：`11 passed`。
- D3 兼容、M2、M4 与 M5 联合回归：`49 passed`。
- 全量回归：`625 passed, 5 skipped, 1 warning`。
- 5 个 skip 为既有可选姿态权重/real_3 observation、P3 shared cache、
  两项旧 shared-geometry smoke 产物和未提交 strict-v2 六视频 fixture。
- warning 为 WSL NVIDIA driver 与 PyTorch CUDA build 不匹配；M5 未运行
  新模型推理。
- strict-v1/v2 及 P4-C0/C1/C2 冻结内容未修改。

### 当前限制

- 真实内部结构点缺少稳定跨帧 point identity，因此真实 `R_edge_length` 和
  `R_local_rigidity` 尚未形成有效证据；当前仅由合成真值验证。
- 真实对象本体朝向没有可靠 provider，真实 `R_relative_orientation` 不执行。
- 当前 Mask 是 visible instance mask，不是 amodal mask；真实短片段未发现
  满足严格条件的 partial/full occlusion 或 reappearance。
- 米制深度和内参仍是模型预测，不是传感器真值；坐标只在短片段局部对齐，
  没有长期 world frame。
- 本阶段没有建立伪造检测有效性结论。

### 下一步计划

- 在不降低质量门控的前提下物化稳定跨帧几何 point identity，并用真实短片段
  执行 edge-length 和 local-rigidity 分支。
- 扫描具备正式 visible mask、D2 支撑和深度次序的事件片段，补足真实遮挡与
  重现覆盖。
- 证据融合只消费 valid/quality/coverage/localization 字段；缺失、阻断、
  provider failure 和 not-applicable 分支继续保持掩码。

## 2026-07-24 - P4-C3B-M6 Missing-Aware Evidence Fusion and Localization

### 目的

在不训练分类器、不拟合分布、不选择正式阈值和不评价真假性能的前提下，
把已有静态、时序、D1、D2、D3、遮挡与重现残差统一为可审计 Evidence，
建立缺失感知的确定性风险/置信度双输出，并完成时间、空间、对象和轨迹定位
接口。

### 新增文件

- `src/semantic3d/evidence_fusion/contracts.py`：定义九类分支和统一 Evidence
  数据契约。
- `src/semantic3d/evidence_fusion/routing.py`：实现静态、低运动、动态且有
  位姿、动态但无位姿及 motion-unreliable 路由。
- `src/semantic3d/evidence_fusion/fusion.py`：实现质量加权、缺失感知的
  确定性融合及 branch dropout 审计。
- `src/semantic3d/evidence_fusion/audits.py`：提供 missingness-only、
  分层覆盖率和 provider failure balance 审计接口。
- `src/semantic3d/evidence_fusion/temporal.py`：实现帧级聚合、因果中值平滑
  和显式诊断阈值下的区间生成/合并接口。
- `src/semantic3d/evidence_fusion/localization.py`：实现对象、边界、点、轨迹
  和帧级空间证据图及对象/轨迹排名。
- `src/semantic3d/evidence_fusion/adapters.py`：将 M3/M4/M5 持久化产物离线
  适配为统一 Evidence，不运行 provider。
- `src/semantic3d/evidence_fusion/smoke.py`：执行离线融合、定位和审计。
- `src/semantic3d/evidence_fusion/__init__.py`：导出 M6 公共接口。
- `configs/p4c3b_evidence_fusion_v1.yaml`：记录输入产物、逐视频原始尺寸、
  路由、固定分支权重及无正式阈值约束。
- `scripts/run_p4c3b_evidence_fusion_smoke.py`：M6 离线 smoke 入口。
- `tests/test_p4c3b_evidence_fusion.py`：覆盖 NaN、provider failure、质量
  降权、缺失不改风险、路由、时间序列、空间回映和完整 smoke。

### 修改文件

- `docs/DEV_LOG.md`：追加本记录。

### 删除文件

- 无。

### 主要变化

- 每条证据统一保存 residual、applicability、valid、confidence、
  uncertainty、provider status、failure reason、branch、对象、轨迹、
  帧、空间/时间引用和 provenance。
- 风险只使用 applicable 且 valid 的有限残差。缺失分支不进入风险公式，
  仅降低 available weight ratio 和 evidence confidence；provider failure
  不能构成有效证据。
- 分支残差先使用 P4-A 的 median/top-k 稳健聚合，再以固定权重和证据质量
  加权；输出风险不是概率，也不是真假结论。
- static/low-motion 保留静态几何、边界、内部结构和尺度稳定性；动态且位姿
  不可用时保留静态、D1 和局部时序，D2/D3 被显式阻断。
- missingness-only、branch dropout、分组覆盖率及 provider failure balance
  仅作为审计接口；当前 smoke 不读取 authenticity label。
- 时间定位不配置正式阈值时只输出原始和因果平滑序列，不产生异常区间。
  空间定位只栅格化持久化的点、边界、Mask 或可见边界支撑，不伪造 Mask；
  无空间支撑的证据仍可按对象/轨迹 ID 排名和追溯。
- 修复了横屏固定尺寸用于竖屏 `real_1` 的坐标错误。现在按视频保存原始
  image shape，空间栅格证据由 178 条增至 306 条，对象/轨迹排名为 6 条。
- 最终离线 smoke 读取 1004 条持久化证据，路由后 1017 条，其中 325 条
  valid；invalid finite count 和 provider-failure valid count 均为 0。
- `fake_1` 静态 smoke 有 2 个有效分支，风险 0.006822、证据置信度
  0.185495；`real_1` 相机运动 smoke 有 2 个有效分支，风险 0.001081、
  证据置信度 0.102614；`fake_2` motion-unreliable 控制项无有效证据，
  风险保持 NaN。数值仅用于功能闭环，不比较真假。

### 生成结果

- `outputs/p4c3b_evidence_fusion/evidence_schema_audit.json`
- `outputs/p4c3b_evidence_fusion/branch_availability_audit.csv`
- `outputs/p4c3b_evidence_fusion/risk_confidence_baseline.csv`
- `outputs/p4c3b_evidence_fusion/branch_contribution_audit.csv`
- `outputs/p4c3b_evidence_fusion/temporal_evidence_sequences.json`
- `outputs/p4c3b_evidence_fusion/spatial_evidence_manifest.csv`
- `outputs/p4c3b_evidence_fusion/spatial_evidence_maps.npz`
- `outputs/p4c3b_evidence_fusion/object_track_rankings.csv`
- `outputs/p4c3b_evidence_fusion/missingness_shortcut_audit.json`
- `outputs/p4c3b_evidence_fusion/localization_interface_audit.json`
- `outputs/p4c3b_evidence_fusion/validation_report.json`
- `outputs/p4c3b_evidence_fusion/EVIDENCE_FUSION_REPORT.md`

### 验证命令

```bash
.venv/bin/python scripts/run_p4c3b_evidence_fusion_smoke.py
.venv/bin/python -m pytest tests/test_p4c3b_evidence_fusion.py -v
.venv/bin/python -m pytest \
  tests/test_p4c3b_evidence_fusion.py \
  tests/test_p4c3b_d3_occlusion.py \
  tests/test_p4c3b_pose_d2.py \
  tests/test_p4c3b_view_scale_history.py \
  tests/test_aggregation_v2.py -q -rs
.venv/bin/python -m pytest -q -rs
```

### 验证结果

- M6 专项：`11 passed`。
- M3/M4/M5/M6 与 P4-A 联合回归：`46 passed`。
- 全量回归：`636 passed, 5 skipped, 1 warning`。
- 5 个 skip 均为既有本地可选工件缺失：人体姿态/real_3 observation、
  P3 shared cache、两项旧 shared-geometry smoke 产物和未提交 strict-v2
  六视频 fixture；M6 关键测试无 skip。
- warning 为 WSL NVIDIA driver 与 PyTorch CUDA build 不匹配；M6 不运行
  provider 或模型推理。
- strict-v1 SHA-256 保持
  `e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b`；
  strict-v2 SHA-256 保持
  `3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b`。

### 当前限制

- 各分支残差尚未统计校准到共同分布，固定变换和权重只是无学习诊断基线。
- 当前真实 smoke 只有两个具有有效融合证据的短片段，不能说明方法有效性。
- D1 没有进入当前持久化 smoke；真实 D3 内部结构、遮挡和重现覆盖仍不足。
- 当前对象定位中 M4 object D2 使用可见边界支撑点，不等价于完整对象 Mask；
  reference-only 证据不会被伪造成像素热图。
- 未执行 real/fake missingness balance；接口已存在，但当前配置故意只使用
  无标签 source groups。
- 时间区间接口尚未选择正式阈值，因此当前不输出最终异常片段判定。

### 下一步计划

- 在正式数据 split 和冻结阈值协议就绪后，先做分支尺度校准与 missingness
  平衡审计，再考虑学习融合。
- 扩充真实 D1、D3、遮挡和重现事件覆盖，同时保留 provider failure 与
  not-applicable 的独立状态。
- 在不读取真假标签的定位验证中补充正式 Mask、边界、点和轨迹的可视化，
  检查证据能否完整追溯到原观测。

## 2026-07-25 - P4-C3C-A1 Formal Data Interface and Training Readiness

### 目的

在不下载正式数据、不运行模型推理和不训练的前提下，审计真实训练能力，
并建立可接入服务器外部数据根目录的统一视频样本与数据集 adapter 基础。

### 新增文件

- `src/semantic3d/dataset_builder/formal_schema.py`：定义正式视频样本字段、
  三种 split、显式缺失状态和一致性检查。
- `src/semantic3d/dataset_adapters/`：定义通用 adapter 接口及
  GenVideo-100K 未解析官方 schema 的显式骨架。
- `configs/data_registry/genvideo_100k_adapter_skeleton_v1.yaml`：记录
  `adapter_status: skeleton`、未下载且官方 schema 未核验。
- `tests/test_p4c3c_a1_formal_data.py`：使用临时极小文件验证外部路径、递归
  扫描、同名文件、稳定 ID、split、缺失标签、lineage、checksum、adapter
  骨架及旧 YAML 路径兼容。

### 修改文件

- `src/semantic3d/dataset_builder/manifest.py`：增加通用视频扩展名、递归扫描、
  绝对/数据根相对路径、正式样本构建和同名源消歧；旧 manifest 外部路径
  不再强制 `relative_to(project_root)`。
- `src/semantic3d/dataset_builder/pipeline.py` 与 `p4b5_pipeline.py`：支持
  `sources.data_root` 和持久化绝对源路径，同时保留旧 YAML 相对路径以项目根
  解析的语义。
- `scripts/build_video_manifest.py`：保留旧 pilot 模式并增加显式 formal 模式。
- `scripts/build_labels_manifest.py`：优先使用稳定 ID，拒绝同 stem 歧义，并用
  null 加 `metadata_status` 表示缺失标签/标注。
- `src/semantic3d/dataset_builder/__init__.py`：导出正式数据公共接口。

### 训练能力审计结论

- 未发现 torch `Dataset`/`DataLoader`、可训练 `nn.Module`、loss、backward、
  optimizer、scheduler、AMP 或配置化训练入口。
- 现有 cache/checkpoint/resume 是离线数据阶段与批次状态恢复，不包含模型、
  optimizer 或训练 epoch 状态。
- 现有验证与 metrics 是结构证据、数据完整性和 deterministic fusion 审计，
  不是训练 validation loop 或真实性性能评估。
- `SMALL_DRY_RUN_READY=NO`。
- `FORMAL_TRAINING_READY=NO`。
- 完整逐项证据报告为 `p4c3c_a1_training_readiness_audit.md`
  （服务器外部审计产物，不纳入 Git）。

### 验证

```bash
.venv/bin/python -m pytest tests/test_p4c3c_a1_formal_data.py -q
.venv/bin/python -m pytest \
  tests/test_p4c3c_a1_formal_data.py \
  tests/test_structural_enhancement_dataset.py \
  tests/test_dataset_builder_cache.py \
  tests/test_p4b5_full_observation.py -q
.venv/bin/python scripts/build_video_manifest.py --help
.venv/bin/python scripts/build_labels_manifest.py --help
git diff --check
```

### 验证结果

- P4-C3C-A1 新增专项：`9 passed`。
- 新增与受影响数据构建联合专项：`38 passed`。
- 两个 manifest 脚本的 CLI 解析正常，`git diff --check` 通过。
- 未运行依赖六视频、模型权重或历史产物的测试；未运行完整测试、推理、数据
  构建或训练。

## 2026-07-25 - P4-C3C-A2 Minimal Trainable Loop

### 目的

在不下载正式数据或模型、不运行视频 provider 和不执行正式训练的前提下，
建立基于预计算 M6 分支证据的最小二分类训练工程闭环。此阶段只验证
Dataset、missing-aware 模型、loss、优化、validation、checkpoint 和 resume
接口，不评价方法有效性。

### 新增文件

- `configs/p4c3c_a2_m6_feature_contract_v1.yaml`：冻结九个 M6 分支顺序，
  仅允许分支 `bounded_risk` 作为特征、`confidence` 作为可靠性门控，
  并显式排除 ID、来源、label、split、provider failure 和 deterministic
  fusion 总分。
- `configs/p4c3c_a2_minimal_training_v1.yaml`：记录数据入口、小模型、
  optimizer、scheduler、AMP、确定性、checkpoint 和输出配置。
- `src/semantic3d/minimal_training/`：实现 feature contract、A1 formal
  manifest 与 evidence manifest join、DataLoader/collate、missing-aware
  evidence head、masked BCE、validation 指标、严格 checkpoint/resume、
  synthetic fixture 和训练循环。
- `scripts/train_p4c3c_a2_minimal.py`：提供配置化训练入口及显式
  `--synthetic-fixture` 工程模式。
- `tests/test_p4c3c_a2_minimal_training.py`：覆盖 split/泄漏、缺失语义、
  forward/loss/backward、CPU/CUDA/AMP、checkpoint/resume、checksum、
  feature contract、固定 seed、validation 不更新状态和 metrics 追加。

### 修改文件

- `docs/DEV_LOG.md`：追加本记录。

### 主要变化

- Dataset 只读取 A1 formal metadata 和预计算 evidence manifest，不读取或
  解码视频，不运行 YOLO、UniDepth 或 M1–M6 provider。
- 输入张量固定为九分支 `features`、`feature_mask`、`missing_mask`、
  `observability` 和 `reliability`；shape 不一致直接报错，不截断。
- 所有 evidence 缺失的样本由 `valid_sample_mask=false` 标记，logit 保持
  NaN 且不进入 loss；未知 label 使用独立 `label_mask`，不会变成负样本。
- 无有效监督样本的 batch 不执行 backward 或 optimizer step，并记录
  skipped batch。可选 `pos_weight` 只能从 train split 的已知标签派生。
- validation 固定使用 0.5 工程阈值并输出 loss、accuracy、precision、
  recall、F1 和类别齐全时的 ROC-AUC；不搜索阈值，不读取 test split。
- checkpoint 保存模型、optimizer、scheduler、AMP scaler、epoch、
  global step、best validation loss、配置、随机状态、feature contract、
  Git 提交和 train/validation manifest checksum；不匹配时拒绝 resume。
- checkpoint 以 CUDA `map_location` 加载时，会先把保存的 CPU/CUDA RNG
  ByteTensor 规范化回 CPU 再恢复，避免 CUDA resume 的 RNG 类型错误。
- synthetic fixture 的 train/validation sample ID 严格分开，包含双分支、
  部分缺失、全 evidence 缺失、缺失 label 和 0/1 label。fixture 不包含
  视频或私有资产。

### 验证命令

```bash
.venv/bin/python -m pytest tests/test_p4c3c_a2_minimal_training.py -q -rs
.venv/bin/python -m pytest \
  tests/test_p4c3c_a1_formal_data.py \
  tests/test_p4c3b_evidence_fusion.py -q -rs
.venv/bin/python scripts/train_p4c3c_a2_minimal.py \
  --synthetic-fixture --device cpu --epochs 1 \
  --max-train-batches 1 --max-validation-batches 1 \
  --output-dir outputs/p4c3c_a2_synthetic/cpu_single_step
.venv/bin/python scripts/train_p4c3c_a2_minimal.py \
  --synthetic-fixture --device cpu --epochs 2 \
  --output-dir outputs/p4c3c_a2_synthetic/cpu_two_epoch
.venv/bin/python scripts/train_p4c3c_a2_minimal.py \
  --synthetic-fixture --device cuda --epochs 1 \
  --max-train-batches 1 --max-validation-batches 1 \
  --output-dir outputs/p4c3c_a2_synthetic/cuda_single_step
.venv/bin/python scripts/train_p4c3c_a2_minimal.py \
  --synthetic-fixture --device cuda --amp --epochs 1 \
  --max-train-batches 1 --max-validation-batches 1 \
  --output-dir outputs/p4c3c_a2_synthetic/cuda_amp_single_step
```

### 验证结果

- A2 专项：`32 passed`，CUDA 和 CUDA AMP 测试在可用设备上实际执行。
- A1 formal schema 与 M6 evidence fusion 回归：`20 passed`。
- CPU 单步、CPU 两 epoch、CUDA 单步和 CUDA AMP 单步均完成。
- checkpoint resume 从 epoch 1 继续 epoch 2，global step 从 2 增至 4，
  metrics JSONL 从 2 行追加到 4 行。
- CUDA AMP resume 从 epoch 1 继续 epoch 2，scaler state 非空，
  optimizer step 与 global step 均从 1 连续到 2。
- 所有工程输出位于 `outputs/p4c3c_a2_synthetic/` 且被 Git 忽略。
- 未下载数据或模型，未运行视频推理、M1–M6 真实流水线或正式训练。

### 当前限制

- 当前服务器没有真实 formal manifest 或真实 M6 evidence manifest，
  因此 real-data dry-run 尚不可执行。
- M6 原始分支残差尚未跨分支统计校准；A2 只使用 M6 固定单调变换后的
  `bounded_risk`，不拟合正式统计量。
- synthetic 指标只证明工程闭环可运行，不代表真假检测性能。
- 时间、空间、对象和分支辅助监督仅保留 metadata 接口，未实现多任务 loss。
- 正式阈值、正式训练、test evaluation 和方法效果结论均未建立。

### 下一步计划

- 在官方 schema 核验且真实数据按 A1 contract 接入后，先生成预计算 M6
  evidence manifest 并执行不更新参数的 real-data dry-run。
- 对 train split 做 missingness shortcut 和分支覆盖平衡审计，再决定是否
  允许任何统计拟合或正式训练。

## 2026-07-25 - P4-C3C-A3-A Real-Data Dry-Run Readiness

### 目的

在不下载正式视频、不运行 M1–M6 批处理和不执行训练的前提下，核验官方
GenVideo-100K 源、恢复当前有效配置所需的公共模型资产，并建立真实 M6
审计表到 A2 evidence manifest 的确定性转换桥。

### 官方数据源结论

- 官方 DeMamba 仓库确实链接到 ModelScope 的
  `cccnju/GenVideo-100K`，数据卡许可为 `CC-BY-NC-SA-4.0`。
- 固定读取的数据仓库 revision 为
  `b31a22ff523263c6915a480966cbab1c87eca345`。
- 官方树只有 README、Git 属性文件及 15 个压缩包/分片的 Git LFS
  pointer；没有 CSV、JSON、JSONL、Parquet、record manifest 或 split
  定义。
- label、official split、fake generator、real source、video path 和
  record lineage 字段均未核验，因此
  `official_schema_verified=false`，既有 GenVideo adapter 继续保持
  skeleton，未根据压缩包名猜测字段。
- 小样本计划因缺少 record metadata、单视频下载入口和 official split
  而保持未就绪；本阶段没有下载任何视频。

### 公共模型资产与环境

- `yolov8n-seg.pt` 与冻结 registry 的大小和 SHA-256 完全一致。
- `yolov8n.pt`、`yolov8n-pose.pt` 从 registry 声明的官方 URL 获取后，
  实际大小和 SHA-256 与冻结值不同，状态保持 `MISMATCH`，未进入正式
  checkpoint 位置。
- UniDepth 固定源码 revision
  `8d8cfe4c7ee15297099983607febf0d4f32eb3d6` 保持干净；
  ViT-S14 weight/config 与 registry SHA-256 完全一致。
- 有效 P4-B.5 配置需要 Depth Anything V2 Small。其规模按官方说明为
  24.8M parameters，许可为 Apache-2.0；由于项目 registry 未声明该资产
  revision、大小或 SHA，下载后状态为 `DOWNLOADED_UNVERIFIED`，没有冒充
  冻结校验成功。
- Python、torch、torchvision、transformers 与 triton 的既有版本未被
  替换。UniDepth 使用固定源码 editable `--no-deps`，随后仅按真实 import
  缺失补入最小依赖。
- 上游 UniDepth 包元数据把训练、可视化、评测和可选加速依赖统一声明为
  required；为避免改动已验证 torch/CUDA 栈，本阶段未安装 torchaudio、
  xformers 或 evaluation-only 依赖，因此 `pip check` 仍明确报告这些
  未满足项。
- 未编译 UniDepth KNN/extract-patches 扩展；其旧 CUDA 架构列表未经
  RTX 5090 适配，不在本阶段盲目编译。

### 模型 load smoke

- 三个 YOLO 文件都能识别为对应 detect/segment/pose 模型并切换到 CUDA；
  两个 registry mismatch 文件的 load 成功不改变其资产状态。
- UniDepth 从固定本地源码与固定本地文件离线加载，在 RTX 5090 上对
  56×56 合成 RGB tensor 完成一次工程推理；depth 与 intrinsics 存在、
  shape 合法且无 NaN/Inf。
- UniDepth confidence 只记录为单输入内相对排序信息，不视为跨视频绝对
  校准置信度。
- Depth Anything V2 Small 从本地缓存离线加载并对 56×56 合成 tensor
  完成 forward，relative depth 输出存在且有限；未解释为 metric depth
  或性能结果。
- smoke 输出位于 Git 忽略的 `outputs/p4c3c_a3_model_smoke/`。

### M6 到 A2 evidence bridge

- 新增 `configs/p4c3c_a3_evidence_bridge_v1.yaml`、
  `src/semantic3d/minimal_training/evidence_bridge.py` 和
  `scripts/build_p4c3c_a3_evidence_manifest.py`。
- bridge 读取 M6 `branch_contribution_audit` 与混合型
  `branch_availability_audit`，不读取 `risk_confidence_baseline`。
- 九分支顺序严格来自 A2 feature contract；每个分支显式输出
  `bounded_risk`、`feature_available`、`observability`、`confidence`
  和 `missing_reason`，并保留 A2 所需的 `status`/`observable`。
- legacy M6 `(video_id, clip_id)` 必须通过独立 mapping manifest 严格映射
  到 A1 `sample_id`；重复、漏 join、未知分支和 split 不一致均报错。
- label、source dataset、generator 和 lineage 只来自 formal manifest。
  provider failure 保持 null feature 与显式 missing reason，deterministic
  fusion 总分不进入输出。
- 输出 JSONL 保存所有输入 checksum 与转换配置 checksum；转换不拟合
  normalization、统计量或阈值。

### 验证

```bash
.venv/bin/python -m pytest tests/test_p4c3c_a3_evidence_bridge.py -q -rs
.venv/bin/python -m pytest \
  tests/test_p4c3c_a2_minimal_training.py \
  tests/test_p4c3c_a1_formal_data.py \
  tests/test_p4c3b_evidence_fusion.py -q -rs
.venv/bin/python scripts/build_p4c3c_a3_evidence_manifest.py --help
```

- A3 bridge 专项：`22 passed`。
- A2、A1 与 M6 指定回归：`52 passed`。
- bridge CLI help 正常。
- 未下载 GenVideo-100K 或 BrokenVideos 视频，未运行真实视频流水线，
  未运行训练，未选择阈值，未拟合正式统计量。

### 当前状态

- `PUBLIC_MODEL_ASSETS_READY=NO`：两个冻结 YOLO 权重 mismatch，且 Depth
  Anything 仍无项目 registry 冻结元数据。
- `GENVIDEO_OFFICIAL_SCHEMA_VERIFIED=NO`。
- `M6_TO_A2_BRIDGE_READY=YES`。
- `SMALL_SAMPLE_DOWNLOAD_READY=NO`。
- `REAL_DATA_DRY_RUN_READY=NO`。
- `FORMAL_TRAINING_READY=NO`。

## 2026-07-25 - P4-C3C-A3-A.2 Public Asset Remediation

### 资产冻结

- 保留 `yolo_weights_v1.yaml` 作为历史记录，未改写其大小或 SHA-256。
- 新增活跃 `yolo_weights_v2.yaml`。detect、segment、pose 三份权重均来自
  Ultralytics 官方 `v8.4.0` release；官方 release API 声明的大小与
  SHA-256 分别和本地文件完全一致，三项状态均为 `VERIFIED`。
- 新增 `depth_anything_v2_small_v1.yaml`，固定官方源 revision、Hugging
  Face model revision、weight/config/preprocessor 的大小与 SHA-256。
  该 registry 只声明 Small（24.8M，Apache-2.0），输出语义保持
  relative depth，不标记为 metric depth。
- 新增公共资产 active profile，通过 `MODEL_ROOT`、`HF_HOME` 和
  `SEMANTIC3D_UNIDEPTH_SOURCE_ROOT` 选择外部资产；配置不包含机器绝对路径，
  也不允许隐式下载。

### UniDepth 环境收敛

- 采用方案 B：仅移除 UniDepth editable distribution 的 `.pth` 和
  `dist-info` 元数据，保留固定 vendor 源码、weight 和 config。
- 新增受控 source-root helper。只有环境显式提供源码根、Git HEAD 等于
  registry revision 且 vendor worktree 干净时，当前进程才会激活
  UniDepth import。
- 当前 UniDepthV2 推理路径所需的 torch、torchvision、triton、einops、
  timm、scipy 和 wandb 保持不变。未安装训练、评测、UI 或可选加速路径
  才需要的完整依赖，也未安装 xformers 或 torchaudio。
- 环境变更前后 package diff 只有 editable UniDepth 分发元数据被移除；
  torch、torchvision、transformers、triton 和 CUDA 栈未变化。
- `python -m pip check` 从 UniDepth 上游完整依赖元数据导致的非零恢复为
  `No broken requirements found`。

### GPU smoke 与测试

- YOLO detect、segment、pose 三份新 registry 权重分别完成反序列化和
  CUDA load；没有执行图像或视频 inference。
- UniDepth 从固定、干净的源码 revision 和固定本地模型目录离线加载，
  对 64×64 合成 RGB tensor 输出有限的 depth 和 intrinsics。
- Depth Anything V2 Small 从固定本地 snapshot 离线加载，对 224×224
  合成 tensor 完成 GPU forward，输出有限且仅解释为 relative depth。
- smoke 只验证加载、接口、数值有限性和 CUDA 可执行性，不代表性能或
  方法效果；输出保存在 Git 忽略目录。
- A3-A.2 资产专项、受控 source handoff、A3 bridge、A2、A1、M6 以及必要
  registry 保护测试联合结果为 `90 passed`。
- 未下载任何视频，未运行 M1–M6 真实批处理，未执行真实训练。

`KNOWN_DEFERRED_TEST_DEPENDENCY`：旧 metric-provider 测试需要未声明的
`pandas`；在真实流水线测试阶段单独决定是否加入测试依赖组。该依赖不是
本阶段模型 runtime 必需项，不影响三类公共模型的 load smoke 或资产状态。

### 当前状态

- `YOLO_ASSETS_READY=YES`。
- `DEPTH_ANYTHING_ASSET_READY=YES`。
- `UNIDEPTH_ENV_READY=YES`。
- `PUBLIC_MODEL_ASSETS_READY=YES`。
- `GENVIDEO_OFFICIAL_SCHEMA_VERIFIED=NO`。
- `M6_TO_A2_BRIDGE_READY=YES`。
- `REAL_DATA_DRY_RUN_READY=NO`。
- `FORMAL_TRAINING_READY=NO`。

## 2026-07-25 - P4-C3C-A3-B0 Formal Dataset Selection Audit

### 官方 metadata 与 split

- 复核 GenVideo-100K 后，官方树仍只有 README、Git 属性文件和 15 个
  archive/part；没有 record metadata 或 official split。既有 adapter
  保持 skeleton，`official_schema_verified=false`。
- GenVidBench 官方 Hugging Face 提供的 `Pair1_labels.txt` 和
  `Pair2_labels.txt` 分别有 74,135 与 67,860 条 path/label 记录。论文将
  Pair1 定义为 train、Pair2 定义为 test，但未定义 validation；两文件合计
  141,995 条，并不构成论文 6,784,490 视频的完整 record manifest。
- DeCoF 固定仓库 revision 下的 prompts metadata 有 964 个唯一
  `video_id`。官方 split 为 train=771、validation=96、test=97，三者互斥
  且穷尽全部 ID；metadata 保留 MSVD/MSRVTT source 和 prompt lineage。
- BrokenVideos 论文核验了 frame-level PNG pixel mask、multi-instance
  artifact region 和五类 artifact taxonomy，但项目页 3,141 与论文
  3,254 视频的差异未解释，也没有核验到可下载 record/split/layout
  manifest。

### 路线选择与阻塞项

- 检测主候选选择 DeCoF：record identity、source lineage 和 official
  split 最完整，适合先实现 formal metadata adapter；但数据许可、真实视频
  恢复、archive member 对齐和无 test 下载仍阻塞。
- 空间定位主候选选择 BrokenVideos：pixel mask 与 J/F/J&F 目标直接匹配；
  但数据许可、下载版本、目录对应和 split schema 未核验。
- GenVidBench 作为后续 cross-source/cross-generator 泛化候选；其完整媒体
  依赖 archive 化的官方 HF/VidProM 源，不适合作为首次 20–50 样本
  dry-run。
- GenVideo-100K 继续 deferred，不把原 GenVideo CSV 或 archive 名称当作
  100K record schema。

### 审计基础设施

- 新增四候选统一 registry，所有记录均保持
  `selection_status=candidate`、`data_downloaded=false`、
  `production_adapter_ready=false` 和
  `official_schema_verified=false`。
- 新增 metadata-only registry validator 与 inspection CLI。默认只读
  stdout，不访问网络、不下载媒体、不推断 label/generator、不生成随机
  split。
- 新增安全不变量测试，覆盖评分字段、许可状态、official-test 排除、
  `UNRESOLVED_SCHEMA`、零媒体字节和未核验 schema 不得 ready。
- A3-B0、A3 bridge、A2 与 A1 指定联合回归结果为 `82 passed`。
- 官方 README、LICENSE、API tree、论文、JSONL/TXT/CSV 与 split 审计缓存
  保存在 Git 仓库之外；本阶段媒体下载量为 0 字节。
- 未运行模型、M1–M6、真实流水线或训练。

### 当前状态

- `DATASET_SELECTION_AUDIT_COMPLETE=YES`。
- `PRIMARY_DETECTION_DATASET_SELECTED=YES`。
- `PRIMARY_SPATIAL_DATASET_SELECTED=YES`。
- `SMALL_SAMPLE_SOURCE_IDENTIFIED=YES`。
- `SMALL_SAMPLE_DOWNLOAD_READY=NO`。
- `GENVIDEO_OFFICIAL_SCHEMA_VERIFIED=NO`。
- `REAL_DATA_DRY_RUN_READY=NO`。
- `FORMAL_TRAINING_READY=NO`。
