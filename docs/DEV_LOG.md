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
