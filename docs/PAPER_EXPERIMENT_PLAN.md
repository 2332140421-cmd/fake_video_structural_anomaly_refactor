# Paper Experiment Plan

## Method under evaluation

The final method path is: shared single-frame and continuous-frame
observations → model-predicted metric depth and camera parameters →
camera-frame metric backprojection → semantic metric residuals → D1/D2/D3
structural residuals → missing-aware temporal fusion → temporal, object,
track, and region anomaly outputs.

This plan does not reinstate the historical object-pair relative
scale–depth (`R_sd`) formulation as the main method. A frozen residual
sequence is the training interface. Its authenticity label is used only for
loss and evaluation, never to produce, normalize, mask, or select residuals.
Every split is fixed at source-video/group level before training.

## A. Overall video detection

- `experiment_id`: VIDEO-DETECTION
- `research_question`: Can the complete 3D spatiotemporal residual sequence
  support video-level forgery classification?
- `required_dataset`: video label, `dataset_name`, `source_video_id`,
  `group_id`, forgery method, scene type, and compression level.
- `required_annotation`: video-level real/fake labels.
- `split_policy`: explicit train/validation/test manifest splits; no
  `source_video_id` or `group_id` crosses a split.
- `metrics`: accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC.
- `saved_files`: per-video logits/probabilities, confusion matrix, raw ROC,
  PR and threshold points, configuration, split snapshot, seed, checkpoint.
- `baselines`: declared conventional video baselines evaluated on identical
  source-level splits.
- `success_criterion`: pre-registered improvement with uncertainty on a
  formal test set, not the six-video engineering smoke.
- `paper_table`: overall video detection.
- `paper_figure`: ROC and PR curves.

## B. Temporal localization

- `experiment_id`: TEMPORAL-LOCALIZATION
- `research_question`: Can the model localize forgery start and end?
- `required_dataset`: original FPS and frame timestamps.
- `required_annotation`: `fake_start_frame`/`fake_end_frame` or frame labels.
- `split_policy`: the source-level video split used for detection.
- `metrics`: frame ROC-AUC, frame PR-AUC, segment hit rate, temporal IoU,
  start/end boundary error, frame F1.
- `saved_files`: `frame_scores.csv`, `clip_scores.csv`,
  `predicted_intervals.csv`, `ground_truth_intervals.csv`,
  `temporal_iou_details.csv`, and threshold-specific localization results.
- `baselines`: frame/clip localization baselines on identical annotations.
- `success_criterion`: pre-registered localization gains on held-out sources.
- `paper_table`: temporal localization.
- `paper_figure`: video timelines and boundary-error distribution.

## C. Spatial, object, and track localization

- `experiment_id`: STRUCTURAL-LOCALIZATION
- `research_question`: Can the method identify anomalous regions, objects,
  and trajectories?
- `required_dataset`: frames with stable object and track identities.
- `required_annotation`: forged mask or box, affected object ID, track ID,
  keypoints/trajectory, and frame-level spatial annotations.
- `split_policy`: source/group-level held-out videos.
- `metrics`: region IoU, pixel AP, object hit rate, track hit rate,
  trajectory deviation, pointing-game accuracy.
- `saved_files`: `object_predictions.csv`, `track_predictions.csv`, region
  predictions, structural heatmaps, per-object/per-track scores, IoU details.
- `baselines`: spatial and trajectory localization baselines.
- `success_criterion`: pre-registered gains with annotation coverage reported.
- `paper_table`: localization metrics.
- `paper_figure`: structural residual heatmaps and trajectory overlays.

If annotations are absent, including on the current six videos, these metrics
are `unavailable`; no artificial IoU or hit-rate value is produced.

## D. Ablation

- `experiment_id`: ABLATION
- `research_question`: Which evidence families and missingness signals matter?
- `required_dataset`: the formal detection dataset with frozen residuals.
- `required_annotation`: video labels; localization labels where applicable.
- `split_policy`: one immutable group/source split across all variants.
- `metrics`: all video metrics, coverage, runtime, and available localization
  metrics.
- `saved_files`: `changed_fields`, base config, seed, split snapshot,
  checkpoint, predictions, metrics, coverage, runtime, source commit.
- `baselines`: Full, w/o semantic metric prior, w/o semantic metric temporal,
  w/o D1, w/o D2, w/o D3, w/o confidence, w/o availability mask, mean pool,
  max pool.
- `success_criterion`: consistent, pre-registered differences on formal data.
- `paper_table`: component ablation.
- `paper_figure`: component/coverage trade-off.

## E. Generalization

- `experiment_id`: GENERALIZATION
- `research_question`: Does the method transfer across sources and domains?
- `required_dataset`: multiple datasets, manipulation methods, scenes,
  resolutions, and codecs, including non-face natural scenes.
- `required_annotation`: video labels and domain metadata.
- `split_policy`: within-dataset, cross-dataset, cross-method, cross-scene,
  resolution, and encoding splits with no group/source leakage.
- `metrics`: video detection metrics with per-domain uncertainty.
- `saved_files`: domain assignments, split snapshots, predictions, metrics,
  checkpoints, and source identities.
- `baselines`: same-data and cross-domain video baselines.
- `success_criterion`: pre-registered transfer behavior on unseen sources.
- `paper_table`: generalization matrix.
- `paper_figure`: domain-wise performance and coverage.

## F. Robustness

- `experiment_id`: ROBUSTNESS
- `research_question`: How stable are predictions and observability under
  H.264 strength, downsampling, FPS reduction, Gaussian blur, noise,
  brightness change, and random frame loss?
- `required_dataset`: formal sources and reproducible derived perturbations.
- `required_annotation`: source labels and perturbation identity/level.
- `split_policy`: every derived sample remains in its source video's split.
- `metrics`: detection metrics, residual coverage, probability shift, runtime.
- `saved_files`: `perturbation_type`, `perturbation_level`,
  `source_sample_id`, `derived_sample_id`, residual coverage, metrics, runtime.
- `baselines`: unperturbed Full and conventional robustness baselines.
- `success_criterion`: pre-registered degradation limits.
- `paper_table`: robustness by perturbation level.
- `paper_figure`: degradation and coverage curves.

## G. Efficiency and observability

- `experiment_id`: EFFICIENCY-OBSERVABILITY
- `research_question`: What is the computational cost and when is each
  structural residual observable?
- `required_dataset`: formal videos and complete provider/residual provenance.
- `required_annotation`: none beyond experiment identities.
- `split_policy`: report all formal splits separately.
- `metrics`: parameters, per-video and per-provider runtime, peak GPU memory,
  per-channel coverage, availability, unavailable reasons, semantic-prior
  observable-object ratio, D1/D2/D3 valid counts, comparison before/after
  missing-aware fusion.
- `saved_files`: runtime, memory, model, channel coverage, sample availability,
  unavailable-reason tables.
- `baselines`: relevant methods measured in the same environment.
- `success_criterion`: complete traceability and practical resource bounds.
- `paper_table`: efficiency and observability.
- `paper_figure`: residual coverage and runtime composition.

## H. Planned tables and figures

Every experiment registry row must contain `experiment_id`,
`research_question`, `required_dataset`, `required_annotation`,
`split_policy`, `metrics`, `saved_files`, `baselines`, `success_criterion`,
`paper_table`, and `paper_figure`. Tables cover overall detection,
localization, ablation, generalization, robustness, and efficiency. Figures
contain raw-data-backed ROC/PR curves, timelines, structural heatmaps,
ablation/coverage plots, robustness curves, and runtime composition. No plot
substitutes for its CSV curve points or structured metric record.
