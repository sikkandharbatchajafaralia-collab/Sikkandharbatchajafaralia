# UTriS-Net reproducibility package (research implementation)

This is a runnable PyTorch research scaffold aligned with the supplied manuscript: 16-frame clips, stride 8, 224-pixel frames, 70:15:15 source-video split, ResNet-18 appearance features, appearance/motion reliability gate, temporal self-attention, proximity graph layers, evidential event head, severity head with an interval-membership-inspired fuzzy branch, and multi-task training. It deliberately does **not** fabricate the paper's reported results.

## 1. Dataset sources

- **DoTA (primary):** https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly . The repository provides annotations, metadata, frame extraction instructions, and extracted object tracks/optical-flow resources. The README links the dataset archives. Cite Yao et al., *DoTA: Unsupervised Detection of Traffic Anomaly in Driving Videos*, IEEE TPAMI, DOI: https://doi.org/10.1109/TPAMI.2022.3150763.
- **DADA-2000 (external):** https://github.com/JWFangit/LOTVS-DADA . The repository describes the benchmark and download options; check current access terms and research-only restrictions before use. Cite Fang et al., *DADA: Driver Attention Prediction in Driving Accident Scenarios*, IEEE T-ITS, DOI: https://doi.org/10.1109/TITS.2020.3044678.

The two datasets are not merged at frame level. Train and select the model using DoTA only, then evaluate on a separately mapped DADA test manifest. Dataset licensing/access conditions remain those of the dataset owners.

## 2. Environment

Recommended: Python 3.10/3.11, PyTorch with a CUDA build matching the installed NVIDIA driver, and a GPU with sufficient memory. Install:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

For a CPU-only installation, install the appropriate PyTorch CPU wheel from https://pytorch.org/get-started/locally/ before installing the remaining requirements.

## 3. Data preparation and manifest contract

Extract the DoTA frames using the official repository instructions. The supplied helper converts the released metadata JSON into a manifest:

```bash
python scripts/make_dota_manifest.py --metadata data/DoTA/metadata_train.json data/DoTA/metadata_val.json --frames data/DoTA/frames --out manifests/dota_raw.csv
python scripts/split_manifest.py --input manifests/dota_raw.csv --output manifests/dota.csv --seed 42 --train 0.70 --val 0.15
```

The manifest is a CSV with these columns:

| Column | Meaning |
|---|---|
| `video_id` | Unique source video ID; must be unique across source recordings |
| `frames_dir` | Directory containing chronologically named JPG/PNG frames |
| `onset_frame` | Annotated anomaly onset frame in the extracted-frame coordinate system |
| `event_class` | Dataset-native event class or a harmonized class label |
| `split` | `train`, `val`, or `test`; assigned at video level |
| `severity` | Optional expert-annotated ordinal label: 0 minor, 1 moderate, 2 severe |
| `boxes_json` | Optional path to per-frame detections JSON (see below) |

**Important:** Check that `anomaly_start` in the downloaded DoTA metadata uses the same frame index as your extracted frame sequence. If not, convert it before training. The helper assumes this correspondence and does not silently rescale it. For DADA, construct a separate CSV with the same columns after mapping its native annotations. Use `split=test` for every DADA row and pass the same event-class mapping used for DoTA. The repository does not guess DADA-to-DoTA category mappings.

### Object detections / graph inputs

The graph branch accepts optional `boxes_json`. Generate one file per video from its extracted frame directory:

```bash
python scripts/detect_tracks.py --frames data/DoTA/frames/VIDEO_ID --weights yolov8n.pt --out annotations/VIDEO_ID_boxes.json
```

Add that JSON path to the corresponding manifest row as `boxes_json`. The JSON is a list of records `{"frame": integer, "boxes": [[x1,y1,x2,y2,confidence,class_id], ...]}`; xyxy coordinates must be normalized to [0,1]. The provided detector script exports per-frame detections, **not persistent identity tracks**. The current graph implementation uses geometry/proximity and does not estimate calibrated physical velocity, acceleration, lane topology, or true TTC. Those require camera calibration/track association or an explicit motion-estimation module.

## 4. Training

Edit `configs/default.yaml` to match the actual class mapping and compute budget. Start with one seed:

```bash
python scripts/train.py --config configs/default.yaml --manifest manifests/dota.csv --seed 42
```

Run the five-seed experiment specified in the manuscript:

```bash
python scripts/train.py --config configs/default.yaml --manifest manifests/dota.csv
```

Best weights, loss history, and test metrics are saved under `outputs/utris_net/seed_<seed>/`. Keep the generated split manifest, class mapping, package versions, GPU model, and logs with the experiment.

## 5. External evaluation on DADA-2000

Prepare `manifests/dada.csv` with the exact same event taxonomy and class IDs. Ensure each row has `split=test`. Then:

```bash
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/utris_net/seed_42/best.pt --manifest manifests/dada.csv --split test --out outputs/dada_external.json
```

Do not report external accuracy if class mappings, onset frame definitions, or positive/negative sample construction differ. For robust external evaluation, report class-wise support and the confusion matrix as well as balanced accuracy, macro-F1, and AUC.

## 6. Single-clip inference

```bash
python scripts/infer_clip.py --config configs/default.yaml --checkpoint outputs/utris_net/seed_42/best.pt --frames path/to/clip_frames --out prediction.json
```

This is a minimal clip-level inference path. It does not yet implement a persistent stream state, temporal risk smoothing, actual responsibility boxes, or operator messaging integration.

## 7. What is and is not implemented

Implemented: clip loading, video-level split helper, visual backbone, appearance-motion gate, temporal transformer, geometry/proximity graph message passing, evidential binary event probability/uncertainty, event-category head, optional ordinal severity loss, fuzzy membership-inspired severity fusion, trend head interface, checkpointing, and evaluation metrics.

Not yet a faithful implementation of every manuscript equation: full optical-flow reliability estimation; identity-consistent multi-object tracking; physical TTC from calibrated trajectories; lane/path relationships; exact interval type-2 Karnik-Mendel type reduction; learned risk-velocity/risk-acceleration targets; differentiable alert cost loss; domain-adversarial alignment; STAUC implementation; event-box localisation/responsibility head; per-hour false-alarm calculation from continuous timestamped streams; edge-device energy/FPS benchmarking. These require the corresponding annotations, engineering and validation. The fuzzy branch here is explicitly an interval-membership-inspired approximation, not a claim of exact IT2-FLS type reduction.

## 8. Labels that the source datasets do not automatically supply

DoTA supplies temporal, spatial and categorical anomaly annotations. DADA-2000 supplies accident-oriented clips and attention-related annotations. Neither dataset should be assumed to provide a shared, clinically/engineering validated 3-level severity label or the four-state risk trend (stable/emerging/escalating/declining). Create and publish an annotation protocol, expert raters, adjudication procedure, inter-rater agreement, and per-class counts before training those heads. Until then, keep severity labels as -1 (masked from loss), and do not claim measured severity performance. The same applies to risk-trend labels.

## 9. Reproducibility checklist before paper submission

1. Archive exact dataset versions, download dates, licence terms, and frame extraction commands.
2. Publish manifests (video IDs and labels only if redistribution is permitted), split seed, and class mapping.
3. State how event-onset frames map to extracted frames and how the 2-second anticipation target is created.
4. State whether object boxes are DoTA-provided tracks or detector-generated boxes; provide detector version, confidence threshold, class filter, and tracking algorithm.
5. Define severity and risk-trend annotations independently of model predictions; report annotator agreement.
6. Report five independent seeds as mean ± standard deviation; never substitute illustrative/synthetic plots for test predictions.
7. Compute all metrics from held-out predictions. For false alarms/hour and latency, evaluate continuous videos with timestamps and synchronize GPU timing.
8. Record Python, PyTorch, CUDA, cuDNN, GPU/CPU, RAM, image size, FPS, batch size, optimiser, scheduler, and all loss weights.
