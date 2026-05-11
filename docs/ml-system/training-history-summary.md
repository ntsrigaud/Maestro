# Training History Summary — Legacy Two-Stream LSTM Runs

Preserved from `experiments/runs/` before artefact cleanup. Only the two final
production-candidate legacy runs are recorded here.

> Current runtime authority: `EnhancedTwoStreamLSTM` loaded via
> `two_stream_attn_inference` artifacts (default file:
> `two_stream_attn_v1_20260503T065204Z_inference.pt`).

---

## Run: `two_stream_lstm_v1_20260420T091431Z`

**Model**: `two_stream_lstm_v1` · input 147 · hidden 64 · layers 2 · dropout 0.3 · classes 8  
**Training**: 30 epochs max, early-stop patience 7, batch 64, lr 0.001 (scheduler ×0.5 at plateau)

| Split      | Loss   | Accuracy | Macro F1 |
| ---------- | ------ | -------- | -------- |
| Validation | 0.9975 | 80.17 %  | 79.27 %  |
| Test       | 1.0283 | 78.46 %  | 77.50 %  |

Best epoch: **8** (val loss 0.9975)  
Calibration ECE (val): 0.0433 · Mean confidence: 0.769

### Per-class validation metrics (best epoch)

| Class          | Precision | Recall |
| -------------- | --------- | ------ |
| open_palm_hold | 85.3 %    | 76.6 % |
| swipe_right    | 72.1 %    | 86.2 % |
| swipe_left     | 92.3 %    | 72.4 % |
| swipe_up       | 77.8 %    | 77.3 % |
| swipe_down     | 66.6 %    | 80.4 % |
| zoom_in        | 70.2 %    | 77.2 % |
| zoom_out       | 86.5 %    | 78.6 % |
| unknown        | 84.8 %    | 90.5 % |

---

## Run: `two_stream_lstm_v1_20260420T091839Z` ← **Legacy production snapshot (archived)**

**Model**: `two_stream_lstm_v1` · input 147 · hidden 64 · layers 2 · dropout 0.3 · classes 8  
**Training**: 30 epochs max, early-stop patience 7, batch 64, lr 0.001 (scheduler ×0.5 at plateau)

| Split      | Loss   | Accuracy | Macro F1 |
| ---------- | ------ | -------- | -------- |
| Validation | 0.9955 | 81.07 %  | 80.17 %  |

Best epoch: **11** (val loss 0.9955)

### Per-class validation metrics (best epoch)

| Class          | Precision | Recall |
| -------------- | --------- | ------ |
| open_palm_hold | 82.9 %    | 82.7 % |
| swipe_right    | 76.7 %    | 83.2 % |
| swipe_left     | 85.3 %    | 77.8 % |
| swipe_up       | 87.2 %    | 75.9 % |
| swipe_down     | 74.7 %    | 80.1 % |
| zoom_in        | 74.9 %    | 70.3 % |
| zoom_out       | 85.1 %    | 76.5 % |
| unknown        | 79.3 %    | 93.5 % |

### Epoch-by-epoch validation loss (all 18 epochs logged)

| Epoch  | Train loss | Val loss   | Val acc     | Val F1      |
| ------ | ---------- | ---------- | ----------- | ----------- |
| 1      | 1.3141     | 1.1055     | 76.19 %     | 75.09 %     |
| 2      | 1.0088     | 1.0557     | 76.95 %     | 75.87 %     |
| 3      | 0.9351     | 1.0546     | 76.78 %     | 75.29 %     |
| 4      | 0.8831     | 1.0273     | 79.01 %     | 77.55 %     |
| 5      | 0.8402     | 1.0407     | 78.51 %     | 76.53 %     |
| 6      | 0.8065     | 1.0288     | 79.82 %     | 79.11 %     |
| 7      | 0.7772     | 1.0066     | 80.50 %     | 79.48 %     |
| 8      | 0.7550     | 1.0230     | 78.96 %     | 78.08 %     |
| 9      | 0.7395     | 1.0159     | 80.48 %     | 79.48 %     |
| 10     | 0.7217     | 1.0224     | 78.51 %     | 77.60 %     |
| **11** | **0.7089** | **0.9955** | **81.07 %** | **80.17 %** |
| 12     | 0.6978     | 1.0146     | 79.98 %     | 78.67 %     |
| 13     | 0.6824     | 1.0427     | 80.81 %     | 79.84 %     |
| 14     | 0.6787     | 1.0396     | 78.89 %     | 77.21 %     |
| 15     | 0.6694     | 1.0471     | 78.94 %     | 77.86 %     |
| 16     | 0.6299     | 1.0154     | 81.02 %     | 79.74 %     |
| 17     | 0.6213     | 1.0290     | 80.93 %     | 79.91 %     |
| 18     | 0.6172     | 1.0255     | 80.22 %     | 78.86 %     |

---

_Exported by system audit cleanup — 2026-04-20. Legacy artifact uploaded to HF Hub as
`two_stream_lstm_v1_20260420T091839Z_inference.pt` (repo: `ntsrigaud/maestro-lstm`)._

---

## May 2026 Experimental Series

Selection criteria for runs appended below:

- Architecturally distinct families (`stgcn_lstm`, `dual_stgcn_lstm`, `two_stream_lstm`, `two_stream_attn`).
- Final or near-final candidates per family plus representative failed/regressed runs.
- Complete metrics present in source JSON (`val_report`, `test_report`, per-class metrics, confusion matrix, mean confidence).

---

## Run: `stgcn_lstm_v1_20260501T144620Z`

**Model**: `stgcn_lstm_v1` · input 126 · hidden 128 · layers 2 · dropout 0.3 · classes 8  
**Training**: 40 epochs max, batch 32, lr 0.001, scheduler ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-06)

| Split      | Loss   | Accuracy | Macro F1 |
| ---------- | ------ | -------- | -------- |
| Validation | 1.1309 | 71.36 %  | 70.88 %  |
| Test       | 1.1382 | 71.48 %  | 70.92 %  |

Best epoch: **10**

Calibration ECE (val): N/A (not present in source JSON) · Mean confidence: 0.716

### Per-class validation metrics

| Class          | Precision | Recall  |
| -------------- | --------- | ------- |
| open_palm_hold | 68.68 %   | 75.63 % |
| swipe_right    | 71.40 %   | 73.78 % |
| swipe_left     | 87.98 %   | 77.12 % |
| swipe_up       | 70.86 %   | 69.55 % |
| swipe_down     | 67.08 %   | 67.28 % |
| zoom_in        | 74.77 %   | 61.03 % |
| zoom_out       | 50.38 %   | 64.94 % |
| unknown        | 80.73 %   | 77.25 % |

### Key Observations

- Overfitting behavior: final train accuracy 91.33% vs final val Macro F1 73.61% (gap indicates generalization pressure after early optimum).
- Weak classes: `zoom_in` and `zoom_out` had the lowest recalls.
- Confusion trend: matrix shows recurrent cross-confusion among dynamic directional classes and zoom-related classes.
- Calibration quality: ECE unavailable in source; mean confidence is moderate (0.716).
- Comparison: this STGCN baseline underperformed later dual-STGCN and final enhanced-attention candidates.

---

## Run: `dual_stgcn_lstm_v1_20260502T021635Z`

**Model**: `dual_stgcn_lstm_v1` · input 201 · hidden 256 · layers 2 · dropout 0.3 · classes 8  
**Training**: 40 epochs max, batch 32, lr 0.001, scheduler ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-06)

| Split      | Loss   | Accuracy | Macro F1 |
| ---------- | ------ | -------- | -------- |
| Validation | 1.2142 | 69.05 %  | 68.58 %  |
| Test       | 1.3000 | 66.71 %  | 65.77 %  |

Best epoch: **27**

Calibration ECE (val): N/A (not present in source JSON) · Mean confidence: 0.681

### Per-class validation metrics

| Class          | Precision | Recall  |
| -------------- | --------- | ------- |
| open_palm_hold | 68.30 %   | 73.79 % |
| swipe_right    | 72.97 %   | 75.54 % |
| swipe_left     | 87.20 %   | 73.52 % |
| swipe_up       | 81.62 %   | 76.90 % |
| swipe_down     | 65.67 %   | 60.80 % |
| zoom_in        | 63.92 %   | 55.15 % |
| zoom_out       | 35.58 %   | 54.87 % |
| unknown        | 84.22 %   | 74.39 % |

### Key Observations

- Overfitting behavior: final train accuracy 89.03% vs final val Macro F1 70.14%.
- Weak classes: `zoom_out` and `zoom_in` were clear bottlenecks.
- Confusion trend: confusion matrix indicates sustained mixing between zoom classes and motion classes.
- Calibration quality: ECE unavailable; mean confidence is low among selected near-final candidates.
- Comparison: this run is a representative failed dual-STGCN configuration and serves as the lower bound for that family.

---

## Run: `dual_stgcn_lstm_v1_20260502T025416Z`

**Model**: `dual_stgcn_lstm_v1` · input 201 · hidden 256 · layers 2 · dropout 0.3 · classes 8  
**Training**: 40 epochs max, batch 32, lr 0.001, scheduler ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-06)

| Split      | Loss   | Accuracy | Macro F1 |
| ---------- | ------ | -------- | -------- |
| Validation | 1.0035 | 81.01 %  | 80.47 %  |
| Test       | 1.1177 | 77.46 %  | 76.01 %  |

Best epoch: **30**

Calibration ECE (val): N/A (not present in source JSON) · Mean confidence: 0.807

### Per-class validation metrics

| Class          | Precision | Recall  |
| -------------- | --------- | ------- |
| open_palm_hold | 73.83 %   | 79.77 % |
| swipe_right    | 80.82 %   | 76.71 % |
| swipe_left     | 79.21 %   | 82.26 % |
| swipe_up       | 83.63 %   | 87.14 % |
| swipe_down     | 77.13 %   | 69.75 % |
| zoom_in        | 77.67 %   | 81.86 % |
| zoom_out       | 81.62 %   | 72.08 % |
| unknown        | 92.29 %   | 93.24 % |

### Key Observations

- Overfitting behavior: final train accuracy 99.27% vs final val Macro F1 79.75% (high train/val gap despite strong val peak).
- Weak classes: `swipe_down` and `zoom_out` remained lower-recall classes.
- Confusion trend: matrix shows improved class separation overall, with residual confusion in directional pairs.
- Calibration quality: ECE unavailable; mean confidence is high (0.807).
- Comparison: this is the strongest dual-STGCN run in the audited set and an offline competitor to enhanced two-stream.

---

## Run: `dual_stgcn_lstm_v1_20260502T035626Z`

**Model**: `dual_stgcn_lstm_v1` · input 201 · hidden 256 · layers 2 · dropout 0.3 · classes 8  
**Training**: 80 epochs max, batch 32, lr 0.0001, scheduler ReduceLROnPlateau (factor=0.5, patience=10, min_lr=1e-06)

| Split      | Loss   | Accuracy | Macro F1 |
| ---------- | ------ | -------- | -------- |
| Validation | 1.1056 | 76.60 %  | 76.02 %  |
| Test       | 1.2386 | 74.05 %  | 72.22 %  |

Best epoch: **48**

Calibration ECE (val): N/A (not present in source JSON) · Mean confidence: 0.891

### Per-class validation metrics

| Class          | Precision | Recall  |
| -------------- | --------- | ------- |
| open_palm_hold | 71.67 %   | 77.93 % |
| swipe_right    | 78.36 %   | 72.99 % |
| swipe_left     | 75.83 %   | 76.61 % |
| swipe_up       | 78.16 %   | 84.51 % |
| swipe_down     | 74.49 %   | 67.59 % |
| zoom_in        | 79.67 %   | 72.06 % |
| zoom_out       | 61.39 %   | 71.75 % |
| unknown        | 89.72 %   | 85.86 % |

### Key Observations

- Overfitting behavior: final train accuracy 97.20% vs final val Macro F1 75.54%.
- Weak classes: `swipe_down` and `zoom_out` persisted as lower-recall classes.
- Confusion trend: matrix suggests residual directional confusion and zoom-related spillover into unknown.
- Calibration quality: ECE unavailable; mean confidence is high (0.891), indicating potential over-confidence risk.
- Comparison: improved over failed dual-STGCN setups but below the best dual-STGCN and top enhanced-attn runs.

---

## Run: `two_stream_lstm_v1_20260502T073950Z`

**Model**: `two_stream_lstm_v1` · input 147 · hidden 64 · layers 2 · dropout 0.3 · classes 8  
**Training**: 80 epochs max, batch 64, lr 0.0001, scheduler ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-06)

| Split      | Loss   | Accuracy | Macro F1 |
| ---------- | ------ | -------- | -------- |
| Validation | 0.8685 | 79.47 %  | 78.12 %  |
| Test       | 0.8091 | 77.86 %  | 77.49 %  |

Best epoch: **33**

Calibration ECE (val): N/A (not present in source JSON) · Mean confidence: 0.914

### Per-class validation metrics

| Class          | Precision | Recall  |
| -------------- | --------- | ------- |
| open_palm_hold | 88.07 %   | 84.30 % |
| swipe_right    | 61.27 %   | 77.59 % |
| swipe_left     | 83.81 %   | 79.94 % |
| swipe_up       | 71.79 %   | 83.36 % |
| swipe_down     | 73.89 %   | 87.93 % |
| zoom_in        | 71.67 %   | 79.62 % |
| zoom_out       | 79.19 %   | 61.39 % |
| unknown        | 91.88 %   | 81.67 % |

### Key Observations

- Overfitting behavior: final train accuracy 97.19% vs final val Macro F1 76.43%.
- Weak classes: `zoom_out` and `swipe_right` recalls trailed other classes.
- Confusion trend: matrix indicates directional class overlap and zoom_out drift into other motion classes.
- Calibration quality: ECE unavailable; very high mean confidence (0.914).
- Comparison: strongest legacy two_stream_lstm run in this May set, but below best enhanced-attn peak.

---

## Run: `two_stream_lstm_v1_20260502T091153Z`

**Model**: `two_stream_lstm_v1` · input 147 · hidden 64 · layers 2 · dropout 0.3 · classes 12  
**Training**: 80 epochs max, batch 64, lr 0.0001, scheduler ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-06)

| Split      | Loss   | Accuracy | Macro F1 |
| ---------- | ------ | -------- | -------- |
| Validation | 1.0184 | 69.16 %  | 67.63 %  |
| Test       | 1.0091 | 68.75 %  | 67.83 %  |

Best epoch: **27**

Calibration ECE (val): N/A (not present in source JSON) · Mean confidence: 0.827

### Per-class validation metrics

| Class            | Precision | Recall  |
| ---------------- | --------- | ------- |
| open_palm_hold   | 81.28 %   | 80.13 % |
| swipe_right      | 68.26 %   | 74.01 % |
| swipe_left       | 83.63 %   | 81.20 % |
| swipe_up         | 78.78 %   | 76.67 % |
| swipe_down       | 78.28 %   | 65.21 % |
| zoom_in          | 67.52 %   | 77.19 % |
| zoom_out         | 68.22 %   | 79.79 % |
| click_one        | 43.74 %   | 45.75 % |
| click_two        | 48.87 %   | 35.90 % |
| double_click_one | 64.48 %   | 62.68 % |
| double_click_two | 57.23 %   | 69.16 % |
| unknown          | 72.03 %   | 69.41 % |

### Key Observations

- Overfitting behavior: final train accuracy 90.41% vs final val Macro F1 66.20%.
- Weak classes: `click_two` and `click_one` recalls were the principal failure points.
- Confusion trend: matrix confirms heavy confusion across click/double-click groups.
- Calibration quality: ECE unavailable; mean confidence 0.827.
- Comparison: this is the weakest run among audited two_stream_lstm variants and marks the 12-class regression boundary.

---

## Run: `two_stream_attn_v1_20260503T041350Z`

**Model**: `two_stream_attn_v1` · input 147 · hidden 96 · layers 2 · dropout 0.35 · classes 12  
**Training**: 100 epochs max, batch 128, lr 9e-05, scheduler ReduceLROnPlateau (factor=0.5, patience=8, min_lr=1e-06)

| Split      | Loss   | Accuracy | Macro F1 |
| ---------- | ------ | -------- | -------- |
| Validation | 1.1916 | 75.72 %  | 73.65 %  |
| Test       | 1.2573 | 71.94 %  | 70.79 %  |

Best epoch: **70**

Calibration ECE (val): N/A (not present in source JSON) · Mean confidence: 0.864

### Per-class validation metrics

| Class            | Precision | Recall  |
| ---------------- | --------- | ------- |
| open_palm_hold   | 91.41 %   | 88.58 % |
| swipe_right      | 83.74 %   | 80.29 % |
| swipe_left       | 85.43 %   | 87.17 % |
| swipe_up         | 87.16 %   | 90.03 % |
| swipe_down       | 91.81 %   | 82.60 % |
| zoom_in          | 79.44 %   | 83.63 % |
| zoom_out         | 81.59 %   | 84.96 % |
| click_one        | 47.70 %   | 42.35 % |
| click_two        | 49.41 %   | 37.69 % |
| double_click_one | 57.96 %   | 52.87 % |
| double_click_two | 57.47 %   | 69.26 % |
| unknown          | 74.50 %   | 85.42 % |

### Key Observations

- Overfitting behavior: final train accuracy 99.99% vs final val Macro F1 73.17%.
- Weak classes: `click_two` and `click_one` remained the dominant low-recall classes.
- Confusion trend: confusion matrix shows dense confusion among click-family labels despite strong core gesture performance.
- Calibration quality: ECE unavailable; mean confidence 0.864.
- Comparison: stronger than early 12-class attention runs, but still clearly below the final 8-class cluster.

---

## Run: `two_stream_attn_v1_20260503T050237Z`

**Model**: `two_stream_attn_v1` · input 147 · hidden 96 · layers 2 · dropout 0.35 · classes 12  
**Training**: 100 epochs max, batch 64, lr 9e-05, scheduler ReduceLROnPlateau (factor=0.5, patience=8, min_lr=1e-06)

| Split      | Loss   | Accuracy | Macro F1 |
| ---------- | ------ | -------- | -------- |
| Validation | 1.3077 | 70.25 %  | 68.88 %  |
| Test       | 1.4184 | 68.28 %  | 67.28 %  |

Best epoch: **44**

Calibration ECE (val): N/A (not present in source JSON) · Mean confidence: 0.814

### Per-class validation metrics

| Class            | Precision | Recall  |
| ---------------- | --------- | ------- |
| open_palm_hold   | 82.29 %   | 85.56 % |
| swipe_right      | 86.71 %   | 74.15 % |
| swipe_left       | 82.90 %   | 82.37 % |
| swipe_up         | 77.17 %   | 73.17 % |
| swipe_down       | 80.74 %   | 82.26 % |
| zoom_in          | 64.83 %   | 81.63 % |
| zoom_out         | 70.86 %   | 83.09 % |
| click_one        | 52.48 %   | 41.69 % |
| click_two        | 37.86 %   | 38.17 % |
| double_click_one | 55.93 %   | 72.56 % |
| double_click_two | 56.25 %   | 51.72 % |
| unknown          | 83.16 %   | 65.29 % |

### Key Observations

- Overfitting behavior: final train accuracy 99.64% vs final val Macro F1 68.50% (largest observed gap in selected attention runs).
- Weak classes: `click_two` and `click_one` recalls remained critically low.
- Confusion trend: matrix indicates severe click-family overlap and unknown leakage.
- Calibration quality: ECE unavailable; mean confidence 0.814.
- Comparison: this is the weakest selected `two_stream_attn` run and is retained as a representative regression case.

---

## Run: `two_stream_attn_v1_20260503T061857Z`

**Model**: `two_stream_attn_v1` · input 147 · hidden 96 · layers 2 · dropout 0.35 · classes 8  
**Training**: 100 epochs max, batch 64, lr 9e-05, scheduler ReduceLROnPlateau (factor=0.5, patience=8, min_lr=1e-06)

| Split      | Loss   | Accuracy | Macro F1 |
| ---------- | ------ | -------- | -------- |
| Validation | 0.8953 | 79.53 %  | 79.31 %  |
| Test       | 0.8422 | 82.97 %  | 81.00 %  |

Best epoch: **50**

Calibration ECE (val): N/A (not present in source JSON) · Mean confidence: 0.846

### Per-class validation metrics

| Class          | Precision | Recall  |
| -------------- | --------- | ------- |
| open_palm_hold | 81.25 %   | 83.69 % |
| swipe_right    | 77.71 %   | 88.69 % |
| swipe_left     | 81.67 %   | 85.22 % |
| swipe_up       | 67.30 %   | 78.66 % |
| swipe_down     | 86.79 %   | 72.43 % |
| zoom_in        | 68.93 %   | 73.95 % |
| zoom_out       | 86.92 %   | 71.81 % |
| unknown        | 83.57 %   | 84.97 % |

### Key Observations

- Overfitting behavior: final train accuracy 99.90% vs final val Macro F1 78.49%.
- Weak classes: `zoom_out` and `swipe_down` remained the lowest-recall classes.
- Confusion trend: confusion matrix shows improved broad separation, but zoom/directional spillover still present.
- Calibration quality: ECE unavailable; mean confidence 0.846.
- Comparison: strong 8-class near-final run and part of the high-performing final cluster.

---

## Run: `two_stream_attn_v1_20260503T064347Z`

**Model**: `two_stream_attn_v1` · input 147 · hidden 96 · layers 2 · dropout 0.35 · classes 8  
**Training**: 100 epochs max, batch 64, lr 9e-05, scheduler ReduceLROnPlateau (factor=0.5, patience=8, min_lr=1e-06)

| Split      | Loss   | Accuracy | Macro F1 |
| ---------- | ------ | -------- | -------- |
| Validation | 0.8722 | 81.34 %  | 80.82 %  |
| Test       | 0.8282 | 83.78 %  | 82.02 %  |

Best epoch: **38**

Calibration ECE (val): N/A (not present in source JSON) · Mean confidence: 0.849

### Per-class validation metrics

| Class          | Precision | Recall  |
| -------------- | --------- | ------- |
| open_palm_hold | 79.96 %   | 84.76 % |
| swipe_right    | 81.25 %   | 82.69 % |
| swipe_left     | 77.84 %   | 87.54 % |
| swipe_up       | 76.04 %   | 75.80 % |
| swipe_down     | 89.12 %   | 73.93 % |
| zoom_in        | 72.68 %   | 72.27 % |
| zoom_out       | 88.52 %   | 83.40 % |
| unknown        | 82.74 %   | 87.06 % |

### Key Observations

- Overfitting behavior: final train accuracy 99.92% vs final val Macro F1 80.30%.
- Weak classes: `zoom_in` and `swipe_down` remained the two lowest recalls.
- Confusion trend: matrix suggests residual directional confusion but materially cleaner class boundaries overall.
- Calibration quality: ECE unavailable; mean confidence 0.849.
- Comparison: best validation Macro F1 among selected enhanced-attention runs and current top candidate by validation criterion.

---

## Run: `two_stream_attn_v1_20260503T065204Z`

**Model**: `two_stream_attn_v1` · input 147 · hidden 96 · layers 2 · dropout 0.35 · classes 8  
**Training**: 100 epochs max, batch 64, lr 9e-05, scheduler ReduceLROnPlateau (factor=0.5, patience=8, min_lr=1e-06)

| Split      | Loss   | Accuracy | Macro F1 |
| ---------- | ------ | -------- | -------- |
| Validation | 0.9491 | 79.12 %  | 78.76 %  |
| Test       | 0.8330 | 83.10 %  | 81.66 %  |

Best epoch: **36**

Calibration ECE (val): N/A (not present in source JSON) · Mean confidence: 0.850

### Per-class validation metrics

| Class          | Precision | Recall  |
| -------------- | --------- | ------- |
| open_palm_hold | 81.20 %   | 81.55 % |
| swipe_right    | 75.47 %   | 84.81 % |
| swipe_left     | 76.62 %   | 85.51 % |
| swipe_up       | 70.18 %   | 76.43 % |
| swipe_down     | 91.28 %   | 73.43 % |
| zoom_in        | 70.44 %   | 71.43 % |
| zoom_out       | 88.65 %   | 72.39 % |
| unknown        | 78.04 %   | 88.31 % |

### Key Observations

- Overfitting behavior: final train accuracy 99.83% vs final val Macro F1 77.61%.
- Weak classes: `zoom_in` and `zoom_out` remained lowest recall in this final run.
- Confusion trend: matrix shows remaining zoom-direction trade-offs but strong unknown separation.
- Calibration quality: ECE unavailable; mean confidence 0.850.
- Comparison: this run remains a strong reference point in the selected two-stream attention series, slightly below the best validation peak but with strong test Macro F1.

---

## Comparative Analysis

- Best performing model (by validation Macro F1): `two_stream_attn_v1_20260503T064347Z` (**80.82%**).
- Most stable model family (among audited multi-run families): `two_stream_attn` in the selected set (population std of selected validation Macro F1 ≈ **0.0442**), with the near-final subgroup (`061857Z`, `064347Z`, `065204Z`) showing tighter stability (std ≈ **0.0087**).
- Most improved classes across selected runs (core gesture subset in this report, min-to-max recall spread): `zoom_out` (+0.3009), `zoom_in` (+0.2848), `swipe_down` (+0.2713).
- Persistent failure cases: click-family classes (`click_one`, `click_two`) in extended-vocabulary setups; `zoom_in`/`zoom_out` remain recurrent weak points in several runs.
- Production recommendation: keep `EnhancedTwoStreamLSTM` as authoritative runtime path, prioritize top-performing recent checkpoints, and continue targeted error reduction on zoom/directional confusions plus click-family robustness in extended-vocabulary experiments.
