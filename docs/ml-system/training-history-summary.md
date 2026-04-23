# Training History Summary — Two-Stream LSTM

Preserved from `experiments/runs/` before artefact cleanup. Only the two final
production-candidate runs are recorded here.

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

## Run: `two_stream_lstm_v1_20260420T091839Z` ← **Production model**

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

_Exported by system audit cleanup — 2026-04-20. Artifact uploaded to HF Hub as
`two_stream_lstm_v1_20260420T091839Z_inference.pt` (repo: `ntsrigaud/gesture-control-lstm`)._
