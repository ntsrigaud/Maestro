# Training Pipeline: Dataset Preparation → Model Training → Hugging Face Deployment

This document describes the offline ML pipeline that produces the production
`two_stream_attn_v1` artefact — from raw IPN Hand video files through to the
published model on Hugging Face Hub. This pipeline is entirely separate from the
real-time inference path: **no webcam or live camera is involved at any stage.**

---

## Pipeline Diagram

```mermaid
flowchart TD
    classDef rawNode    fill:#1a237e,color:#fff,stroke:#283593,stroke-width:2px
    classDef procNode   fill:#004d40,color:#fff,stroke:#00695c,stroke-width:2px
    classDef buildNode  fill:#1b5e20,color:#fff,stroke:#2e7d32,stroke-width:2px
    classDef trainNode  fill:#4a148c,color:#fff,stroke:#6a1b9a,stroke-width:2px
    classDef evalNode   fill:#bf360c,color:#fff,stroke:#870000,stroke-width:2px
    classDef exportNode fill:#880e4f,color:#fff,stroke:#560027,stroke-width:2px
    classDef hfNode     fill:#e65100,color:#fff,stroke:#bf360c,stroke-width:3px
    classDef fileNode   fill:#37474f,color:#fff,stroke:#546e7a,stroke-width:1px,stroke-dasharray:4 2
    classDef configNode fill:#f9a825,color:#000,stroke:#f57f17,stroke-width:1px,stroke-dasharray:4 2

    %% ════════════════════════════════════════════════════════════════════════
    %% PHASE 1 — RAW DATA SOURCES
    %% ════════════════════════════════════════════════════════════════════════

    subgraph PH1["Phase 1 — Raw Data Sources  (IPN Hand Database)"]
        direction TB
        VIDEOS["IPN Hand Videos<br/>200 AVI files · 640×480 · 30 FPS<br/>50 subjects · lab environment"]
        ANNOT["Annotation Files<br/>Annot_TrainList.txt · Annot_TestList.txt<br/>Columns: video, label, t_start, t_end<br/>14 IPN gesture labels total"]
        VOCAB["Vocabulary Config<br/>configs/gestures/vocabulary_hybrid.yaml<br/>production 10-class hybrid vocabulary<br/>no_gesture + point_one/point_two + swiping_*/zooming_*/stop_sign"]
    end

    class VIDEOS rawNode
    class ANNOT rawNode
    class VOCAB configNode

    %% ════════════════════════════════════════════════════════════════════════
    %% PHASE 2 — LANDMARK EXTRACTION  (build_ipn_dataset.py — Phase 1)
    %% ════════════════════════════════════════════════════════════════════════

    subgraph PH2["Phase 2 — Per-Video Landmark Extraction  (parallelised)"]
        direction TB
        DECODE["Video Decoding<br/>cv2.VideoCapture · every frame in source order<br/>Temporal tracker requires sequential processing"]
        MP["MediaPipe Holistic Extractor (legacy solutions API)<br/>sequential per-frame extraction<br/>67 × 3 keypoints per frame + valid mask"]
        HS["Frame Parsing + Validity Mask<br/>graph-schema node packing (67 nodes)<br/>No runtime HandSelector in this offline extraction step"]
        CACHE["Per-Video Cache<br/>cache/&lt;video_stem&gt;_holistic.npz<br/>Arrays: fps · frame_ids · keypoints (N,67,3)<br/>valid_mask (N,67)<br/>Skipped on subsequent runs if cache exists"]
    end

    class DECODE procNode
    class MP procNode
    class HS procNode
    class CACHE fileNode

    VIDEOS --> DECODE --> MP --> HS --> CACHE

    %% ════════════════════════════════════════════════════════════════════════
    %% PHASE 3 — LABEL MAPPING  (build_ipn_dataset.py — Phase 2 preamble)
    %% ════════════════════════════════════════════════════════════════════════

    subgraph PH3["Phase 3 — Label Mapping and Segment Filtering"]
        direction TB
        LMAP["IPN → Vocabulary Mapping<br/>B0A → point_one     B0B → point_two<br/>D0X → no_gesture<br/>G03 → swiping_up      G04 → swiping_down<br/>G05 → swiping_left    G06 → swiping_right<br/>G07 → stop_sign (Jester)<br/>G10 → zooming_in_full_hand       G11 → zooming_out_full_hand"]
        SUBSAMP["Background Class Subsampling<br/>no_gesture sequences subsampled to prevent domination<br/>Residual imbalance handled by config (class caps / optional class weights)"]
    end

    class LMAP buildNode
    class SUBSAMP buildNode

    ANNOT --> LMAP --> SUBSAMP
    VOCAB --> LMAP

    %% ════════════════════════════════════════════════════════════════════════
    %% PHASE 4 — FEATURE ENGINEERING + WINDOWING
    %% ════════════════════════════════════════════════════════════════════════

    subgraph PH4["Phase 4 — Feature Engineering and Sliding Window Assembly"]
        direction TB
        FT["LandmarkFeatureTransformer<br/>per frame · feature-schema-v5<br/>────────────────────────────────────<br/>1. Wrist centering    l_i = l_i - l_0<br/>2. Scale norm         l_i = l_i / dist(l_0, l_9)<br/>3. Orientation norm   DISABLED<br/>4. Velocity           v_i = Δpos / Δt<br/>Output: 147-dim vector per frame"]
        WIN["SlidingWindowSequenceBuffer<br/>window_size = 32 frames (default)<br/>step = 3 (default; configurable)<br/>min_valid_ratio = 0.8<br/>Flat output: (32, 147) numpy array per window"]
        LABEL["Window Labelling<br/>Strategy: majority label in window<br/>Windows with &lt; 80% valid frames → discarded"]
    end

    class FT buildNode
    class WIN buildNode
    class LABEL buildNode

    CACHE --> FT
    SUBSAMP --> WIN
    FT --> WIN --> LABEL

    %% ════════════════════════════════════════════════════════════════════════
    %% PHASE 5 — MIRROR AUGMENTATION + NPZ SERIALISATION
    %% ════════════════════════════════════════════════════════════════════════

    subgraph PH5["Phase 5 — Mirror Augmentation and Dataset Serialisation"]
        direction TB
        MIRR["Mirror Augmentation (optional)<br/>Label-aware horizontal flip<br/>swiping_left ↔ swiping_right are swapped<br/>Doubles training set size"]
        NPZ["Output NPZ Dataset<br/>data/processed/*.npz (configurable output path)<br/>────────────────────────────────────<br/>sequences   (N, 32, 147)  float32<br/>labels      (N,)          str<br/>label_ids   (N,)          int64<br/>recording_ids (N,)        str<br/>valid_ratio (N,)          float32<br/>metadata_json             JSON blob"]
    end

    class MIRR buildNode
    class NPZ fileNode

    LABEL --> MIRR --> NPZ

    %% ════════════════════════════════════════════════════════════════════════
    %% PHASE 6 — DATA LOADING AND SPLIT
    %% ════════════════════════════════════════════════════════════════════════

    subgraph PH6["Phase 6 — Dataset Loading and Stratified Splits"]
        direction TB
        DM["GestureDataModule<br/>Stratified split by class + recording<br/>────────────────────────────────────<br/>Train : 70%  of sequences<br/>Val   : 15%  of sequences<br/>Test  : 15%  of sequences<br/>Recording-level split to reduce leakage"]
        CW["Class Weight Computation<br/>w_c = N_total / (N_c × num_classes)<br/>Applied to CrossEntropyLoss<br/>Counteracts no_gesture class imbalance"]
    end

    class DM trainNode
    class CW trainNode

    NPZ --> DM --> CW

    %% ════════════════════════════════════════════════════════════════════════
    %% PHASE 7 — MODEL CONFIGURATION
    %% ════════════════════════════════════════════════════════════════════════

    subgraph PH7["Phase 7 — Training Configuration"]
        direction LR
        CFG_MODEL["Model Hyperparameters<br/>two_stream_attn.yaml (default)<br/>────────────────────────────<br/>input_size    : 147<br/>hidden_size   : 128 (per direction)<br/>num_layers    : 2<br/>num_heads     : 8<br/>proj_dim      : 96<br/>num_classes   : 10 (runtime profile)"]
        CFG_TRAIN["Training Hyperparameters<br/>────────────────────────────<br/>epochs        : 100 (max)<br/>batch_size    : 64<br/>learning_rate : 0.0002<br/>weight_decay  : 0.0005<br/>grad_clip_norm: 0.5<br/>seed          : 42"]
        CFG_STOP["Early Stopping<br/>────────────────────────────<br/>patience      : 25 epochs<br/>min_delta     : 0.0001<br/>Monitor       : val_loss"]
        CFG_SCHED["LR Scheduler<br/>ReduceLROnPlateau<br/>────────────────────────────<br/>factor        : 0.5<br/>patience      : 3 epochs<br/>min_lr        : 1e-6"]
        CFG_AUG["Augmentation (online)<br/>────────────────────────────<br/>temporal_scale: 0.8 – 1.2×<br/>spatial_jitter: σ = 0.005<br/>apply_prob    : 0.5<br/>mirror_flip   : 0.0  (done offline)"]
    end

    class CFG_MODEL configNode
    class CFG_TRAIN configNode
    class CFG_STOP configNode
    class CFG_SCHED configNode
    class CFG_AUG configNode

    %% ════════════════════════════════════════════════════════════════════════
    %% PHASE 8 — TRAINING LOOP
    %% ════════════════════════════════════════════════════════════════════════

    subgraph PH8["Phase 8 — Training Loop"]
        direction TB
        INIT["Model Initialisation<br/>EnhancedTwoStreamLSTM<br/>BiLSTM + MHA + cross-stream gate · device: CPU / MPS / CUDA<br/>Optimizer: AdamW (default config-driven)"]
        EPOCH["Epoch Loop  (max 100 epochs)<br/>For each mini-batch (size 64):<br/>  → online augmentation applied to training batches<br/>  → forward pass through both enhanced streams<br/>  → CrossEntropyLoss with class weights<br/>  → gradient clip at norm = 1.0<br/>  → AdamW parameter update"]
        VAL["Validation Pass<br/>After each epoch — no augmentation<br/>Metrics: val_loss · val_accuracy · val_macro_f1<br/>ReduceLROnPlateau checks val_loss<br/>Best-epoch checkpoint saved on val_loss improvement"]
        ES["Early Stopping Check<br/>Stop if val_loss has not improved by<br/>min_delta = 0.0001 for configured patience window"]
    end

    class INIT trainNode
    class EPOCH trainNode
    class VAL trainNode
    class ES trainNode

    CW --> INIT
    CFG_MODEL --> INIT
    CFG_TRAIN --> EPOCH
    CFG_STOP --> ES
    CFG_SCHED --> VAL
    CFG_AUG --> EPOCH
    INIT --> EPOCH --> VAL --> ES
    ES -->|"continue"| EPOCH

    %% ════════════════════════════════════════════════════════════════════════
    %% PHASE 9 — EVALUATION
    %% ════════════════════════════════════════════════════════════════════════

    subgraph PH9["Phase 9 — Test Set Evaluation"]
        direction TB
        CKPT["Best Checkpoint Restore<br/>Epoch with lowest val_loss reloaded<br/>Validation metrics are run-specific"]
        EVAL["Test Set Inference (no augmentation)<br/>Full forward pass on held-out 15% split<br/>Metrics computed:<br/>  accuracy · macro F1 · per-class precision/recall<br/>  calibration ECE · confusion matrix"]
        REPORT["Evaluation Report<br/>Per-class F1 · confusion matrix<br/>Swipe directional parity analysis<br/>Overfitting gap (train vs val accuracy)"]
    end

    class CKPT evalNode
    class EVAL evalNode
    class REPORT evalNode

    ES -->|"stopped"| CKPT
    CKPT --> EVAL --> REPORT

    %% ════════════════════════════════════════════════════════════════════════
    %% PHASE 10 — ARTEFACT EXPORT
    %% ════════════════════════════════════════════════════════════════════════

    subgraph PH10["Phase 10 — Inference Artefact Export"]
        direction TB
        EXPORT["export_inference_artifact()<br/>Strips training state (optimizer, scheduler)<br/>Bundles into a single .pt dict:<br/>  model_state_dict<br/>  hyperparameters  (input_size, hidden_size, …)<br/>  class_labels     (task vocabulary)<br/>  feature_schema_version  (feature-schema-v5)<br/>  model_name       (two_stream_attn_v1 default)<br/>  artifact_type    (two_stream_attn_inference default)"]
        PTFILE["Inference Artefact<br/>models/exported/<br/>two_stream_attn_v1_*_inference.pt<br/>Loadable with load_inference_artifact()<br/>No training dependencies required"]
        REG["Model Registry Update (optional)<br/>configs/model_source.yaml<br/>Points to new artefact name<br/>Used by runtime to locate the checkpoint"]
    end

    class EXPORT exportNode
    class PTFILE fileNode
    class REG fileNode

    REPORT --> EXPORT --> PTFILE --> REG

    %% ════════════════════════════════════════════════════════════════════════
    %% PHASE 11 — HUGGING FACE HUB PUBLISH
    %% ════════════════════════════════════════════════════════════════════════

    subgraph PH11["Phase 11 — Hugging Face Hub Publishing"]
        direction TB
        HF_UP["publish_to_huggingface()<br/>Uploads .pt inference artefact<br/>Generates model card with:<br/>  architecture description<br/>  test accuracy · macro F1 · per-class metrics<br/>  feature schema version<br/>  dataset provenance (IPN Hand)<br/>  class label list"]
        HF_REPO["Hugging Face Hub<br/>ntsrigaud/maestro-lstm-hybrid<br/>────────────────────────────<br/>Artefact: two_stream_attn_*_inference.pt<br/>Access: public<br/>Cached locally on first load by runtime"]
        HF_CFG["model_source.yaml updated<br/>Runtime HFModelSource reads this<br/>to locate + download the published artefact<br/>Falls back to local cache if offline"]
    end

    class HF_UP hfNode
    class HF_REPO hfNode
    class HF_CFG fileNode

    REG --> HF_UP --> HF_REPO --> HF_CFG
```

---

## Phase Summary

| Phase                                   | Script / Notebook                                                                     | Key Output                                                              |
| --------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 1 — Raw data                            | External (IPN Hand download)                                                          | 200 AVI videos + annotation TXT files                                   |
| 2 — Landmark extraction                 | `scripts/build_ipn_dataset.py` Phase 1                                                | `cache/<stem>_holistic.npz` per video (parallelised)                    |
| 3 — Label mapping                       | `scripts/build_ipn_dataset.py` Phase 2                                                | Default 10-class vocabulary mapping (configurable via `--vocab-config`) |
| 4 — Feature engineering + windowing     | `build_ipn_dataset.py` → `LandmarkFeatureTransformer` + `SlidingWindowSequenceBuffer` | Sequence windows (N, 32, 147) with majority labels                      |
| 5 — Mirror augmentation + serialisation | `build_ipn_dataset.py`                                                                | `data/processed/*.npz` (output path configurable)                       |
| 6 — Data splits                         | `notebooks/02_model_training_two-stream-lstm.ipynb` → `GestureDataModule`             | Stratified 70/15/15 train/val/test split                                |
| 7 — Configuration                       | `configs/model/two_stream_attn.yaml`                                                  | Default EnhancedTwoStreamLSTM hyperparameters                           |
| 8 — Training loop                       | `notebooks/02_model_training_two-stream-lstm.ipynb` → `train_lstm()`                  | Per-epoch checkpoints; best saved by `val_loss`                         |
| 9 — Evaluation                          | `notebooks/02_model_training_two-stream-lstm.ipynb`                                   | Test metrics, confusion matrix, per-class report                        |
| 10 — Export                             | `training/export_checkpoint.py`                                                       | `two_stream_attn_v1_*_inference.pt` (no training state)                 |
| 11 — Publish                            | `training/hf_publish.py`                                                              | `ntsrigaud/maestro-lstm-hybrid` on Hugging Face Hub                     |

---

## LD-CONG Hybrid Fine-tuning Pipeline (Phase 2 — May 2026)

This pipeline extends the IPN fine-tuned model to the LD-CONG hybrid vocabulary
via Jester pretraining followed by LD-CONG fine-tuning. It runs **after** the IPN pipeline
above and reuses the same pretrained backbone.

```mermaid
flowchart TD
    classDef rawNode    fill:#1a237e,color:#fff,stroke:#283593,stroke-width:2px
    classDef procNode   fill:#004d40,color:#fff,stroke:#00695c,stroke-width:2px
    classDef trainNode  fill:#4a148c,color:#fff,stroke:#6a1b9a,stroke-width:2px
    classDef exportNode fill:#880e4f,color:#fff,stroke:#560027,stroke-width:2px
    classDef fileNode   fill:#37474f,color:#fff,stroke:#546e7a,stroke-width:1px,stroke-dasharray:4 2
    classDef configNode fill:#f9a825,color:#000,stroke:#f57f17,stroke-width:1px,stroke-dasharray:4 2

    JESTER_CKPT["Jester Pretrained Checkpoint<br/>two_stream_attn_v1_&lt;ts&gt;.pt<br/>27-class · W=32 · feature-schema-v5"]
    LDCONG_RAW["LD-CONG Raw Recordings<br/>data/ld-cong/ · CSV annotations"]
    JESTER_CLIPS["Jester W16 Clips<br/>center-cropped W32→16 from Jester dataset"]
    IPN_CLIPS["IPN W16 Clips<br/>center-cropped W32→16<br/>point_one · point_two · unknown"]

    BUILD["build_ld_cong_dataset.py<br/>Merges three sources:<br/>LD-CONG + Jester (W16) + IPN (W16)<br/>→ max_samples_per_class balancing<br/>→ mirror augmentation<br/>→ 10-class exclusion via exclude_classes"]
    NPZ["ld_cong_hybrid_dataset.npz<br/>train / val / test split<br/>classes: 12 (fist + thumb_up excluded)"]

    STAGE_A["Stage A — Head warm-up<br/>backbone frozen · 10 epochs<br/>lr = 1e-3 · new head only"]
    STAGE_B["Stage B — Full fine-tune<br/>all layers · backbone LR warmup<br/>60 epochs · lr = 3e-5<br/>backbone_lr_multiplier = 0.1"]

    CKPT_LD["two_stream_attn_v1_finetune_&lt;ts&gt;.pt"]

    JESTER_CKPT --> STAGE_A
    LDCONG_RAW --> BUILD
    JESTER_CLIPS --> BUILD
    IPN_CLIPS --> BUILD
    BUILD --> NPZ
    NPZ --> STAGE_A --> STAGE_B --> CKPT_LD

    class JESTER_CKPT fileNode
    class LDCONG_RAW rawNode
    class JESTER_CLIPS rawNode
    class IPN_CLIPS rawNode
    class BUILD procNode
    class NPZ fileNode
    class STAGE_A trainNode
    class STAGE_B trainNode
    class CKPT_LD exportNode
```

### Phase Summary — LD-CONG Hybrid

| Phase                        | Script / Notebook                                                | Key Output                                  |
| ---------------------------- | ---------------------------------------------------------------- | ------------------------------------------- |
| 1 — Dataset build            | `scripts/build_ld_cong_dataset.py`                               | `data/processed/ld_cong_hybrid_dataset.npz` |
| 2 — Head warm-up (Stage A)   | `notebooks/08_model_training_jester_ldcong_hybrid_12class.ipynb` | Warm-started head weights                   |
| 3 — Full fine-tune (Stage B) | `notebooks/08_model_training_jester_ldcong_hybrid_12class.ipynb` | `two_stream_attn_v1_finetune_<ts>.pt`       |
| 4 — Evaluation               | `notebooks/08_model_training_jester_ldcong_hybrid_12class.ipynb` | Test metrics, confusion matrix              |

### Key Design Decisions — LD-CONG Hybrid

**Window length mismatch**: Jester was pretrained at W=32; LD-CONG uses W=16.
`EnhancedTwoStreamLSTM` is sequence-length-agnostic (BiLSTM + temporal pooling),
so the pretrained backbone loads and runs cleanly on shorter sequences.

**Class exclusion**: `fist` and `thumb_up` are excluded from training via
`training.exclude_classes` in `configs/model/two_stream_attn_ld_cong_12class.yaml`.
The hybrid dataset NPZ still contains all 14 source classes; exclusion happens at
dataloader construction time inside `finetune_lstm()`.

**Anti-forgetting**: backbone LR warmup (`stage_b_warmup_epochs=5`) and discriminative learning rates (`backbone_lr_multiplier=0.1`) limit catastrophic forgetting during fine-tuning.

---

## Key Design Decisions in the Data Pipeline

### Orientation Normalisation: Disabled

Rotating the landmark frame to a canonical hand orientation before computing velocity
destroys swipe-direction coherence: a leftward and rightward swipe would produce
identical velocity vectors after rotation. This is deliberately disabled at the
`LandmarkFeatureTransformer` level.

### Mirror Augmentation: Offline, Label-Aware

Mirror augmentation (horizontal flip) doubles the dataset size and balances
left-vs-right representation. It is applied **offline** at dataset build time rather
than online during training, because a naive horizontal flip must also swap
`swiping_left` ↔ `swiping_right` labels — a label-aware operation that the standard
augmentation transforms do not support.

### Background Class Subsampling

The IPN Hand `D0X` (non-gesture background) class is substantially more frequent than
the active gesture classes. It is subsampled at dataset build time, and residual
imbalance is controlled by training configuration (for example,
`max_samples_per_class` and optional class weighting).

### Sliding Window Step During Build vs. Inference

During **dataset build**, the sliding window uses the CLI-configured `window_step`
(default `3`) and `window_length` (default `32`).
During **inference**, the runtime uses `configs/inference.yaml` values (currently
`window_step=3`, `window_length=16`). These are independent parameter sources.

---

_See also: [enhanced-two-stream-lstm-architecture.md](enhanced-two-stream-lstm-architecture.md) for the
model architecture diagram, and [data-flow.md](data-flow.md) for the real-time
inference pipeline._
