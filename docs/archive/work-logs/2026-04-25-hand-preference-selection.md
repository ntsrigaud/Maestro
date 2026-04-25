# Work Log — Hand Preference Selection: `HandSelector` Pipeline Component

**Date:** 2026-04-25  
**Branch:** `feat/hand-preference-selection`  
**Author:** @ntsrigaud

---

## Context

The `hand_preference` configuration field (`"left"`, `"right"`, `"both"`) in
`AppConfig` was defined in the schema but had no effect on inference: the hand
landmark extractor returned at most one hand using an internally selected
handedness filter and produced a single `LandmarkPacket` directly, conflating
extraction with selection. This made it impossible to honour a preferred hand
reliably in two-hand scenarios and prevented preference-agnostic testing of the
extractor in isolation. The branch corrects this by separating extraction from
selection: the extractor now always returns all detected hands in a new
`MultiHandPacket`, and a new `HandSelector` pipeline component applies the
configured preference and produces the single `LandmarkPacket` consumed by the
downstream feature transformer.

---

## Scope of Work

### Domain

- Added `HandLandmarks` dataclass to `src/maestro/domain/entities.py`: per-hand
  data container holding the MediaPipe label, 21 × 3 normalised landmark
  coordinates, handedness classification score, and track confidence ratio.
- Added `MultiHandPacket` dataclass to `src/maestro/domain/entities.py`:
  aggregates a list of `HandLandmarks` (0–2 entries) and a timestamp. Represents
  the raw, unfiltered output of the extractor.

### Infrastructure Refactoring

- Refactored `MediaPipeHandsExtractor` to be preference-agnostic: `max_num_hands`
  is fixed at `2`; all detected hands are packaged into a `MultiHandPacket` via a
  strict `zip(..., strict=True)` of the `multi_handedness` and
  `multi_hand_landmarks` parallel lists. Per-hand `track_conf` is computed as the
  fraction of finite landmark coordinates (`valid_points / 21`).
- Updated `IHandLandmarkExtractor.extract()` return type from `LandmarkPacket` to
  `MultiHandPacket` to match the new extractor contract.

### Pipeline Feature Addition

- Implemented `HandSelector` in `src/maestro/pipeline/hand_selection.py`:
  - Accepts `hand_preference` (`"left"`, `"right"`, `"both"`) in user-facing
    coordinates and applies the MediaPipe selfie-view label inversion
    (`"right"` → MP `"Left"`, `"left"` → MP `"Right"`) via a module-level
    mapping table.
  - `"both"` mode selects the hand with the highest detection confidence from
    the packet.
  - Specific-side mode finds the first hand whose MediaPipe label matches the
    inverted preference; if absent, returns an invalid `LandmarkPacket` with the
    first detected hand's metadata preserved for diagnostics.
  - Validates the selected hand against configurable `min_handedness_conf` and
    `min_track_conf` thresholds before setting `valid=True`.
- Exported `HandSelector` from `src/maestro/pipeline/__init__.py`.

### Application Integration

- Added `hand_selector: HandSelector` to `AppContainer` and `RuntimeDependencies`
  in `src/maestro/app/bootstrap.py` and `src/maestro/app/runtime.py`.
- Updated `build_container()` to instantiate `HandSelector` from
  `config.app.hand_preference` with `min_handedness_conf=0.7` and
  `min_track_conf=0.5`.
- Updated the `Runtime.run()` hot loop to call
  `hand_selector.select(landmark_extractor.extract(frame))` in sequence,
  replacing the direct `landmark_extractor.extract()` → `LandmarkPacket` path.

### Build Script

- Updated `scripts/build_ipn_dataset.py` to instantiate `HandSelector` and
  call `hand_selector.select(extractor.extract(packet))` during IPN dataset
  construction, aligning offline preprocessing with the runtime pipeline.

---

## Technical Changes

**`src/maestro/domain/entities.MultiHandPacket`** holds
`hands: list[HandLandmarks]` (0–2 entries, ordered by MediaPipe detection
confidence) and `timestamp_ms`. Preference selection is explicitly deferred to
`HandSelector` so the extractor remains stateless.

**`src/maestro/infrastructure/mediapipe/hands_extractor.MediaPipeHandsExtractor`**
replaces the previous single-hand extraction path. `zip(result.multi_handedness, result.multi_hand_landmarks, strict=True)` enforces that the two parallel
lists have equal length; a mismatched result would indicate a MediaPipe API
violation and raises `ValueError` immediately rather than silently producing
corrupt data.

**`src/maestro/pipeline/hand_selection.HandSelector`** centralises all
preference filtering and validity logic previously embedded in the extractor.
The `_USER_TO_MEDIAPIPE_LABEL` module-level dict documents the selfie-view
inversion explicitly. When the preferred hand is absent, the fallback branch
preserves the first detected hand's metadata in an invalid `LandmarkPacket`,
allowing callers to log which hand was actually seen without the need for
separate diagnostic instrumentation.

**`src/maestro/app/runtime.RuntimeDependencies`** gains a `hand_selector: HandSelector`
slot. The hot loop changes from:

```python
landmark_packet = self._container.landmark_extractor.extract(frame)
```

to:

```python
multi_packet = self._container.landmark_extractor.extract(frame)
landmark_packet = self._container.hand_selector.select(multi_packet)
```

All downstream code (`feature_transformer`, `sequence_buffer`, `valid_landmarks`
counter) is unchanged — it still receives a `LandmarkPacket`.

---

## Testing and Validation

- Added `tests/unit/test_hand_selector.py` covering: no detections, `"both"`
  mode (highest confidence selected), `"right"` and `"left"` preference with
  correct label inversion, absent-preference fallback, validity threshold
  enforcement at and below boundary, and `LandmarkPacket` field pass-through.
- Updated `tests/unit/test_mediapipe_extractor.py` to assert that `extract()`
  returns a `MultiHandPacket`; removed tests that previously asserted single-hand
  `LandmarkPacket` fields from the extractor.
- Updated `tests/integration/test_phase2_pipeline.py` to supply `MultiHandPacket`
  mock returns from the extractor and exercise `container.hand_selector.select()`
  explicitly, verifying the handedness label-inversion for `hand_preference="right"`.
- Updated `tests/unit/test_ui_runtime_on_state.py` container fixtures to include a
  mocked `hand_selector`.
- Updated latency benchmark tests to include `HandSelector` in the extraction
  timing fixture.
- Applied Ruff formatting to the hand selection and extraction test modules.

---

## Outcome

Hand preference selection is now a first-class, isolated pipeline step with a
documented handedness label-inversion convention. The extractor is stateless and
preference-agnostic, simplifying its unit tests and making it independently
reusable. The `hand_preference` configuration field is fully honoured at runtime
and during IPN dataset construction. No breaking changes were introduced to any
interface downstream of `HandSelector`; all callers continue to receive a
`LandmarkPacket`.
