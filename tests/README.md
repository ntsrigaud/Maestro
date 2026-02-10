# Tests

Unit tests for the Semantic Gesture Understanding System.

## Running Tests

### All Tests

```bash
python -m unittest discover tests -v
```

### Specific Test Module

```bash
python -m unittest tests.test_clip_encoder -v
python -m unittest tests.test_semantic_intent -v
```

### Integration Validation

```bash
KMP_DUPLICATE_LIB_OK=TRUE python scripts/validate_clip_integration.py
```

## Test Coverage

### Layer 2: Representation (`test_clip_encoder.py`)

**16 tests** covering:

- Model initialization and configuration
- Image encoding (uint8, float32)
- Batch encoding
- Deterministic behavior
- Input validation and error handling
- Edge cases (small/large/non-square images)

### Layer 3: Semantic Intent (`test_semantic_intent.py`)

**6 tests** covering:

- Intent extractor initialization
- Semantic matching with embeddings
- Top-K ranking and filtering
- Batch processing
- Input validation and error handling
- Vocabulary structure validation

## Test Principles

1. **Deterministic**: All tests produce consistent results
2. **Minimal Network**: Avoids external dependencies; CLIP weights may be downloaded once then cached
3. **Fast**: Complete test suite runs in <1 minute
4. **Isolated**: Each test is independent
5. **Comprehensive**: Validates both success and failure paths

## Status

✅ **22/22 tests passing** (as of February 10, 2026)

## Adding New Tests

When adding new functionality:

1. Create test file: `tests/test_<component>.py`
2. Import component from `src/`
3. Write test class inheriting from `unittest.TestCase`
4. Test success paths, edge cases, and error handling
5. Ensure deterministic behavior
6. Run full test suite to check for regressions

## Notes

- Tests run on CPU for determinism
- OpenMP conflict on macOS requires `KMP_DUPLICATE_LIB_OK=TRUE`
- Mock CLIP model in semantic intent tests to avoid repeated model loading
