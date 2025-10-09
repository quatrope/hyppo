# Testing Guide for HYPPO

This document provides detailed testing guidelines and fixture documentation for the HYPPO project.

## Testing Standards

### AAA Pattern (Arrange, Act, Assert)

All tests must follow the AAA pattern with descriptive comments explaining what each section does:

```python
def test_feature_extraction(self, small_hsi):
    """Test that feature extraction produces expected results."""
    # Arrange: Create extractor instance
    extractor = MyFeatureExtractor()

    # Act: Execute extraction
    result = extractor.extract(small_hsi)

    # Assert: Verify correct output structure
    assert "feature_name" in result
    assert result["feature_name"] == expected_value
```

**Comment Guidelines:**
- Keep comments concise (e.g., "Arrange: Create test extractor", "Act: Call extract method")
- Describe *what* is being done, not *why* (the docstring explains the why)
- Use action verbs: "Create", "Execute", "Verify", "Setup", "Call"

### Test Organization

#### Class-Based Grouping

Group related tests in classes to improve organization and enable shared setup:

```python
class TestExtractor:
    """Test cases for the base Extractor class."""

    def test_extract_basic(self, small_hsi):
        """Test basic extraction."""
        # ...

    def test_extract_with_inputs(self, small_hsi):
        """Test extraction with dependency inputs."""
        # ...

class TestFeatureSpace:
    """Test cases for FeatureSpace orchestration."""

    def test_single_extractor(self, small_hsi):
        """Test extraction with single extractor."""
        # ...
```

#### Auxiliary Test Classes

Define concrete implementations outside the test class for testing abstract classes:

```python
class ConcreteExtractor(Extractor):
    """Concrete implementation for testing purposes."""

    def _extract(self, data: HSI, **inputs) -> dict:
        return {"feature": "value"}


class TestExtractor:
    """Test cases for base Extractor."""

    def test_extract_calls_validate(self):
        extractor = ConcreteExtractor()
        # ...
```

## Available Fixtures

Fixtures are defined in `tests/fixtures/hsi.py` and automatically available in all tests via `tests/conftest.py`.

### 1. `small_hsi`

**Dimensions:** 3x3 spatial, 5 spectral bands
**Wavelengths:** [450, 550, 650, 750, 850] nm
**Reflectance range:** 0.1 to 1.0

**Use for:**
- Unit tests requiring minimal data
- Fast execution tests
- Interface validation tests
- Error/edge case tests

**Example:**
```python
def test_extractor_basic(self, small_hsi):
    """Test basic extractor functionality."""
    # Arrange: Create extractor
    extractor = MeanExtractor()

    # Act: Extract features
    result = extractor.extract(small_hsi)

    # Assert: Verify output
    assert "mean" in result
```

### 2. `sample_hsi`

**Dimensions:** 10x10 spatial, 50 spectral bands
**Wavelengths:** 400-1000 nm (linear spacing)
**Reflectance range:** Random values 0.0 to 0.8

**Use for:**
- Integration tests
- Performance tests
- Realistic extraction validation
- Multi-extractor workflows

**Example:**
```python
def test_feature_space_extraction(self, sample_hsi):
    """Test complete feature extraction workflow."""
    # Arrange: Setup feature space with multiple extractors
    fs = FeatureSpace.from_features(["mean", "std", "pca"])

    # Act: Extract all features
    results = fs.extract(sample_hsi)

    # Assert: Verify all features extracted
    assert len(results) == 3
```

### 3. `sample_hsi_data`

**Returns:** Tuple `(reflectance, wavelengths)`
**Type:** Raw numpy arrays (not HSI object)

**Use for:**
- Tests requiring manual HSI construction
- Testing HSI initialization edge cases
- Custom reflectance/wavelength combinations

**Example:**
```python
def test_hsi_construction(self, sample_hsi_data):
    """Test HSI object construction."""
    # Arrange: Get raw data
    reflectance, wavelengths = sample_hsi_data

    # Act: Create HSI with custom parameters
    hsi = HSI(reflectance=reflectance, wavelengths=wavelengths, metadata={"source": "test"})

    # Assert: Verify correct initialization
    assert hsi.reflectance.shape == reflectance.shape
    assert hsi.metadata["source"] == "test"
```

## Coverage Requirements

### Target Coverage

- **Core classes (Extractor, FeatureSpace, HSI):** 100% coverage
- **Extractor implementations:** 95%+ coverage
- **Utility modules:** 90%+ coverage

### Running Coverage

**Full test suite with coverage:**
```bash
pytest tests/ --cov=hyppo --cov-report=term-missing
```

**Specific module coverage:**
```bash
pytest tests/extractor/test_base.py --cov=hyppo.extractor.base --cov-report=term-missing
```

**HTML coverage report:**
```bash
pytest tests/ --cov=hyppo --cov-report=html
# Open htmlcov/index.html in browser
```

### Coverage Interpretation

- **100%**: All lines executed (ideal for core classes)
- **Missing lines**: Lines shown in report that need test coverage
- **Abstract methods**: `pass` statements in abstract methods may show as uncovered (acceptable)

## Reference Implementation

See `tests/extractor/test_base.py` for a complete reference implementation demonstrating:

- 100% coverage of `hyppo.extractor.base.Extractor`
- AAA pattern with descriptive comments
- Class-based test organization
- Fixture usage (`small_hsi`)
- Auxiliary test classes for concrete implementations
- Comprehensive test coverage including:
  - Abstract class instantiation prevention
  - Method execution flow
  - Class method behavior
  - Edge cases (naming conventions, validation, defaults)

## Best Practices

1. **One assertion per logical concept** (but multiple assertions for one concept is OK)
2. **Descriptive test names** starting with `test_` describing what is being tested
3. **Specific docstrings** explaining the test's purpose
4. **Use fixtures** instead of creating test data in each test
5. **Test edge cases** not just happy paths
6. **Keep tests independent** - no test should depend on another test's execution
7. **Fast tests** - use `small_hsi` when possible for speed
8. **Clear failure messages** - assertions should make failures easy to diagnose

## Docstring Standards

### Module Docstrings
- Keep test module docstrings concise and professional
- Format: `"""Tests for [ClassName]."""`
- **DO NOT include:** coverage goals, implementation details, meta-information, or "prompt leaks"
- **Focus on:** what is being tested, not how or how thoroughly

**Examples:**
```python
# Good - concise and professional
"""Tests for DWT1DExtractor."""

# Bad - verbose with implementation details
"""
Comprehensive tests for DWT1DExtractor.
Ensures 100% code coverage and validates all functionality.
"""
```

### Test Docstrings
- One line describing what the test validates
- Be specific about the scenario being tested
- Avoid redundancy with the test name

## Test Consolidation Patterns

### When to Consolidate Tests

Consolidate multiple simple tests into one when they:
- Test the same method/function with different inputs
- Share identical arrange/act/assert structure
- Validate related aspects of the same behavior

### Consolidation Strategy 1: Parametrize

Use `@pytest.mark.parametrize` for tests that differ only in input values:

```python
# Before - 3 separate tests
def test_validate_negative_levels(self, small_hsi):
    extractor = DWT1DExtractor(levels=-1)
    with pytest.raises(ValueError):
        extractor.extract(small_hsi)

def test_validate_zero_levels(self, small_hsi):
    extractor = DWT1DExtractor(levels=0)
    with pytest.raises(ValueError):
        extractor.extract(small_hsi)

def test_validate_float_levels(self, small_hsi):
    extractor = DWT1DExtractor(levels=2.5)
    with pytest.raises(ValueError):
        extractor.extract(small_hsi)

# After - 1 parametrized test
@pytest.mark.parametrize("levels", [-1, 0, 2.5])
def test_validate_invalid_levels(self, small_hsi, levels):
    """Test validation fails with invalid decomposition levels."""
    # Arrange: Create extractor with invalid levels
    extractor = DWT1DExtractor(levels=levels)

    # Act & Assert: Verify validation raises ValueError
    with pytest.raises(ValueError, match="levels must be a positive integer"):
        extractor.extract(small_hsi)
```

### Consolidation Strategy 2: Comprehensive Basic Test

Expand the main "basic" test to include assertions that were in separate simple tests:

```python
def test_extract_basic_with_defaults(self, small_hsi):
    """Test extraction with default parameters."""
    # Arrange: Create extractor with defaults
    extractor = DWT1DExtractor()

    # Act: Execute extraction
    result = extractor.extract(small_hsi)

    # Assert: Verify output structure (was separate test)
    assert "features" in result
    assert "wavelet" in result

    # Assert: Verify default values (was separate test)
    assert result["wavelet"] == "db4"
    assert result["mode"] == "symmetric"

    # Assert: Verify shape correctness (was separate test)
    assert result["features"].shape[0] == small_hsi.height
    assert result["features"].ndim == 3

    # Assert: Verify metadata consistency (was separate test)
    assert result["n_features"] == result["features"].shape[1]
```

### Consolidation Strategy 3: Cross-Product Parametrize

For testing combinations of parameters, use multi-parameter parametrize:

```python
# Instead of separate loops or tests for wavelets and modes
@pytest.mark.parametrize("wavelet,mode", [
    ("haar", "symmetric"),
    ("haar", "periodic"),
    ("db4", "symmetric"),
    ("db4", "zero"),
    ("sym5", "constant"),
])
def test_extract_wavelet_mode_combinations(self, small_hsi, wavelet, mode):
    """Test extraction with cross-product of wavelets and modes."""
    # Arrange: Create extractor with specific combination
    extractor = DWT1DExtractor(wavelet=wavelet, mode=mode)

    # Act: Execute extraction
    result = extractor.extract(small_hsi)

    # Assert: Verify both parameters used correctly
    assert result["wavelet"] == wavelet
    assert result["mode"] == mode
    assert "features" in result
```

### When NOT to Consolidate

Keep tests separate when:
- They test fundamentally different behaviors or code paths
- They require significantly different setup (arrange phase)
- Failure of one test should not obscure failures in others
- The test logic becomes complex or hard to understand when combined
- They serve as documentation for different use cases
