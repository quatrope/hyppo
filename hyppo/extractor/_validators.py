"""Reusable validation helpers for extractor parameters."""


def validate_positive_int(value, name):
    """Validate that value is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def validate_non_negative_int(value, name):
    """Validate that value is a non-negative integer."""
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def validate_positive_number(value, name):
    """Validate that value is a positive number."""
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number.")


def validate_non_empty_list(value, name):
    """Validate that value is a non-empty list or tuple."""
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty list or tuple.")


def validate_optional_non_empty_list(value, name):
    """Validate that value is None or a non-empty list."""
    if value is not None and (not isinstance(value, list) or not value):
        raise ValueError(
            f"{name} must be None or a non-empty list of integers."
        )


def validate_window_sizes(window_sizes):
    """Validate that all window sizes are odd integers >= 3."""
    validate_non_empty_list(window_sizes, "window_sizes")
    for w in window_sizes:
        if not isinstance(w, int) or w < 3 or w % 2 == 0:
            raise ValueError(
                f"Each window size must be an odd integer "
                f"\u2265 3. Got: {w}"
            )


def validate_sufficient_bands(data, n_components):
    """Validate that the HSI has enough bands for n_components."""
    n_bands = data.reflectance.shape[-1]
    if n_bands < n_components:
        raise ValueError(
            f"Number of spectral bands ({n_bands}) "
            f"is less than n_components ({n_components})."
        )


def validate_membership(value, valid_set, name):
    """Validate that value is a member of valid_set."""
    if value not in valid_set:
        raise ValueError(
            f"Invalid {name} '{value}'. " f"Valid {name}s: {valid_set}"
        )


def validate_all_in_set(values, valid_set, name):
    """Validate that all values are members of valid_set."""
    for v in values:
        if v not in valid_set:
            raise ValueError(
                f"Invalid {name} '{v}'. " f"Valid {name}s: {valid_set}"
            )


def validate_positive_int_list(values, name):
    """Validate that all elements are positive integers."""
    for v in values:
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"All {name} must be positive integers.")


def validate_band_indices(bands, n_bands):
    """Resolve and validate band indices. None means all bands."""
    if bands is None:
        return list(range(n_bands))
    max_idx = n_bands - 1
    for band in bands:
        if band < 0 or band > max_idx:
            raise ValueError(
                f"Band index {band} is out of range for input "
                f"with {n_bands} bands."
            )
    return list(bands)
