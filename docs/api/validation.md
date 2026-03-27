# Validation

::: crosscat.validate
    options:
      show_source: false

## Overview

State consistency checking utilities.

## `validate_state`

```python
validate_state(state, data=None) -> list[str]
```

Check state consistency. Verifies column assignments, view states, sufficient statistics, and hyperparameters.

**Returns**: List of error messages (empty if valid).

## `assert_valid_state`

```python
assert_valid_state(state, data=None) -> None
```

Raise `ValidationError` if state is invalid. Calls `validate_state` and raises on any errors.

**Raises**: `ValidationError` (subclass of `ValueError`).
