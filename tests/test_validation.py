"""Tests for state validation and pack_state validation."""

from __future__ import annotations

import pytest


class TestValidation:
    def test_valid_state(self, simple_state):
        from crosscat.validate import validate_state

        state, data, _ = simple_state
        errors = validate_state(state, data)
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_assert_valid(self, simple_state):
        from crosscat.validate import assert_valid_state

        state, data, _ = simple_state
        assert_valid_state(state, data)  # Should not raise


class TestPackStateValidation:
    """Tests for pack_state validation."""

    def test_rejects_too_many_views(self, simple_state):
        from crosscat.packed.state import pack_state

        state, _, _ = simple_state
        with pytest.raises(ValueError, match="max_views"):
            pack_state(state, max_views=1)

    def test_rejects_too_many_clusters(self, simple_state):
        from crosscat.packed.state import pack_state

        state, _, _ = simple_state
        with pytest.raises(ValueError, match="max_clusters"):
            pack_state(state, max_clusters=1)
