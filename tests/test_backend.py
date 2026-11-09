from __future__ import annotations

import numpy as np
import pytest

from polyxios import _backend, _core_fallback


def _rows(*, rng: np.random.Generator, n: int, dtype: type) -> np.ndarray:
    """Rows drawn from a few values, so most repeat, with signed zeros and NaN."""
    rows = rng.integers(-2, 3, size=(n, 3)).astype(dtype)
    rows[rng.random((n, 3)) < 0.2] *= -1
    rows[rng.random(n) < 0.02, 1] = np.nan
    return np.ascontiguousarray(rows)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [0, 1, 7, 5000])
def test_merge_rows_groups_as_the_sort_does(dtype, n) -> None:
    """The hash table and the ``np.unique`` fallback agree row for row."""
    rows = _rows(rng=np.random.default_rng(n), n=n, dtype=dtype)

    first, inverse = _backend.merge_rows(rows, seed=12345)
    want_first, want_inverse = _core_fallback.merge_rows(rows, seed=0)

    np.testing.assert_array_equal(first, want_first)
    np.testing.assert_array_equal(inverse, want_inverse)
    assert inverse.dtype == np.int32


def test_merge_rows_does_not_depend_on_its_seed() -> None:
    """The seed spreads the rows over the table; it never changes the groups."""
    rows = _rows(rng=np.random.default_rng(1), n=2000, dtype=np.float64)
    a = _backend.merge_rows(rows, seed=0)
    b = _backend.merge_rows(rows, seed=2**64 - 1)
    for got, want in zip(a, b, strict=True):
        np.testing.assert_array_equal(got, want)


def test_merge_rows_keeps_every_distinct_row() -> None:
    """Rows that all differ grow the table past its first size, losing none."""
    rows = np.ascontiguousarray(np.random.default_rng(2).random((50_000, 3)))
    first, inverse = _backend.merge_rows(rows, seed=7)
    np.testing.assert_array_equal(first, np.arange(50_000))
    np.testing.assert_array_equal(inverse, np.arange(50_000))
