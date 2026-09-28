import numpy as np
import pytest

from scripts.prepare_h1_cell_holdout import split_indices


def test_h1_cell_holdout_is_disjoint_and_stratified():
    labels = np.asarray(["non-targeting"] * 20 + ["AKT2"] * 10 + ["SIN3B"] * 10)
    batches = np.asarray((["a"] * 10 + ["b"] * 10) + (["a"] * 5 + ["b"] * 5) * 2)
    train, val = split_indices(labels, batches, {"AKT2", "SIN3B"}, fraction=0.2, seed=7)
    assert not np.intersect1d(train, val).size
    assert len(train) + len(val) == len(labels)
    for label in ("non-targeting", "AKT2", "SIN3B"):
        for batch in ("a", "b"):
            assert np.any((labels[train] == label) & (batches[train] == batch))
            assert np.any((labels[val] == label) & (batches[val] == batch))
    other_train, other_val = split_indices(labels, batches, {"AKT2", "SIN3B"}, fraction=0.2, seed=7)
    np.testing.assert_array_equal(train, other_train)
    np.testing.assert_array_equal(val, other_val)


def test_h1_cell_holdout_rejects_missing_target():
    with pytest.raises(ValueError, match="absent"):
        split_indices(np.asarray(["non-targeting"] * 4), np.asarray(["a"] * 4), {"AKT2"}, fraction=0.2, seed=7)
