
"""Tests for depth-blocked cross-validation."""

import unittest

import numpy as np
import pandas as pd

from src.depth_blocked_validation import DepthBlockedKFold
from src.nested_validation import run_nested_depth_cv


class TestDepthBlockedKFold(unittest.TestCase):

    def test_all_samples_appear_once_in_test(self):
        X = np.arange(23).reshape(-1, 1)
        cv = DepthBlockedKFold(n_splits=5)

        all_test = []

        for train_idx, test_idx in cv.split(X):
            self.assertEqual(
                len(np.intersect1d(train_idx, test_idx)),
                0,
            )

            self.assertTrue(
                np.all(np.diff(test_idx) == 1)
            )

            all_test.extend(test_idx.tolist())

        self.assertEqual(
            sorted(all_test),
            list(range(len(X))),
        )

    def test_too_few_samples(self):
        X = np.arange(3).reshape(-1, 1)
        cv = DepthBlockedKFold(n_splits=5)

        with self.assertRaises(ValueError):
            list(cv.split(X))

    def test_invalid_split_count(self):
        with self.assertRaises(ValueError):
            DepthBlockedKFold(n_splits=1)


class TestNestedValidation(unittest.TestCase):

    def test_duplicate_depth_rejected(self):
        df = pd.DataFrame({
            "depth": [1, 2, 2, 4, 5, 6, 7, 8],
            "GR": np.arange(8, dtype=float),
            "target": np.arange(8, dtype=float),
        })

        with self.assertRaises(ValueError):
            run_nested_depth_cv(
                data=df,
                feature_columns=["GR"],
                target_column="target",
                depth_column="depth",
                outer_splits=5,
                inner_splits=3,
            )

    def test_target_as_feature_rejected(self):
        df = pd.DataFrame({
            "depth": np.arange(10),
            "GR": np.arange(10),
            "target": np.arange(10),
        })

        with self.assertRaises(ValueError):
            run_nested_depth_cv(
                data=df,
                feature_columns=["GR", "target"],
                target_column="target",
                depth_column="depth",
            )


if __name__ == "__main__":
    unittest.main()
