
"""Depth-blocked cross-validation for ordered subsurface observations."""

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class DepthBlockedKFold(BaseCrossValidator):
    """
    Split samples into contiguous depth-ordered test blocks.

    Each fold holds out one contiguous block of observations.
    Remaining observations form the training set.

    Note:
        This is a within-well validation strategy.
        It does not establish generalization to independent wells.
    """

    def __init__(self, n_splits=5):
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2.")
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)

        if n_samples < self.n_splits:
            raise ValueError(
                "Number of samples must be >= n_splits."
            )

        # Assumes X is already sorted by depth.
        indices = np.arange(n_samples)
        test_blocks = np.array_split(indices, self.n_splits)

        for test_indices in test_blocks:
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_indices] = False

            train_indices = indices[train_mask]

            yield train_indices, test_indices
