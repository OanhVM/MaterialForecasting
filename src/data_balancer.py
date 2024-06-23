from collections import defaultdict
from enum import IntEnum
from typing import List, Tuple, Dict, Optional

import numpy as np
from numpy import ndarray
from numpy.random import default_rng


# TODO: reintroduce DTW
class BalanceStrategy(IntEnum):
    VARIANCE = 0
    DTW = 1


def get_distr(data: ndarray, n_bin: int = 10) -> Tuple[ndarray, List[ndarray], ndarray]:
    bin_edges = np.histogram_bin_edges(data, bins=n_bin)
    binned_inds = np.digitize(data, np.asarray(bin_edges[:-1]))
    idx_bins = [np.flatnonzero(binned_inds == bin_idx) for bin_idx in range(1, len(bin_edges))]
    idx_bins_sizes = np.asarray([len(idx_bin) for idx_bin in idx_bins])

    return bin_edges, idx_bins, idx_bins_sizes


def _balance_by_variance(inputs: List[ndarray]) -> Tuple[List[str], List[ndarray], List[ndarray]]:
    variances = np.asarray([_input.var() for _input in inputs])

    bin_edges, original_idx_bins, idx_bins_sizes = get_distr(data=variances)

    balanced_coeffs = idx_bins_sizes ** 0.25
    balanced_coeffs /= balanced_coeffs.sum()

    balanced_bin_sizes = np.round(balanced_coeffs * idx_bins_sizes.sum()).astype(int)
    balanced_idx_bins = [
        np.random.choice(idx_bin, size=bin_size) if len(idx_bin) > 0 else np.asarray([])
        for idx_bin, bin_size in zip(original_idx_bins, balanced_bin_sizes)
    ]

    balance_labels = [
        f"{np.format_float_scientific(float(bin_edges[i]), precision=1, exp_digits=1)}"
        f"\n..\n"
        f"{np.format_float_scientific(float(bin_edges[i + 1]), precision=1, exp_digits=1)}"
        for i in range(len(bin_edges) - 1)
    ]

    return balance_labels, original_idx_bins, balanced_idx_bins


def balance_input_label_pairs(
        inputs: List[ndarray], labels: List[ndarray], strategy: BalanceStrategy,
) -> Tuple[
    List[Tuple[ndarray, ndarray]],
    Tuple[List[str], List[ndarray], List[ndarray]],
]:
    if strategy == BalanceStrategy.VARIANCE:
        balance_labels, original_idx_bins, balanced_idx_bins = _balance_by_variance(inputs=inputs)

    else:
        # TODO: balance by DTW
        pass

    balanced_input_label_pairs = []
    for balanced_idx_bin in balanced_idx_bins:
        balanced_input_label_pairs.extend([
            (inputs[i], labels[i]) for i in balanced_idx_bin
        ])

    return (
        balanced_input_label_pairs,
        (balance_labels, original_idx_bins, balanced_idx_bins),
    )


def get_input_label_pairs_per_input_width(
        inputs: List[ndarray], labels: List[ndarray],
        balance_strategy: Optional[BalanceStrategy] = None,
) -> Dict[
    int, List[Tuple[ndarray, ndarray]],
]:
    if balance_strategy is not None:
        input_label_pairs, _ = balance_input_label_pairs(
            inputs=inputs, labels=labels, strategy=balance_strategy,
        )
    else:
        input_label_pairs = list(zip(inputs, labels))

    input_label_pairs_per_input_width = defaultdict(list)
    for _input, label in input_label_pairs:
        input_label_pairs_per_input_width[len(_input)].append((_input, label))

    return input_label_pairs_per_input_width
