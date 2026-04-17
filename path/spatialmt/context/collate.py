"""
spatialmt.context.collate — icl_collate and ICLBatch

Collate function for the ICL training DataLoader.
Not yet implemented — tests define the contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from spatialmt.context.builder import CellTable, TrainingTargets


@dataclass
class ICLBatch:
    """Batched model inputs and training targets as torch.Tensors.

    Fields
    ------
    context_expression  : (B, n_anchors, n_genes)  float32
    context_pseudotime  : (B, n_anchors)            float32
    context_soft_labels : (B, n_anchors, K)         float32
    query_expression    : (B, n_genes)              float32
    query_pseudotime    : (B,)                      float32
    query_soft_labels   : (B, K)                    float32
    """
    context_expression:  torch.Tensor
    context_pseudotime:  torch.Tensor
    context_soft_labels: torch.Tensor
    query_expression:    torch.Tensor
    query_pseudotime:    torch.Tensor
    query_soft_labels:   torch.Tensor


def icl_collate(
    batch: list[tuple[CellTable, TrainingTargets]],
) -> ICLBatch:
    """Stack a list of (CellTable, TrainingTargets) pairs into an ICLBatch.

    Parameters
    ----------
    batch : list of (CellTable, TrainingTargets)
        Output of CellTableBuilder.build(), one item per training example.
        All items must have the same n_anchors — ragged batches are not supported.

    Returns
    -------
    ICLBatch
        All fields are float32 torch.Tensors on CPU.

    Raises
    ------
    ValueError
        If items have mismatched n_anchors (ragged context windows).
    RuntimeError
        Propagated from torch.stack if shapes are inconsistent.
    """
    tables, targets = zip(*batch)

    # Guard against ragged context windows before stacking
    n_anchors = tables[0].context_expression.shape[0]
    for i, t in enumerate(tables[1:], start=1):
        if t.context_expression.shape[0] != n_anchors:
            raise ValueError(
                f"Ragged context: item 0 has {n_anchors} anchors, "
                f"item {i} has {t.context_expression.shape[0]}. "
                "All items in a batch must have the same n_anchors."
            )

    def _stack(arrays) -> torch.Tensor:
        return torch.from_numpy(np.stack(arrays, axis=0))

    return ICLBatch(
        context_expression  = _stack([t.context_expression  for t in tables]),
        context_pseudotime  = _stack([t.context_pseudotime  for t in tables]),
        context_soft_labels = _stack([t.context_soft_labels for t in tables]),
        query_expression    = _stack([t.query_expression    for t in tables]),
        query_pseudotime    = torch.tensor([float(tgt.query_pseudotime)  for tgt in targets], dtype=torch.float32),
        query_soft_labels   = _stack([tgt.query_soft_labels for tgt in targets]),
    )
