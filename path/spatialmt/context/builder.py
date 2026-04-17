"""
spatialmt.context.builder — CellTableBuilder, CellTable, TrainingTargets

Assembles the ICL input table for a single training step.

Separation of concerns
-----------------------
CellTable       — model inputs only (query labels are absent)
TrainingTargets — query labels only (never fed into the model)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatialmt.data_preparation.dataset import ProcessedDataset


@dataclass
class CellTable:
    """Model input for one training step.

    Fields
    ------
    context_expression  : (n_anchors, n_genes) float32
    context_pseudotime  : (n_anchors,)         float32
    context_soft_labels : (n_anchors, K)       float32
    query_expression    : (n_genes,)           float32
    """
    context_expression:  np.ndarray
    context_pseudotime:  np.ndarray
    context_soft_labels: np.ndarray
    query_expression:    np.ndarray


@dataclass
class TrainingTargets:
    """Query labels used for loss computation only — never fed into the model.

    Fields
    ------
    query_pseudotime  : float32 scalar
    query_soft_labels : (K,) float32
    """
    query_pseudotime:  np.floating
    query_soft_labels: np.ndarray


class CellTableBuilder:
    """Builds (CellTable, TrainingTargets) for a single ICL training step.

    Parameters
    ----------
    dataset : ProcessedDataset
        Schema-validated dataset.  An index mapping cell_id → row position
        is built once at construction and reused across all build() calls.
    """

    def __init__(self, dataset: ProcessedDataset) -> None:
        self._dataset = dataset
        # O(1) lookup: cell_id → integer row index in dataset arrays
        self._id_to_idx: dict[str, int] = {
            cid: i for i, cid in enumerate(dataset.cell_ids)
        }

    # ------------------------------------------------------------------

    def build(
        self,
        query_cell_id: str,
        anchor_ids: list[str],
    ) -> tuple[CellTable, TrainingTargets]:
        """Assemble model inputs and training targets for one step.

        Parameters
        ----------
        query_cell_id : str
            The cell being predicted.
        anchor_ids : list[str]
            Ordered anchor cell ids from ContextSampler.sample().
            Duplicates are allowed (replacement sampling).

        Returns
        -------
        table : CellTable
            Model inputs — expression and pseudotime for context + query
            expression only.  No query labels.
        targets : TrainingTargets
            Query pseudotime and soft labels for loss computation.

        Raises
        ------
        KeyError
            If query_cell_id or any anchor id is not in the dataset.
        """
        ds = self._dataset
        id_to_idx = self._id_to_idx

        # Validate query
        if query_cell_id not in id_to_idx:
            raise KeyError(f"query_cell_id '{query_cell_id}' not found in dataset.")

        # Validate anchors
        for cid in anchor_ids:
            if cid not in id_to_idx:
                raise KeyError(f"anchor id '{cid}' not found in dataset.")

        query_idx = id_to_idx[query_cell_id]

        # --- context arrays (n_anchors rows, positionally aligned to anchor_ids) ---
        if anchor_ids:
            anchor_indices = [id_to_idx[cid] for cid in anchor_ids]
            context_expression  = ds.expression[anchor_indices].astype(np.float32)
            context_pseudotime  = ds.pseudotime[anchor_indices].astype(np.float32)
            context_soft_labels = ds.soft_labels[anchor_indices].astype(np.float32)
        else:
            K = ds.soft_labels.shape[1]
            context_expression  = np.empty((0, ds.n_genes), dtype=np.float32)
            context_pseudotime  = np.empty((0,),            dtype=np.float32)
            context_soft_labels = np.empty((0, K),          dtype=np.float32)

        # --- query inputs (expression only — no labels) ---
        query_expression = ds.expression[query_idx].astype(np.float32)

        # --- targets (kept separate from model inputs) ---
        query_pseudotime  = np.float32(ds.pseudotime[query_idx])
        query_soft_labels = ds.soft_labels[query_idx].astype(np.float32)

        table = CellTable(
            context_expression=context_expression,
            context_pseudotime=context_pseudotime,
            context_soft_labels=context_soft_labels,
            query_expression=query_expression,
        )
        targets = TrainingTargets(
            query_pseudotime=query_pseudotime,
            query_soft_labels=query_soft_labels,
        )
        return table, targets
