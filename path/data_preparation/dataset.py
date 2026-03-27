"""
path/spatialmt/data/dataset.py

ProcessedDataset  — Jain et al. days 5–21 WT timecourse (training data)
PerturbationDataset — He et al. day-45 GLI3-KO (perturbation inference target)

Pipeline (both classes)
-----------------------
  _filter_cells
      → _remove_blacklist_genes
          → _normalise           (CP10k + log1p, sparse-native)
              → _select_genes    (HVG or frozen manifest)
                  → check_memory_feasibility
                      → _build   (densify, assemble dataclass)

"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from typing import Literal

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse

from spatialmt.errors import ConfigurationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory feasibility — module-level (Decision 16A)
# ---------------------------------------------------------------------------

def check_memory_feasibility(
    n_cells: int,
    n_genes: int,
    gpu_memory_bytes: int,
    label: str = "dataset",
) -> None:
    """
    Raise ConfigurationError if the dense float32 expression matrix would
    exceed 80% of the available GPU memory.

    Called from both ProcessedDataset._build() and PerturbationDataset._build()
    with the `label` parameter making errors actionable.

    Parameters
    ----------
    n_cells:
        Number of cells after QC filtering.
    n_genes:
        Number of genes after HVG / manifest selection.
    gpu_memory_bytes:
        Total GPU memory in bytes (e.g. 8 * 1024**3 for 8 GB).
    label:
        Human-readable name for the dataset being checked, included in
        the error message ("ProcessedDataset" or "PerturbationDataset").
    """
    required_bytes = n_cells * n_genes * 4  # float32 = 4 bytes
    limit_bytes    = gpu_memory_bytes * 0.8

    if required_bytes > limit_bytes:
        raise ConfigurationError(
            f"{label}: dense expression matrix would require "
            f"{required_bytes / 1e9:.2f} GB but only "
            f"{limit_bytes / 1e9:.2f} GB is available "
            f"(80 % of {gpu_memory_bytes / 1e9:.2f} GB). "
            f"Reduce DataConfig.max_genes or use a higher hardware tier."
        )


# ---------------------------------------------------------------------------
# Manifest hash — module-level (Decision 15A)
# ---------------------------------------------------------------------------

def _compute_manifest_hash(gene_names: list[str]) -> str:
    """
    SHA-256 of the sorted gene name list.

    Only gene_names are hashed (not preprocessing_config) so that
    wt.manifest_hash == ko.manifest_hash holds even when QC configs differ
    between datasets — which is expected and correct.

    Parameters
    ----------
    gene_names:
        List of gene name strings in any order. Sorted before hashing so
        that ordering differences do not produce spurious mismatches.
    """
    payload = json.dumps(sorted(gene_names), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# GENOTYPE_MAP and _parse_genotype (Decision 7A)
# ---------------------------------------------------------------------------

#: Allowlist mapping raw AnnData organoid-column labels to canonical literals.
#: Add new organoid lines here — never perform ad-hoc string matching elsewhere.
GENOTYPE_MAP: dict[str, Literal["WT", "GLI3_KO"]] = {
    "H9":        "WT",
    "H9_GLI3KO": "GLI3_KO",
}


def _parse_genotype(raw_label: str) -> Literal["WT", "GLI3_KO"]:
    """
    Map a raw AnnData organoid column value to its canonical genotype literal.

    Raises
    ------
    ValueError
        If raw_label is not present in GENOTYPE_MAP. The error message
        names both the offending label and GENOTYPE_MAP so the user knows
        exactly where to register the new label.
    """
    if not raw_label:
        raise ValueError(
            f"Empty string is not a valid genotype label. "
            f"Add the correct label to GENOTYPE_MAP in spatialmt.data.dataset."
        )
    try:
        return GENOTYPE_MAP[raw_label]
    except KeyError:
        raise ValueError(
            f"Unknown genotype label {raw_label!r}. "
            f"Add it to GENOTYPE_MAP in spatialmt.data.dataset. "
            f"Known labels: {list(GENOTYPE_MAP)}"
        ) from None


# ---------------------------------------------------------------------------
# QCConfig (Decision 6A)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class QCConfig:
    """
    Quality-control thresholds for cell filtering.

    All values are inclusive bounds unless noted. Use the named class-method
    presets for paper-derived configurations.

    Attributes
    ----------
    min_genes:
        Minimum number of detected genes per cell (complexity filter).
    max_genes_per_cell:
        Maximum number of detected genes per cell (doublet proxy).
    min_umi:
        Minimum total UMI count per cell.
    max_umi:
        Maximum total UMI count per cell.
    max_pct_mt:
        Maximum fraction of counts from mitochondrial genes (0.0–1.0).
    """
    min_genes:          int
    max_genes_per_cell: int
    min_umi:            int
    max_umi:            int
    max_pct_mt:         float   # fraction, not percentage

    @classmethod
    def jain_timecourse(cls) -> "QCConfig":
        """
        QC thresholds for the Jain et al. 2025 Matrigel WT timecourse.

        Source: Morphodynamics of human early brain organoid development
        (Jain et al., Nature 2025) — WLS-KO condition used as closest proxy
        for WT preprocessing choices (WLS-KO not used in training).

        Thresholds
        ----------
        min_genes   > 500    : filters low-complexity cells and empty droplets
        max_pct_mt  < 10 %   : removes apoptotic cells corrupting pseudotime
        """
        return cls(
            min_genes=501,         # > 500 genes (paper: "more than 500 detected genes")
            max_genes_per_cell=10_000,
            min_umi=1_000,
            max_umi=100_000,
            max_pct_mt=0.099,      # < 10 % (paper: "less than 10 % mitochondrial counts")
        )

    @classmethod
    def he_gli3_ko_day45(cls) -> "QCConfig":
        """
        QC thresholds for the He et al. 2022 GLI3-KO day-45 dataset.

        Source: Inferring and perturbing cell fate regulomes in human brain
        organoids (He et al. 2022). Thresholds to be confirmed by inspecting
        the real AnnData — these are conservative placeholders.
        """
        return cls(
            min_genes=200,
            max_genes_per_cell=10_000,
            min_umi=500,
            max_umi=100_000,
            max_pct_mt=0.20,
        )


# ---------------------------------------------------------------------------
# GeneBlacklist (Decision 8A)
# ---------------------------------------------------------------------------

# Prefix patterns used by get_genes_to_remove().
# Extend here if new gene categories need to be supported.
_MITO_PREFIX    = ("MT-",)
_RIBO_PREFIX    = ("RPL", "RPS")
_HISTONE_PREFIX = ("HIST",)


@dataclasses.dataclass(frozen=True)
class GeneBlacklist:
    """
    Configuration for gene exclusion before HVG selection.

    Attributes
    ----------
    mito:
        Remove MT- prefixed mitochondrial genes.
    ribo:
        Remove RPL/RPS prefixed ribosomal genes.
    histone:
        Remove HIST prefixed histone genes.
    """
    mito:    bool
    ribo:    bool
    histone: bool

    @classmethod
    def jain_timecourse(cls) -> "GeneBlacklist":
        """
        Jain et al. 2025 WT timecourse: remove mito + ribo + histone.

        Histone genes are cell-cycle correlated and bias pseudotime ordering
        in proliferating early organoid cells.
        """
        return cls(mito=True, ribo=True, histone=True)

    @classmethod
    def he_gli3_ko_day45(cls) -> "GeneBlacklist":
        """
        He et al. 2022 GLI3-KO day-45: remove mito + ribo only.

        Histone removal is not documented in the He et al. preprocessing;
        omitting it avoids introducing undocumented differences from the paper.
        """
        return cls(mito=True, ribo=True, histone=False)

    def get_genes_to_remove(self, adata: ad.AnnData) -> list[str]:
        """
        Return list of gene names in adata that match the active flags.

        Parameters
        ----------
        adata:
            AnnData whose var_names will be scanned.

        Returns
        -------
        list[str]
            Gene names to remove. Empty if all flags are False.
            Never includes 'WLS' — protected by prefix non-overlap.
        """
        prefixes: list[str] = []
        if self.mito:
            prefixes.extend(_MITO_PREFIX)
        if self.ribo:
            prefixes.extend(_RIBO_PREFIX)
        if self.histone:
            prefixes.extend(_HISTONE_PREFIX)

        if not prefixes:
            return []

        remove = [
            g for g in adata.var_names
            if any(g.startswith(p) for p in prefixes)
        ]
        logger.debug("GeneBlacklist: %d genes flagged for removal.", len(remove))
        return remove


# ---------------------------------------------------------------------------
# FeatureManifest duck-type (used in type hints)
# ---------------------------------------------------------------------------

class FeatureManifest:
    """
    Minimal protocol — any object with a .gene_names list[str] attribute
    is accepted as a gene_manifest argument.

    This class provides a concrete implementation usable directly. Callers
    may also pass any object satisfying the same interface (e.g. the mock
    duck-type in conftest.py).
    """

    def __init__(self, gene_names: list[str]) -> None:
        self.gene_names: list[str] = gene_names


# ---------------------------------------------------------------------------
# ProcessedDataset (Decision 1A, 2A, 5A)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ProcessedDataset:
    """
    WT timecourse dataset ready for TabGRN-ICL training.

    Produced by ProcessedDataset.from_anndata() from raw count AnnData.
    Immutable after construction (frozen dataclass).

    Attributes
    ----------
    expression:
        float32 ndarray, shape (n_cells, n_genes). CP10k + log1p normalised.
    gene_names:
        Ordered list of gene names — column identity for expression.
    cell_ids:
        Ordered list of cell barcode strings — row identity for expression.
    pseudotime:
        float32 ndarray, shape (n_cells,). DC1 diffusion pseudotime from
        the source AnnData obs column.
    collection_day:
        int ndarray, shape (n_cells,). Organoid collection day per cell.
    test_mask:
        bool ndarray, shape (n_cells,). True where collection_day equals
        DataConfig.test_timepoint (day 11).
    manifest_hash:
        SHA-256 hex digest of sorted(gene_names). Used to verify gene-space
        compatibility with PerturbationDataset at runtime.
    preprocessing_config:
        Plain dict recording all decisions made during construction —
        QC thresholds, removed genes, selected genes, normalisation method.
        Must be JSON-serialisable.
    """
    expression:           np.ndarray
    gene_names:           list[str]
    cell_ids:             list[str]
    pseudotime:           np.ndarray
    collection_day:       np.ndarray
    test_mask:            np.ndarray
    manifest_hash:        str
    preprocessing_config: dict

    # ------------------------------------------------------------------ #
    # Public constructor                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_anndata(
        cls,
        adata:             ad.AnnData,
        data_config,                        # duck-typed DataConfig
        qc_config:         QCConfig,
        blacklist_config:  GeneBlacklist,
        gene_manifest,                      # FeatureManifest | None
        dc1_col:           str = "DC1",
        day_col:           str = "day",
        gpu_memory_bytes:  int = 8 * 1024 ** 3,
    ) -> "ProcessedDataset":
        """
        Build a ProcessedDataset from a raw-count AnnData.

        Pipeline (hardcoded order — Decision 14A)
        -----------------------------------------
        1. _filter_cells          (QC thresholds)
        2. _remove_blacklist_genes (mito / ribo / histone)
        3. _normalise              (CP10k + log1p, sparse-native)
        4. _select_genes           (HVG or frozen manifest)
        5. check_memory_feasibility
        6. _build                  (densify, assemble dataclass)

        Parameters
        ----------
        adata:
            Raw UMI count AnnData. X must be non-negative integers.
        data_config:
            Object with .max_genes (int) and .test_timepoint (int) attributes.
        qc_config:
            QCConfig instance defining cell-filtering thresholds.
        blacklist_config:
            GeneBlacklist instance defining genes to remove.
        gene_manifest:
            If provided, bypasses HVG selection and uses the manifest gene set.
            If None, selects top `data_config.max_genes` HVGs.
        dc1_col:
            obs column name containing DC1 diffusion pseudotime values.
        day_col:
            obs column name containing organoid collection day integers.
        gpu_memory_bytes:
            Available GPU memory for feasibility check.
        """
        preprocessing_config: dict = {
            "normalisation": "CP10k+log1p",
            "qc": dataclasses.asdict(qc_config),
            "blacklist": dataclasses.asdict(blacklist_config),
        }

        # Step 1
        adata = cls._filter_cells(adata, qc_config)

        # Step 2
        adata, removed_genes = cls._remove_blacklist_genes(
            adata, blacklist_config, return_removed=True
        )
        preprocessing_config["removed_genes"] = removed_genes

        # Step 3
        adata = cls._normalise(adata)

        # Step 4
        gene_names = cls._select_genes(adata, data_config, gene_manifest)
        preprocessing_config["selected_genes"] = gene_names

        # Step 5
        check_memory_feasibility(
            n_cells=adata.n_obs,
            n_genes=len(gene_names),
            gpu_memory_bytes=gpu_memory_bytes,
            label="ProcessedDataset",
        )

        # Step 6
        return cls._build(
            adata, gene_names, data_config,
            dc1_col=dc1_col,
            day_col=day_col,
            preprocessing_config=preprocessing_config,
        )

    # ------------------------------------------------------------------ #
    # Private pipeline steps                                               #
    # ------------------------------------------------------------------ #

    @classmethod
    def _filter_cells(cls, adata: ad.AnnData, qc: QCConfig) -> ad.AnnData:
        """
        Remove cells failing QC thresholds.

        Computes per-cell metrics in-place on adata.obs, then subsets.

        Raises
        ------
        ValueError
            If no cells remain after filtering.
        """
        # Compute QC metrics (modifies adata.obs in-place).
        # percent_top=[] avoids IndexError when n_genes < max(percent_top)
        # default values — common with small synthetic AnnDatas in tests.
        sc.pp.calculate_qc_metrics(
            adata,
            qc_vars=[],
            percent_top=[],
            inplace=True,
        )

        # Compute mitochondrial fraction
        mito_genes = adata.var_names.str.startswith("MT-")
        adata.obs["pct_counts_mt"] = (
            np.array(adata[:, mito_genes].X.sum(axis=1)).flatten()
            / np.array(adata.X.sum(axis=1)).flatten().clip(min=1)  # clip avoids 0/0 → NaN; zero-count cells get pct_mt=0.0
        )

        mask = (
            (adata.obs["n_genes_by_counts"] >= qc.min_genes)
            & (adata.obs["n_genes_by_counts"] <= qc.max_genes_per_cell)
            & (adata.obs["total_counts"]      >= qc.min_umi)
            & (adata.obs["total_counts"]      <= qc.max_umi)
            & (adata.obs["pct_counts_mt"]     <= qc.max_pct_mt)
        )

        filtered = adata[mask].copy()

        if filtered.n_obs == 0:
            raise ValueError(
                "No cells remain after QC filtering. "
                f"Thresholds used: {dataclasses.asdict(qc)}. "
                "Loosen QCConfig thresholds or inspect the input AnnData."
            )

        logger.info(
            "_filter_cells: %d → %d cells (removed %d).",
            adata.n_obs, filtered.n_obs, adata.n_obs - filtered.n_obs,
        )
        return filtered

    @classmethod
    def _remove_blacklist_genes(
        cls,
        adata:          ad.AnnData,
        blacklist:      GeneBlacklist,
        return_removed: bool = False,
    ) -> ad.AnnData | tuple[ad.AnnData, list[str]]:
        """
        Remove blacklisted genes from adata.

        Parameters
        ----------
        return_removed:
            If True, return (filtered_adata, removed_gene_names).
            If False, return filtered_adata only.
            Defaults to False for backward compatibility with callers that
            do not need the removed gene list. from_anndata() always passes
            return_removed=True to capture removed genes in preprocessing_config.
        """
        to_remove = set(blacklist.get_genes_to_remove(adata))
        keep_mask = ~adata.var_names.isin(to_remove)
        filtered  = adata[:, keep_mask].copy()
        removed   = [g for g in adata.var_names if g in to_remove]

        logger.info(
            "_remove_blacklist_genes: removed %d genes (%s...).",
            len(removed), removed[:3],
        )

        if return_removed:
            return filtered, removed
        return filtered

    @classmethod
    def _normalise(cls, adata: ad.AnnData) -> ad.AnnData:
        """
        Apply CP10k normalisation followed by log1p.

        Operates on the sparse matrix in-place (scanpy sparse-native ops).
        The result remains sparse — densification happens only in _build()
        after gene selection reduces the gene dimension.

        Normalisation contract
        ----------------------
        output[i, j] = log1p( X[i, j] / sum_j(X[i, j]) * 10_000 )

        Zero-count cells produce 0.0 (not NaN) because scanpy clips to avoid
        division by zero.
        """
        adata = adata.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata

    @classmethod
    def _select_genes(
        cls,
        adata:         ad.AnnData,
        data_config,
        gene_manifest,                   # FeatureManifest | None
    ) -> list[str]:
        """
        Return the ordered list of gene names to use as the feature space.

        Two paths
        ---------
        gene_manifest is None:
            Select top `data_config.max_genes` HVGs using scanpy's
            seurat_v3 dispatcher. Genes are returned in the order scanpy
            ranks them (descending by normalised dispersion).

        gene_manifest is provided:
            Use exactly the genes in gene_manifest.gene_names.
            All manifest genes must be present in adata.var_names — missing
            genes raise ValueError immediately with the gene name.

        Raises
        ------
        ValueError
            If any manifest gene is absent from adata.var_names.
        """
        if gene_manifest is None:
            # HVG path — must be called on normalised data.
            # When max_genes >= n_vars all genes are selected; seurat_v3
            # LOESS regression is near-singular in that case, so bypass it.
            if data_config.max_genes >= adata.n_vars:
                logger.debug(
                    "_select_genes: max_genes (%d) >= n_vars (%d) — returning all genes.",
                    data_config.max_genes, adata.n_vars,
                )
                return list(adata.var_names)

            # flavor="seurat" is designed for log-normalised data (CP10k+log1p).
            # seurat_v3 expects raw counts and emits a UserWarning on normalised
            # input — avoid that by using the correct flavor for this pipeline.
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=data_config.max_genes,
                flavor="seurat",
                subset=False,
            )
            hvg_mask = adata.var["highly_variable"]
            return list(adata.var_names[hvg_mask])

        # Manifest path — bypass HVG, validate all genes present
        avar = set(adata.var_names)
        missing = [g for g in gene_manifest.gene_names if g not in avar]
        if missing:
            raise ValueError(
                f"Gene(s) in manifest are absent from AnnData: {missing}. "
                f"The manifest was built from a different gene space. "
                f"Ensure the KO AnnData contains all genes in the WT manifest."
            )
        return list(gene_manifest.gene_names)

    @classmethod
    def _build(
        cls,
        adata:                ad.AnnData,
        gene_names:           list[str],
        data_config,
        dc1_col:              str  = "DC1",
        day_col:              str  = "day",
        preprocessing_config: dict | None = None,
    ) -> "ProcessedDataset":
        """
        Densify expression, assemble and return the frozen dataclass.

        Densification happens here — after gene selection — so the matrix is
        (n_cells × n_selected_genes) not (n_cells × n_all_genes).
        Peak memory: ~16 MB for 8095 × 512 float32.
        """
        subset = adata[:, gene_names]

        expression: np.ndarray
        if scipy.sparse.issparse(subset.X):
            expression = subset.X.toarray().astype(np.float32)
        else:
            expression = np.asarray(subset.X, dtype=np.float32)

        collection_day = adata.obs[day_col].to_numpy().astype(np.int32)
        test_mask      = collection_day == data_config.test_timepoint

        return cls(
            expression=expression,
            gene_names=list(gene_names),
            cell_ids=list(adata.obs_names),
            pseudotime=adata.obs[dc1_col].to_numpy().astype(np.float32),
            collection_day=collection_day,
            test_mask=test_mask,
            manifest_hash=_compute_manifest_hash(gene_names),
            preprocessing_config=preprocessing_config or {},
        )


# ---------------------------------------------------------------------------
# PerturbationDataset (Decision 1A — sibling, not subclass)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PerturbationDataset:
    """
    GLI3-KO day-45 dataset for perturbation inference.

    Produced by PerturbationDataset.from_anndata() from raw count AnnData.
    Requires a frozen gene_manifest derived from a ProcessedDataset.

    Attributes
    ----------
    expression:
        float32 ndarray, shape (n_ko_cells, n_genes). CP10k + log1p.
    gene_names:
        Ordered list of gene names — must match ProcessedDataset.gene_names.
    cell_ids:
        Ordered list of cell barcode strings.
    collection_day:
        Scalar int. All cells are from the same collection day (45).
    inferred_pseudotime:
        None at construction. Populated post-training by model inference.
        (Decision 3A mod — no diffusion map projection at construction.)
    manifest_hash:
        SHA-256 of sorted(gene_names). Must equal ProcessedDataset.manifest_hash
        for compatible datasets.
    preprocessing_config:
        Plain dict recording construction decisions. JSON-serialisable.
    """
    expression:           np.ndarray
    gene_names:           list[str]
    cell_ids:             list[str]
    collection_day:       int
    inferred_pseudotime:  np.ndarray | None
    manifest_hash:        str
    preprocessing_config: dict

    # ------------------------------------------------------------------ #
    # Public constructor                                                   #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_anndata(
        cls,
        adata:             ad.AnnData,
        qc_config:         QCConfig,
        blacklist_config:  GeneBlacklist,
        gene_manifest,                      # FeatureManifest — required
        target_genotype:   str,             # canonical Literal passed as str
        collection_day:    int,
        organoid_col:      str = "organoid",
        gpu_memory_bytes:  int = 8 * 1024 ** 3,
    ) -> "PerturbationDataset":
        """
        Build a PerturbationDataset from a raw-count AnnData.

        gene_manifest is required — PerturbationDataset has no independent
        gene selection. Passing None raises ValueError immediately.

        Pipeline (same ordering as ProcessedDataset — Decision 14A)
        -----------------------------------------------------------
        0. _filter_genotype        (subset to target_genotype cells)
        1. _filter_cells           (QC thresholds)
        2. _remove_blacklist_genes
        3. _normalise
        4. _select_genes           (manifest path only — HVG not permitted)
        5. check_memory_feasibility
        6. _build_perturbation

        Parameters
        ----------
        adata:
            Raw UMI count AnnData containing WT and KO cells.
        qc_config:
            QCConfig for cell filtering (may differ from WT thresholds).
        blacklist_config:
            GeneBlacklist (may differ from WT — e.g. no histone removal).
        gene_manifest:
            FeatureManifest from the WT ProcessedDataset. Required.
        target_genotype:
            Canonical genotype string ("WT" or "GLI3_KO"). Cells where
            organoid_col maps to this genotype are kept.
        collection_day:
            Scalar int labelling this dataset's collection timepoint (45).
        organoid_col:
            obs column name containing organoid / genotype labels.
        gpu_memory_bytes:
            Available GPU memory for feasibility check.
        """
        if gene_manifest is None:
            raise ValueError(
                "PerturbationDataset.from_anndata() requires a gene_manifest. "
                "Build a ProcessedDataset first and pass its gene_names as a "
                "FeatureManifest to ensure gene-space alignment."
            )

        # Validate organoid_col exists before doing anything else
        if organoid_col not in adata.obs.columns:
            raise KeyError(
                f"Column {organoid_col!r} not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

        preprocessing_config: dict = {
            "normalisation": "CP10k+log1p",
            "qc": dataclasses.asdict(qc_config),
            "blacklist": dataclasses.asdict(blacklist_config),
            "target_genotype": target_genotype,
            "collection_day": collection_day,
        }

        # Step 0 — genotype filter
        adata = cls._filter_genotype(adata, target_genotype, organoid_col)

        # Step 1
        adata = cls._filter_cells(adata, qc_config)

        # Step 2
        adata, removed_genes = cls._remove_blacklist_genes(
            adata, blacklist_config, return_removed=True
        )
        preprocessing_config["removed_genes"] = removed_genes

        # Step 3
        adata = cls._normalise(adata)

        # Step 4 — manifest path only
        gene_names = cls._select_genes(adata, data_config=None, gene_manifest=gene_manifest)
        preprocessing_config["selected_genes"] = gene_names

        # Step 5
        check_memory_feasibility(
            n_cells=adata.n_obs,
            n_genes=len(gene_names),
            gpu_memory_bytes=gpu_memory_bytes,
            label="PerturbationDataset",
        )

        # Step 6
        return cls._build_perturbation(
            adata, gene_names, collection_day, preprocessing_config
        )

    # ------------------------------------------------------------------ #
    # Private pipeline steps (shared logic delegated to ProcessedDataset) #
    # ------------------------------------------------------------------ #

    @classmethod
    def _filter_genotype(
        cls,
        adata:           ad.AnnData,
        target_genotype: str,
        organoid_col:    str,
    ) -> ad.AnnData:
        """
        Subset adata to cells whose organoid_col value maps to target_genotype.

        Each raw label is validated through _parse_genotype — unknown labels
        raise ValueError with a GENOTYPE_MAP reference.

        Raises
        ------
        ValueError
            If no cells of target_genotype are present after filtering.
        """
        raw_labels      = adata.obs[organoid_col]
        canonical_labels = raw_labels.map(
            lambda lbl: _parse_genotype(lbl)
        )
        mask    = canonical_labels == target_genotype
        subset  = adata[mask].copy()

        if subset.n_obs == 0:
            raise ValueError(
                f"No cells with genotype {target_genotype!r} found in "
                f"adata.obs[{organoid_col!r}]. "
                f"Unique values present: {list(raw_labels.unique())}. "
                f"Check GENOTYPE_MAP or the target_genotype argument."
            )

        logger.info(
            "_filter_genotype(%r): %d → %d cells.",
            target_genotype, adata.n_obs, subset.n_obs,
        )
        return subset

    # Shared pipeline steps — forward to the module-level implementations
    # via explicit @classmethods rather than class-level assignment.
    # Class-level assignment would bind to ProcessedDataset's descriptor at
    # definition time, making monkeypatching in tests unpredictable.

    @classmethod
    def _filter_cells(cls, adata: ad.AnnData, qc: "QCConfig") -> ad.AnnData:
        return ProcessedDataset._filter_cells(adata, qc)

    @classmethod
    def _remove_blacklist_genes(
        cls,
        adata:          ad.AnnData,
        blacklist:      "GeneBlacklist",
        return_removed: bool = False,
    ):
        return ProcessedDataset._remove_blacklist_genes(adata, blacklist, return_removed)

    @classmethod
    def _normalise(cls, adata: ad.AnnData) -> ad.AnnData:
        return ProcessedDataset._normalise(adata)

    @classmethod
    def _select_genes(cls, adata: ad.AnnData, data_config, gene_manifest) -> list:
        return ProcessedDataset._select_genes(adata, data_config, gene_manifest)

    @classmethod
    def _build_perturbation(
        cls,
        adata:                ad.AnnData,
        gene_names:           list[str],
        collection_day:       int,
        preprocessing_config: dict,
    ) -> "PerturbationDataset":
        """
        Densify expression and assemble the frozen PerturbationDataset.
        """
        subset = adata[:, gene_names]

        if scipy.sparse.issparse(subset.X):
            expression = subset.X.toarray().astype(np.float32)
        else:
            expression = np.asarray(subset.X, dtype=np.float32)

        return cls(
            expression=expression,
            gene_names=list(gene_names),
            cell_ids=list(adata.obs_names),
            collection_day=int(collection_day),
            inferred_pseudotime=None,           # Decision 3A mod
            manifest_hash=_compute_manifest_hash(gene_names),
            preprocessing_config=preprocessing_config,
        )