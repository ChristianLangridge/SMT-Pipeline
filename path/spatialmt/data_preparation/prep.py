import dataclasses
import warnings
import psutil
import scanpy as sc
import numpy as np
import pandas as pd


class DataIntegrityError(ValueError):
    """Raised when the expression matrix contains invalid values."""
from spatialmt.config.paths import Dirs, setup_output_dirs, validate_raw_inputs


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def load_h5ad(path) -> sc.AnnData:
    """Load an .h5ad file and return the AnnData object."""
    adata = sc.read_h5ad(path)
    return adata


def extract_expression_matrix(adata: sc.AnnData) -> np.ndarray:
    """Return the expression matrix as a dense numpy array (cells x genes)."""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(np.float32)
    if np.any(np.isnan(X)):
        raise DataIntegrityError("NaN detected in expression matrix")
    if np.any(np.isinf(X)):
        raise DataIntegrityError("Inf detected in expression matrix")
    return X


def extract_cell_labels(adata: sc.AnnData) -> pd.Index:
    """Return cell (observation) labels."""
    return adata.obs_names.copy()


def extract_gene_labels(adata: sc.AnnData) -> pd.Index:
    """Return gene (variable) labels."""
    return adata.var_names.copy()


def extract_cell_type_labels(adata: sc.AnnData, cell_type_key: str = "cell_type") -> pd.Series:
    """
    Return the target cell-type label for each cell.

    Parameters
    ----------
    adata : AnnData
    cell_type_key : str
        Key in adata.obs that holds the cell-type annotation.
    """
    if cell_type_key not in adata.obs.columns:
        raise KeyError(
            f"Cell-type key '{cell_type_key}' not found in adata.obs. "
            f"Available keys: {list(adata.obs.columns)}"
        )
    return adata.obs[cell_type_key].copy()


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_highly_variable_genes(
    adata: sc.AnnData,
    n_top_genes: int = 2000,
    flavor: str = "seurat",
) -> sc.AnnData:
    """
    Filter adata to the top highly variable genes.

    Parameters
    ----------
    adata : AnnData
        Log-normalised counts expected. Use 'seurat' or 'cell_ranger'; 'seurat_v3' expects raw counts and is not used here.
    n_top_genes : int
        Number of HVGs to retain.
    flavor : str
        HVG method passed to scanpy. Defaults to 'seurat', which expects log-normalised input.

    Returns
    -------
    AnnData
        Filtered to HVG columns only (copy).
    """
    import warnings
    adata = adata.copy()
    xmax = adata.X.max() if not hasattr(adata.X, 'toarray') else adata.X.toarray().max()
    if flavor == "seurat" and xmax > 20.0:
        warnings.warn(f"flavor='seurat' expects log-normalised data but X.max()={xmax:.1f}")
    if flavor == "seurat_v3" and xmax < 20.0:
        warnings.warn(f"flavor='seurat_v3' expects raw counts but X.max()={xmax:.1f}")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor)
    adata = adata[:, adata.var["highly_variable"]]
    return adata


# ---------------------------------------------------------------------------
# Pseudotime labelling
# ---------------------------------------------------------------------------


def generate_pseudotime_labels(
    orig_ident: pd.Series,
    min_label: str = "HB4_D5",
    max_label: str = "HB4_D30",
) -> pd.Series:
    """
    Scaffold pseudotime — linearly scaled collection day. Will be replaced by
    diffusion pseudotime. See TDD v1.2.0 §S3.

    Assign a linearly scaled pseudotime value to each cell based on its
    ``orig.ident`` timepoint string (e.g. ``"HB4_D7"``).

    Day numbers are parsed from the trailing integer in each label.
    ``min_label`` maps to 0.0 and ``max_label`` maps to 1.0; all other
    recognised ``HB4_D{N}`` values are scaled proportionally.

    Parameters
    ----------
    orig_ident : pd.Series
        Per-cell timepoint strings, e.g. from ``adata.obs["orig.ident"]``.
    min_label : str
        The timepoint string that should map to 0.0.
    max_label : str
        The timepoint string that should map to 1.0.

    Returns
    -------
    pd.Series
        Float pseudotime values in [0, 1], named ``"pseudotime"``.
        Cells whose ``orig_ident`` cannot be parsed receive ``NaN``.
    """
    def _parse_day(label: str) -> float:
        parts = str(label).split("_D")
        if len(parts) == 2 and parts[1].isdigit():
            return float(parts[1])
        return float("nan")

    days = orig_ident.astype(str).map(_parse_day)
    min_day = _parse_day(min_label)
    max_day = _parse_day(max_label)

    pseudotime = (days - min_day) / (max_day - min_day)
    pseudotime.name = "pseudotime"
    return pseudotime


# ---------------------------------------------------------------------------
# Memory pre-check
# ---------------------------------------------------------------------------

def check_memory_feasibility(n_cells: int, n_genes: int, n_top_genes: int) -> None:
    """Warn if estimated peak memory exceeds 80% of available RAM."""
    peak_sparse_bytes = n_cells * n_genes * 0.1 * 8    # ~10% density, float64
    peak_dense_bytes = n_cells * n_top_genes * 4        # float32 after HVG
    peak_total = peak_sparse_bytes + peak_dense_bytes

    available = psutil.virtual_memory().available
    if peak_total > available * 0.8:
        warnings.warn(
            f"Estimated peak {peak_total/1e9:.1f}GB exceeds 80% of available {available/1e9:.1f}GB"
        )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PreparedData:
    X: np.ndarray        # (n_cells, n_hvgs), float32
    cell_labels: pd.Index
    gene_labels: pd.Index  # HVG subset
    y: pd.Series           # cell-type labels
    orig_ident: pd.Series  # timepoint strings

    def __post_init__(self):
        assert self.X.shape[0] == len(self.cell_labels) == len(self.y)
        assert self.X.shape[1] == len(self.gene_labels)
        assert self.X.dtype == np.float32


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def prepare_dataset(
    h5ad_path,
    cell_type_key: str = "cell_type",
    n_top_genes: int = 2000,
    hvg_flavor: str = "seurat",
) -> dict:
    """
    Full preprocessing pipeline: load → extract → HVG filter.

    Returns
    -------
    dict with keys:
        X           : np.ndarray, shape (n_cells, n_hvgs)
        cell_labels : pd.Index
        gene_labels : pd.Index   (HVG subset)
        y           : pd.Series  (cell-type labels)
    """
    adata = load_h5ad(h5ad_path)
    check_memory_feasibility(adata.n_obs, adata.n_vars, n_top_genes)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata_hvg = select_highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=hvg_flavor)

    X = extract_expression_matrix(adata_hvg)
    cell_labels = extract_cell_labels(adata_hvg)
    gene_labels = extract_gene_labels(adata_hvg)
    y = extract_cell_type_labels(adata_hvg, cell_type_key=cell_type_key)

    return PreparedData(
        X=X,
        cell_labels=cell_labels,
        gene_labels=gene_labels,
        y=y,
        orig_ident=adata_hvg.obs["orig.ident"].copy(),
    )



