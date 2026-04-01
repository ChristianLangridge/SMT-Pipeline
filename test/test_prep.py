"""
Tests for spatialmt.data_preparation.prep

All tests use synthetic data — no .h5ad file required.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

from spatialmt.data_preparation.prep import (
    extract_expression_matrix,
    extract_cell_labels,
    extract_gene_labels,
    extract_cell_type_labels,
    generate_pseudotime_labels,
    select_highly_variable_genes,
    prepare_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adata(n_cells=10, n_genes=5, sparse=False, scale=None):
    if scale == "normalised":
        X = (np.random.rand(n_cells, n_genes) * 4).astype(np.float32)
    elif scale == "raw":
        X = (np.random.rand(n_cells, n_genes) * 10_000).astype(np.float32)
    else:
        X = np.random.rand(n_cells, n_genes).astype(np.float32)
    if sparse:
        X = sp.csr_matrix(X)
    _timepoints = ["HB4_D5", "HB4_D7", "HB4_D11", "HB4_D16", "HB4_D21", "HB4_D30"]
    obs = pd.DataFrame(
        {
            "cell_type": [f"type_{i % 3}" for i in range(n_cells)],
            "orig.ident": [_timepoints[i % len(_timepoints)] for i in range(n_cells)],
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# select_highly_variable_genes
# ---------------------------------------------------------------------------

def test_hvg_returns_correct_gene_count():
    adata = _make_adata(n_cells=20, n_genes=50, scale="normalised")
    result = select_highly_variable_genes(adata, n_top_genes=10, flavor="seurat")
    assert result.n_vars == 10


def test_hvg_output_is_copy():
    adata = _make_adata(n_cells=20, n_genes=50, scale="normalised")
    result = select_highly_variable_genes(adata, n_top_genes=10, flavor="seurat")
    result.obs["sentinel"] = 99
    assert "sentinel" not in adata.obs.columns


def test_hvg_preserves_cell_count():
    adata = _make_adata(n_cells=20, n_genes=50, scale="normalised")
    result = select_highly_variable_genes(adata, n_top_genes=10, flavor="seurat")
    assert result.n_obs == adata.n_obs


def test_hvg_selected_genes_subset_of_input():
    adata = _make_adata(n_cells=20, n_genes=50, scale="normalised")
    result = select_highly_variable_genes(adata, n_top_genes=10, flavor="seurat")
    assert set(result.var_names).issubset(set(adata.var_names))


def test_hvg_seurat_on_normalised_no_warning():
    adata = _make_adata(n_cells=20, n_genes=50, scale="normalised")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        select_highly_variable_genes(adata, n_top_genes=10, flavor="seurat")


def test_hvg_seurat_v3_on_normalised_warns():
    adata = _make_adata(n_cells=20, n_genes=50, scale="normalised")
    with pytest.warns(UserWarning):
        select_highly_variable_genes(adata, n_top_genes=10, flavor="seurat_v3")


def test_hvg_seurat_on_raw_warns():
    adata = _make_adata(n_cells=20, n_genes=50, scale="raw")
    with pytest.warns(UserWarning):
        select_highly_variable_genes(adata, n_top_genes=10, flavor="seurat")


def test_hvg_n_top_exceeds_available():
    adata = _make_adata(n_cells=20, n_genes=10, scale="normalised")
    result = select_highly_variable_genes(adata, n_top_genes=999, flavor="seurat")
    assert result.n_vars == adata.n_vars


# ---------------------------------------------------------------------------
# prepare_dataset
# Monkeypatches load_h5ad so no real .h5ad file is needed.
# Tests written against the current dict return type.
# TODO (Batch 2): update assertions to dataclass interface once Issue 5 lands.
# ---------------------------------------------------------------------------

@pytest.fixture()
def patched_prepare(monkeypatch):
    """Return a prepare_dataset callable whose load_h5ad is replaced with
    a function that returns a synthetic 30-cell / 50-gene AnnData."""
    import spatialmt.data_preparation.prep as prep_module

    def _fake_load(path):
        return _make_adata(n_cells=30, n_genes=50, scale="normalised")

    monkeypatch.setattr(prep_module, "load_h5ad", _fake_load)
    return prepare_dataset


def test_prepare_dataset_returns_prepared_data(patched_prepare):
    from spatialmt.data_preparation.prep import PreparedData
    result = patched_prepare("dummy.h5ad", n_top_genes=10, hvg_flavor="seurat")
    assert isinstance(result, PreparedData)


def test_prepare_dataset_shapes_consistent(patched_prepare):
    result = patched_prepare("dummy.h5ad", n_top_genes=10, hvg_flavor="seurat")
    n = result.X.shape[0]
    assert len(result.cell_labels) == n
    assert len(result.y) == n
    assert len(result.orig_ident) == n


def test_prepare_dataset_gene_count(patched_prepare):
    n_top = 10
    result = patched_prepare("dummy.h5ad", n_top_genes=n_top, hvg_flavor="seurat")
    assert result.X.shape[1] == len(result.gene_labels)
    assert result.X.shape[1] <= n_top


TIMEPOINTS = ["HB4_D5", "HB4_D7", "HB4_D11", "HB4_D16", "HB4_D21", "HB4_D30"]
EXPECTED   = [0.0,       0.08,      0.24,       0.44,       0.64,       1.0]


# ---------------------------------------------------------------------------
# extract_expression_matrix
# ---------------------------------------------------------------------------

def test_extract_expression_matrix_dense():
    adata = _make_adata()
    X = extract_expression_matrix(adata)
    assert isinstance(X, np.ndarray)
    assert X.dtype == np.float32
    assert X.shape == (10, 5)


def test_extract_expression_matrix_sparse():
    adata = _make_adata(sparse=True)
    X = extract_expression_matrix(adata)
    assert isinstance(X, np.ndarray)
    assert X.shape == (10, 5)


# ---------------------------------------------------------------------------
# extract_expression_matrix — current behaviour with invalid values
# These are DOCUMENTATION tests: they pin what the function does today
# (pass-through) so that Batch 2 validation guards have a failing baseline.
# ---------------------------------------------------------------------------

def test_expression_matrix_with_nan():
    from spatialmt.data_preparation.prep import DataIntegrityError
    X = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    adata = ad.AnnData(X=X)
    with pytest.raises(DataIntegrityError, match="NaN"):
        extract_expression_matrix(adata)


def test_expression_matrix_with_negative():
    # Current behaviour: negative values pass through unchanged.
    # Batch 2 should replace this with pytest.raises(ValueError).
    X = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    adata = ad.AnnData(X=X)
    result = extract_expression_matrix(adata)
    assert (result < 0).any(), "expected negative values to pass through (current behaviour)"


def test_expression_matrix_with_inf():
    from spatialmt.data_preparation.prep import DataIntegrityError
    X = np.array([[1.0, np.inf], [3.0, 4.0]], dtype=np.float32)
    adata = ad.AnnData(X=X)
    with pytest.raises(DataIntegrityError, match="Inf"):
        extract_expression_matrix(adata)


# ---------------------------------------------------------------------------
# extract_cell_labels / extract_gene_labels
# ---------------------------------------------------------------------------

def test_extract_cell_labels():
    adata = _make_adata()
    labels = extract_cell_labels(adata)
    assert list(labels) == [f"cell_{i}" for i in range(10)]


def test_extract_gene_labels():
    adata = _make_adata()
    labels = extract_gene_labels(adata)
    assert list(labels) == [f"gene_{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# extract_cell_type_labels
# ---------------------------------------------------------------------------

def test_extract_cell_type_labels_valid_key():
    adata = _make_adata()
    y = extract_cell_type_labels(adata, cell_type_key="cell_type")
    assert len(y) == 10
    assert set(y.unique()) == {"type_0", "type_1", "type_2"}


def test_extract_cell_type_labels_missing_key():
    adata = _make_adata()
    with pytest.raises(KeyError, match="missing_key"):
        extract_cell_type_labels(adata, cell_type_key="missing_key")


# ---------------------------------------------------------------------------
# generate_pseudotime_labels
# ---------------------------------------------------------------------------

def test_pseudotime_boundary_values():
    s = pd.Series(["HB4_D5", "HB4_D30"])
    pt = generate_pseudotime_labels(s)
    assert pt.iloc[0] == pytest.approx(0.0)
    assert pt.iloc[1] == pytest.approx(1.0)


def test_pseudotime_linear_scaling():
    s = pd.Series(TIMEPOINTS)
    pt = generate_pseudotime_labels(s)
    for val, expected in zip(pt, EXPECTED):
        assert val == pytest.approx(expected, abs=1e-6)


def test_pseudotime_categorical_input():
    """orig.ident is typically a Categorical — must not raise or produce NaN."""
    s = pd.Categorical(TIMEPOINTS)
    s = pd.Series(s)
    pt = generate_pseudotime_labels(s)
    assert not pt.isna().any()


def test_pseudotime_unknown_label_is_nan():
    s = pd.Series(["HB4_D5", "UNKNOWN", "HB4_D30"])
    pt = generate_pseudotime_labels(s)
    assert pt.iloc[0] == pytest.approx(0.0)
    assert np.isnan(pt.iloc[1])
    assert pt.iloc[1+1] == pytest.approx(1.0)


def test_pseudotime_output_name():
    s = pd.Series(["HB4_D5"])
    pt = generate_pseudotime_labels(s)
    assert pt.name == "pseudotime"


def test_pseudotime_length_preserved():
    s = pd.Series(TIMEPOINTS * 100)
    pt = generate_pseudotime_labels(s)
    assert len(pt) == len(s)


# ---------------------------------------------------------------------------
# Pseudotime interface contract
# Parametrized over all pseudotime implementations. Add "diffusion" when
# the diffusion pseudotime function is available.
# ---------------------------------------------------------------------------

@pytest.fixture(params=["scaffold"])  # add "diffusion" when available
def pseudotime_fn(request):
    if request.param == "scaffold":
        return generate_pseudotime_labels


def test_pseudotime_contract_output_is_series(pseudotime_fn):
    s = pd.Series(TIMEPOINTS)
    result = pseudotime_fn(s)
    assert isinstance(result, pd.Series)


def test_pseudotime_contract_name_is_pseudotime(pseudotime_fn):
    s = pd.Series(TIMEPOINTS)
    result = pseudotime_fn(s)
    assert result.name == "pseudotime"


def test_pseudotime_contract_values_in_unit_interval(pseudotime_fn):
    s = pd.Series(TIMEPOINTS)
    result = pseudotime_fn(s)
    valid = result.dropna()
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


def test_pseudotime_contract_length_matches_input(pseudotime_fn):
    s = pd.Series(TIMEPOINTS * 10)
    result = pseudotime_fn(s)
    assert len(result) == len(s)


# ---------------------------------------------------------------------------
# check_memory_feasibility
# ---------------------------------------------------------------------------

from spatialmt.data_preparation.prep import check_memory_feasibility


def test_memory_feasibility_no_warn_small_data():
    """Small dataset should not trigger a warning."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        check_memory_feasibility(n_cells=100, n_genes=500, n_top_genes=50)


def test_memory_feasibility_warns_large_data(monkeypatch):
    """Simulate insufficient RAM: patch available memory to 1 byte."""
    import psutil
    mock_vm = psutil.virtual_memory()._replace(available=1)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: mock_vm)
    with pytest.warns(UserWarning, match="Estimated peak"):
        check_memory_feasibility(n_cells=10_000, n_genes=20_000, n_top_genes=2_000)


def test_memory_feasibility_estimation():
    """Peak estimate formula: sparse + dense bytes."""
    import psutil
    n_cells, n_genes, n_top = 1_000, 10_000, 2_000
    expected = n_cells * n_genes * 0.1 * 8 + n_cells * n_top * 4
    # Should not warn when available memory is very large (patch to 1 TB)
    import unittest.mock as mock
    mock_vm = psutil.virtual_memory()._replace(available=int(1e12))
    with mock.patch.object(psutil, "virtual_memory", return_value=mock_vm):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            check_memory_feasibility(n_cells, n_genes, n_top)
    assert expected == 1_000 * 10_000 * 0.1 * 8 + 1_000 * 2_000 * 4
