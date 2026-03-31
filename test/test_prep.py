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
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adata(n_cells=10, n_genes=5, sparse=False):
    X = np.random.rand(n_cells, n_genes).astype(np.float32)
    if sparse:
        X = sp.csr_matrix(X)
    obs = pd.DataFrame({"cell_type": [f"type_{i % 3}" for i in range(n_cells)]},
                       index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


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
