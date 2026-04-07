from spatialmt.data_preparation.prep import prepare_dataset, load_h5ad
from spatialmt.data_preparation.diffusion_trajectory import compute_dpt_from_css_embedding
from spatialmt.config.paths import Dirs, Paths, setup_output_dirs, validate_raw_inputs
import numpy as np
import pandas as pd
import scanpy as sc

if __name__ == "__main__":
    setup_output_dirs()
    validate_raw_inputs()

    # Load raw counts and normalise once — shared by both prep and diffusion
    print("── Loading and normalising ──")
    adata = load_h5ad(Dirs.model_data_anndata / "neurectoderm_complete.h5ad")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

    # HVG selection + feature extraction for ML
    dataset = prepare_dataset(
        adata,
        cell_type_key="class3",
        n_top_genes=2000,
    )

    X = dataset.X
    cell_labels = dataset.cell_labels
    gene_labels = dataset.gene_labels
    y = dataset.y

    # Diffusion pseudotime using CSS embedding from css_pseudotime.R
    pseudotime = compute_dpt_from_css_embedding(
        adata,
        css_embedding_path=Paths.css_embedding,
        cell_type_key="class3",
    )
    pseudotime = pseudotime.reindex(cell_labels)

    print(f"\nExpression matrix : {X.shape}  (cells x HVGs)")
    print(f"Cell labels        : {len(cell_labels)}")
    print(f"Gene labels        : {len(gene_labels)}")
    print(f"Cell-type classes  : {y.nunique()}  →  {y.unique().tolist()}")
    print(f"Pseudotime range   : {pseudotime.min():.4f} – {pseudotime.max():.4f}")

    out_dir = Dirs.model_data_ml
    np.savez_compressed(out_dir / "expression_matrix.npz", X=X)
    pd.Series(cell_labels, name="cell_id").to_csv(out_dir / "cell_labels.csv", index=False)
    pd.Series(gene_labels, name="gene_id").to_csv(out_dir / "gene_labels.csv", index=False)
    y.reset_index(drop=True).to_csv(out_dir / "cell_type_labels.csv", index=False)
    pseudotime.reset_index(drop=True).to_csv(
        out_dir / "diffusion_component_pseudotime_labels.csv", index=False
    )

    print(f"\nSaved preprocessed data to {out_dir}")