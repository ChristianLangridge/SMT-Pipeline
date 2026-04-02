"""
Batch effect diagnostic — run before diffusion pseudotime computation.
Determines whether CSS/Harmony integration is needed.

Usage:
    python batch_effect_diagnostic.py
"""
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

# ── Load ──────────────────────────────────────────────────────────────
h5ad_path = "data/training_data/AnnData/neurectoderm_complete.h5ad"  # adjust if needed
adata = sc.read_h5ad(h5ad_path)
print(f"Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
print(f"Collection days: {adata.obs['orig.ident'].value_counts().sort_index().to_dict()}")
print(f"Cell types: {adata.obs['class3'].nunique()}")

# ── HVG + PCA ─────────────────────────────────────────────────────────
adata_work = adata.copy()
sc.pp.normalize_total(adata_work, target_sum=1e4)
sc.pp.log1p(adata_work)

# HVG selection (seurat flavor for normalised data)
sc.pp.highly_variable_genes(adata_work, n_top_genes=2000, flavor="seurat")
adata_work = adata_work[:, adata_work.var["highly_variable"]].copy()

# Scale (z-score) — optional cell-cycle regression if scores available
scale_keys = []
if "S.Score" in adata_work.obs.columns and "G2M.Score" in adata_work.obs.columns:
    scale_keys = ["S.Score", "G2M.Score"]
    print("Regressing out cell-cycle scores during scaling")

sc.pp.scale(adata_work)
# Note: if you want to regress out cell cycle, use:
# sc.pp.regress_out(adata_work, ['S.Score', 'G2M.Score']) before scaling

sc.tl.pca(adata_work, n_comps=30, svd_solver="arpack")

# ── Diagnostic plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(37, 7))

# Plot 1: PCA coloured by collection day
sc.pl.pca(adata_work, color="orig.ident", ax=axes[0], show=False, title="PC1 vs PC2 — Collection day", legend_loc="right margin")
axes[0].get_legend().set_bbox_to_anchor((1.02, 0.5))
axes[0].get_legend().set_loc("center left")

# Plot 2: PCA coloured by cell type
sc.pl.pca(adata_work, color="class3", ax=axes[1], show=False, title="PC1 vs PC2 — Cell type (class3)", legend_loc="right margin")
axes[1].get_legend().set_bbox_to_anchor((1.02, 0.5))
axes[1].get_legend().set_loc("center left")

# Plot 3: Variance explained — scree plot
variance_ratio = adata_work.uns["pca"]["variance_ratio"]
axes[2].bar(range(1, 31), variance_ratio[:30], color="steelblue", alpha=0.7)
axes[2].set_xlabel("PC")
axes[2].set_ylabel("Variance explained")
axes[2].set_title("Scree plot (top 30 PCs)")
cumulative = np.cumsum(variance_ratio[:30])
ax2 = axes[2].twinx()
ax2.plot(range(1, 31), cumulative, color="coral", marker="o", markersize=3)
ax2.set_ylabel("Cumulative variance")
ax2.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5)

plt.subplots_adjust(left=0.05, right=0.95, wspace=0.5)
plt.savefig("batch_effect_diagnostic.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Quantitative batch mixing metric ─────────────────────────────────
# Simple check: per-day mean PC1 — if days are well-separated on PC1,
# batch effect is strong
print("\n── Per-day PC1 summary ──")
adata_work.obs["PC1"] = adata_work.obsm["X_pca"][:, 0]
day_stats = adata_work.obs.groupby("orig.ident")["PC1"].agg(["mean", "std", "count"])
print(day_stats.to_string())
