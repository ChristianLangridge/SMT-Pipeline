install.packages("rlang")
install.packages("devtools")
devtools::install_github("quadbiolab/simspec")
install.packages("dplyr")

library(Seurat)
library(simspec)
library(dplyr)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

rds_path      <- "data/training_data/Seurat/Timecourse.rds"
out_css_embed <- "data/training_data/ML_data/css_embedding.csv"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

cat("Loading Seurat object...\n")
Timecourse <- readRDS(rds_path)
cat(sprintf("Loaded: %d cells x %d genes\n", ncol(Timecourse), nrow(Timecourse)))

# ---------------------------------------------------------------------------
# HVG selection — exclude cell cycle, MT, RP genes (mirrors prep.py)
# ---------------------------------------------------------------------------

cat("Selecting HVGs...\n")
Timecourse <- FindVariableFeatures(Timecourse, nfeatures = 2000)

blacklist <- c(
  unlist(cc.genes.updated.2019),
  grep("^MT-",  rownames(Timecourse), value = TRUE),
  grep("^RPS",  rownames(Timecourse), value = TRUE),
  grep("^RPL",  rownames(Timecourse), value = TRUE)
)
VariableFeatures(Timecourse) <- setdiff(VariableFeatures(Timecourse), blacklist)
cat(sprintf("HVGs after blacklist: %d\n", length(VariableFeatures(Timecourse))))

# ---------------------------------------------------------------------------
# Scale + PCA with cell-cycle regression
# ---------------------------------------------------------------------------

cat("Scaling and running PCA (regressing out S.Score, G2M.Score)...\n")
Timecourse <- ScaleData(Timecourse, vars.to.regress = c("G2M.Score", "S.Score")) %>%
  RunPCA(npcs = 20)

# ---------------------------------------------------------------------------
# CSS — batch label is 'dataset' (one entry per collection day sample)
# ---------------------------------------------------------------------------

cat("Running CSS...\n")
Timecourse <- cluster_sim_spectrum(
  Timecourse,
  label_tag         = "dataset",
  cluster_resolution = 1
) %>%
  run_PCA(
    reduction      = "css",
    npcs           = 10,
    reduction.name = "csspca",
    reduction.key  = "CSSPCA_"
  ) %>%
  regress_out_from_embeddings(
    reduction      = "csspca",
    vars_to_regress = c("G2M.Score", "S.Score"),
    reduction.name = "csspcacc",
    reduction.key  = "CSSPCACC_"
  )

# Sanity check: residual correlation with cell cycle scores
cor_css2cc <- cor(
  Embeddings(Timecourse, "csspcacc"),
  Timecourse@meta.data[, c("G2M.Score", "S.Score")],
  method = "spearman"
)
cat("Max residual |correlation| with cell cycle after regression:\n")
cat(sprintf("  G2M.Score: %.4f\n", max(abs(cor_css2cc[, "G2M.Score"]))))
cat(sprintf("  S.Score  : %.4f\n", max(abs(cor_css2cc[, "S.Score"]))))

# ---------------------------------------------------------------------------
# Save CSS-PCA-CC embedding for DPT computation in Python (scanpy)
# destiny is not available for R 4.5 — DPT is handled by diffusion_trajectory.py
# ---------------------------------------------------------------------------

cat("Saving CSS-PCA-CC embedding...\n")

css_embed <- as.data.frame(Embeddings(Timecourse, "csspcacc"))
css_embed <- cbind(cell_id = colnames(Timecourse), css_embed)

write.csv(css_embed, file = out_css_embed, row.names = FALSE)
cat(sprintf("Saved CSS embedding to %s\n", out_css_embed))

n_dims <- ncol(css_embed) - 1
cat(sprintf("Cell count: %d | Dimensions: %d\n", nrow(css_embed), n_dims))

# Sanity check — per-day cell counts
cat("\n── Per-day cell counts ──\n")
print(table(Timecourse$orig.ident))
