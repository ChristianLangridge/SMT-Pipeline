"""
spatialmt.config.experiment — ExperimentConfig and all sub-configs.

Single source of truth for every hyperparameter. Every training run serialises
its config to experiments/{run_id}/config.json, making runs fully reproducible
from that file alone.

Sub-configs
-----------
DataConfig          Data shape, feature selection, target definitions
ContextConfig       ICL context window layout
ModelConfig         Learning rates, warmup, head initialisation
ExplainabilityConfig  SHAP background, biological plausibility gate
PerturbationConfig  In-silico knockout definitions and thresholds
BenchmarkConfig     Baseline ladder for dual-axis justification

Named presets
-------------
ExperimentConfig.debug_preset()         128 genes, CPU, 2 cells/bin
ExperimentConfig.rotation_finetune()    512 genes, V100, dual-head
ExperimentConfig.rotation_baselines()   Baseline ladder only
ExperimentConfig.full_finetune()        1024 genes, A100, extended training
ExperimentConfig.scratch_preset()       No pretrained weights [Phase 6]
ExperimentConfig.no_icl_preset()        Single cell, no context [Phase 6]
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Sentinel exception
# ---------------------------------------------------------------------------

class ConfigurationError(ValueError):
    """Raised for invalid ExperimentConfig values or memory budget violations."""


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    max_genes: int
    test_timepoint: int = 11
    hardware_tier: str = "standard"
    n_cell_states: int = 8
    label_softening_temperature: float = 1.0
    log1p_transform: bool = True

    def __post_init__(self) -> None:
        if not self.log1p_transform:
            raise ValueError(
                "log1p_transform must be True — raw counts are not supported. "
                "Apply sc.pp.normalize_total() + sc.pp.log1p() before constructing DataConfig."
            )


@dataclass
class ContextConfig:
    n_bins: int = 6
    cells_per_bin: int = 5
    max_context_cells: int = 50
    allow_replacement: bool = True

    def __post_init__(self) -> None:
        total = self.n_bins * self.cells_per_bin
        if total > self.max_context_cells:
            raise ValueError(
                f"n_bins ({self.n_bins}) × cells_per_bin ({self.cells_per_bin}) = {total} "
                f"exceeds max_context_cells ({self.max_context_cells}). "
                "Increase max_context_cells or reduce n_bins/cells_per_bin."
            )


@dataclass
class ModelConfig:
    lr_col: float = 1e-5       # column attention — gene × gene
    lr_row: float = 1e-4       # row attention — feature → cell repr
    lr_icl: float = 5e-5       # ICL attention — cell × cell
    lr_emb: float = 1e-3       # column embeddings (always re-initialised)
    lr_head: float = 1e-3      # pseudotime head + composition head
    warmup_col_steps: int = 500   # steps before column attention is unfrozen
    warmup_icl_steps: int = 100   # steps before ICL attention is unfrozen
    output_head_init_bias: float = 0.5   # PseudotimeHead bias → sigmoid(0.5)≈0.62
    output_head_init_std: float = 0.01   # near-zero weight init for both heads
    bio_plausibility_passed: Optional[bool] = None  # populated post-training


@dataclass
class ExplainabilityConfig:
    shap_background_size: int = 100
    shap_background_seed: int = 42
    bio_plausibility_required: tuple[str, ...] = ("SOX2",)


@dataclass
class PerturbationConfig:
    perturbation_mask: dict[str, float] = field(default_factory=lambda: {"WLS": 0.0})
    pseudotime_delta_threshold: float = -0.05
    attention_drop_fraction: float = 0.1
    composition_shift_threshold: float = 0.05


@dataclass
class BenchmarkConfig:
    baselines: tuple[str, ...] = (
        "mean",
        "ridge_pca",
        "xgboost_regressor",
        "tabicl_finetune",
    )


# ---------------------------------------------------------------------------
# Hardware tier defaults (used by named presets)
# ---------------------------------------------------------------------------

HARDWARE_TIERS: dict[str, dict] = {
    # max_context_cells = n_bins × cells_per_bin (theoretical max; day-11 bin is excluded
    # at runtime, so actual cells fed = (n_bins-1) × cells_per_bin).
    "debug":    {"max_genes": 128,  "batch_size": 2,  "max_context_cells": 12},
    "standard": {"max_genes": 512,  "batch_size": 16, "max_context_cells": 50},
    "full":     {"max_genes": 1024, "batch_size": 32, "max_context_cells": 100},
}


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    run_id: str
    data: DataConfig
    context: ContextConfig
    model: ModelConfig
    explainability: ExplainabilityConfig
    perturbation: PerturbationConfig
    benchmark: BenchmarkConfig

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write config to experiments/{run_id}/config.json under PROJECT_ROOT."""
        from spatialmt.config.paths import PROJECT_ROOT
        run_dir = PROJECT_ROOT / "experiments" / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        with open(config_path, "w") as fh:
            json.dump(dataclasses.asdict(self), fh, indent=2)

    # ------------------------------------------------------------------
    # Hash — SHA-256 of serialised hyperparameters (run_id excluded)
    # ------------------------------------------------------------------

    @property
    def config_hash(self) -> str:
        d = dataclasses.asdict(self)
        d.pop("run_id", None)
        s = json.dumps(d, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Named presets
    # ------------------------------------------------------------------

    @classmethod
    def debug_preset(cls, run_id: str = "debug") -> "ExperimentConfig":
        tier = HARDWARE_TIERS["debug"]
        return cls(
            run_id=run_id,
            data=DataConfig(
                max_genes=tier["max_genes"],
                hardware_tier="debug",
            ),
            context=ContextConfig(
                n_bins=6,
                cells_per_bin=2,
                max_context_cells=tier["max_context_cells"],
            ),
            model=ModelConfig(),
            explainability=ExplainabilityConfig(),
            perturbation=PerturbationConfig(),
            benchmark=BenchmarkConfig(),
        )

    @classmethod
    def rotation_finetune(cls, run_id: str = "rotation_001") -> "ExperimentConfig":
        tier = HARDWARE_TIERS["standard"]
        return cls(
            run_id=run_id,
            data=DataConfig(
                max_genes=tier["max_genes"],
                hardware_tier="standard",
            ),
            context=ContextConfig(
                n_bins=6,
                cells_per_bin=5,
                max_context_cells=tier["max_context_cells"],
            ),
            model=ModelConfig(),
            explainability=ExplainabilityConfig(),
            perturbation=PerturbationConfig(),
            benchmark=BenchmarkConfig(),
        )

    @classmethod
    def rotation_baselines(cls, run_id: str = "rotation_baselines") -> "ExperimentConfig":
        tier = HARDWARE_TIERS["standard"]
        return cls(
            run_id=run_id,
            data=DataConfig(
                max_genes=tier["max_genes"],
                hardware_tier="standard",
            ),
            context=ContextConfig(
                n_bins=6,
                cells_per_bin=5,
                max_context_cells=tier["max_context_cells"],
            ),
            model=ModelConfig(),
            explainability=ExplainabilityConfig(),
            perturbation=PerturbationConfig(),
            benchmark=BenchmarkConfig(
                baselines=("mean", "ridge_pca", "xgboost_regressor"),
            ),
        )

    @classmethod
    def full_finetune(cls, run_id: str = "full_finetune") -> "ExperimentConfig":
        tier = HARDWARE_TIERS["full"]
        return cls(
            run_id=run_id,
            data=DataConfig(
                max_genes=tier["max_genes"],
                hardware_tier="full",
            ),
            context=ContextConfig(
                n_bins=6,
                cells_per_bin=5,
                max_context_cells=tier["max_context_cells"],
            ),
            model=ModelConfig(),
            explainability=ExplainabilityConfig(),
            perturbation=PerturbationConfig(),
            benchmark=BenchmarkConfig(),
        )

    @classmethod
    def scratch_preset(cls, run_id: str = "scratch") -> "ExperimentConfig":
        """No pretrained weights — ablation [Phase 6]."""
        cfg = cls.rotation_finetune(run_id=run_id)
        # Phase 6: add finetune_strategy flag to ModelConfig when needed
        return cfg

    @classmethod
    def no_icl_preset(cls, run_id: str = "no_icl") -> "ExperimentConfig":
        """Single cell input — no in-context learning [Phase 6]."""
        tier = HARDWARE_TIERS["standard"]
        return cls(
            run_id=run_id,
            data=DataConfig(
                max_genes=tier["max_genes"],
                hardware_tier="standard",
            ),
            context=ContextConfig(
                n_bins=1,
                cells_per_bin=1,
                max_context_cells=tier["max_context_cells"],
            ),
            model=ModelConfig(),
            explainability=ExplainabilityConfig(),
            perturbation=PerturbationConfig(),
            benchmark=BenchmarkConfig(),
        )
