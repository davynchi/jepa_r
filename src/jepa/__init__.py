"""Compact JEPA research primitives for time-series experiments."""

from jepa.config import (
    ExperimentConfig,
    PairedReplicateSeeds,
    apply_paired_replicate,
    config_from_dict,
    config_identity_hash,
    derive_paired_replicate,
    load_config,
)
from jepa.data import LatentDynamicsDataset, build_dataset_splits, generate_system
from jepa.metrics import (
    MetricValue,
    RepresentationMetrics,
    RidgeProbe,
    WeightedMean,
    compute_representation_metrics,
    fit_ridge_probe,
    global_gradient_norm,
    ridge_r2_score,
)
from jepa.models import (
    LinearEncoder,
    LinearPredictor,
    TanhEncoder,
    TanhPredictor,
    build_model_pair,
)
from jepa.reporting import SweepResult, aggregate_rows, run_sweep, variant_label
from jepa.training import (
    JEPACore,
    OptimizationPolicy,
    TrainResult,
    build_jepa_core,
    build_run_id,
    compute_loss,
    optimizer_parameters,
    resolve_policy,
    train_experiment,
    train_step,
)

__all__ = [
    "ExperimentConfig",
    "JEPACore",
    "LatentDynamicsDataset",
    "LinearEncoder",
    "LinearPredictor",
    "MetricValue",
    "OptimizationPolicy",
    "PairedReplicateSeeds",
    "RepresentationMetrics",
    "RidgeProbe",
    "TanhEncoder",
    "TanhPredictor",
    "TrainResult",
    "SweepResult",
    "WeightedMean",
    "apply_paired_replicate",
    "aggregate_rows",
    "build_dataset_splits",
    "build_jepa_core",
    "build_model_pair",
    "build_run_id",
    "compute_loss",
    "compute_representation_metrics",
    "config_from_dict",
    "config_identity_hash",
    "derive_paired_replicate",
    "generate_system",
    "global_gradient_norm",
    "load_config",
    "fit_ridge_probe",
    "optimizer_parameters",
    "resolve_policy",
    "ridge_r2_score",
    "run_sweep",
    "train_experiment",
    "train_step",
    "variant_label",
]

__version__ = "0.1.0"
