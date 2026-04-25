"""
Unbias AI Decision — Configuration Loader
==========================================
Loads config.yaml and provides typed access to all pipeline settings.
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class DataAuditConfig:
    underrepresentation_threshold: float = 0.10
    missing_data_gap_threshold: float = 0.15


@dataclass
class BiasDetectionConfig:
    cramers_v_threshold: float = 0.30
    correlation_threshold: float = 0.30
    chi_square_alpha: float = 0.05
    small_sample_cutoff: int = 30
    fisher_cell_cutoff: int = 5


@dataclass
class FairnessMetricsConfig:
    demographic_parity_gap_threshold: float = 0.10
    disparate_impact_threshold: float = 0.80
    equalized_odds_threshold: float = 0.10


@dataclass
class RiskScoringConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {
        "cramers_v": 0.30, "dp_gap": 0.30, "imbalance_ratio": 0.15,
        "missing_gap": 0.10, "explainability_disparity": 0.10, "label_bias": 0.05
    })
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.30, "high": 0.60
    })
    sensitivity_perturbation: float = 0.20
    sensitivity_runs: int = 100


@dataclass
class MitigationConfig:
    smote_neighbors: int = 5
    pareto_models: int = 25
    accuracy_loss_tolerance: float = 0.05


@dataclass
class MonitoringConfig:
    drift_threshold: float = 0.15
    check_interval_days: int = 30
    alert_on_di_below: float = 0.80


@dataclass
class PipelineConfig:
    sensitive_attributes: List[str] = field(default_factory=lambda: ["gender", "caste", "region"])
    target_column: str = "approved"
    data_audit: DataAuditConfig = field(default_factory=DataAuditConfig)
    bias_detection: BiasDetectionConfig = field(default_factory=BiasDetectionConfig)
    fairness_metrics: FairnessMetricsConfig = field(default_factory=FairnessMetricsConfig)
    risk_scoring: RiskScoringConfig = field(default_factory=RiskScoringConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


def load_config(config_path: str = None) -> PipelineConfig:
    """Load pipeline configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

    if not os.path.exists(config_path):
        print(f"[Config] No config.yaml found at {config_path}, using defaults.")
        return PipelineConfig()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = PipelineConfig()
    cfg.sensitive_attributes = raw.get("sensitive_attributes", cfg.sensitive_attributes)
    cfg.target_column = raw.get("target_column", cfg.target_column)

    if "data_audit" in raw:
        cfg.data_audit = DataAuditConfig(**raw["data_audit"])
    if "bias_detection" in raw:
        cfg.bias_detection = BiasDetectionConfig(**raw["bias_detection"])
    if "fairness_metrics" in raw:
        cfg.fairness_metrics = FairnessMetricsConfig(**raw["fairness_metrics"])
    if "risk_scoring" in raw:
        cfg.risk_scoring = RiskScoringConfig(**raw["risk_scoring"])
    if "mitigation" in raw:
        cfg.mitigation = MitigationConfig(**raw["mitigation"])
    if "monitoring" in raw:
        cfg.monitoring = MonitoringConfig(**raw["monitoring"])

    return cfg
