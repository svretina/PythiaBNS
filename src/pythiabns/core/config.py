from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from pythiabns.core import constants


class SamplerConfig(BaseModel):
    plugin: str = "pocomc"
    settings: dict[str, Any] = Field(default_factory=dict)


class PriorConfig(BaseModel):
    mode: str = "file"  # "file" or "empirical"
    source: str = "easter.priors"  # Filename or relation name
    constraints: str | None = None  # Optional constraints file


class InjectionConfig(BaseModel):
    """Configuration for waveform injection source."""

    # "nr": load from STRAIN_PATH via ID
    # "file": load from absolute/relative path
    # "analytic": simulate using a model from registry
    mode: str = "nr"
    target: str | None = None  # NR ID, File Path, or Model Name
    parameters: dict[str, float] = Field(default_factory=dict)


class JobMatrix(BaseModel):
    """Configuration for expanding into multiple simulations."""

    # Support both legacy 'waveform' and new modular 'injection'
    waveform: list[str] | None = None
    injection: list[InjectionConfig] | None = None

    snr: list[float] = Field(default_factory=lambda: [50.0])
    model: list[str]

    sampler: SamplerConfig
    priors: PriorConfig

    # Extra model parameters like nfreqs
    model_params: dict[str, Any] = Field(default_factory=dict)
    # Legacy: injection_parameters at top level
    injection_parameters: dict[str, float] = Field(default_factory=dict)


class SimulationConfig(BaseModel):
    """Configuration for a single simulation run."""

    # Single instance of injection source
    injection: InjectionConfig

    snr: float
    model: str
    sampler: SamplerConfig
    priors: PriorConfig

    model_params: dict[str, Any] = Field(default_factory=dict)
    # Legacy support
    waveform: str | None = None
    injection_parameters: dict[str, float] = Field(default_factory=dict)


class PlottingConfig(BaseModel):
    enabled: bool = True
    plots: list[str] = Field(default_factory=lambda: ["corner", "trace"])
    settings: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    name: str = "GW_Experiment"
    output_dir: Path = constants.RESULTS_PATH
    imports: list[str] = Field(default_factory=list)

    plotting: PlottingConfig = Field(default_factory=PlottingConfig)

    matrix: JobMatrix


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(**data)
