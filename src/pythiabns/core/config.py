from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import yaml
from pydantic import BaseModel, Field

from pythiabns.core import constants

class SamplerConfig(BaseModel):
    plugin: str = "pocomc"
    settings: Dict[str, Any] = Field(default_factory=dict)

class PriorConfig(BaseModel):
    mode: str = "file" # "file" or "empirical"
    source: str = "easter.priors" # Filename or relation name
    constraints: Optional[str] = None # Optional constraints file

class JobMatrix(BaseModel):
    """Configuration for expanding into multiple simulations."""
    waveform: List[str]
    snr: List[float] = Field(default_factory=lambda: [50.0])
    model: List[str]
    
    sampler: SamplerConfig
    priors: PriorConfig
    
    # Extra model parameters like nfreqs
    model_params: Dict[str, Any] = Field(default_factory=dict)
    injection_parameters: Dict[str, float] = Field(default_factory=dict)

class SimulationConfig(BaseModel):
    """Configuration for a single simulation run."""
    waveform: str
    snr: float
    model: str
    sampler: SamplerConfig
    priors: PriorConfig
    
    model_params: Dict[str, Any] = Field(default_factory=dict)
    injection_parameters: Dict[str, float] = Field(default_factory=dict)

class ExperimentConfig(BaseModel):
    name: str = "GW_Experiment"
    output_dir: Path = constants.RESULTS_PATH
    imports: List[str] = Field(default_factory=list)
    matrix: JobMatrix

def load_config(path: Union[str, Path]) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    return ExperimentConfig(**data)
