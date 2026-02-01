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

class InjectionConfig(BaseModel):
    """Configuration for waveform injection source."""
    # "nr": load from STRAIN_PATH via ID
    # "file": load from absolute/relative path
    # "analytic": simulate using a model from registry
    mode: str = "nr"
    target: Optional[str] = None # NR ID, File Path, or Model Name
    parameters: Dict[str, float] = Field(default_factory=dict)

class JobMatrix(BaseModel):
    """Configuration for expanding into multiple simulations."""
    # Support both legacy 'waveform' and new modular 'injection'
    waveform: Optional[List[str]] = None
    injection: Optional[List[InjectionConfig]] = None
    
    snr: List[float] = Field(default_factory=lambda: [50.0])
    model: List[str]
    
    sampler: SamplerConfig
    priors: PriorConfig
    
    # Extra model parameters like nfreqs
    model_params: Dict[str, Any] = Field(default_factory=dict)
    # Legacy: injection_parameters at top level
    injection_parameters: Dict[str, float] = Field(default_factory=dict)

class SimulationConfig(BaseModel):
    """Configuration for a single simulation run."""
    # Single instance of injection source
    injection: InjectionConfig
    
    snr: float
    model: str
    sampler: SamplerConfig
    priors: PriorConfig
    
    model_params: Dict[str, Any] = Field(default_factory=dict)
    # Legacy support
    waveform: Optional[str] = None 
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
