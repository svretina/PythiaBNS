# GW Pipe: Robust BNS Post-Merger Parameter Estimation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/poetry-managed-blue)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GW Pipe** is a modular, production-grade Python library designed for Bayesian Parameter Estimation (PE) of Binary Neutron Star (BNS) post-merger signals. It is architected to support next-generation (3G) gravitational wave observatories like Einstein Telescope (ET) and Cosmic Explorer (CE).

## Use Cases

- **Analytic Modeling**: Fit complex time-domain or frequency-domain waveform models to Numerical Relativity (NR) data.
- **Inspiral-Informed Priors**: Constrain post-merger parameters using empirical relations (e.g., [Vretinaris et al. 2020](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.084039])) derived from inspiral measurements (mass, tidal deformability).
- **High-Efficiency Sampling**: Leverage `pocomc` (Preconditioned Monte Carlo) for efficient sampling of difficult posteriors.
- **Benchmarking**: Validate models against a curated catalog of NR waveforms.

## 🚀 Installation

This project is managed with [Poetry](https://python-poetry.org/).

```bash
# Clone the repository
git clone https://github.com/your-username/gw_pipe.git
cd gw_pipe

# Install dependencies (including dev tools)
poetry install

# Activate the virtual environment
poetry shell
```

## 📂 Project Structure

The library is organized into specialized modules within `src/gw_pipe`:

- `core/`: Core infrastructure.
  - `config.py`: Configuration schemas (`Pydantic`) and loading logic.
  - `registry.py`: Central registry for models and relations.
  - `constants.py`: Physical constants and path definitions.
- `models/`: Waveform and physics models.
  - `waveforms.py`: Analytic waveform implementations (e.g., `easter`, `lorentzian`).
  - `relations.py`: Empirical relations linking physical EOS params to waveform features.
  - `interface.py`: `WaveformModel` Protocol definition.
- `inference/`: Bayesian inference tools.
  - `priors.py`: Prior generation factory, supporting file-based and empirical priors.
  - `samplers/pocomc.py`: Robust wrapper for `pocomc` sampler.
  - `likelihood.py`: Custom likelihood classes (extending `bilby`).
- `detectors/`: Detector network management (`network.py`).
- `data_utils/`: Data processing and NR waveform loading (`nr.py`, `processing.py`).

## 🛠️ Configuration

Simulations are configured using YAML files, validated by Pydantic schemas.

**Example `config.yaml`:**

```yaml
name: "Experiment_01"
output_dir: "results/run1"

# Import custom plugins (optional)
imports: 
  - "my_custom_models" 

matrix:
  # Batch processing: Generate simulations for all combinations
  waveform: 
    - "BAM:0088:R01"  # NR Simulation ID
  snr: 
    - 50.0  # Target SNR
  model: 
    - "easter_half_reparam" # Analytic Model Name
  
  sampler:
    plugin: "pocomc"
    settings:
      npoints: 1000
      corr_threshold: 0.75
      n_cpus: 16

  priors:
    mode: "file" # or "empirical" to use relations
    source: "easter_half_reparam.priors"
    
  # Specific model arguments (passed to get_model)
  model_params:
      nfreqs: 3
```

## ⚡ Usage

### Running the Pipeline

The main orchestrator is `spine.py`. It reads the config, expands the job matrix, and executes simulations.

```bash
# Ensure src is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run with config
poetry run python3 src/gw_pipe/spine.py config.yaml
```

### Adding New Models

GW Pipe uses a registry system. You can add new models without modifying the core code by using the `@register_model` decorator.

**Create a new file `my_models.py`:**

```python
import numpy as np
from gw_pipe.core.registry import ModelRegistry

def my_conversion_func(params):
    # logic ...
    return params, ["added_key"]

@ModelRegistry.register("my_model", nfreqs=2, conversion_func=my_conversion_func, domain="time")
def my_model_func(time, param1, param2, **kwargs):
    # Compute h_plus, h_cross
    return {"plus": h_plus, "cross": h_cross}
```

**In your `config.yaml`:**

```yaml
imports:
  - "my_models" # Tells spine.py to import this module
matrix:
  model: ["my_model"]
```

## 🔍 Validation

To verify the installation, you can run the simplified test configuration:

```bash
poetry run python3 src/gw_pipe/spine.py test_config.yaml
```

Check `results/verification` for the output.
