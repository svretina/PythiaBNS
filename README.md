# PythiaBNS: Robust BNS Post-Merger Parameter Estimation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PythiaBNS** is a modular, production-grade Python library designed for Bayesian Parameter Estimation (PE) of Binary Neutron Star (BNS) post-merger signals. It is architected to support next-generation (3G) gravitational wave observatories like Einstein Telescope (ET) and Cosmic Explorer (CE).

## Use Cases

- **Analytic Modeling**: Fit complex time-domain or frequency-domain waveform models to Numerical Relativity (NR) data.
- **Inspiral-Informed Priors**: Constrain post-merger parameters using empirical relations (e.g., [Vretinaris et al. 2020](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.084039])) derived from inspiral measurements (mass, tidal deformability).
- **High-Efficiency Sampling**: Leverage `pocomc` (Preconditioned Monte Carlo) for efficient sampling of difficult posteriors.
- **Benchmarking**: Validate models against a curated catalog of NR waveforms.

## 🚀 Installation

This project is managed with [uv](https://github.com/astral-sh/uv).

```bash
# Clone the repository
git clone https://github.com/svretina/pythiabns.git
cd pythiabns

# Install dependencies and create virtual environment
uv sync
```

## 📂 Project Structure

The library is organized into specialized modules within `src/pythiabns`:

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
  - `samplers/`: Pluggable sampler wrappers.
    - `pocomc.py`: Robust wrapper for `pocomc`.
    - `zeus.py`: Wrapper for `zeus-mcmc`.
    - `numpyro_sampler.py`: JAX-based `NumPyro` (SA) wrapper.
    - `blackjax_sampler.py`: JAX-based `BlackJAX` (RWM) wrapper.
    - `tempest.py`: `Tempest` Persistent Sampler wrapper.
    - `stan_sampler.py`: `CmdStanPy` wrapper.
    - `nutpie_sampler.py`: `Nutpie` (NUTS) wrapper.
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
# Run with config using uv
uv run python src/pythiabns/spine.py config.yaml
```

### Supported Samplers

PythiaBNS supports multiple backends via its plugin system:

| Sampler | Plugin Name | Type | Status |
| :--- | :--- | :--- | :--- |
| **PocoMC** | `pocomc` | Preconditioned MC | ✅ Functional |
| **Zeus** | `zeus` | Ensemble Slice | ✅ Functional |
| **NumPyro** | `numpyro` | JAX (SA) | ✅ Functional |
| **BlackJAX** | `blackjax` | JAX (RWM) | ✅ Functional |
| **Tempest** | `tempest` | Persistent Sampler | ✅ Functional |
| **Stan** | `stan` | HMC/NUTS | 🏗️ Wrapper (Plugin) |
| **Nutpie** | `nutpie` | Rust NUTS | 🏗️ Wrapper (Plugin) |
| **Bilby Natives** | `dynesty`, etc. | Various | ✅ Functional |

### Adding New Models

GW Pipe uses a registry system. You can add new models without modifying the core code by using the `@register_model` decorator.

**Create a new file `my_models.py`:**

```python
import numpy as np
from pythiabns.core.registry import ModelRegistry

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
uv run python src/pythiabns/spine.py test_config.yaml
```

Check `results/verification` for the output.

## Tutorial: Custom Models

PythiaBNS makes it easy to define custom waveform models and perform infernece.

### 1. Define your model

Create a file like `src/pythiabns/models/tutorial_models.py`:

```python
from pythiabns.core.registry import ModelRegistry
import numpy as np

@ModelRegistry.register("three_sines")
def three_sines(time, a1, f1, p1, a2, f2, p2, a3, f3, p3, **kwargs):
    plus = (
        a1 * np.sin(2 * np.pi * f1 * time + p1) +
        a2 * np.sin(2 * np.pi * f2 * time + p2) +
        a3 * np.sin(2 * np.pi * f3 * time + p3)
    )
    return {"plus": plus, "cross": plus} # Simplified cross
```

### 2. Configure Priors

Define your priors in `src/pythiabns/priors/tutorial.priors`:

```python
a1 = bilby.core.prior.Uniform(1e-23, 2e-22, name="a1", latex_label="$A_1$")
f1 = bilby.core.prior.Uniform(100, 200, name="f1", latex_label="$f_1$")
p1 = bilby.core.prior.Uniform(0.0, 2*np.pi, name="p1", latex_label="$\phi_1$", boundary="periodic")
# ... add for a2, a3 ...
```

### 3. Run Simulation

Create a `tutorial.yaml` and run:

```bash
uv run python src/pythiabns/spine.py tutorial.yaml
```

### 4. Results

Inference on 3 sine waves demonstrates excellent parameter recovery:

![Corner Plot](docs/images/tutorial_corner_plot.png)

## 📄 Citation

If you use **PythiaBNS** in your research, please cite:

```bibtex
@article{g1qs-j74x,
  title = {Robust and fast parameter estimation for gravitational waves from binary neutron star merger remnants},
  author = {Vretinaris, Stamatis and Vretinaris, Georgios and Mermigkas, Christos and Karamanis, Minas and Stergioulas, Nikolaos},
  journal = {Phys. Rev. D},
  volume = {113},
  issue = {2},
  pages = {024012},
  numpages = {19},
  year = {2026},
  month = {Jan},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevD.113.024012},
  url = {https://link.aps.org/doi/10.1103/PhysRevD.113.024012}
}
```
