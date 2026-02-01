# PythiaBNS: Robust BNS Post-Merger Parameter Estimation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![CI](https://github.com/svretina/pythiabns/actions/workflows/ci.yml/badge.svg)](https://github.com/svretina/pythiabns/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/svretina/pythiabns/branch/main/graph/badge.svg)](https://codecov.io/gh/svretina/pythiabns)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PythiaBNS** is a modular Python library designed for Bayesian Parameter Estimation (PE) of Binary Neutron Star (BNS) post-merger waveforms. It is architected to support next-generation (3G) gravitational wave observatories like Einstein Telescope (ET) and Cosmic Explorer (CE). **PythiaBNS** implements the method of [Vretinaris et al. 2020](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.101.084039), where a set of informed priors are used to constrain Parameter Estimation of post-merger waveforms leading to robust and fast inference..

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
  # Modular Injection
  injection:
    - mode: "nr"
      target: "BAM:0088:R01" # Presets from STRAIN_PATH
    - mode: "file" 
      target: "/path/to/my/waveform/folder" # NR format folder or .txt file
    - mode: "analytic"
      target: "three_sines" # Simulated from registry
      
  snr: [50.0, 100.0]
  model: ["easter_half_reparam"] 
  
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
uv run python src/pythiabns/spine.py examples/test_config.yaml
```

Check `results/verification` for the output.

## 📚 Documentation

PythiaBNS uses `pdoc` to generate its API documentation directly from docstrings.

### Local Generation

To generate and view the documentation locally, use the provided `Makefile`:

```bash
make docs
```

This will generate HTML files in `docs/html/`. You can open `docs/html/index.html` in your browser.

### Automated Deployment

A GitHub Action is configured to automatically generate and deploy the latest documentation to **GitHub Pages** whenever changes are pushed to the `main` branch.

## 📁 Custom Waveforms

PythiaBNS allows you to inject custom waveforms for PE studies. Specify `mode: "file"` in your configuration and provide a `target` path.

### Supported Formats

1. **NR Format (Standard)**:
    A directory containing:
    - `metadata.txt`: Contains parameters like `id_mass_starA`.
    - `data.h5`: HDF5 file with `/rh_22/l2_m2_rXXX` groups containing time-series data.
2. **Simple Text Format**:
    A plain text file with three whitespace-separated columns:
    - `time` (seconds)
    - `h_plus` (dimensionless strain at 1 Mpc)
    - `h_cross` (dimensionless strain at 1 Mpc)

Example `injection` config for a text file:

```yaml
matrix:
  injection:
    - mode: "file"
      target: "data/my_waveform.txt"
```

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
uv run python src/pythiabns/spine.py examples/tutorial.yaml
```

### 4. Results

Inference on 3 sine waves demonstrates excellent parameter recovery. Using an **SNR of 100** and **nlive=100**, we successfully retrieve the injected values:

| Parameter | Injected | Retrieved (Median) |
| :--- | :--- | :--- |
| **$A_1$** | $1.00 \times 10^{-22}$ | $1.00 \times 10^{-22} \pm 2.8 \times 10^{-24}$ |
| **$f_1$** | $150.0$ Hz | $150.0 \pm 0.016$ Hz |
| **$\phi_1$** | $0.0$ rad | $0.11 \pm 3.1$ rad |
| **$A_2$** | $0.50 \times 10^{-22}$ | $0.50 \times 10^{-22} \pm 2.5 \times 10^{-24}$ |
| **$f_2$** | $350.0$ Hz | $350.0 \pm 0.028$ Hz |
| **$\phi_2$** | $1.0$ rad | $1.01 \pm 0.10$ rad |
| **$A_3$** | $0.80 \times 10^{-22}$ | $0.79 \times 10^{-22} \pm 2.8 \times 10^{-24}$ |
| **$f_3$** | $550.0$ Hz | $550.0 \pm 0.020$ Hz |
| **$\phi_3$** | $2.0$ rad | $2.01 \pm 0.07$ rad |

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
