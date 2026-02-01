# GW Pipe Architecture Overview

This document provides a high-level overview of the `gw_pipe` library architecture, designed for modularity, extensibility, and 3G-detector readiness.

## 🏗️ High-Level Structure

The library is organized into five main pillars located in `src/gw_pipe`:

| Module | Purpose | Key Components |
| :--- | :--- | :--- |
| **`core`** | Configuration & Infrastructure | `ExperimentConfig`, `ModelRegistry`, `constants` |
| **`models`** | Physics & Waveforms | `WaveformModel` (Protocol), `EmpiricalRelation` |
| **`inference`** | Bayesian Inference Engine | `PriorFactory`, `PostMergerLikelihood`, `PocoMCWrapper` |
| **`detectors`** | Instrument Modeling | `DetectorNetwork` (LIGO/Virgo/ET/CE) |
| **`data_utils`** | Data I/O & Processing | `NumericalWaveform` (NR loader), Signal Processing |

## 🔄 Execution Flow

The pipeline execution is orchestrated by `spine.py`.

1. **Configuration Loading**:
    * `core.config.load_config` reads the YAML file.
    * Data is validated against `pydantic` schemas (`ExperimentConfig`, `JobMatrix`).
    * Plugins are imported dynamically if specified.

2. **Job Matrix Expansion**:
    * The `JobMatrix` from the config is expanded into individual `SimulationConfig` objects (Cartesian product of waveforms, models, SNRs, etc.).

3. **Simulation Initialization** (`run_simulation`):
    * **Data**: `NumericalWaveform` loads NR data (HDF5/ASCII), converts to SI units, and scales to 1Mpc.
    * **Detectors**: `DetectorNetwork` initializes interferometers (e.g., H1, L1, V1) and sets up noise/PSDs.
    * **Model**: The requested analytic model is retrieved from `ModelRegistry` using user-specified parameters (e.g., `nfreqs=3`).

4. **Inference Setup**:
    * **Priors**: `PriorFactory` builds the `Bilby` prior dictionary.
        * *File Mode*: Loads from `.priors` files.
        * *Empirical Mode*: Uses `EmpiricalRelation` (e.g., VSB, Koutalios) to predict parameters properties from EOS/mass data.
    * **Likelihood**: `PostMergerLikelihood` (subclass of `bilby.GravitationalWaveTransient`) is instantiated.

5. **Sampling**:
    * If `sampler: pocomc` is selected, the `PocoMCWrapper` handles the interface between Bilby's likelihood and the Preconditioned Monte Carlo sampler, managing multiprocessing and serialization.
    * Results are saved as JSON and Pickle files in the output directory.

## 🧩 Key Components

### 1. `core.registry.ModelRegistry`

A central registry pattern that allows decoupling model implementation from the orchestrator.

* **Usage**: Decorate functions in `models/waveforms.py` with `@ModelRegistry.register("name", domain="time", ...)`.
* **Benefit**: Users can add new models in external scripts without modifying `gw_pipe` source code.

### 2. `inference.priors.PriorFactory`

Abstracts the complex logic of prior generation.

* It handles the conversion of strings (from config) to executable code or file loads.
* It injects "Empirical Priors" by querying `relations.py` for physics-informed constraints (e.g., relating peak frequency to tidal deformability).

### 3. `inference.samplers.PocoMCWrapper`

A custom wrapper designed to bridge `bilby` and `pocomc`.

* Handles the translation of `bilby.PriorDict` (and constraints) to `pocomc`'s expected `Prior` interface.
* Manages `multiprocess` pools to ensure picklability of likelihood functions during parallel sampling.

## 📊 Data Flow

```mermaid
graph TD
    Config[YAML Config] -->|Parse & Validate| Spine(Orchestrator)
    NR[NR Data (HDF5)] -->|Load| Spine
    
    subgraph "Phase 1: Setup"
    Spine -->|Init| Detectors[DetectorNetwork]
    Spine -->|Get| Registry[ModelRegistry]
    Spine -->|Create| Priors[PriorFactory]
    end
    
    subgraph "Phase 2: Sampling"
    Priors --> Sampler[PocoMCSampler]
    Detectors --> Likelihood
    Registry --> Likelihood
    Likelihood --> Sampler
    end
    
    Sampler -->|Output| Results[JSON / Pickle]
```
