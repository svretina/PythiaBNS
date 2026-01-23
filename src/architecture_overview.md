# gw_pipe: Architectural Overview

This document provides a sketch of the library's functionality, component interactions, and execution flow.

## 1. High-Level Hierarchy

The library follows a modular design where core orchestration is decoupled from physics models and configuration parsing.

```mermaid
graph TD
    A[spine.py: Entry Point] --> B[config.py: Parser]
    A --> C[spine.run: Orchestrator]
    C --> D[spine.main: Simulation Task]
    
    subgraph "Simulation Task Components"
        D --> E[NR_strains.py: Data Loading]
        D --> F[ifo.py: Detector Handling]
        D --> G[source_model.py: Registry/Models]
        D --> H[priors.py: Bayesian Priors]
        D --> I[bilby: Inference Engine]
    end

    G -.-> J[registry.py: Global Registry]
    H -.-> J
    K[models.py: User Extensions] -.-> J
```

## 2. Execution Flow

The typical workflow follows these steps:

1. **Configuration Parsing**:
    * `spine.py` reads the `.cfg` file via `config.py`.
    * `create_iterator_dict()` generates a Cartesian product of all parameters (e.g., multiple SNR values x multiple models).
2. **Simulation Orchestration**:
    * `spine.run()` manages a `multiprocess.Pool` to run several simulations in parallel.
3. **Individual Simulation (`main`)**:
    * **Data Preparation**: `NumericalData` loads the NR waveform (plus/cross) and metadata.
    * **Injection**: `InterferometerHandler` takes the NR signal and injects it into a simulated detector network (H1, L1, V1) with or without Gaussian noise.
    * **Model Lookup**: The `ModelRegistry` retrieves the requested waveform model and its parameter conversion function.
    * **Prior Setup**: `get_priors()` finds the relevant `.priors` file and applies empirical relations (e.g., $f_{\mathrm{peak}}$ predictions).
    * **Sampling**: Bilby runs the chosen sampler (`dynesty` or `pocomc`) to perform the Bayesian parameter estimation.
    * **Cleanup**: Results are saved as JSON and PDF plots in a structured output directory.

## 3. Component Breakdown

| Component | Responsibility | Key Class/Function |
| :--- | :--- | :--- |
| **spine.py** | Entry point and parallel orchestration. | `run()`, `main()` |
| **config.py** | Handles `.cfg` files and produces task iterators. | `Config` |
| **registry.py** | Central hub for model and conversion discovery. | `ModelRegistry` |
| **source_model.py**| Built-in physics models for post-merger signals. | `lorentzian`, `easter_model` |
| **models.py** | User-facing file to add custom GW models. | `@register_model` |
| **priors.py** | Maps physical parameters to Bayesian priors. | `get_priors()` |
| **ifo.py** | Interfaces with Bilby detectors and injections. | `InterferometerHandler` |
| **NR_strains.py** | Low-level loading of NR data (BAM, THC, etc.). | `NumericalData` |
| **global_vars.py** | Consistent paths and constant management. | `project_path` |

## 4. How Models are Called

When `bilby` evaluates the likelihood, it calls the `WaveformGenerator`. The generator then calls the function registered in our logic:

```mermaid
sequenceDiagram
    participant B as Bilby Sampler
    participant WG as WaveformGenerator
    participant SM as source_model.py
    participant R as registry.py
    
    B->>WG: Request frequency_domain_strain(parameters)
    WG->>WG: Apply parameter_conversion_func()
    WG->>SM: Call registered model function
    SM-->>B: Return {"plus": h+, "cross": hx}
```

---
*Created by Antigravity*
