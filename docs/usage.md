# Usage Guide

## Running the Pipeline

The main orchestrator is `spine.py`. It reads the config, expands the job matrix, and executes simulations.

```bash
# Run with config using uv
uv run python src/pythiabns/spine.py config.yaml
```

## Configuration

Simulations are configured using YAML files, validated by Pydantic schemas.

### Example `config.yaml`

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
      target: "BAM:0088:R01" 
    - mode: "file" 
      target: "/path/to/my/waveform/folder"
    - mode: "analytic"
      target: "three_sines"
      
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

## Combinatorial Matrix

PythiaBNS is designed to make parameter studies effortless through its combinatorial matrix system. Any argument in the `matrix` section of the configuration file can be provided as a list. The pipeline will automatically generate and run simulations for **all Cartesian products** of these lists.

### Example Study

```yaml
matrix:
  # ...
  snr: [20, 50, 100]            # 3 values
  model: ["model_A", "model_B"] # 2 models
  sampler:
    plugin: "pocomc"
    settings:
      n_cpus: [8, 16]           # 2 settings
```

This configuration will automatically trigger **3 × 2 × 2 = 12 distinct simulations**, covering every combination of SNR, model, and CPU count.
