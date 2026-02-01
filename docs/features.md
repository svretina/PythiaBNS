# Features

## Output Structure & Smart Naming

PythiaBNS automatically organizes results to keep large campaigns structured.

- **Study Folder**: Named after the configuration file (or the `name` field).
- **Run Subfolders**: Each simulation gets a dedicated subfolder with an informative name, automatically generated from the varying parameters (e.g., `run001_inj_three_sines_snr50.0_model_three_sines`).

**Resulting Directory Tree:**

```text
results/
└── MyStudy/
    ├── run000_inj_three_sines_snr50.0_model_A/
    ├── run001_inj_three_sines_snr100.0_model_A/
    └── ...
```

## Automated Plotting

You can configure automated plot generation directly in your YAML config. Plots are generated at the end of each simulation and saved in the respective run folder.

### Configuration Example

```yaml
plotting:
  enabled: true
  plots: ["corner", "trace"] # Supported: corner, trace
  settings:
    corner:
      show_titles: true
      quantiles: [0.16, 0.5, 0.84]
    trace:
      dpi: 150
```

## Sampler Plugins

PythiaBNS supports multiple sampling backends via its plugin system.

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
