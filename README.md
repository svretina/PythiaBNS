# gw_pipe

A Python library for Gravitational Wave (GW) post-merger signal analysis.

## Features

- Numerical Relativity (NR) data handling (BAM, THC, Soultanis datasets).
- Parameter estimation using `bilby`, `dynesty`, and `pocomc`.
- Multi-frequency source modeling for post-merger signals.
- Automated PSD generation and visualization.
- High-level interface for running large suites of simulations in parallel.

## Installation

```bash
pip install gw-pipe
```

Or using poetry:

```bash
poetry add gw-pipe
```

## Usage

```bash
python -m gw_pipe.spine -c config.cfg
```

## Configuration

Standard configuration is done through a `.cfg` file. See `config.cfg` for examples of:

- Sampler settings (`pocomc`, `dynesty`).
- Model parameters and frequency counts.
- Injection parameters (SNR, noise).

## Custom Models

Users can define new models in `src/gw_pipe/models.py`.
Use the `@register_model` decorator to make them available:

```python
from gw_pipe.registry import register_model

@register_model("my_new_model")
def my_waveform(f, param1, param2, **kwargs):
    # logic to generate plus/cross strains
    return {"plus": ..., "cross": ...}
```

You can then use `name = my_new_model` in the `[Model]` section of your configuration file.

## License

GNU GPL v3
