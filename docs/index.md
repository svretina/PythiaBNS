# PythiaBNS: Robust BNS Post-Merger Parameter Estimation

**PythiaBNS** is a modular Python library designed for Bayesian Parameter Estimation (PE) of Binary Neutron Star (BNS) post-merger waveforms. It is architected to support next-generation (3G) gravitational wave observatories like Einstein Telescope (ET) and Cosmic Explorer (CE).

**PythiaBNS** implements the method of **Empirical Priors** of [Vretinaris et al. 2026](https://journals.aps.org/prd/abstract/10.1103/g1qs-j74x), where a set of informed priors are used to constrain Parameter Estimation of post-merger waveforms leading to robust and fast inference.

## Key Use Cases

- **Analytic Modeling**: Fit complex time-domain or frequency-domain waveform models to Numerical Relativity (NR) data.
- **Inspiral-Informed Priors**: Constrain post-merger parameters using empirical relations derived from inspiral measurements (mass, tidal deformability).
- **High-Efficiency Sampling**: Leverage `pocomc` (Preconditioned Monte Carlo) for efficient sampling of difficult posteriors.
- **Benchmarking**: Validate models against a curated catalog of NR waveforms.

## Quick Install

This project is managed with [uv](https://github.com/astral-sh/uv).

```bash
# Clone the repository
git clone https://github.com/svretina/pythiabns.git
cd pythiabns

# Install dependencies
uv sync
```

See the [Usage](usage.md) guide for detailed instructions.

## Citation

If you use **PythiaBNS** in your research, please cite:

```bibtex
@article{g1qs-j74x,
  title = {Robust and fast parameter estimation for gravitational waves from binary neutron star merger remnants},
  author = {Vretinaris, Stamatis and Vretinaris, Georgios and Mermigkas, Christos and Karamanis, Minas and Stergioulas, Nikolaos},
  journal = {Phys. Rev. D},
  volume = {113},
  issue = {2},
  pages = {024012},
  year = {2026},
  doi = {10.1103/PhysRevD.113.024012},
  url = {https://link.aps.org/doi/10.1103/PhysRevD.113.024012}
}
```
