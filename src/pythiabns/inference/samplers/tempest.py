import logging
from pathlib import Path
from typing import Any

import bilby
import dill
import pandas as pd

from pythiabns.inference.samplers.pocomc import BilbyPocomcPrior

logger = logging.getLogger(__name__)


class TempestWrapper:
    """Wrapper for tempest sampler."""

    def __init__(
        self,
        likelihood: bilby.Likelihood,
        priors: bilby.core.prior.PriorDict,
        outdir: Path,
        label: str,
        settings: dict[str, Any] = None,
    ):
        self.likelihood = likelihood
        self.priors = priors
        self.outdir = Path(outdir)
        self.label = label
        self.settings = settings or {}

        self.outdir.mkdir(parents=True, exist_ok=True)
        self.wrapped_prior = BilbyPocomcPrior(self.priors)

        self.likelihood.parameters.update(self.wrapped_prior.fixed_params)

    def log_likelihood(self, x):
        params = dict(zip(self.wrapped_prior.keys, x))
        self.likelihood.parameters.update(params)
        return self.likelihood.log_likelihood()

    def run(self):
        import multiprocess
        import tempest as tp

        # Extract settings for run/pool
        n_samples = self.settings.get("n_samples", 1000)
        n_cpus = self.settings.get("n_cpus", 1)

        # Filter settings for Sampler.__init__
        sampler_settings = self.settings.copy()
        sampler_settings.pop("n_samples", None)
        sampler_settings.pop("n_cpus", None)

        # Tempest requires a prior_transform (unit cube -> physical)
        def prior_transform(u):
            # Bilby priors rescale takes unit cube values
            return self.priors.rescale(self.wrapped_prior.keys, u)

        logger.info(f"Tempest sampling started with {n_samples} samples.")

        with multiprocess.Pool(n_cpus) as pool:
            sampler = tp.Sampler(
                prior_transform=prior_transform,
                log_likelihood=self.log_likelihood,
                n_dim=self.wrapped_prior.dim,
                pool=pool,
                **sampler_settings,
            )

            sampler.run(n_total=n_samples)

            self._save_results(sampler)

    def _save_results(self, sampler):
        results = sampler.results()
        samples = results.get("samples")

        if samples is not None:
            df = pd.DataFrame(samples, columns=self.wrapped_prior.keys)
            df["log_likelihood"] = results.get("log_likelihood")
            df["log_prior"] = results.get("log_prior")

            df.to_json(self.outdir / f"{self.label}_result.json")

        with open(self.outdir / f"{self.label}_tempest.pickle", "wb") as f:
            dill.dump(results, f)
