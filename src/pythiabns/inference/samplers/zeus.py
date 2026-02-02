import logging
from pathlib import Path
from typing import Any

import bilby
import dill
import pandas as pd

from pythiabns.inference.samplers.pocomc import BilbyPocomcPrior  # Reusing the prior wrapper

logger = logging.getLogger(__name__)


class ZeusWrapper:
    """Wrapper for zeus sampler."""

    def __init__(
        self,
        likelihood: bilby.Likelihood,
        priors: bilby.core.prior.PriorDict,
        outdir: Path,
        label: str,
        settings: dict[str, Any] | None = None,
    ):
        self.likelihood = likelihood
        self.priors = priors
        self.outdir = Path(outdir)
        self.label = label
        self.settings = settings or {}

        self.outdir.mkdir(parents=True, exist_ok=True)
        self.wrapped_prior = BilbyPocomcPrior(self.priors)  # Zeus also needs prior samples/logpdf

        self.likelihood.parameters.update(self.wrapped_prior.fixed_params)

    def log_likelihood(self, x):
        params = dict(zip(self.wrapped_prior.keys, x))
        self.likelihood.parameters.update(params)
        return self.likelihood.log_likelihood()

    def run(self):
        import multiprocess
        import zeus

        nwalkers = self.settings.get("nwalkers", 2 * self.wrapped_prior.dim + 2)
        nsteps = self.settings.get("nsteps", 1000)
        n_cpus = self.settings.get("n_cpus", 1)

        # Initial positions
        start_pos = self.wrapped_prior.rvs(nwalkers)

        with multiprocess.Pool(n_cpus) as pool:
            sampler = zeus.Sampler(
                logprob_fn=self.log_likelihood, n_dim=self.wrapped_prior.dim, n_walkers=nwalkers, pool=pool
            )

            sampler.run_mcmc(start_pos, nsteps)

            self._save_results(sampler)

    def _save_results(self, sampler):
        samples = sampler.get_chain(flat=True)
        log_prob = sampler.get_log_prob(flat=True)

        df = pd.DataFrame(samples, columns=self.wrapped_prior.keys)
        df["log_likelihood"] = log_prob  # Note: zeus doesn't separate prior/likelihood by default if using logprob_fn

        df.to_json(self.outdir / f"{self.label}_result.json")

        with open(self.outdir / f"{self.label}_zeus.pickle", "wb") as f:
            dill.dump(sampler, f)
