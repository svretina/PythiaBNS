import logging
from pathlib import Path
from typing import Any

import bilby
import dill
import multiprocess
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BilbyPocomcPrior:
    """Wrapper to make bilby priors compatible with PocoMC Prior interface."""

    def __init__(self, bilby_priors):
        self.priors = bilby_priors

        # Identify stochastic keys
        self.keys = []
        self.fixed_params = {}

        for k in self.priors.keys():
            p = self.priors[k]
            if isinstance(p, (bilby.core.prior.Constraint, bilby.core.prior.DeltaFunction)):
                if isinstance(p, bilby.core.prior.DeltaFunction):
                    self.fixed_params[k] = p.peak
                continue
            elif isinstance(p, (float, int)):
                self.fixed_params[k] = p
                continue
            else:
                self.keys.append(k)

        self.dim = len(self.keys)
        # Sort keys to ensure consistency
        self.keys.sort()

        # Precompute bounds
        self._bounds = []
        for k in self.keys:
            p = self.priors[k]
            if hasattr(p, "minimum") and hasattr(p, "maximum"):
                self._bounds.append([p.minimum, p.maximum])
            else:
                self._bounds.append([-np.inf, np.inf])
        self.bounds = np.array(self._bounds)

    def logpdf(self, x):
        x = np.atleast_2d(x)
        n = x.shape[0]
        res = np.zeros(n)

        for i in range(n):
            params = dict(zip(self.keys, x[i]))
            # Merge fixed params so constraints/dependencies work
            params.update(self.fixed_params)
            res[i] = self.priors.ln_prob(params)
        return res

    def rvs(self, size=1):
        samples = [self.priors.sample() for _ in range(size)]
        res = np.array([[s[k] for k in self.keys] for s in samples])
        return res

    def __call__(self, x):
        return self.logpdf(x)


class PocoMCWrapper:
    """Wrapper for PocoMC sampler."""

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

        # Populate fixed parameters into likelihood initially
        self.likelihood.parameters.update(self.wrapped_prior.fixed_params)

        # Periodicity logic
        self.periodicity = []
        for i, key in enumerate(self.wrapped_prior.keys):
            p = self.priors[key]
            if hasattr(p, "boundary") and p.boundary == "periodic":
                self.periodicity.append(i)

    def log_likelihood(self, x):
        x = np.atleast_1d(x)
        if x.ndim > 1:
            res = []
            for xi in x:
                res.append(self._log_likelihood_single(xi))
            return np.array(res)
        return self._log_likelihood_single(x)

    def _log_likelihood_single(self, x):
        params = dict(zip(self.wrapped_prior.keys, x))
        # Fixed params already in likelihood.parameters, but safety:
        # If likelihood code modifies them? Usually not.
        # But if we update with `params`, it overwrites stochastic ones.
        self.likelihood.parameters.update(params)
        return self.likelihood.log_likelihood()

    def run(self):
        import pocomc as pc

        nwalkers = self.settings.get("npoints", 1000)

        # Using 'multiprocess' with dill support
        with multiprocess.Pool(self.settings.get("n_cpus", 1)) as pool:
            sampler = pc.Sampler(
                prior=self.wrapped_prior,
                likelihood=self.log_likelihood,
                n_dim=self.wrapped_prior.dim,
                n_effective=nwalkers,
                n_active=nwalkers,
                periodic=self.periodicity,
                pool=pool,
                vectorize=False,
            )

            sampler.run(progress=True)

            self._save_results(sampler)

    def _save_results(self, sampler):
        results = sampler.results
        samples = results.get("samples", results.get("posterior_samples"))
        if samples is None:
            # Fallback logic based on version behavior
            if "posterior_samples" in results and results["posterior_samples"] is not None:
                samples = results["posterior_samples"]
            else:
                # sampler might have 'samples' attribute?
                pass

        if samples is not None:
            df = pd.DataFrame(samples, columns=self.wrapped_prior.keys)
            df["log_prior"] = results.get("posterior_logp", results.get("log_prior"))
            df["log_likelihood"] = results.get("posterior_logl", results.get("log_likelihood"))

            df.to_json(self.outdir / f"{self.label}_result.json")

        with open(self.outdir / f"{self.label}_pocomc.pickle", "wb") as f:
            dill.dump(results, f)
