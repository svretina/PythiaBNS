import logging
from pathlib import Path
from typing import Any

import bilby
import dill
import numpy as np
import pandas as pd

from pythiabns.inference.samplers.pocomc import BilbyPocomcPrior

logger = logging.getLogger(__name__)


class NumPyroWrapper:
    """Wrapper for NumPyro sampler."""

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
        import jax
        import jax.numpy as jnp
        import numpyro
        from numpyro.infer import MCMC

        n_samples = self.settings.get("n_samples", 1000)
        n_warmup = self.settings.get("n_warmup", n_samples // 2)
        n_chains = self.settings.get("n_chains", 1)

        # We use a potential_fn because we have a custom log-likelihood/prior
        def potential_fn(params):
            # Convert jax dict to list in correct order
            x = [params[k] for k in self.wrapped_prior.keys]

            # Use pure_callback to call non-jax likelihood
            def wrapped_logp(x_np):
                lp = self.wrapped_prior.logpdf(np.atleast_2d(x_np))[0]
                ll = self.log_likelihood(x_np)
                return -(lp + ll)

            val = jax.pure_callback(wrapped_logp, jnp.float64(0.0), jnp.array(x))
            return val

        # For NUTS, we need gradients. pure_callback doesn't support them by default.
        # If we use NUTS, it might fail unless we use a sampler that doesn't need gradients
        # or we provide numerical gradients.

        logger.info(f"NumPyro sampling started with {n_samples} samples and {n_chains} chains.")

        # Initial values from prior
        rng_key = jax.random.PRNGKey(0)
        init_params = self.wrapped_prior.rvs(n_chains)
        init_dict = {
            self.wrapped_prior.keys[i]: jnp.array(init_params[:, i]) for i in range(len(self.wrapped_prior.keys))
        }

        # Use SA (Simulated Annealing) or another gradient-free sampler
        # because the models are not JAX-traceable.
        kernel = numpyro.infer.SA(potential_fn=potential_fn)
        mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains)
        mcmc.run(rng_key, init_params=init_dict)

        self._save_results(mcmc)

    def _save_results(self, mcmc):
        samples = mcmc.get_samples()
        # Convert to pandas. Handle potential chain/sample dimension overlap
        data = {}
        for k, v in samples.items():
            v_np = np.array(v)
            if v_np.ndim > 1:
                # Flatten if it's (num_samples, 1) or similar, else keep but pandas might complain
                if v_np.shape[1] == 1:
                    data[k] = v_np.flatten()
                else:
                    # Multi-dim param, might need special handling but for now just take it
                    data[k] = list(v_np)
            else:
                data[k] = v_np

        df = pd.DataFrame(data)

        df.to_json(self.outdir / f"{self.label}_result.json")

        with open(self.outdir / f"{self.label}_numpyro.pickle", "wb") as f:
            dill.dump(samples, f)
