import logging
from pathlib import Path
from typing import Any

import bilby
import dill
import numpy as np
import pandas as pd

from pythiabns.inference.samplers.pocomc import BilbyPocomcPrior

logger = logging.getLogger(__name__)


class BlackJAXWrapper:
    """Wrapper for BlackJAX sampler."""

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
        import blackjax
        import jax
        import jax.numpy as jnp

        n_samples = self.settings.get("n_samples", 1000)
        # n_warmup = self.settings.get("n_warmup", n_samples // 2)

        # BlackJAX is very flexible. Requires a log-density function.
        def log_density(x):
            # Wrapper to use pure_callback for non-jax likelihood/prior
            def wrapped_logp(x_np):
                lp = self.wrapped_prior.logpdf(np.atleast_2d(x_np))[0]
                ll = self.log_likelihood(x_np)
                return lp + ll

            return jax.pure_callback(wrapped_logp, jnp.float64(0.0), x)

        logger.info(f"BlackJAX sampling initialized with RMH, {n_samples} samples.")

        # Initial values
        rng_key = jax.random.PRNGKey(0)
        init_params = self.wrapped_prior.rvs(1)[0]
        init_params = jnp.array(init_params)

        # Using Random Walk Metropolis Hastings as it doesn't require gradients
        # Use additive_step_random_walk with a simple normal step
        def random_step(key, x):
            return x + jax.random.normal(key, x.shape) * 0.1

        rw = blackjax.additive_step_random_walk(log_density, random_step)
        state = rw.init(init_params)

        def step(state, key):
            state, _ = rw.step(key, state)
            return state, state

        keys = jax.random.split(rng_key, n_samples)
        _, states = jax.lax.scan(step, state, keys)

        self._save_results(states.position)

    def _save_results(self, samples):
        # Convert to pandas
        df = pd.DataFrame(np.array(samples), columns=self.wrapped_prior.keys)
        df.to_json(self.outdir / f"{self.label}_result.json")

        with open(self.outdir / f"{self.label}_blackjax.pickle", "wb") as f:
            dill.dump(samples, f)
