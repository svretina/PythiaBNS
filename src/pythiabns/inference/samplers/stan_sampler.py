import bilby
import numpy as np
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import dill

from pythiabns.inference.samplers.pocomc import BilbyPocomcPrior

logger = logging.getLogger(__name__)

class StanWrapper:
    """Wrapper for Stan sampler using CmdStanPy."""
    
    def __init__(self, likelihood: bilby.Likelihood, 
                 priors: bilby.core.prior.PriorDict,
                 outdir: Path,
                 label: str,
                 settings: Dict[str, Any] = None):
         
        self.likelihood = likelihood
        self.priors = priors
        self.outdir = Path(outdir)
        self.label = label
        self.settings = settings or {}
        
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.wrapped_prior = BilbyPocomcPrior(self.priors)
        
        self.likelihood.parameters.update(self.wrapped_prior.fixed_params)

    def run(self):
        from cmdstanpy import CmdStanModel
        
        # Stan requires a .stan file. 
        # For a generic Bilby likelihood, we would need to write the model in Stan.
        # This wrapper expects a .stan file named {label}.stan in the outdir or current dir.
        
        stan_file = self.outdir / f"{self.label}.stan"
        if not stan_file.exists():
            stan_file = Path(f"{self.label}.stan")
            
        if not stan_file.exists():
            logger.error(f"Stan file {stan_file} not found. Stan sampler requires a model definition.")
            return

        logger.info(f"Stan sampler starting with model: {stan_file}")
        
        model = CmdStanModel(stan_file=str(stan_file))
        
        # Data for Stan
        # This would usually include the interferometers data, etc.
        # For now, we pass empty dict or generic data.
        stan_data = self.settings.get("stan_data", {})
        
        fit = model.sample(data=stan_data, 
                           chains=self.settings.get("n_chains", 4),
                           iter_sampling=self.settings.get("n_samples", 1000))
        
        self._save_results(fit)
            
    def _save_results(self, fit):
        if fit is None:
            return
            
        # Extract samples
        samples = fit.draws_pd()
        samples.to_json(self.outdir / f"{self.label}_result.json")
        
        with open(self.outdir / f"{self.label}_stan.pickle", "wb") as f:
            dill.dump(fit, f)
