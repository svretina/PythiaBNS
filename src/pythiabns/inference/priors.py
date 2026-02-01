import bilby
import numpy as np
import logging
from typing import Dict, Any, Optional

from pythiabns.core import constants, registry
from pythiabns.core.config import PriorConfig

logger = logging.getLogger(__name__)

class PriorFactory:
    """Factory to generate bilby PriorDicts based on configuration."""
    
    @staticmethod
    def create_priors(config: PriorConfig, 
                      model_name: str, 
                      metadata: Optional[Dict[str, Any]] = None,
                      model_params: Dict[str, Any] = None) -> bilby.core.prior.PriorDict:
        
        # Fetch conversion function from registry
        reg_meta = registry.ModelRegistry.get_metadata(model_name, **(model_params or {}))
        conversion_func = reg_meta.get("conversion_func")

        # 1. Load base priors
        priors = bilby.core.prior.PriorDict(conversion_function=conversion_func)
        
        # Load from file if specified
        base_filename = f"{model_name}.priors"
        # Check explicit source or default locations
        if config.source:
             if (constants.PRIORS_PATH / config.source).exists():
                 priors.from_file(str(constants.PRIORS_PATH / config.source))
             elif (constants.PROJECT_ROOT / config.source).exists():
                 priors.from_file(str(constants.PROJECT_ROOT / config.source))
        
        if not priors and (constants.PRIORS_PATH / base_filename).exists():
            priors.from_file(str(constants.PRIORS_PATH / base_filename))
            
        # 2. Apply Empirical Relations if requested
        if config.mode == "empirical" and metadata:
            PriorFactory._apply_empirical_relations(priors, config, metadata)
            
        return priors

    @staticmethod
    def _apply_empirical_relations(priors: bilby.core.prior.PriorDict, 
                                   config: PriorConfig, 
                                   metadata: Dict[str, Any]):
        
        method = config.source # e.g. "VSB_R"
        relation_cls = registry.RelationRegistry.get(method)
        if not relation_cls:
            logger.warning(f"Relation {method} not found in registry.")
            return

        relation = relation_cls()
        eos = metadata.get("id_eos", "SLY")
        m1 = metadata.get("id_mass_starA")
        m2 = metadata.get("id_mass_starB")
        
        if m1 is None or m2 is None:
            logger.warning("Masses not found in metadata, skipping empirical priors.")
            return

        preds = relation.predict(m1, m2, eos)
        fpeak = preds.get("f_peak")
        
        # Update f_peak prior
        if "f_peak" in priors and fpeak:
            # Assume 10% width or similar logic from original
            # Original used specific logic based on Distribution type (Gaussian/Uniform)
            # Here we simplify or need to inspect prior type.
            prior = priors["f_peak"]
            if isinstance(prior, bilby.core.prior.Uniform):
                width = 500 # Arbitrary default or derived?
                priors["f_peak"] = bilby.core.prior.Uniform(fpeak - width, fpeak + width, name="f_peak")
            elif isinstance(prior, bilby.core.prior.Gaussian):
                priors["f_peak"] = bilby.core.prior.Gaussian(mu=fpeak, sigma=200, name="f_peak") # sigma??
                
            # TODO: Add more sophisticated logic matching priors.py
