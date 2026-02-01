import numpy as np
import numpy.linalg as linalg
if not hasattr(linalg, "linalg"):
    linalg.linalg = linalg

import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import bilby

from pythiabns.core import config, constants, registry
from pythiabns.data_utils.nr import NumericalWaveform
from pythiabns.detectors.network import DetectorNetwork
from pythiabns.inference.priors import PriorFactory
from pythiabns.inference.likelihood import PostMergerLikelihood
from pythiabns.inference.samplers.pocomc import PocoMCWrapper
from pythiabns.inference.samplers.zeus import ZeusWrapper
from pythiabns.inference.samplers.tempest import TempestWrapper
from pythiabns.inference.samplers.numpyro_sampler import NumPyroWrapper
from pythiabns.inference.samplers.nutpie_sampler import NutpieWrapper
from pythiabns.inference.samplers.blackjax_sampler import BlackJAXWrapper
from pythiabns.inference.samplers.stan_sampler import StanWrapper

# Ensure models are registered
import pythiabns.models.waveforms 
import pythiabns.models.relations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pythiabns.spine")

def main():
    parser = argparse.ArgumentParser(description="GW Post-Merger Pipeline")
    parser.add_argument("config", type=str, help="Path to config file (YAML)")
    args = parser.parse_args()
    
    # Load Configuration
    cfg = config.load_config(args.config)
    
    # Load Plugins/Imports
    import importlib
    for module_name in cfg.imports:
        try:
            importlib.import_module(module_name)
            logger.info(f"Imported plugin: {module_name}")
        except ImportError as e:
            logger.error(f"Failed to import plugin {module_name}: {e}")
    
    # Expand Matrix
    # We need a helper to expand the matrix into list of SimulationConfigs
    # For now, manual expansion as placeholder for the logic in config.py or here
    # Assuming config.config contains list expansion logic or we do it here.
    # JobMatrix in config.py defines lists.
    
    matrix = cfg.matrix
    
    # 1. Normalize Injection/Waveform
    if matrix.injection is None:
        if matrix.waveform is not None:
            # Convert legacy waveform list to injection list
            matrix.injection = [
                config.InjectionConfig(mode="nr", target=w) 
                for w in matrix.waveform
            ]
        else:
            # Fallback for analytic-only tutorial-like cases
            matrix.injection = [
                config.InjectionConfig(mode="analytic", target=matrix.model[0])
            ]

    # Cartesian product logic
    import itertools
    
    # We expand over: injection, snr, model
    # model_params and injection_parameters (legacy) are usually fixed but could be lists too if we want.
    # For now, let's keep it simple and expand the core three.
    
    list_keys = ["injection", "snr", "model"]
    list_values = [getattr(matrix, k) for k in list_keys]
    
    combinations = list(itertools.product(*list_values))
    
    logger.info(f"Generated {len(combinations)} simulations.")
    
    for combo in tqdm(combinations):
        combo_dict = dict(zip(list_keys, combo))
        
        # Merge with other matrix settings
        sim_settings = {
            "sampler": matrix.sampler,
            "priors": matrix.priors,
            "model_params": matrix.model_params,
            "injection_parameters": matrix.injection_parameters # Legacy top-level
        }
        sim_settings.update(combo_dict)
        
        # Instantiate SimulationConfig
        try:
            sim_config = config.SimulationConfig(**sim_settings)
        except Exception as e:
            logger.error(f"Config Error: {e}")
            continue
            
        run_simulation(sim_config, cfg.output_dir)

def run_simulation(sim_config: config.SimulationConfig, base_outdir: Path):
    inj = sim_config.injection
    logger.info(f"Running Simulation | Injection: {inj.mode}:{inj.target} | Model: {sim_config.model} | SNR: {sim_config.snr}")
    
    # 1. Load/Prepare Injection Source
    # We still need nr_data object (even if dummy) for priors and masses in some models
    nr_data = None
    
    if inj.mode in ["nr", "file"]:
        try:
            # NumericalWaveform should be updated to handle paths
            nr_data = NumericalWaveform(inj.target)
        except Exception as e:
            logger.error(f"Failed to load injection waveform {inj.target}: {e}")
            return
    
    # 2. Setup Detectors
    # In a real pipeline, network params might come from config. 
    # For now, keeping the tutorial/test defaults.
    network = DetectorNetwork(["H1", "L1", "V1"])
    
    # 3. Setup Waveform Generator for Inference
    model_func = registry.ModelRegistry.get(sim_config.model, **sim_config.model_params)
    if model_func is None:
        logger.error(f"Recovery model {sim_config.model} not found.")
        return
        
    meta = registry.ModelRegistry.get_metadata(sim_config.model, **sim_config.model_params)
    domain = meta.get("domain", "frequency")

    wg = bilby.gw.waveform_generator.WaveformGenerator(
        duration=network.duration,
        sampling_frequency=network.sampling_frequency,
        frequency_domain_source_model=model_func if domain=="frequency" else None,
        time_domain_source_model=model_func if domain=="time" else None,
        parameter_conversion=meta.get("conversion_func"),
        start_time=network.start_time
    )
    
    # 4. Prepare Injection Parameters
    # Merge injection-specific parameters with top-level ones
    inj_params = sim_config.injection_parameters.copy()
    inj_params.update(inj.parameters)
    
    # 5. Inject Signal
    if inj.mode == "analytic":
        # Inject using a registered model (might be different from recovery model)
        inj_model_name = inj.target or sim_config.model
        inj_model_func = registry.ModelRegistry.get(inj_model_name, **sim_config.model_params)
        inj_meta = registry.ModelRegistry.get_metadata(inj_model_name, **sim_config.model_params)
        inj_domain = inj_meta.get("domain", "frequency")
        
        inj_wg = bilby.gw.waveform_generator.WaveformGenerator(
            duration=network.duration,
            sampling_frequency=network.sampling_frequency,
            frequency_domain_source_model=inj_model_func if inj_domain=="frequency" else None,
            time_domain_source_model=inj_model_func if inj_domain=="time" else None,
            parameter_conversion=inj_meta.get("conversion_func"),
            start_time=network.start_time
        )
        logger.info(f"Injecting analytic signal: {inj_model_name}")
        network.set_data(noise=False)
        network.inject_signal(inj_wg, inj_params)
        
    elif inj.mode in ["nr", "file"]:
        # Logically, we should have a way to inject NR data directly.
        # For now, we continue to use the current pattern where we might 
        # use the NR data to set defaults for the analytic model injection 
        # if the user specifically asked for that (like in the tutorial).
        # BUT the goal is to allow NR injection. 
        
        # REAL NR INJECTION (Placeholder logic)
        # network.inject_signal_from_waveform_object(nr_data, ...)
        
        # Current logic for tutorial/legacy:
        if inj_params:
            logger.info(f"Injecting analytic signal informed by {inj.mode}:{inj.target}")
            network.set_data(noise=False)
            network.inject_signal(wg, inj_params)
        else:
            logger.warning("NR injection requested but no parameters provided for analytic model. "
                           "Direct NR injection not yet fully implemented.")
    
    # 6. Setup Priors
    # Use nr_data metadata if available
    nr_meta = nr_data.metadata_dict if nr_data else {}
    priors = PriorFactory.create_priors(sim_config.priors, sim_config.model, nr_meta, sim_config.model_params)
    
    # 7. Setup Likelihood
    likelihood = PostMergerLikelihood(network.ifos, wg)
    
    # 8. Run Sampler
    # Output dir logic: use target name if possible
    target_label = (inj.target or "sim").replace(":", "_").replace("/", "_")
    outdir = base_outdir / f"{target_label}_{sim_config.model}_{sim_config.snr}"
    
    if sim_config.sampler.plugin == "pocomc":
        sampler = PocoMCWrapper(
            likelihood, 
            priors, 
            outdir=outdir, 
            label=sim_config.model,
            settings=sim_config.sampler.settings
        )
        sampler.run()
    elif sim_config.sampler.plugin == "zeus":
        sampler = ZeusWrapper(
            likelihood, 
            priors, 
            outdir=outdir, 
            label=sim_config.model,
            settings=sim_config.sampler.settings
        )
        sampler.run()
    elif sim_config.sampler.plugin == "tempest":
        sampler = TempestWrapper(
            likelihood, 
            priors, 
            outdir=outdir, 
            label=sim_config.model,
            settings=sim_config.sampler.settings
        )
        sampler.run()
    elif sim_config.sampler.plugin == "numpyro":
        sampler = NumPyroWrapper(
            likelihood, 
            priors, 
            outdir=outdir, 
            label=sim_config.model,
            settings=sim_config.sampler.settings
        )
        sampler.run()
    elif sim_config.sampler.plugin == "nutpie":
        sampler = NutpieWrapper(
            likelihood, 
            priors, 
            outdir=outdir, 
            label=sim_config.model,
            settings=sim_config.sampler.settings
        )
        sampler.run()
    elif sim_config.sampler.plugin == "blackjax":
        sampler = BlackJAXWrapper(
            likelihood, 
            priors, 
            outdir=outdir, 
            label=sim_config.model,
            settings=sim_config.sampler.settings
        )
        sampler.run()
    elif sim_config.sampler.plugin == "stan":
        sampler = StanWrapper(
            likelihood, 
            priors, 
            outdir=outdir, 
            label=sim_config.model,
            settings=sim_config.sampler.settings
        )
        sampler.run()
    else:
        bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler=sim_config.sampler.plugin,
            outdir=str(outdir),
            label=sim_config.model,
            **sim_config.sampler.settings
        )

if __name__ == "__main__":
    main()
