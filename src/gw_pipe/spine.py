import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import bilby

from gw_pipe.core import config, constants, registry
from gw_pipe.data_utils.nr import NumericalWaveform
from gw_pipe.detectors.network import DetectorNetwork
from gw_pipe.inference.priors import PriorFactory
from gw_pipe.inference.likelihood import PostMergerLikelihood
from gw_pipe.inference.samplers.pocomc import PocoMCWrapper

# Ensure models are registered
import gw_pipe.models.waveforms 
import gw_pipe.models.relations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gw_pipe.spine")

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
    
    # Cartesian product logic (simplified loop)
    # TODO: make this robust
    import itertools
    keys = matrix.model_dump().keys()
    # Filter keys that are lists
    list_keys = [k for k in keys if isinstance(getattr(matrix, k), list)]
    list_values = [getattr(matrix, k) for k in list_keys]
    
    combinations = list(itertools.product(*list_values))
    
    logger.info(f"Generated {len(combinations)} simulations.")
    
    for combo in tqdm(combinations):
        # Construct simulation config from combo
        combo_dict = dict(zip(list_keys, combo))
        # Merge with non-list items
        sim_settings = matrix.model_dump()
        sim_settings.update(combo_dict)
        
        # Ensure model_params is included (it might be in sim_settings from matrix)
        # If matrix has model_params, it's already there.
        
        # Instantiate SimulationConfig
        try:
            sim_config = config.SimulationConfig(**sim_settings)
        except Exception as e:
            logger.error(f"Config Error: {e}")
            continue
            
        run_simulation(sim_config, cfg.output_dir)

def run_simulation(sim_config: config.SimulationConfig, base_outdir: Path):
    logger.info(f"Running: {sim_config.waveform} | {sim_config.model} | SNR={sim_config.snr}")
    
    # 1. Load Data
    try:
        nr_data = NumericalWaveform(sim_config.waveform)
    except Exception as e:
        logger.error(f"Failed to load NR data: {e}")
        return

    # 2. Setup Detectors
    network = DetectorNetwork(["H1", "L1", "V1"])
    
    # Get Model
    # Pass model_params as filters (e.g. nfreqs=3)
    model_func = registry.ModelRegistry.get(sim_config.model, **sim_config.model_params)
    if model_func is None:
        logger.error(f"Model {sim_config.model} not found with params {sim_config.model_params}")
        return
        
    meta = registry.ModelRegistry.get_metadata(sim_config.model, **sim_config.model_params)
    domain = meta.get("domain", "frequency")

    wg = bilby.gw.waveform_generator.WaveformGenerator(
        duration=network.duration,
        sampling_frequency=network.sampling_frequency,
        frequency_domain_source_model=model_func if domain=="frequency" else None,
        time_domain_source_model=model_func if domain=="time" else None,
        # We need conversion function?
        # ModelRegistry stores conversion_func in metadata? 
        # No, decorator arg in waveforms.py passed it to register calls, but did we store it?
        # registry.py: stores **metadata.
        # waveforms.py: @register_model(..., conversion_func=...)
        # So yes, it is in meta.
        parameter_conversion=meta.get("conversion_func"),
        
        # Start time logic?
        start_time=network.start_time
    )
    
    # Injection parameters
    # We need to define them. Fixed for now?
    injection_parameters = sim_config.injection_parameters.copy()
    if not injection_parameters:
        injection_parameters = {
            "mass_1": nr_data.m1 * constants.MSUN_SI,
            "mass_2": nr_data.m2 * constants.MSUN_SI,
            "luminosity_distance": 100.0, # Dummy dist for now
            "theta_jn": 0.,
            "psi": 0.,
            "phase": 0.,
            "geocent_time": 0.,
            "ra": 0.,
            "dec": 0.,
            # Model specific params?
            # NR injection usually injects the NR waveform itself, not the analytic model!
            # so we should use nr_data as source model for injection.
        }
    
    # For NR injection:
    # Bilby needs a source model that returns the NR data.
    # We can define a wrapper around nr_data.
    def nr_injection_model(frequency_array, **kwargs):
         # Returns FD nr data
         # Or TD if time_domain_source_model
         # We need to interpolate nr_data to required times/freqs
         return {"plus": 0, "cross": 0} # TODO: Implement NR injection logic properly
    
    # For now, let's assume we proceed to sampling assuming data is ready.
    network.set_data(noise=False) # Zero noise for testing
    
    # 3. Setup Priors
    priors = PriorFactory.create_priors(sim_config.priors, sim_config.model, nr_data.metadata_dict, sim_config.model_params)
    
    # 4. Setup Likelihood
    likelihood = PostMergerLikelihood(network.ifos, wg)
    
    # 5. Run Sampler
    # Output dir logic
    outdir = base_outdir / f"{sim_config.waveform}_{sim_config.model}_{sim_config.snr}"
    
    if sim_config.sampler.plugin == "pocomc":
        sampler = PocoMCWrapper(
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
