from typing import Any

import numpy.linalg as linalg

if not hasattr(linalg, "linalg"):
    linalg.linalg = linalg  # type: ignore

import argparse
import logging
from pathlib import Path

import bilby
from tqdm import tqdm

# Ensure models are registered
from pythiabns.core import config, plotting, registry
from pythiabns.data_utils.nr import NumericalWaveform
from pythiabns.detectors.network import DetectorNetwork
from pythiabns.inference.likelihood import PostMergerLikelihood
from pythiabns.inference.priors import PriorFactory
from pythiabns.inference.samplers.blackjax_sampler import BlackJAXWrapper
from pythiabns.inference.samplers.numpyro_sampler import NumPyroWrapper
from pythiabns.inference.samplers.nutpie_sampler import NutpieWrapper
from pythiabns.inference.samplers.pocomc import PocoMCWrapper
from pythiabns.inference.samplers.stan_sampler import StanWrapper
from pythiabns.inference.samplers.tempest import TempestWrapper
from pythiabns.inference.samplers.zeus import ZeusWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pythiabns.spine")


def main():
    parser = argparse.ArgumentParser(description="GW Post-Merger Pipeline")
    parser.add_argument("config", type=str, help="Path to config file (YAML)")
    args = parser.parse_args()

    # Load Configuration
    config_path = Path(args.config)
    cfg = config.load_config(config_path)

    # Determine Study Name (Folder Name)
    # Prefer config 'name' if explicit, otherwise file stem
    study_name = cfg.name if cfg.name != "GW_Experiment" else config_path.stem
    study_outdir = cfg.output_dir / study_name

    logger.info(f"Study: {study_name} | Output: {study_outdir}")

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
            matrix.injection = [config.InjectionConfig(mode="nr", target=w) for w in matrix.waveform]
        else:
            # Fallback for analytic-only tutorial-like cases
            matrix.injection = [config.InjectionConfig(mode="analytic", target=matrix.model[0])]

    # Cartesian product logic
    import itertools

    # We expand over: injection, snr, model
    # model_params and injection_parameters (legacy) are usually fixed but could be lists too if we want.
    # For now, let's keep it simple and expand the core three.

    list_keys = ["injection", "snr", "model"]
    list_values = [getattr(matrix, k) for k in list_keys]

    combinations = list(itertools.product(*list_values))

    logger.info(f"Generated {len(combinations)} simulations for study '{study_name}'.")

    for i, combo in enumerate(tqdm(combinations)):
        combo_dict = dict(zip(list_keys, combo))

        # Smart Folder Naming
        # Construct a name based on the varying parameters
        inj_str = (combo_dict["injection"].target or "sim").replace(":", "_").replace("/", "_")
        if combo_dict["injection"].mode == "analytic":
            inj_str = f"inj_{inj_str}"
        elif combo_dict["injection"].mode == "nr":
            inj_str = f"nr_{inj_str}"

        run_name = f"run{i:03d}_{inj_str}_snr{combo_dict['snr']}_model_{combo_dict['model']}"
        run_outdir = study_outdir / run_name

        # Merge with other matrix settings
        sim_settings = {
            "sampler": matrix.sampler,
            "priors": matrix.priors,
            "model_params": matrix.model_params,
            "injection_parameters": matrix.injection_parameters,  # Legacy top-level
        }
        sim_settings.update(combo_dict)

        # Instantiate SimulationConfig
        try:
            sim_config = config.SimulationConfig(**sim_settings)
        except Exception as e:
            logger.error(f"Config Error: {e}")
            continue

        run_simulation(sim_config, cfg.plotting, run_outdir)


def run_simulation(sim_config: config.SimulationConfig, plot_config: config.PlottingConfig, outdir: Path):
    inj = sim_config.injection
    logger.info(f"Running Simulation -> {outdir.name}")

    # 1. Load/Prepare Injection Source
    # We still need nr_data object (even if dummy) for priors and masses in some models
    nr_data = None

    if inj.mode in ["nr", "file"]:
        try:
            # NumericalWaveform should be updated to handle paths
            if inj.target is None:
                raise ValueError("Injection target must be provided for 'nr' or 'file' mode.")
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
        frequency_domain_source_model=model_func if domain == "frequency" else None,
        time_domain_source_model=model_func if domain == "time" else None,
        parameter_conversion=meta.get("conversion_func"),
        start_time=network.start_time,
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
            frequency_domain_source_model=inj_model_func if inj_domain == "frequency" else None,
            time_domain_source_model=inj_model_func if inj_domain == "time" else None,
            parameter_conversion=inj_meta.get("conversion_func"),
            start_time=network.start_time,
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
            logger.warning(
                "NR injection requested but no parameters provided for analytic model. "
                "Direct NR injection not yet fully implemented."
            )

    # 6. Setup Priors
    # Use nr_data metadata if available
    nr_meta = nr_data.metadata_dict if nr_data else {}
    priors = PriorFactory.create_priors(sim_config.priors, sim_config.model, nr_meta, sim_config.model_params)

    # 7. Setup Likelihood
    likelihood = PostMergerLikelihood(network.ifos, wg)

    # 8. Run Sampler
    # Use passed outdir

    sampler: Any = None
    if sim_config.sampler.plugin == "pocomc":
        sampler = PocoMCWrapper(
            likelihood, priors, outdir=outdir, label=sim_config.model, settings=sim_config.sampler.settings
        )
        sampler.run()
    elif sim_config.sampler.plugin == "zeus":
        sampler = ZeusWrapper(
            likelihood, priors, outdir=outdir, label=sim_config.model, settings=sim_config.sampler.settings
        )
        sampler.run()
    elif sim_config.sampler.plugin == "tempest":
        sampler = TempestWrapper(
            likelihood, priors, outdir=outdir, label=sim_config.model, settings=sim_config.sampler.settings
        )
        sampler.run()
    elif sim_config.sampler.plugin == "numpyro":
        sampler = NumPyroWrapper(
            likelihood, priors, outdir=outdir, label=sim_config.model, settings=sim_config.sampler.settings
        )
        sampler.run()
    elif sim_config.sampler.plugin == "nutpie":
        sampler = NutpieWrapper(
            likelihood, priors, outdir=outdir, label=sim_config.model, settings=sim_config.sampler.settings
        )
        sampler.run()
    elif sim_config.sampler.plugin == "blackjax":
        sampler = BlackJAXWrapper(
            likelihood, priors, outdir=outdir, label=sim_config.model, settings=sim_config.sampler.settings
        )
        sampler.run()
    elif sim_config.sampler.plugin == "stan":
        sampler = StanWrapper(
            likelihood, priors, outdir=outdir, label=sim_config.model, settings=sim_config.sampler.settings
        )
        sampler.run()
    else:
        bilby.run_sampler(
            likelihood=likelihood,
            priors=priors,
            sampler=sim_config.sampler.plugin,
            outdir=str(outdir),
            label=sim_config.model,
            **sim_config.sampler.settings,
        )

    # 9. Plotting
    # If sampler provides a uniform result object/path, we read it
    # Note: Wrappers typically save a `_result.json`
    # We reuse the `sampler` object if it has a `result` property, or load from file.

    # Attempt to load the result.
    # Most Bilby samplers save to outdir/label_result.json
    result_path = outdir / f"{sim_config.model}_result.json"
    if result_path.exists():
        try:
            result = bilby.result.read_in_result(filename=str(result_path))
            plotting.generate_plots(result, plot_config, outdir)
        except Exception as e:
            logger.error(f"Plotting failed: Could not read result file {result_path}: {e}")
    else:
        logger.warning(f"No result file found at {result_path}, skipping plots.")


if __name__ == "__main__":
    main()
