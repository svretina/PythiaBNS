import logging
from pathlib import Path

logger = logging.getLogger("pythiabns.plotting")


def generate_plots(result, config, outdir: Path):
    """
    Generate plots based on configuration.

    Args:
        result: bilby.core.result.Result object
        config: PlottingConfig object
        outdir: Path to output directory
    """
    if not config.enabled:
        return

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)  # Should already exist but good check

    for plot_type in config.plots:
        try:
            if plot_type == "corner":
                plot_corner(result, outdir, **config.settings.get("corner", {}))
            elif plot_type == "trace":
                plot_trace(result, outdir, **config.settings.get("trace", {}))
            elif plot_type == "waveform":
                # This requires the waveform generator which isn't passed here easily
                # typically stored in result object metadata if using standard bilby
                # or we skip for now/handle in spine
                logger.warning(
                    "Waveform plotting from this module requires access to the generator. Handled in spine.py?"
                )
            else:
                logger.warning(f"Unknown plot type: {plot_type}")
        except Exception as e:
            logger.error(f"Failed to generate {plot_type} plot: {e}")


def plot_corner(result, outdir, **kwargs):
    logger.info("Generating corner plot...")
    # Bilby's plot_corner is robust
    result.plot_corner(outdir=str(outdir), **kwargs)


def plot_trace(result, outdir, **kwargs):
    logger.info("Generating trace plot...")
    # Bilby's plot_walkers (often called trace)
    # Check if sampler supports it or if we just plot samples
    # result.plot_walkers() works for some MCMC
    try:
        result.plot_walkers(outdir=str(outdir), **kwargs)
    except Exception:
        # Fallback if walker data not available
        logger.warning(f"Trace plot (plot_walkers) not available for {result.sampler}")


def plot_waveform_posterior(result, waveform_generator, descriptors, outdir, n_samples=100):
    """
    Plot the waveform posterior against injection/data.

    Args:
        result: Bilby Result
        waveform_generator: Initiailized WaveformGenerator
        descriptors: Dictionary of {'label': [list of ifos]} or similar to know what to plot?
                     Actually, usually we plot strain vs time/freq.
        outdir: Output path
    """
    logger.info("Generating waveform posterior plot...")

    # Select random samples
    if len(result.posterior) >= n_samples:
        result.posterior.sample(n=n_samples)

    # We need to compute waveforms for these samples
    # accessing waveform_generator.time_domain_source_model(parameter)

    # This is quite specific to the domain (time/freq) and IFOs
    # For now, let's just implement a simple Time Domain viewer if available

    pass
