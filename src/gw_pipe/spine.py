import os
import sys
import re
import argparse
import logging
import warnings
import gc
import inspect
from time import time
from functools import partialmethod
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.constants import c, G, M_sun

import bilby
import dill
import pymp
from dynesty import plotting as dyplot

# Domain-specific imports (gw_pipe)
from gw_pipe import ifo
from gw_pipe import priors
from gw_pipe import config
from gw_pipe import source_model
from gw_pipe import post_analysis
from gw_pipe import NR_strains as nr
from gw_pipe import global_vars as glb
from gw_pipe import utils as ut

# PyCBC imports
import pycbc
from pycbc.detector import Detector
from pycbc.types import TimeSeries
from pycbc.filter import make_frequency_series
from scipy.integrate import simps
from scipy import signal

# Environment configuration
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_AFFINITY"] = "disabled"
pymp.config.nested = True

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def setup_bilby_logger(quiet=False):
    log_level = 40 if quiet else 20
    bilby.core.utils.log.setup_logger(log_level=log_level)

parser = argparse.ArgumentParser()
parser.add_argument('-c',type=str,help='Config file')
conf_file = parser.parse_args().c
if conf_file is None:
    config_obj = config.Config()
    config_dict = config_obj.config_dict
else:
    config_obj = config.Config(conf_file)
    config_dict = config_obj.config_dict

#np.random.seed(1234)

quiet_flag = config_dict["Run"]["quiet"]

# if quiet_flag:
    # tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
bilby.core.utils.log.setup_logger(log_level=40)

def calc_pycbc_snr(data):
    """Calculate Signal-to-Noise Ratio (SNR) using PyCBC."""
    hplus = data.hp
    hcross = data.hc
    time_SI = data.time
    amplitude = np.sqrt(data.hp**2 + data.hc**2)
    idx = np.argmax(amplitude)
    
    hp_pm = hplus[idx:]
    hc_pm = hcross[idx:]
    time_step = time_SI[1] - time_SI[0]
    start_time = 0
    
    hp = TimeSeries(np.array(hp_pm), delta_t=time_step, epoch=start_time)
    hc = TimeSeries(np.array(hc_pm), delta_t=time_step, epoch=start_time)
    signal_duration = 0.125  # 1/8

    hp.resize(int(signal_duration / hp.delta_t))
    hc.resize(int(signal_duration / hc.delta_t))

    # Set the desired sampling rate
    sampling_rate = 16384  # 2**14

    # Find the number of samples in the resampled signal
    num_samples = int(sampling_rate * hp.duration) + 1
    new_times = np.arange(0, signal_duration, 1/sampling_rate)

    hp_resampled = ut.interpolate(hp.sample_times.numpy(), hp.numpy(), new_times)
    hc_resampled = ut.interpolate(hc.sample_times.numpy(), hc.numpy(), new_times)
    
    hp_16khz = TimeSeries(hp_resampled, delta_t=1/sampling_rate, epoch=start_time)
    hc_16khz = TimeSeries(hc_resampled, delta_t=1/sampling_rate, epoch=start_time)

    inclination = 0
    hp_16khz *= (1 + np.cos(inclination)**2) / 2
    hc_16khz *= np.cos(inclination)
    
    det_h1 = Detector('H1')
    det_l1 = Detector('L1')
    det_v1 = Detector('V1')

    # Sky location
    declination = 0
    right_ascension = 0
    polarization = 0

    signal_h1 = det_h1.project_wave(hp_16khz, hc_16khz, right_ascension, declination, polarization)
    signal_l1 = det_l1.project_wave(hp_16khz, hc_16khz, right_ascension, declination, polarization)
    signal_v1 = det_v1.project_wave(hp_16khz, hc_16khz, right_ascension, declination, polarization)

    signal_h1.resize(int(signal_duration / signal_h1.delta_t))
    signal_l1.resize(int(signal_duration / signal_l1.delta_t))
    signal_v1.resize(int(signal_duration / signal_v1.delta_t))

    signal_h1_fd = make_frequency_series(signal_h1)
    signal_l1_fd = make_frequency_series(signal_l1)
    signal_v1_fd = make_frequency_series(signal_v1)

    # Use bilby noise curves dynamically
    ligo_asd_file = ut.get_bilby_noise_path('aLIGO_O4_high_asd.txt')
    virgo_asd_file = ut.get_bilby_noise_path('AdV_psd.txt')
    
    ligo_asd_data = np.loadtxt(ligo_asd_file)
    virgo_asd_data = np.loadtxt(virgo_asd_file)

    asd_ligo = ut.interpolate(ligo_asd_data.T[0], ligo_asd_data.T[1],
                             np.arange(ligo_asd_data.T[0, 0], ligo_asd_data.T[0, -1], signal_h1_fd.delta_f))

    psd_ligo = pycbc.types.frequencyseries.FrequencySeries(asd_ligo**2, delta_f=signal_h1_fd.delta_f)

    asd_virgo = ut.interpolate(virgo_asd_data.T[0], virgo_asd_data.T[1],
                              np.arange(virgo_asd_data.T[0, 0], virgo_asd_data.T[0, -1], signal_h1_fd.delta_f))

    psd_virgo = pycbc.types.frequencyseries.FrequencySeries(asd_virgo, delta_f=signal_h1_fd.delta_f)

    mask_ligo = min(psd_ligo.sample_frequencies.numpy().size, signal_h1_fd.sample_frequencies.numpy().size)
    mask_virgo = min(psd_virgo.sample_frequencies.numpy().size, signal_v1_fd.sample_frequencies.numpy().size)

    integrand_h1 = abs(signal_h1_fd[:mask_ligo])**2 / psd_ligo[:mask_ligo]
    integrand_l1 = abs(signal_l1_fd[:mask_ligo])**2 / psd_ligo[:mask_ligo]
    integrand_v1 = abs(signal_v1_fd[:mask_virgo])**2 / psd_virgo[:mask_virgo]

    flower_post_merger = 500
    index_flower = int(flower_post_merger / psd_ligo.delta_f)
    fhighest_post_merger = 5000
    index_fhighest = int(fhighest_post_merger / psd_ligo.delta_f)

    rho2_h1 = 4 * simps(integrand_h1[index_flower:index_fhighest].numpy(), dx=psd_ligo.delta_f)
    rho2_l1 = 4 * simps(integrand_l1[index_flower:index_fhighest].numpy(), dx=psd_ligo.delta_f)
    rho2_v1 = 4 * simps(integrand_v1[index_flower:index_fhighest].numpy(), dx=psd_virgo.delta_f)

    return np.sqrt(rho2_h1 + rho2_l1 + rho2_v1)


def load_config(conf_file=None):
    """Load configuration from a file or use default."""
    if conf_file is None:
        cfg = config.Config()
    else:
        cfg = config.Config(conf_file)
    return cfg, cfg.config_dict

def run():
    """Main entry point for running the GW pipeline."""
    parser = argparse.ArgumentParser(description="Gravitational Wave Post-Merger Pipeline")
    parser.add_argument('-c', type=str, help='Config file path')
    args = parser.parse_args()

    cfg_obj, cfg_dict = load_config(args.c)
    setup_bilby_logger(cfg_dict["Run"]["quiet"])

    dict_list = cfg_obj.create_iterator_dict()
    ncpus = cfg_dict["Run"]["n_cpus"]
    
    start_time = time()
    if ncpus > 1:
        logger.info(f"Running in parallel with {ncpus} CPUs")
        # Ensure we don't exceed physical limits
        try:
            mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            mem_gib = mem_bytes / (1024.0 ** 3)
            max_processes = int(mem_gib // 1.5)
            if ncpus > max_processes:
                logger.warning(f"Limiting CPUs to {max_processes} due to memory constraints")
                ncpus = max_processes
        except:
            pass

        with Pool(processes=ncpus) as pool:
            # Use tqdm to monitor progress across simulations
            results = list(tqdm(pool.imap(partial_main, [(d, cfg_dict) for d in dict_list]),
                               total=len(dict_list), ascii=" =>"))
            pool.close()
            pool.join()
        
        logger.info(f"Total execution time: {time() - start_time:.2f} seconds")
    else:
        logger.info("Running in serial mode")
        for dict_item in dict_list:
            main(dict_item, cfg_dict)

def partial_main(args):
    """Helper for parallel execution."""
    sim_dict, cfg_dict = args
    return main(sim_dict, cfg_dict)

def main(simulation_dict, cfg_dict, p=None):
    """Run a single simulation."""
    data = nr.NumericalData(simulation_dict["waveform"], sampling_frequency=2*8192)
    data.metadata_dict["id_name"] = simulation_dict["waveform"]

    # Construct output directory using robust path logic from glb
    base_out = f"{glb.results_path}/{data.metadata_dict['id_eos']}_{data.metadata_dict['id_mass_starA']}_{data.metadata_dict['id_mass_starB']}_{simulation_dict['model']}"
    
    sampler_type = simulation_dict['sampler'].lower()
    if sampler_type == 'dynesty':
        sampler_params = f"{int(simulation_dict['npoints'])}_{simulation_dict['dlogz']}_{simulation_dict['times']}"
    elif sampler_type == 'pocomc':
        sampler_params = f"{int(simulation_dict['npoints'])}_{simulation_dict['corr_threshold']}_{simulation_dict['times']}"
    else:
        raise ValueError(f"{simulation_dict['sampler']} is not implemented yet.")

    mode_params = f"nfreqs_{simulation_dict['number_of_freqs']}_hardcoded_{simulation_dict['from_file']}"
    if not simulation_dict['from_file']:
        mode_params += f"_{simulation_dict['method']}_{simulation_dict['distribution']}"
    
    outdir = f"{base_out}/{simulation_dict['waveform']}_{sampler_type}_{sampler_params}/{mode_params}"

    # Refined output directory structure for PocoMC/Dynesty
    out_parts = outdir.split('/')
    try:
        idx = out_parts.index('results')
        if simulation_dict['from_file']:
            out_parts[idx] = "results/hardcoded"
        else:
            model_suffix = "_plus" if simulation_dict['number_of_freqs'] == 4 else ""
            param_key = "corr_threshold" if sampler_type == 'pocomc' else "dlogz"
            out_parts[idx] = f"results/{sampler_type}_{simulation_dict['model']}{model_suffix}_{simulation_dict['method']}_classifier_{simulation_dict['classifier']}_{simulation_dict['npoints']}_{simulation_dict[param_key]}"
        
        out_parts[idx] += f"_{simulation_dict['snr']}"
        outdir = '/'.join(out_parts)
    except ValueError:
        pass # Keep original outdir if 'results' not found
    #print([i for i in (os.listdir('/'.join((outdir.split('/')[:-4])) )) if 'poco' in i and f"{simulation_dict['snr']}" in i][0])
    #print( outdir.split('/')[-4] )
    #exit()
    ################ FOR USE ####################

    
    ##### Our method #####

    #from scipy.signal import butter, lfilter

    #def butter_bandpass(lowcut, highcut, fs, order=5):
        #return butter(order, [lowcut, highcut], fs=fs, btype='band')

    #def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        #b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        #y = lfilter(b, a, data)
        #return y

    #lowcut = 1000.0
    #highcut = 5000.0

    #data.get_post_merger()  # inplace
    #htilde, freqs = utils.nfft(
        #butter_bandpass_filter(data.hp*40,lowcut,highcut,1/(data.time[1]-data.time[0]),order=4),
        #1/(data.time[1]-data.time[0])
        #)
    #plt.figure(figsize=(6,4),dpi=100)
    #plt.plot(freqs,
             #(np.abs(htilde)*freqs)
             #)
    #plt.yscale('log')
    #plt.xlim(0,5000)
    #plt.show()
    #exit()

    #data.tukey_window(a=0.03)
    full_wave = (data.hp**2 + data.hp ** 2) ** 0.5
    postmerger_start_index = np.argmax(full_wave)
    t0 = data.time[postmerger_start_index+1]

    del full_wave
    del postmerger_start_index
    gc.collect()

    injection_parameters_data = dict(
        psi=0,
        geocent_time=0,
        ra=0,
        dec=0,
        mass_1=data.m1 * M_sun.value,
        mass_2=data.m2 * M_sun.value,
        t0=t0
    )

    # Calculate SNR and rescale data
    snr_1mpc = calc_pycbc_snr(data)
    data.rescale_to_SNR(snr_1mpc, 1, simulation_dict["snr"])
    
    handler = ifo.InterferometerHandler(
        data,
        injection_parameters_data,
        noise=simulation_dict["noise"],
        ifo_list=["L1", "H1", "V1"],
    )
    
    gw_model = source_model.model_picker(
        simulation_dict["model"], simulation_dict["number_of_freqs"]
    )

    # Filter simulation_dict for extra hyperparameters to pass to the source model
    known_keys = ["times", "model", "sampler", "npoints", "dlogz", "corr_threshold", 
                  "number_of_freqs", "classifier", "waveform", "snr", "noise", 
                  "from_file", "method", "distribution", "priors"]
    waveform_kwargs = {k: v for k, v in simulation_dict.items() if k not in known_keys}
    
    waveform_model = bilby.gw.waveform_generator.WaveformGenerator(
        duration=handler.duration,
        sampling_frequency=handler.sampling_frequency,
        frequency_domain_source_model=gw_model,
        start_time=handler.start_time_wv,
        waveform_arguments=waveform_kwargs
    )

    likelihood = bilby.gw.GravitationalWaveTransient(handler.ifos, waveform_model)

    priors_dist = priors.get_priors(simulation_dict,
                                    handler=handler,
                                    metadata_dict=data.metadata_dict)

    priors_dist['t0'].peak = handler.t0

    # Handle custom frequency priors
    region = priors.classifier(data.metadata_dict)
    if not cfg_dict['Priors']['from_file'] and cfg_dict['Model']['number_of_freqs'] > 2:
        if region != 2:
            if region == 1:
                f1, sigma = priors.VSB_f20(data.metadata_dict, simulation_dict["method"])
                latex_name = "$f_1$"
            elif region == 3:
                f1, sigma = priors.VSB_fspiral(data.metadata_dict, simulation_dict["method"])
                latex_name = "$f_1$"

            priors_dist["f_1"] = bilby.core.prior.Gaussian(name="f_1", mu=f1, sigma=sigma, latex_label=latex_name)
            priors_dist['f_2'] = bilby.core.prior.Uniform(1000, priors_dist['f_peak'].mu, name="f_2", latex_label="$f_2$")
            priors_dist.pop('f12', None)
            
    handler.data.metadata_dict['SNR'] = simulation_dict["snr"]
    simulation_dict['priors'] = priors_dist
    os.makedirs(outdir, exist_ok=True)
    
    result_file_json = f"{outdir}/{simulation_dict['model']}_result.json"

    if not os.path.exists(result_file_json):
        if sampler_type == "dynesty":
            result = run_dynesty(likelihood, priors_dist, simulation_dict, outdir)
        elif sampler_type == "pocomc":
            result = run_pocomc(likelihood, priors_dist, simulation_dict, outdir)
    
    # Post-analysis plotting
    try:
        post_analysis.plot_data_with_psd(handler, outdir, simulation_dict)
        post_analysis.plot_posterior(
            outdir=outdir,
            handler=handler,
            simulation_dict=simulation_dict,
            domain=["frequency"],
            mode=[0.05, "maximum likelihood"],
            resample=False,
        )
    except Exception as e:
        logger.error(f"Post-analysis failed for {simulation_dict['waveform']}: {e}")

    plt.close('all')
    gc.collect()
    return None

def run_dynesty(likelihood, priors_obj, simulation_dict, outdir):
    """Run Dynesty sampler via bilby."""
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors_obj,
        sampler="dynesty",
        npoints=simulation_dict["npoints"],
        outdir=outdir,
        label=simulation_dict["model"],
        dlogz=simulation_dict["dlogz"],
        maxmcmc=10000,
    )
    result.plot_corner()
    return result

def run_pocomc(likelihood, priors_dist, simulation_dict, outdir):
    """Run PocoMC sampler."""
    import pocomc as pc
    import multiprocess
    
    model = bilby.core.sampler.base_sampler.Sampler(
        likelihood=likelihood,
        priors=priors_dist,
        outdir=outdir,
        label=simulation_dict["model"],
        skip_import_verification=True,
    )

    nwalkers = simulation_dict["npoints"]
    start = np.array([model.get_random_draw_from_prior() for _ in range(nwalkers)])
    ndim = start.shape[1]

    def logprior(x):
        lp = model.log_prior(x)
        return -np.inf if np.isnan(lp) else lp

    def loglike(x):
        lp = logprior(x)
        return -np.inf if not np.isfinite(lp) else model.log_likelihood(x)

    bounds = []
    periodicity = []
    poco_labels = []
    for i, key in enumerate(priors_dist.keys()):
        if i >= ndim: break
        p = priors_dist[key]
        bounds.append([p.minimum, p.maximum] if isinstance(p, bilby.core.prior.Uniform) else [np.nan, np.nan])
        if p.boundary is not None: periodicity.append(i)
        poco_labels.append(fr"{p.latex_label}")

    with multiprocess.Pool(10) as pool:
        sampler = pc.Sampler(nwalkers, ndim, loglike, logprior, threshold=1.0,
                            bounds=np.array(bounds), corr_threshold=simulation_dict["corr_threshold"],
                            periodic=periodicity, pool=pool)
        sampler.run(start, ess=0.95, nmin=ndim//2, nmax=1000, progress=True)
        sampler.add_samples(3000)

    # Save results
    df = pd.DataFrame(sampler.results['posterior_samples'], columns=list(priors_dist.keys())[:ndim])
    df['log_prior'] = sampler.results['posterior_logp']
    df['log_likelihood'] = sampler.results['posterior_logl']
    df.to_json(f"{outdir}/{simulation_dict['model']}_result.json")
    
    with open(f"{outdir}/{simulation_dict['model']}_pocomc.pickle", "wb") as f:
        dill.dump(sampler.results, f)

    # Plots
    pc.plotting.trace(sampler.results, labels=poco_labels).savefig(f"{outdir}/{simulation_dict['model']}_trace.png")
    pc.plotting.corner(sampler.results, labels=poco_labels, quantiles=[0.16, 0.84], smooth=1.0).savefig(f"{outdir}/{simulation_dict['model']}_corner.png")
    pc.plotting.run(sampler.results).savefig(f"{outdir}/{simulation_dict['model']}_sampling.png")
    
    return sampler.results

if __name__ == "__main__":
    run()
