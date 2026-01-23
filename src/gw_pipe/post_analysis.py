#!/usr/bin/env python3

# Copyright (C) 2022 Georgios Vretinaris, Stamatis Vretinaris
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/>.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import bilby
from bilby.core import utils as butils
from bilby.gw import utils as gwutils
from bilby.gw.result import CBCResult
from bilby.gw.utils import noise_weighted_inner_product,inner_product
import inspect
import itertools
from gw_pipe import source_model
from gw_pipe import global_vars as glb
from gw_pipe import priors
from gw_pipe import utils
from gw_pipe import NR_strains as nr
from astropy.constants import c, G, M_sun
from gw_pipe import ifo
from scipy.stats import gaussian_kde
import matplotlib.ticker as mt
import gc
from scipy.integrate import simpson
#from memory_profiler import profile
##### pycbc #####
import pycbc
from pycbc.detector import Detector
from pycbc.types import TimeSeries
from pycbc.psd import interpolate
from scipy import signal
from pycbc.filter import make_frequency_series
from gw_pipe import utils as ut
#################
#from memory_profiler import profile

rcparams = {}
rcparams["axes.linewidth"] = 0.5
rcparams["font.family"] = "serif"
rcparams["font.size"] = 16
rcparams['legend.fontsize'] = 12
rcparams["mathtext.fontset"] = "stix"
rcparams["text.usetex"] = True
rcparams["text.latex.preamble"] = r"\usepackage{fontenc}"
rcparams["figure.dpi"] = 300
rcparams["figure.figsize"] = (
    1920 / rcparams["figure.dpi"],
    1440 / rcparams["figure.dpi"],
)
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
rcparams['lines.linewidth'] = 1

plt.rcParams.update(rcparams)  # update plot parameters



def json_checker():
    """Check for directories containing JSON result files."""
    has_json = []
    has_not_json = []
    
    if not os.path.exists(glb.results_path):
        return [], []

    for root, dirs, files in os.walk(glb.results_path):
        if any(f.endswith('.json') for f in files):
            has_json.append(root)
        elif not dirs and not files:
            # Leaf directory with no files
            has_not_json.append(root)
            
    # This is a bit different from the original but safer than 'find'
    return has_json, has_not_json

#@profile
def get_time_domain_posterior(outdir, handler, simulation_dict, mode, resample=False):

    if simulation_dict["model"].lower() == "lorentzian" or simulation_dict["model"].lower() == "last_hope":

        f, htilde = get_frequency_domain_posterior(outdir, handler, simulation_dict, mode, resample)
        h = butils.infft(htilde, 2**14)
        t = handler.waveform_generator.time_array

    else:

        json_result = str([file for file in os.listdir(outdir) if ".json" in file][0])

        try:
            result = CBCResult.from_json(f"{outdir}/{json_result}")
            posterior = result.posterior
        except:
            posterior = pd.read_json(f"{outdir}/{json_result}")

        #parameters = inspect.getfullargspec(
            #source_model.model_picker(
                #simulation_dict["model"], simulation_dict["number_of_freqs"]
            #)
        #)[0][1:]

        if not resample:
            t = handler.waveform_generator.time_array
        else:
            t = handler.waveform_generator.time_array
            t = np.arange(t[0],t[-1],1/(2*handler.sampling_frequency))

        args = get_source_model_args(posterior, handler, simulation_dict, mode)
        args = tuple([t]) + args

        h = source_model.model_picker(
            simulation_dict["model"], simulation_dict["number_of_freqs"]
        )(*args)["plus"]

        try:
            del result
        except:
            pass
        del posterior
        del args
        # gc.collect()

    return t, h

#@profile
def get_frequency_domain_posterior(outdir, handler, simulation_dict, mode, resample=False):

    if simulation_dict["model"].lower() == "lorentzian" or  simulation_dict["model"].lower() == "last_hope":

        json_result = str([file for file in os.listdir(outdir) if ".json" in file][0])

        try:
            result = CBCResult.from_json(f"{outdir}/{json_result}")
            posterior = result.posterior
        except:
            posterior = pd.read_json(f"{outdir}/{json_result}")

        #parameters = inspect.getfullargspec(
            #source_model.model_picker(
                #simulation_dict["model"], simulation_dict["number_of_freqs"]
            #)
        #)[0][1:]

        f = handler.waveform_generator.frequency_array

        args = get_source_model_args(posterior, handler, simulation_dict, mode)
        args = tuple([f]) + args

        h = source_model.model_picker(
            simulation_dict["model"], simulation_dict["number_of_freqs"]
        )(*args)["plus"]

    else:
        t, h = get_time_domain_posterior(outdir, handler, simulation_dict, mode, resample)
        sampling_frequency = handler.sampling_frequency
        if resample:
            #tnew = np.arange(t[0],t[-1], 1/(2*sampling_frequency))
            #h = utils.interpolate(t,h,tnew)
            sampling_frequency *= 2

        freqs, signal_asd = butils.nfft(h, sampling_frequency)
        idx = min(len(signal_asd), len(freqs))
        df = freqs[1] - freqs[0]
        f = freqs[:idx]
        h = signal_asd[:idx]

    return f,h

def get_ASD_from_freq_domain(outdir, handler, simulation_dict, mode, resample=False):
    freqs, signal = get_frequency_domain_posterior(outdir, handler, simulation_dict, mode)
    df = freqs[1] - freqs[0]
    signal_asd = gwutils.asd_from_freq_series(
            freq_data=signal, df=df)
    idx = min(len(signal_asd), len(freqs))

    f = freqs[:idx]
    h = signal_asd[:idx]
    return f,h

#@profile
def get_source_model_args(posterior, handler, simulation_dict, mode):

    parameters = inspect.getfullargspec(
        source_model.model_picker(
            simulation_dict["model"], simulation_dict["number_of_freqs"]
        )
    )[0][1:]

    args = ()

    if isinstance(mode, float):
        if mode > 1:
            raise ValueError("Mode value cannot exceed 1.")
        for parameter in parameters:
            if parameter == 't0':
                args = args + (handler.t0,)
            else:
                args = args + (np.quantile(posterior[parameter].values, mode),)

    elif mode.lower() == "maximum likelihood":
        idx = np.argmax(posterior["log_likelihood"])

        for parameter in parameters:
            if parameter == 't0':
                args = args + (handler.t0,)
            else:
                args = args + (posterior[parameter].values[idx],)

    elif mode.upper() == "MAP" or mode.lower() == "maximum a posteriori":
        idx = np.argmax(posterior["log_likelihood"] + posterior["log_prior"])

        for parameter in parameters:
            if parameter == 't0':
                args = args + (handler.t0,)
            else:
                args = args + (posterior[parameter].values[idx],)

    elif 'opt' in mode.lower():
        for parameter in parameters:
            if parameter == 't0':
                args = args + (handler.t0,)
            else:
                args = args + (posterior[parameter].values[0],)



    elif mode.lower() == 'random':
        freqs = [i for i in simulation_dict['priors'].keys() if 'f_' in i]
        if simulation_dict['number_of_freqs'] > 2:
            logBFs = np.array([get_Bayes_Factor(freq,simulation_dict['priors'][freq],posterior[freq]) for freq in freqs])

            idx = np.where(logBFs >  -1e3)[0]

            if idx.size != 0:
                for index in idx:
                    descriptor = freqs[index].split('_')[-1]
                    names = [i for i in simulation_dict['priors'].keys() if descriptor in i]
                    shape = posterior.shape
                    posterior[names] = np.ones((shape[0],len(names)))*(-50)
        else:
            pass
        try:
            posterior = posterior[((posterior['h_high']<=-19)&
                                (posterior['h_1']<=-19)&
                                (posterior['h_2']<=-19))]
        except:
            if simulation_dict['number_of_freqs'] > 2:
                posterior = posterior[((posterior['h_1']<=-19)&
                                       (posterior['h_2']<=-19))]

            else:
                posterior = posterior[posterior['h_2']<=-19]



        idx = np.random.randint(0,posterior.shape[0])
        for parameter in parameters:
            if parameter == 't0':
                args = args + (handler.t0,)
            else:
                args = args + (posterior[parameter].values[idx],)

    del posterior
    del parameters
    #gc.collect()

    return args

### Just Wrong
def get_Bayes_Factor(feature_name,prior,posterior):

    #### posterior is a pandas DataFrame


    posterior_logpdf = gaussian_kde(posterior)
    posterior_logpdf_at_zero = posterior_logpdf.logpdf([0,0])[0]
    if prior.maximum!=np.inf or prior.minimum!=-np.inf:
        prior_logpdf_at_zero = np.log(1.0 / (prior.maximum - prior.minimum))
    else:
        prior_logpdf_at_zero = np.log(1.0 / (prior.sigma * 6))

    logBF = posterior_logpdf_at_zero - prior_logpdf_at_zero
    return logBF

#@profile
def plot_posterior(outdir, handler, simulation_dict, domain, mode,resample):

    if isinstance(domain, list):
        for entry_domain in domain:
            if isinstance(mode, list):
                for entry_mode in mode:
                    plot_posterior(
                        outdir, handler, simulation_dict, entry_domain, entry_mode,resample
                    )
        return None

    if isinstance(mode, list):
        for entry_mode in mode:
            plot_posterior(outdir, handler, simulation_dict, domain, entry_mode,resample)
        return None

    if isinstance(mode, float):
        if mode > 1:
            raise ValueError("Mode value cannot exceed 1.")
        else:
            if domain == "frequency":
                mode = [mode / 2, 0.5, 1 - mode / 2]
                _plotter(outdir, handler, simulation_dict, domain, mode,resample)
            elif domain == "time":
                _plotter(outdir, handler, simulation_dict, domain, 0.5,resample)
            mode = "median"
    elif mode == "maximum likelihood":
        _plotter(outdir, handler, simulation_dict, domain, mode,resample)
    elif mode == "MAP":
        _plotter(outdir, handler, simulation_dict, domain, mode,resample)
    elif 'opt' in mode.lower():
        _plotter(outdir, handler, simulation_dict, domain, mode,resample)
    else:
        raise ValueError("Mode is not supported.")
    plt.tight_layout()
    save_path = f"{outdir}/{mode}_{domain}_domain.pdf"
    plt.savefig(save_path, dpi=300)
    plt.close()
    #gc.collect()
    #os.system("echo 'save done'")

#@profile
def _plotter(outdir, handler, simulation_dict, domain, mode,resample,ax=None):

    type = priors.classifier(handler.data.metadata_dict)

    if type == 1:
        EOS = handler.data.metadata_dict["id_eos"]
        mass = handler.data.metadata_dict["id_mass"]
        if EOS == "SLy" and 2.825/2 - mass < 0.1:
            type = "Ib"
        elif EOS == "LS220" and 2.975/2 - mass < 0.1:
            type = "Ib"
        elif EOS == "MPA1" and 3.225/2 - mass < 0.1:
            type = "Ib"
        else:
            type = "I" 
    elif type == 2:
        type = "II"
    else:
        type = "III"

    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    flag = 0
    if ax == None:
        fig,ax = plt.subplots()
        flag = 1

    if domain == "time":

        t, h = get_time_domain_posterior(outdir, handler, simulation_dict, mode)
        if isinstance(mode, float):
            mode = "median"
        elif mode == "MAP":
            mode = "Maximum A Posteriori"


        #plt.title(
           # f"FF = {get_FittingFactor(outdir, handler, simulation_dict, mode):.5f}"
        #)

        #plt.plot(t,h*1e20)

        mask = ((t>=handler.t0-5*(handler.data.time[1]-handler.data.time[0])) & (t<=handler.t_end))
        if mode == "Maximum A Posteriori":
            mode = "MAP"
        plt.plot(
            t[mask],
            handler.waveform_generator.time_domain_strain()["plus"][mask]*1e21,
            label="Numerical Waveform",
            color="#2B2E7D"
        )
        plt.plot(t[mask], h[mask]*1e21, color="#C33F04", label=f"{mode.title()}",
                 linewidth=2, alpha=0.6)
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Strain $\\times 10^{{21}}$")
        plt.tick_params(bottom=True, left=True, direction="in")
        plt.legend()

    elif domain == "frequency":

        if isinstance(mode, list):
            signal_list = np.zeros(3, dtype=object)
            for i, mode_item in enumerate(mode):
                freqs, signal = get_ASD_from_freq_domain(
                    outdir,
                    handler,
                    simulation_dict,
                    mode_item,
                    resample=resample
                )
                signal_list[i] = signal

            #fig, ax = plt.subplots()
            ls = ['-','','--']
            nn = ['aLIGO','','AdV']
            ii = [1,0,-1]
            for i,ifo in enumerate(handler.ifos):
                if i == 1:
                    continue

                x = ifo.strain_data.frequency_array[ifo.strain_data.frequency_mask]
                y = ifo.amplitude_spectral_density_array[ifo.strain_data.frequency_mask]
                mask = (x>=500)&(x<=5000)
                ax.plot(x[mask],y[mask],ls=ls[i],
                        color='k',alpha=0.7)

                ax.annotate(xy=(x[mask][-50],y[mask][-50]), xytext=(-10,ii[i]*15), textcoords='offset points', text=fr"$\mathrm{{{nn[i]}}}$", va='center',fontsize=12)

            #FFlist = [get_FittingFactor(outdir, handler, simulation_dict, 'random') for it in range(1000)]
            #print(np.mean(FFlist))
            #print(get_FittingFactor(outdir, handler, simulation_dict, mode[1]))
            #print(np.std(FFlist))

            #FFlist = [get_FittingFactor(outdir, handler, simulation_dict, it) for it in [0.05,0.5,0.95]]
            #print(FFlist)
            #print(FFlist[1]-FFlist[0],FFlist[1]-FFlist[-1])

            #exit()


            FF = get_FittingFactor(outdir, handler, simulation_dict, mode[1])
            plt.text(550, 1.2e-25, fr"$\mathcal{{F}}=${FF:.5f}", ha='left', va='bottom')
            plt.text(550, 4e-22, fr"Type {type}", ha='left', va='top')

            mask = ((freqs>=500) & (freqs<=5000))
            ax.plot(
                freqs[mask],
                np.abs(signal_list[1][mask]),
                color="#C33F04",
                linewidth=1,
                label=f"Median",
                zorder=3
            )
            for i in range(1000):
                freqs, signal = get_ASD_from_freq_domain(
                    outdir,
                    handler,
                    simulation_dict,
                    "random",
                    resample=resample
                )
                ax.plot(
                    freqs[mask],
                    np.abs(signal[mask]),
                    color="gray",
                    linewidth=0.5,
                    alpha=0.2,
                )
            #[print(i) for i in locals()]
            #exit()
            #plt.fill_between(
                #freqs,
                #np.abs(signal_list[0]) * np.sqrt(freqs),
                #np.abs(signal_list[-1]) * np.sqrt(freqs),
                #color="darkorange",
                #linewidth=1,
                #linestyle="--",
                #alpha=0.6,
                #label=f"{int((1-mode[0]*2)*100)}\% Credible Interval",
                #zorder=2,
            #)

        elif 'opt' in mode.lower():
            freqs, signal = get_ASD_from_freq_domain(
                outdir, handler, simulation_dict, mode, resample=resample
            )

            mask = ((freqs>=500) & (freqs<=5000))
            #plt.title(
                #f"FF = {get_FittingFactor(outdir, handler, simulation_dict, mode):.5f}"
            #)
            plt.plot(
                freqs[mask],
                np.abs(signal[mask]),
                color="#C33F04",
                linewidth=2,
                label=f"{mode.title()}",
            )
            f_peak, s_peak = priors.VSB_fpeak(
                handler.data.metadata_dict, simulation_dict["method"]
            )
            plt.axvline(f_peak, ls="--", color="#F09D78", lw=1.2)
            plt.axvspan(
                f_peak - 3 * s_peak, f_peak + 3 * s_peak, color="#F09D78", alpha=0.1
            )

            f_spiral, s_spiral = priors.VSB_fspiral(
                handler.data.metadata_dict, simulation_dict["method"]
            )
            plt.axvline(f_spiral, ls="--", color="#787AB0", lw=1.2)
            plt.axvspan(
                f_spiral - 3 * s_spiral,
                f_spiral + 3 * s_spiral,
                color="#787AB0",
                alpha=0.1,
            )

            f_20, s_20 = priors.VSB_f20(
                handler.data.metadata_dict, simulation_dict["method"]
            )
            plt.axvline(f_20, ls="--", color="#89B2B8", lw=1.2)
            plt.axvspan(f_20 - 3 * s_20, f_20 + 3 * s_20, color="#89B2B8", alpha=0.1)
        elif mode == "maximum likelihood":

            freqs, signal = get_ASD_from_freq_domain(
                outdir, handler, simulation_dict, mode, resample=resample
            )

            plt.text(4950, 4e-22, f"Type {type}", ha='right', va='top')

            #fig, ax = plt.subplots()
            mask = ((freqs>=500) & (freqs<=5000))
            #plt.title(
                #f"FF = {get_FittingFactor(outdir, handler, simulation_dict, mode):.5f}"
            #)

            plt.plot(
                freqs[mask],
                np.abs(signal[mask]),
                color="#C33F04",
                linewidth=1,
                label=f"Maximum\n Likelihood",
            )

            json_result = str([file for file in os.listdir(outdir) if ".json" in file][0])

            try:
                result = CBCResult.from_json(f"{outdir}/{json_result}")
                posterior = result.posterior
            except:
                posterior = pd.read_json(f"{outdir}/{json_result}")

            parameters = inspect.getfullargspec(
                source_model.model_picker(
                    simulation_dict["model"], simulation_dict["number_of_freqs"]
                )
            )[0]

            args = tuple([handler.waveform_generator.time_array],)+get_source_model_args(posterior, handler, simulation_dict, mode)

            labels = ['peak','sec.1','sec.2','post-peak']
            style = ['-','--',':','-.']

            ### bad way of coding but too bored to change it now
            if simulation_dict["number_of_freqs"] == 4:
                names = ['peak','1','2','high']
            elif simulation_dict["number_of_freqs"] == 3:
                names = ['peak','1','2']
            elif simulation_dict["number_of_freqs"] == 2:
                names = ['peak','2']

            import re
            for j,name in enumerate(names):
                regex = re.compile(f".*{name}")
                matches = [string for string in parameters if re.match(regex, string)]
                rest = [string for string in parameters if string not in matches]

                idxs = [parameters.index(elem) for elem in rest]
                args2 = [-50 if i in idxs else args[i] for i in range(len(args))]
                args2[0] = handler.waveform_generator.time_array
                args2[-1] = handler.t0
                args2 = tuple(args2)

                h = source_model.model_picker(
                    simulation_dict["model"], simulation_dict["number_of_freqs"]
                )(*args2)["plus"]

                signal, freqs = butils.nfft(h, handler.sampling_frequency)
                signal = gwutils.asd_from_freq_series(
                        freq_data=signal, df=(freqs[1]-freqs[0])
                        )

                idx = min(len(signal), len(freqs))
                signal = signal[:idx]; freqs = freqs[:idx]
                mask1 = ((freqs>=500) & (freqs<=5000))
                plt.plot(
                        freqs[mask1],
                        np.abs(signal[mask1]),
                        color="gray",
                        linewidth=1,
                        alpha=0.6,
                        label=fr"$f_{{\small\textrm{{{labels[j]}}}}}$",
                        ls=f"{style[j]}",
                        zorder = 3
                    )

            f_peak, s_peak = priors.VSB_fpeak(
                handler.data.metadata_dict, simulation_dict["method"]
            )
            plt.axvline(f_peak, ls="--", color="#F09D78", lw=1.2)
            plt.text(f_peak+50, 1.5e-22 ,r'$\bar{{f}}_{{\mathrm{{peak}}}}$',color="#F09D78",rotation=270)

            plt.axvspan(
                f_peak - 3 * s_peak, f_peak + 3 * s_peak, color="#F09D78", alpha=0.1
            )

            f_spiral, s_spiral = priors.VSB_fspiral(
                handler.data.metadata_dict, simulation_dict["method"]
            )

            plt.text(f_spiral+50, 1.5e-22 ,r'$\bar{{f}}_{{\mathrm{{spiral}}}}$',color="#787AB0",rotation=270)

            plt.axvline(f_spiral, ls="--", color="#787AB0", lw=1.2)
            plt.axvspan(
                f_spiral - 3 * s_spiral,
                f_spiral + 3 * s_spiral,
                color="#787AB0",
                alpha=0.1,
            )

            f_20, s_20 = priors.VSB_f20(
                handler.data.metadata_dict, simulation_dict["method"]
            )
            plt.axvline(f_20, ls="--", color="#89B2B8", lw=1.2)
            plt.text(f_20+50, 1.5e-22 ,r'$\bar{{f}}_{{2-0}}$',color="#89B2B8",rotation=270)

            plt.axvspan(f_20 - 3 * s_20, f_20 + 3 * s_20, color="#89B2B8", alpha=0.1)
            #plt.plot([],[],color=(0,0,0,0), label=" ")

            def flip(items, ncol):
                return itertools.chain(*[items[i::ncol] for i in range(ncol)])

            handles, labels = ax.get_legend_handles_labels()
            temp1 = handles[0]
            temp2 = labels[0]
            for i in range(len(handles)-1):
                handles[i] = handles[i+1]
                labels[i] = labels[i+1]

            handles[-1] = temp1
            labels[-1] = temp2
            fig.set_figwidth(2120 / rcparams["figure.dpi"])
            plt.legend(#flip(handles, 4), flip(labels, 4),
                       #ncol=2,columnspacing=-1./rcparams["figure.dpi"],
                bbox_to_anchor=(1.025, .5),
                loc="center left",
                borderaxespad=0)#, mode="expand")

        elif mode == "MAP":
            freqs, signal = get_ASD_from_freq_domain(
                outdir, handler, simulation_dict, mode, resample = resample
            )
            #fig, ax = plt.subplots()
            mask = ((freqs>=500) & (freqs<=5000))
            #plt.title(
                #f"FF = {get_FittingFactor(outdir, handler, simulation_dict, mode):.5f}"
            #)
            plt.plot(
                freqs[mask],
                np.abs(signal[mask]),
                color="#C33F04",
                linewidth=2,
                label=f"Maximum A Posteriori",
            )
            f_peak, s_peak = priors.VSB_fpeak(
                handler.data.metadata_dict, simulation_dict["method"]
            )
            plt.axvline(f_peak, ls="--", color="#F09D78", lw=1.2)

            plt.axvspan(
                f_peak - 3 * s_peak, f_peak + 3 * s_peak, color="#F09D78", alpha=0.1
            )

            f_spiral, s_spiral = priors.VSB_fspiral(
                handler.data.metadata_dict, simulation_dict["method"]
            )
            plt.axvline(f_spiral, ls="--", color="#787AB0", lw=1.2)
            plt.axvspan(
                f_spiral - 3 * s_spiral,
                f_spiral + 3 * s_spiral,
                color="#787AB0",
                alpha=0.1,
            )

            f_20, s_20 = priors.VSB_f20(
                handler.data.metadata_dict, simulation_dict["method"]
            )
            plt.axvline(f_20, ls="--", color="#89B2B8", lw=1.2)
            plt.axvspan(f_20 - 3 * s_20, f_20 + 3 * s_20, color="#89B2B8", alpha=0.1)

        if resample:
            data = nr.NumericalData(simulation_dict["waveform"],handler.sampling_frequency*2)
            data.get_post_merger()
            injection_parameters_data = dict(
                psi=0,
                geocent_time=0,
                ra=0,
                dec=0,
                mass_1=data.m1 * M_sun.value,
                mass_2=data.m2 * M_sun.value,
            )

            handler = ifo.InterferometerHandler(
                data,
                injection_parameters_data,
                noise=simulation_dict["noise"],
                ifo_list=["L1", "H1", "V1"],
            )

            data.rescale_to_SNR(handler.Hsnr, 1, simulation_dict["snr"])
            # data.rescale_to_dist(04)
            handler = ifo.InterferometerHandler(
                data,
                injection_parameters_data,
                noise=simulation_dict["noise"],
                ifo_list=["L1", "H1", "V1"],
            )

        data = handler.waveform_generator.frequency_domain_strain()['plus']
        data_freqs = handler.waveform_generator.frequency_array
        signal_asd = gwutils.asd_from_freq_series(freq_data=data, df=(data_freqs[1]-data_freqs[0]))
        mask = ((data_freqs>500)&(data_freqs<5000))

        ax.plot(
            data_freqs[mask],
            signal_asd[mask],
            label="Numerical Waveform",
            color="#2B2E7D",
        )

        # plt.grid(True, which="major", linestyle="--")
        if flag == 1:
            ax.set_xlabel("Frequency [Hz]")

        ax.set_ylabel(r"ASD [1/$\sqrt{\mathrm{Hz}}$]")
        ax.set_yscale("log")
        ax.set_xlim(500, 5000)
        if simulation_dict['snr'] != 50:
            ax.set_ylim([5e-26,3e-22])
        else:
            ax.set_ylim([1e-25,5e-22])

        ax.set_ylim([1e-25,5e-22])
        plt.tick_params(bottom=True, left=True, direction="in")
        plt.tick_params(axis="y", direction="in", which="minor")
        plt.title(f"EOS {handler.data.metadata_dict['id_eos']}, {handler.data.metadata_dict['id_mass_starA']}+{handler.data.metadata_dict['id_mass_starB']}, SNR={handler.data.metadata_dict['SNR']}")
        #plt.tight_layout()
        ax.set_xticks(np.arange(1000, 4001, 1000))
        ax.set_xticks(np.arange(1500, 4501, 1000), minor=True)
        #return ax
        if mode != 'maximum likelihood':
            ax.legend(loc='lower right')
    else:
        raise ValueError("Provide correct argument for the domain [time/frequency].")

    #gc.collect()



snr_sq = 0
#@profile
def plot_data_with_psd(handler, outdir ,simulation_dict,ax=None):
    global snr_sq
    if np.all(ax != None):
        ncols = 3
    else:
        ncols = 1

    #fig = plt.figure()
    #ax = fig.subplots(3,3,sharex=True)#,gridspec_kw={'hspace': 0})
    for i,ifo in enumerate(handler.ifos):
        if i == 0:
            plt.title(f"EOS {handler.data.metadata_dict['id_eos']}, {handler.data.metadata_dict['id_mass_starA']}+{handler.data.metadata_dict['id_mass_starB']}, SNR={handler.data.metadata_dict['SNR']}")

        if ncols == 3:
            ax = plt.subplot(3,ncols,i*3+1)#,sharex=True,gridspec_kw={'hspace': 0})
        else:
            ax = plt.subplot(3,ncols,i+1)
        df = ifo.strain_data.frequency_array[1] - ifo.strain_data.frequency_array[0]
        asd = gwutils.asd_from_freq_series(
            freq_data=ifo.strain_data.frequency_domain_strain, df=df)

        freqs = ifo.strain_data.frequency_array[ifo.strain_data.frequency_mask]

        ax.plot(freqs,
                asd[ifo.strain_data.frequency_mask],
                color='gray',alpha=0.8, label=ifo.name + ' Simulated Noise + Injected Signal')
        ax.plot(freqs,
                ifo.amplitude_spectral_density_array[ifo.strain_data.frequency_mask],
                color='black', lw=1.0, label=ifo.name + ' ASD')

        signal_asd = gwutils.asd_from_freq_series(
            freq_data=handler.waveform_generator.frequency_domain_strain()["plus"],
            df=df)

        ax.plot(freqs,
                    signal_asd[ifo.strain_data.frequency_mask],
                    color="#2B2E7D",
                    label='Injected Signal')

        freqs, signal = get_frequency_domain_posterior(
                outdir, handler, simulation_dict, 'maximum likelihood')

        #signal_asd = gwutils.asd_from_freq_series(
        #    freq_data=signal, df=(freqs[1]-freqs[0]))

        mask = ((freqs>=500) & (freqs<=5000))
        ax.plot(freqs[mask],
                    signal[mask],
                    color="#C33F04",
                    label='Maximum Likelihood')

        # t, h = get_time_domain_posterior(outdir, handler, simulation_dict, 0.5)
        # sampling_frequency = handler.sampling_frequency
        # signal, freqs = butils.nfft(h, sampling_frequency)
        # idx = min(len(signal), len(freqs))
        # df = freqs[1] - freqs[0]

        # mask = ((freqs>=500) & (freqs<=5000))


        # a1 = ifo.strain_data.frequency_domain_strain[ifo.strain_data.frequency_mask]
        # a2 = handler.waveform_generator.frequency_domain_strain()["plus"][ifo.strain_data.frequency_mask]
        # a3 = signal[mask]

        # np.savetxt(f'{outdir}/{ifo.name}.dat',np.c_[a1,a2],delimiter=',')

        ax.grid(True,which='major')
        #ax.grid(True,which='minor')
        #ax.set_yticks(np.arange(1,2,1),minor=True)
        minor_locator = mt.AutoMinorLocator(2)
        ax.yaxis.set_minor_locator(minor_locator)
        ax.set_yscale('log')
        #ax.legend(loc='best')
        ax.set_xlim([500,5000])
        if simulation_dict['snr'] != 50:
            ax.set_ylim([5e-26,3e-22])

        ax.set_ylim([1e-25,5e-22])
        ax.set_xticks(np.arange(1000, 4501, 500))
        ax.set_ylabel(f'{ifo.name}')
        #ax.set_xticks(np.arange(1500, 4501, 1000), minor=True)

    #fig.supylabel(r'ASD [1/$\sqrt{\rm Hz}$]')
    plt.xlabel(r'Frequency [Hz]')
    #plt.title(f"EOS {handler.data.metadata_dict['id_eos']}, {handler.data.metadata_dict['id_mass_starA']}+{handler.data.metadata_dict['id_mass_starB']}, SNR={handler.data.metadata_dict['SNR']}")

    #plt.xticks(np.arange(1000, 4001, 1000))
    if ncols != 3:
        plt.tight_layout()
        plt.savefig(
            '{}/detector_network_data.pdf'.format(outdir))

    #plt.close(fig)
    #del asd
    #del signal_asd
    #del freqs
    #del signal
    #del mask
    #gc.collect()
        #handler.ifos.plot_data(
            #handler.waveform_generator.frequency_domain_strain()["plus"], outdir=outdir
        #)
    #except:
        #print("Another occation where their method fails")

def inner_product(a,b,psd,freqs):
    return 4*simps(np.conj(a)*b/psd,x=freqs).real

def get_FittingFactor(outdir, handler, simulation_dict, mode,dummy=None):

    if mode == 'Median' or mode == 'median':
        mode = 0.5

    t, hp = get_time_domain_posterior(
        outdir, handler, simulation_dict, mode
    )

    hc = utils.generate_cross(hp)
    h = hp - 1j * hc

    signal = np.fft.fft(h)[:h.size//2]*(t[1]-t[0])
    freqs = np.fft.fftfreq(h.size,t[1]-t[0])[:h.size//2]

    t = handler.waveform_generator.time_array
    data = np.fft.fft(
        handler.waveform_generator.time_domain_strain()["plus"]
        -1j*handler.waveform_generator.time_domain_strain()["cross"]
        )[:handler.waveform_generator.time_domain_strain()["plus"].size//2]*(t[1]-t[0])

    #f_peak, s_peak = priors.VSB_fpeak(
        #handler.data.metadata_dict, simulation_dict["method"]
    #)
    f_cut = 5000 #f_peak+3*s_peak

    # Use shared helper for noise curves
    ligo_asd_file = utils.get_bilby_noise_path('aLIGO_O4_high_asd.txt')
    LIGO_asd = np.loadtxt(ligo_asd_file)

    asd = ut.interpolate(LIGO_asd.T[0],LIGO_asd.T[1],
                    np.arange(LIGO_asd.T[0,0],LIGO_asd.T[0,-1],1/handler.duration))

    psd_aLIGO = pycbc.types.frequencyseries.FrequencySeries(
        pow(asd,2),delta_f=1/handler.duration)

    #idx = min(
        #len(signal),
        #len(data),
        ##len(psd_aLIGO)
        #len(handler.ifos[0].power_spectral_density_array),
    #)

    delta_frequency = 1/handler.duration
    idx_start = int(1000 / delta_frequency)
    idx_end = int(5000 / delta_frequency)

    #idx_end = int(min(5000,psd_aLIGO.sample_frequencies.numpy()[-1]) / delta_frequency)


    #idx_start = min(
        #len(signal[:idx][freqs[:idx]<=1000]),
        #len(data[:idx][freqs[:idx]<=1000]),
        ##len(psd_aLIGO[:idx][psd_aLIGO.sample_frequencies.numpy()[:idx]<=1000])
        #len(handler.ifos[0].power_spectral_density_array[:idx][freqs[:idx]<=1000]),
    #)
    #idx_end = min(
        #len(signal[:idx][freqs[:idx]<=f_cut]),
        #len(data[:idx][freqs[:idx]<=f_cut]),
        ##len(psd_aLIGO[:idx][psd_aLIGO.sample_frequencies.numpy()[:idx]<=f_cut])
        #len(handler.ifos[0].power_spectral_density_array[:idx][freqs[:idx]<=f_cut]),
    #)

    integrant =  np.conj(
        signal[idx_start:idx_end])*data[idx_start:idx_end]/handler.ifos[0].power_spectral_density_array[idx_start:idx_end]


    numerator = inner_product(
        signal[idx_start:idx_end],
        data[idx_start:idx_end],
        #psd_aLIGO[idx_start:idx_end],
        handler.ifos[0].power_spectral_density_array[idx_start:idx_end],
        freqs[idx_start:idx_end],
    )

    denominator = np.sqrt(
        inner_product(
            signal[idx_start:idx_end],
            signal[idx_start:idx_end],
            handler.ifos[0].power_spectral_density_array[idx_start:idx_end],
            freqs[idx_start:idx_end],
            #1/handler.duration,
        )
        * inner_product(
            data[idx_start:idx_end],
            data[idx_start:idx_end],
            handler.ifos[0].power_spectral_density_array[idx_start:idx_end],
            freqs[idx_start:idx_end],
            #1/handler.duration,
        )
    )

    #numerator = noise_weighted_inner_product(
        #signal[idx_start:idx_end],
        #data[idx_start:idx_end],
        #handler.ifos[0].power_spectral_density_array[idx_start:idx_end],
        #handler.duration,
    #)

    #denominator = np.sqrt(
        #noise_weighted_inner_product(
            #signal[idx_start:idx_end],
            #signal[idx_start:idx_end],
            #handler.ifos[0].power_spectral_density_array[idx_start:idx_end],
            #handler.duration,
        #)
        #* noise_weighted_inner_product(
            #data[idx_start:idx_end],
            #data[idx_start:idx_end],
            #handler.ifos[0].power_spectral_density_array[idx_start:idx_end],
            #handler.duration,
        #)
    #)

    #print(sum(integrant).real*4*delta_frequency/ denominator)
    #print((numerator / denominator).real)

    return (numerator / denominator).real

### old
def old_get_FittingFactor(outdir, handler, simulation_dict, mode, dummy=None):

    if simulation_dict["model"].lower() == "lorentzian" or simulation_dict["model"].lower() == "last_hope":
        freqs, signal = get_frequency_domain_posterior(outdir, handler, simulation_dict, mode)
        data = handler.waveform_generator.frequency_domain_strain()

    else:

        if mode == 'Median' or mode == 'median':
            mode = 0.5

        t, hp = get_time_domain_posterior(
            outdir, handler, simulation_dict, mode
        )

        hc = utils.generate_cross(hp)
        h = hp - 1j * hc

        signal = np.fft.fft(h)[:h.size//2]*(t[1]-t[0])
        freqs = np.fft.fftfreq(h.size,t[1]-t[0])[:h.size//2]

        t = handler.waveform_generator.time_array
        data = np.fft.fft(
            handler.waveform_generator.time_domain_strain()["plus"]
            -1j*handler.waveform_generator.time_domain_strain()["cross"]
            )[:handler.waveform_generator.time_domain_strain()["plus"].size//2]*(t[1]-t[0])

        #f_peak, s_peak = priors.VSB_fpeak(
            #handler.data.metadata_dict, simulation_dict["method"]
        #)

    f_cut = 5000 #f_peak+3*s_peak

    idx = min(
        len(signal),
        len(data),
        len(handler.ifos[0].power_spectral_density_array),
    )

    idx_start = min(
        len(signal[:idx][freqs[:idx]<=1000]),
        len(data[:idx][freqs[:idx]<=1000]),
        len(handler.ifos[0].power_spectral_density_array[:idx][freqs[:idx]<=1000]),
    )
    idx_end = min(
        len(signal[:idx][freqs[:idx]<=f_cut]),
        len(data[:idx][freqs[:idx]<=f_cut]),
        len(handler.ifos[0].power_spectral_density_array[:idx][freqs[:idx]<=f_cut]),
    )

    numerator = noise_weighted_inner_product(
        signal[idx_start:idx_end],
        data[idx_start:idx_end],
        handler.ifos[0].power_spectral_density_array[idx_start:idx_end],
        handler.duration,
    )

    denominator = np.sqrt(
        noise_weighted_inner_product(
            signal[idx_start:idx_end],
            signal[idx_start:idx_end],
            handler.ifos[0].power_spectral_density_array[idx_start:idx_end],
            handler.duration,
        )
        * noise_weighted_inner_product(
            data[idx_start:idx_end],
            data[idx_start:idx_end],
            handler.ifos[0].power_spectral_density_array[idx_start:idx_end],
            handler.duration,
        )
    )

    return np.real(numerator / denominator)


def plot_joined_fig(outdir, handler, simulation_dict):
    import copy
    import matplotlib as mpl
    fig = plt.figure()
    size = fig.get_size_inches()
    fig.set_size_inches(size[0]*3, size[1])
    ax = np.zeros(3,dtype=object)
    fig.supylabel(r'ASD [1/$\sqrt{\rm Hz}$]')
    plot_data_with_psd(handler, outdir, simulation_dict, plt.subplot(1,3,3))

    for i,val in enumerate([[0.025,0.5,0.925],"maximum likelihood"]):
        i = i+1

        if i == 1:
                ax[i] = plt.subplot(
                mpl.gridspec.SubplotSpec(
                mpl.gridspec.GridSpec(1,3),i,
                ),
                    )

        #ax[i] = plt.subplot(1,3,i+1)
        else:
            ax[i] = plt.subplot(
                mpl.gridspec.SubplotSpec(
                mpl.gridspec.GridSpec(1,3,wspace=0),i,
                ),
                #sharey=ax[0]
                )

        _plotter(outdir, handler, simulation_dict,
                                    domain='frequency',
                                    mode=val,
                                    resample=False,
                                    ax = ax[i])

        if i == 1:
            ax[i].legend(title=f"FF={get_FittingFactor(outdir, handler, simulation_dict,'MAP'):.4f}",loc='lower right')
            #ax[i].set_ylabel("ASD $[1/\sqrt{\mathrm{Hz}}]$")
        else:
            ax[i].set_yticks([])
            ax[i].set_ylabel(None)
            box = ax[1].get_position()
            x0 = box.x1
            box = ax[i].get_position()
            box.x0 = x0
            ax[i].set_position(box)
            #ax[i].set_xlabel([])

    #plot_data_with_psd(handler, outdir, simulation_dict, plt.subplot(1,3,3))

    fig.tight_layout()
    plt.savefig(f"{outdir}/joined_plot.pdf",format='pdf',bbox_tight=True)


