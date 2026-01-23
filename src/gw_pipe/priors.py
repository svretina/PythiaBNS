#!/usr/bin/env python3
# Copyright (C) 2022 Stamatis Vretinaris, Christos Mermigkas, Georgios Vretinaris
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

import os
import logging
import bilby
import numpy as np
import pandas as pd
import inspect
import matplotlib.pyplot as plt
import _pickle as pickle
from gw_pipe import utils
from gw_pipe import global_vars
from gw_pipe import source_model
from gw_pipe import predictors

logger = logging.getLogger(__name__)



from gw_pipe.registry import ModelRegistry

def convert_parameters(parameters, model_name, nfreqs):
    """Convert parameters using the registered conversion function for the model."""
    conversion_func = ModelRegistry.get_conversion_func(model_name, nfreqs)
    if conversion_func:
        return conversion_func(parameters)
    return parameters

def freqs_cutter(simulation_dict, metadata_dict):
    region = classifier(metadata_dict)

    if region == 1 or region == 3:
        simulation_dict['number_of_freqs'] = 2
    elif region == 2:
        simulation_dict['number_of_freqs'] = 3

def classifier(metadata_dict):

    m1 = metadata_dict['id_mass_starA']
    m2 = metadata_dict['id_mass_starB']
    _, r1 = predictors.get_Lambda_and_R(m1, metadata_dict['id_name'])
    _, r2 = predictors.get_Lambda_and_R(m2, metadata_dict['id_name'])

    # function to find if point p is lower than line
    # created by points a and b
    def isabove(p, a, b): return np.cross(p-a, b-a) < 0

    r = np.array([11, 16])
    low_line = 0.12074852048946*r - 0.38801323366469
    high_line = 0.20414622214340*r - 1.27775721544950

    m_fin = (m1+m2)/2
    _,r_fin = predictors.get_Lambda_and_R(m_fin,metadata_dict['id_name'])

    low_a = np.array([r[0],low_line[0]])
    low_b = np.array([r[-1],low_line[-1]])
    high_a = np.array([r[0],high_line[0]])
    high_b = np.array([r[-1],high_line[-1]])

    if isabove(np.array([r_fin,m_fin]),high_a,high_b):
        region = 1
        if metadata_dict['id_eos'] == "SLY" and 2.825/2 - m_fin < 1:
            metadata_dict["Type"] = "Ib"
        elif metadata_dict['id_eos'] == "LS220" and 2.975/2 - m_fin < 1:
            metadata_dict["Type"] = "Ib"
        elif metadata_dict['id_eos'] == "SLY" and 3.225/2 - m_fin < 1:
            metadata_dict["Type"] = "Ib"
        else:
            metadata_dict["Type"] = "I"
    elif ~isabove(np.array([r_fin,m_fin]),low_a,low_b):
        region = 3
        metadata_dict["Type"] = "III"
    else:
        region = 2
        metadata_dict["Type"] = "II"


    return region
    
def empirical_relation(filename):

    ### Read pickle file that contains the regression function
    with open('regress.pickle', 'rb') as f:
        data = pickle.load(f)

    ### Get the features needed to evaluate the function
    pred_X = predictors.get_predictor(filename)
    ### Transform the array properly for use
    pred_X = pred_X.T.loc[range(1,len(pred_X))].T
    pred_tilde = pred_X.rdiv(1)
    pred_tilde.set_axis(
        [f"{head}^-1" for head in pred_X.columns], axis=1, inplace=True,
    )
    pred_X = pred_X.join(pred_tilde)
    
    ### And finally get the requested value
    fpeak = data['pipeline'].predict(pred_X)[0]
    # pd.DataFrame(data['X'].loc[row]).T)[0]

    return fpeak, data['gauss_sigma']

def VSB_R_fpeak(metadata_dict):
    m1 = metadata_dict['id_mass_starA']
    m2 = metadata_dict['id_mass_starB']
    Mchirp = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
    _, R18 = predictors.get_Lambda_and_R(1.8, metadata_dict['id_name'])

    max_res =  0.247*Mchirp*1e3
    sigma = max_res/3
    return Mchirp*(10.942
            -0.369*Mchirp
            -0.987*R18
            +1.095*Mchirp**2
            -0.201*Mchirp*R18
            +0.036*R18**2)*1e3, sigma

def VSB_R_f20(metadata_dict):
    m1 = metadata_dict['id_mass_starA']
    m2 = metadata_dict['id_mass_starB']
    Mchirp = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
    _, R18 = predictors.get_Lambda_and_R(1.8, metadata_dict['id_name'])

    max_res = 0.258*Mchirp*1e3
    sigma = max_res/3
    return Mchirp*(8.007
            +4.356*Mchirp
            -1.241*R18
            +0.558*Mchirp**2
            -0.375*Mchirp*R18
            +0.054*R18**2)*1e3, sigma

def VSB_R_fspiral(metadata_dict):
    m1 = metadata_dict['id_mass_starA']
    m2 = metadata_dict['id_mass_starB']
    Mchirp = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
    _, R18 = predictors.get_Lambda_and_R(1.8, metadata_dict['id_name'])

    max_res = 0.27*Mchirp*1e3
    sigma = max_res/3
    return Mchirp*(5.846
            +1.75*Mchirp
            -0.555*R18
            +1.002*Mchirp**2
            -0.316*Mchirp*R18
            +0.026*R18**2)*1e3, sigma

def VSB_L_fpeak(metadata_dict):
    m1 = metadata_dict['id_mass_starA']
    m2 = metadata_dict['id_mass_starB']
    Mchirp = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
    L1, _ = predictors.get_Lambda_and_R(m1, metadata_dict['id_name'])
    L2, _ = predictors.get_Lambda_and_R(m2, metadata_dict['id_name'])
    temp = (m1+12.0*m2)*m1**4.0 * L1 + (m2+12.0*m1)*m2**4.0 * L2
    temp2 = (m1+m2)**5
    Lambda_Tilde = 16.0/13.0 * (temp/temp2)

    max_res = 0.302*1e3
    sigma = max_res/3

    return 1e3*(1.392
            -0.108*Mchirp
            +51.7*Lambda_Tilde**(-1/2))/Mchirp, sigma

def VSB_L_f20(metadata_dict):
    m1 = metadata_dict['id_mass_starA']
    m2 = metadata_dict['id_mass_starB']
    Mchirp = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
    L1, _ = predictors.get_Lambda_and_R(m1, metadata_dict['id_name'])
    L2, _ = predictors.get_Lambda_and_R(m2, metadata_dict['id_name'])
    temp = (m1+12.0*m2)*m1**4.0 * L1 + (m2+12.0*m1)*m2**4.0 * L2
    temp2 = (m1+m2)**5
    Lambda_Tilde = 16.0/13.0 * (temp/temp2)

    max_res = 0.362111/Mchirp*1e3
    sigma = max_res/3
    ### adjR^2 = 0.939
    return 1e3*(0.5577277
            -0.4064*Mchirp
            +48.6962*Lambda_Tilde**(-1/2))/Mchirp, sigma

def VSB_L_fspiral(metadata_dict):
    m1 = metadata_dict['id_mass_starA']
    m2 = metadata_dict['id_mass_starB']
    Mchirp = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
    L1, _ = predictors.get_Lambda_and_R(m1, metadata_dict['id_name'])
    L2, _ = predictors.get_Lambda_and_R(m2, metadata_dict['id_name'])
    temp = (m1+12.0*m2)*m1**4.0 * L1 + (m2+12.0*m1)*m2**4.0 * L2
    temp2 = (m1+m2)**5
    Lambda_Tilde = 16.0/13.0 * (temp/temp2)

    max_res = 0.4612915/Mchirp*1e3
    sigma = max_res/3
    ### adjR^2 = 0.942
    return 1e3*(1.1998558
            -0.4422*Mchirp
            +45.8192*Lambda_Tilde**(-1/2))/Mchirp, sigma

def VSB_fpeak(metadata_dict,method):
    if method.lower()[-1] == 'r':
        return VSB_R_fpeak(metadata_dict)
    elif method.lower()[-1] == 'l':
        return VSB_L_fpeak(metadata_dict)
    else:
        raise ValueError('Option not implemented yet.')

def VSB_f20(metadata_dict,method):
    if method.lower()[-1] == 'r':
        return VSB_R_f20(metadata_dict)
    elif method.lower()[-1] == 'l':
        return VSB_L_f20(metadata_dict)
    else:
        raise ValueError('Option not implemented yet.')

def VSB_fspiral(metadata_dict,method):
    if method.lower()[-1] == 'r':
        return VSB_R_fspiral(metadata_dict)
    elif method.lower()[-1] == 'l':
        return VSB_L_fspiral(metadata_dict)
    else:
        raise ValueError('Option not implemented yet.')
    
    
def get_fpeak(method, **kwargs):
    """
    :param data: Method Name
    :type: str
    :returns: fpeak, gaussian sigma
    :rtype: float
    """

    if method == "empiricalVretiBros":
        metadata_dict = kwargs["metadata_dict"]
        fpeak, sigma = empirical_relation(metadata_dict['id_name'])

    elif method == "argmax":
        handler = kwargs["handler"]
        fpeak = handler.ifos[0].frequency_array[
            np.argmax(
                handler.fft_amplitude[0][
                    handler.ifos[0].frequency_array<3500])]
        sigma = .2*fpeak/3

    elif "empiricalVSB" in method:
        metadata_dict = kwargs["metadata_dict"]
        fpeak, sigma = VSB_fpeak(metadata_dict,method)
        
    return fpeak, sigma


def get_priors(simulation_dict, *args, **kwargs):
    model_name = simulation_dict["model"]
    priors = bilby.core.prior.PriorDict(
        conversion_function=lambda s: convert_parameters(s, model_name, simulation_dict['number_of_freqs'])
    )
    
    # Try multiple locations for the .priors file
    prior_filename = f"{model_name.lower()}.priors"
    potential_paths = [
        os.path.join(global_vars.priors_path, prior_filename),
        os.path.join(global_vars.project_path, prior_filename),
        os.path.join(os.getcwd(), prior_filename)
    ]
    
    prior_file = None
    for path in potential_paths:
        if os.path.exists(path):
            prior_file = path
            break
            
    if prior_file:
        priors.from_file(prior_file)
    else:
        logger.warning(f"Could not find priors file {prior_filename} in search paths: {potential_paths}")


    if simulation_dict["from_file"]:
        return priors

    ### fpeak
    if simulation_dict["method"].lower() == "argmax":
        fpeak,sigma = get_fpeak("argmax",
                          handler = kwargs["handler"],)

        if simulation_dict["distribution"].lower() == "uniform":
            priors["f_peak"] = bilby.core.prior.Uniform(
                name="f_peak",
                minimum=fpeak - sigma*3,
                maximum=fpeak + sigma*3,
                latex_label=r"$f_{\mathrm{peak}}$",
            )
        elif simulation_dict["distribution"].lower() == "gaussian":
            priors["f_peak"] = bilby.core.prior.Gaussian(
                name="f_peak", mu=fpeak, sigma=sigma, latex_label=r"$f_{\mathrm{peak}}$",
            )
        elif simulation_dict["distribution"].lower() == "truncatedgaussian":
            priors["f_peak"] = bilby.core.prior.TruncatedGaussian(
                name="f_peak", mu=fpeak, sigma=sigma,
                minimum = fpeak-3*sigma, maximum = fpeak + 3*sigma,
                latex_label=r"$f_{\mathrm{peak}}$",
            )

    elif "empiricalvsb" in simulation_dict["method"].lower():
        fpeak,sigma = get_fpeak(simulation_dict["method"],
                              metadata_dict = kwargs["metadata_dict"],)        
        if simulation_dict["distribution"].lower() == "uniform":
            priors["f_peak"] = bilby.core.prior.Uniform(
                name="f_peak",
                minimum=fpeak - sigma*3,
                maximum=fpeak + sigma*3,
                latex_label=r"$f_{\mathrm{peak}}$",
            )
        elif simulation_dict["distribution"].lower() == "gaussian":
            priors["f_peak"] = bilby.core.prior.Gaussian(
                name="f_peak", mu=fpeak, sigma=sigma, latex_label=r"$f_{\mathrm{peak}}$",
            )
        elif simulation_dict["distribution"].lower() == "truncatedgaussian":
            priors["f_peak"] = bilby.core.prior.TruncatedGaussian(
                name="f_peak", mu=fpeak, sigma=sigma,
                minimum = fpeak-3*sigma, maximum = fpeak + 3*sigma,
                latex_label=r"$f_{\mathrm{peak}}$",
            )
        
    elif simulation_dict["method"].lower() == "empiricalvretibros":
        try:
            fpeak, sigma = get_fpeak("empiricalVretiBros",
                              metadata_dict = kwargs["metadata_dict"],)
        except:
            raise KeyError("Data not provided")
        if simulation_dict["distribution"].lower() == "uniform":
            priors["f_peak"] = bilby.core.prior.Uniform(
                name="f_peak",
                minimum=fpeak - sigma*3,
                maximum=fpeak + sigma*3,
                latex_label=r"$f_{\mathrm{peak}}$",
            )
        elif simulation_dict["distribution"].lower() == "gaussian":
            priors["f_peak"] = bilby.core.prior.Gaussian(
                name="f_peak", mu=fpeak, sigma=sigma, latex_label=r"$f_{\mathrm{peak}}$",
            )
        elif simulation_dict["distribution"].lower() == "truncatedgaussian":
            priors["f_peak"] = bilby.core.prior.TruncatedGaussian(
                name="f_peak", mu=fpeak, sigma=sigma,
                minimum = fpeak-3*sigma, maximum = fpeak + 3*sigma,
                latex_label=r"$f_{\mathrm{peak}}$",
            )

    #### Secondary frequencies
    if simulation_dict['classifier']:
        region = classifier(kwargs["metadata_dict"])

    nfreqs = simulation_dict["number_of_freqs"]
    if nfreqs == 2:
        try:
            if region == 1:
                f2, sigma = VSB_f20(kwargs["metadata_dict"],
                                    simulation_dict["method"])
                name = "f_20"
                latex_name = "$f_{2-0}$"
            elif region == 3:
                f2, sigma = VSB_fspiral(kwargs["metadata_dict"],
                                    simulation_dict["method"])
                name = "f_spiral"
                latex_name = r"$f_{\mathrm{spiral}}$"
                
            if isinstance(priors["f_peak"],bilby.core.prior.Uniform):
                priors["f_2"] = bilby.core.prior.Uniform(
                    name=name,
                    minimum=2*f2-1.1*priors["f_peak"].minimum,
                    maximum=1.1*priors["f_peak"].minimum,
                    latex_label=latex_name
                )
            
            elif isinstance(priors["f_peak"],bilby.core.prior.Gaussian):
                priors["f_2"] = bilby.core.prior.Gaussian(
                    name="f_2", mu=f2, sigma=sigma*1.1, latex_label="$f_{2}$",
                )
            elif simulation_dict["distribution"].lower() == "truncatedgaussian":
                priors["f_2"] = bilby.core.prior.TruncatedGaussian(
                    name="f_2", mu=f2, sigma=sigma,
                    minimum = f2-3*sigma, maximum = f2 + 3*sigma,
                    latex_label="$f_{2}$",
                )


        except:
            fpeak,sigma = get_fpeak(simulation_dict["method"],
                                    metadata_dict = kwargs["metadata_dict"],)

            priors["f_2"] = bilby.core.prior.Uniform(
                name="f_sec.",
                minimum=1000,
                maximum=fpeak-0.5*sigma,
                latex_label=priors["f_peak"].latex_label.replace('peak','sec.')
            )

        #gw_model = source_model.model_picker(model_name, nfreqs)
        topop = list()
        for key, item in priors.items():
            if "1" in key or "high" in key:
                topop.append(key)
        for i in topop:
            priors.pop(i)
                    
        
    else:
        f_spiral, sigma_spiral = VSB_fspiral(kwargs["metadata_dict"],
                                    simulation_dict["method"])
        name_spiral = priors['f_1'].name #"f_spiral"
        latex_name_spiral = priors['f_1'].latex_label #"$f_{spiral}$"
        
        f_20, sigma_20 = VSB_f20(kwargs["metadata_dict"],
                                    simulation_dict["method"])
        name_20 = priors['f_2'].name#"f_20"
        latex_name_20 = priors['f_2'].latex_label #"$f_{2-0}$"

        if isinstance(priors["f_peak"],bilby.core.prior.Uniform):
            priors["f_1"] = bilby.core.prior.Uniform(
                name=name_spiral,
                minimum=f_spiral-sigma_spiral*3,
                maximum=f_spiral+sigma_spiral*3,
                latex_label=latex_name_spiral
            )
            priors["f_2"] = bilby.core.prior.Uniform(
                name=name_20,
                minimum=f_20-sigma_20*3,
                maximum=f_20+sigma_20*3,
                latex_label=latex_name_20
            )
            
            
        elif isinstance(priors["f_peak"],bilby.core.prior.Gaussian):
            priors["f_1"] = bilby.core.prior.Gaussian(
                name=name_spiral, mu=f_spiral, sigma=sigma_spiral,
                latex_label=latex_name_spiral,
            )
            priors["f_2"] = bilby.core.prior.Gaussian(
                name=name_20, mu=f_20, sigma=sigma, latex_label=latex_name_20,
            )
        elif simulation_dict["distribution"].lower() == "truncatedgaussian":
            priors["f_1"] = bilby.core.prior.TruncatedGaussian(
                name="f_1", mu=f_spiral, sigma=sigma_spiral,
                minimum = f_spiral-3*sigma_spiral, maximum = f_spiral + 3*sigma_spiral,
                latex_label=latex_name_spiral,
            )
            priors["f_2"] = bilby.core.prior.TruncatedGaussian(
                name="f_2", mu=f_20, sigma=sigma_20,
                minimum = f_20-3*sigma_20, maximum = f_20 + 3*sigma_20,
                latex_label=latex_name_20,
            )

        if nfreqs == 3:
            topop = list()
            for key, item in priors.items():
                if "high" in key:
                    topop.append(key)
            for i in topop:
                priors.pop(i)
        else:
            try:
                priors["f_high"].minimum = priors["f_peak"].mu + 3*priors["f_peak"].sigma
            except:
                priors["f_high"].minimum = priors["f_peak"].maximum

    names = priors.keys()
    for name in names:
        if "01" in name:
            priors[name].name=priors[name].name.replace('02',"peak-20")
            label = priors[name].latex_label.replace('02', r'{\mathrm{peak}-20}')
            priors[name].latex_label=label
        elif "02" in name:
            priors[name].name=priors[name].name.replace('01',"peak-spiral")
            label = priors[name].latex_label.replace('01', r'{\mathrm{peak-spiral}}')
            priors[name].latex_label=label
        elif "0" in name and name != 't0':
            priors[name].name=priors[name].name.replace('0',"peak")
            label = priors[name].latex_label.replace('0', r'{\mathrm{peak}}')
            priors[name].latex_label=label
        elif "2" in name:
            priors[name].name=priors[name].name.replace('1',"spiral")
            label = priors[name].latex_label.replace('1', r'{\mathrm{spiral}}')
            priors[name].latex_label=label
        elif "1" in name:
            priors[name].name=priors[name].name.replace('2',"2-0")
            label = priors[name].latex_label.replace('2','{2-0}')
            priors[name].latex_label=label

    return priors
