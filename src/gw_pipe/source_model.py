import bilby
import numpy as np
import pandas as pd
import logging
from gw_pipe import utils
from gw_pipe import global_vars as glb
from gw_pipe.registry import register_model, ModelRegistry

# --- Parameter Conversion Functions ---

def easter_conversion_3(parameters):
    parameters["w01"] = parameters["w_peak"] + parameters["w_1"]
    parameters["f01"] = parameters["f_1"] / parameters["f_peak"]
    parameters["f12"] = parameters["f_2"] / parameters['f_1']
    return parameters

def easter_conversion_2(parameters):
    parameters["f02"] = parameters["f_2"] / parameters["f_peak"]
    return parameters

def easter_reparam_conversion_3(parameters):
    parameters["h01"] = (
        (parameters["h_1_c"] ** 2 + parameters["h_1_s"] ** 2)
        / (parameters["h_peak_c"] ** 2 + parameters["h_peak_s"] ** 2)
    )
    parameters["h12"] = (
        (parameters["h_2_c"] ** 2 + parameters["h_2_s"] ** 2)
        / (parameters["h_1_c"] ** 2 + parameters["h_2_s"] ** 2)
    )   
    parameters["f01"] = parameters["f_1"] / parameters["f_peak"]
    parameters["f12"] = parameters["f_2"] / parameters['f_1']
    return parameters

def general_conversion_3(parameters):
    parameters["f01"] = parameters["f_1"] / parameters["f_peak"]
    parameters["f12"] = parameters["f_2"] / parameters['f_1']
    return parameters

def general_conversion_2(parameters):
    parameters["f02"] = parameters["f_2"] / parameters["f_peak"]
    return parameters

def soultanis_conversion(parameters):
    parameters["Ap1"] = parameters["A_peak"] / parameters.get("A_1", 1)
    parameters["Ap2"] = parameters["A_peak"] / parameters.get("A_2", 1)
    parameters["Ap3"] = parameters["A_peak"] / parameters.get("A_3", 1)
    return parameters

# Import custom models to trigger registration
try:
    from gw_pipe import models
except ImportError:
    pass

logger = logging.getLogger(__name__)

@register_model("lorentzian", nfreqs=None, conversion_func=general_conversion_3)
def lorentzian(f,
               h_peak, f_peak, T_peak,
               h_1, f_1, T_1,
               h_2, f_2, T_2,
               h_high, f_high, T_high):

    h_peak = 10 ** h_peak
    h_1 = 10 ** h_1
    h_2 = 10 ** h_2
    h_high = 10 ** h_high

    #T_peak = 1/T_peak*1e-3
    #T_1 = 1/T_1*1e-3
    #T_2 = 1/T_2*1e-3
    #T_high = 1/T_high*1e-3

    plus = (
        h_peak / (1 + ((f - f_peak) / T_peak)**2) +
        h_1 / (1 + ((f - f_1) / T_1)**2) +
        h_2 / (1 + ((f - f_2) / T_2)**2) +
        h_high / (1 + ((f - f_high) / T_high)**2)
    )
    cross = plus * np.exp(1j * np.pi / 2)

    # Add noise to the signal (uncomment this section if needed)
    # noise_amplitude =
    # noise_level = noise_amplitude / np.sqrt(2)
    # noise = np.random.normal(0, noise_level, len(signal))
    # plus += noise
    # cross += noise

    return {"plus": plus, "cross": cross}

@register_model("lorentzian", nfreqs=3, conversion_func=general_conversion_3)
def lor_three(f, h_peak, f_peak, T_peak, h_1, f_1, T_1, h_2, f_2, T_2):
    return lorentzian(f, h_peak, f_peak, T_peak, h_1, f_1, T_1, h_2, f_2, T_2, 0, 0, 1)

func = lambda f,A,fa,tau,phi:1/2 * A * ( np.e )**( ( complex( 0,(-0.785398e0) ) * f + ( complex( \
0,(-0.785398e0) ) * fa + ( complex( 0,-1 ) * phi + (-0.125e0) * ( tau \
)**( -1 ) ) ) ) ) * tau * ( ( complex( 0,-1 ) + 2 * ( f + -1 * fa ) * \
np.pi * tau ) )**( -1 ) * ( ( complex( 0,-1 ) + 2 * ( f + fa ) * \
np.pi * tau ) )**( -1 ) * ( complex( 0,1 ) + ( 2 * ( -1 * f + fa ) * \
np.pi * tau + ( ( np.e )**( ( complex( 0,0.15708e1 ) * fa + complex( \
0,2 ) * phi ) ) * ( complex( 0,-1 ) + 2 * ( f + fa ) * np.pi * tau ) \
+ -2 * ( np.e )**( ( complex( 0,0.785398e0 ) * f + ( complex( \
0,0.785398e0 ) * fa + ( complex( 0,1 ) * phi + 0.125e0 * ( tau )**( \
-1 ) ) ) ) ) * ( 2 * fa * np.pi * tau * np.cos( phi ) + ( np.sin( phi \
) + complex( 0,2 ) * f * np.pi * tau * np.sin( phi ) ) ) ) ) )

@register_model("last_hope", conversion_func=general_conversion_3)
def last_hope(f,
              h_peak, f_peak, T_peak, phi_peak,
              h_1, f_1, T_1,phi_1,
              h_2, f_2, T_2,phi_2,
              h_high, f_high, T_high,phi_high):

    h_peak = 10 ** h_peak
    h_1 = 10 ** h_1
    h_2 = 10 ** h_2
    h_high = 10 ** h_high

    T_peak = 10 ** T_peak
    T_1 = 10 ** T_1
    T_2 = 10 ** T_2
    T_2 = 10 ** T_high

    signal = (func(f,h_peak,f_peak,T_peak,phi_peak)+
              func(f,h_1,f_1,T_1,phi_1)+
              func(f,h_2,f_2,T_2,phi_2)+
              func(f,h_high,f_high,T_high,phi_high))

    plus = np.abs(signal)
    cross = np.exp(1j*np.pi)*signal

    # Add noise to the signal (uncomment this section if needed)
    # noise_amplitude =
    # noise_level = noise_amplitude / np.sqrt(2)
    # noise = np.random.normal(0, noise_level, len(signal))
    # plus += noise
    # cross += noise

    return {"plus": plus, "cross": cross}


@register_model("easter", nfreqs=3, conversion_func=easter_conversion_3)
def easter_model(
    time,
    logH,
    w_peak,
    w_1,
    logT_peak,
    logT_1,
    logT_2,
    f_peak,
    f_1,
    f_2,
    a_peak,
    a_1,
    a_2,
    psi_peak,
    psi_1,
    psi_2,
    t0
):
    """Defines an analytic model to describe the time domain strain (Easter et al. 2020)

    :returns: h_plus, h_cross
    :rtype: dict
    """
    H = 10 ** logH
    T_peak = 10 ** logT_peak
    T_1 = 10 ** logT_1
    T_2 = 10 ** logT_2
    w_2 = 1 - w_peak - w_1
    #t0 = time[0]
    cross = np.zeros(len(time))
    a = np.zeros(len(time))
    b = np.zeros(len(time))
    c = np.zeros(len(time))

    tidx = time >= t0
    a[tidx] = (
        H
        * w_peak
        * np.exp(-(time[tidx] - t0) / T_peak)
        * np.cos(
            2 * np.pi * f_peak * (time[tidx] - t0) * (1 + a_peak * (time[tidx] - t0))
            + psi_peak
        )
    )
    b[tidx] = (
        H
        * w_1
        * np.exp(-(time[tidx] - t0) / T_1)
        * np.cos(
            2 * np.pi * f_1 * (time[tidx] - t0) * (1 + a_1 * (time[tidx] - t0)) + psi_1
        )
    )
    c[tidx] = (
        H
        * w_2
        * np.exp(-(time[tidx] - t0) / T_2)
        * np.cos(
            2 * np.pi * f_2 * (time[tidx] - t0) * (1 + a_2 * (time[tidx] - t0)) + psi_2
        )
    )
    plus = a + b + c

    cross[tidx] = utils.generate_cross(plus[tidx])

    return {"plus": plus, "cross": cross}

@register_model("easter_half_reparam", nfreqs=3, conversion_func=general_conversion_3)
def easter_half_reparam(
    time,
    h_peak,
    h_1,
    h_2,
    logT_peak,
    logT_1,
    logT_2,
    f_peak,
    f_1,
    f_2,
    a_peak,
    a_1,
    a_2,
    psi_peak,
    psi_1,
    psi_2,
    t0
):
    """Defines an analytic model to describe the time domain strain (Easter et al. 2020)

    :returns: h_plus, h_cross
    :rtype: dict
    """

    h_peak = 10 ** h_peak
    h_1 = 10 ** h_1
    h_2 = 10 ** h_2

    T_peak = 10 ** logT_peak
    T_1 = 10 ** logT_1
    T_2 = 10 ** logT_2
#    t0 = time[0]
    cross = np.zeros(len(time))
    a = np.zeros(len(time))
    b = np.zeros(len(time))
    c = np.zeros(len(time))

    tidx = time >= t0
    a[tidx] = (
        h_peak
        * np.exp(-(time[tidx] - t0) / T_peak)
        * np.cos(
            2 * np.pi * f_peak * (time[tidx] - t0) * (1 + a_peak * (time[tidx] - t0))
            + psi_peak
        )
    )
    b[tidx] = (
        h_1
        * np.exp(-(time[tidx] - t0) / T_1)
        * np.cos(
            2 * np.pi * f_1 * (time[tidx] - t0) * (1 + a_1 * (time[tidx] - t0)) + psi_1
        )
    )
    c[tidx] = (
        h_2
        * np.exp(-(time[tidx] - t0) / T_2)
        * np.cos(
            2 * np.pi * f_2 * (time[tidx] - t0) * (1 + a_2 * (time[tidx] - t0)) + psi_2
        )
    )
    plus = a + b + c

    cross[tidx] = utils.generate_cross(plus[tidx])

    return {"plus": plus, "cross": cross}

@register_model("easter_half_reparam", nfreqs=4, conversion_func=general_conversion_3)
def easter_half_reparam_plus(
    time,
    h_peak,
    h_1,
    h_2,
    h_high,
    logT_peak,
    logT_1,
    logT_2,
    logT_high,
    f_peak,
    f_1,
    f_2,
    f_high,
    a_peak,
    a_1,
    a_2,
    a_high,
    psi_peak,
    psi_1,
    psi_2,
    psi_high,
    t0
):
    """Defines an analytic model to describe the time domain strain (Easter et al. 2020)

    :returns: h_plus, h_cross
    :rtype: dict
    """

    h_peak = 10 ** h_peak
    h_1 = 10 ** h_1
    h_2 = 10 ** h_2
    h_high = 10 ** h_high

    T_peak = 10 ** logT_peak
    T_1 = 10 ** logT_1
    T_2 = 10 ** logT_2
    T_high = 10 ** logT_high

    cross = np.zeros(len(time))
    a = np.zeros(len(time))
    b = np.zeros(len(time))
    c = np.zeros(len(time))
    d = np.zeros(len(time))

    tidx = time >= t0
    a[tidx] = (
        h_peak
        * np.exp(-(time[tidx] - t0) / T_peak)
        * np.cos(
            2 * np.pi * f_peak * (time[tidx] - t0) * (1 + a_peak * (time[tidx] - t0))
            + psi_peak
        )
    )
    b[tidx] = (
        h_1
        * np.exp(-(time[tidx] - t0) / T_1)
        * np.cos(
            2 * np.pi * f_1 * (time[tidx] - t0) * (1 + a_1 * (time[tidx] - t0)) + psi_1
        )
    )
    c[tidx] = (
        h_2
        * np.exp(-(time[tidx] - t0) / T_2)
        * np.cos(
            2 * np.pi * f_2 * (time[tidx] - t0) * (1 + a_2 * (time[tidx] - t0)) + psi_2
        )
    )

    d[tidx] = (
        h_high
        * np.exp(-(time[tidx] - t0) / T_high)
        * np.cos(
            2 * np.pi * f_high * (time[tidx] - t0) * (1 + a_high * (time[tidx] - t0)) + psi_high
        )
    )

    plus = a + b + c + d

    cross[tidx] = utils.generate_cross(plus[tidx])

    return {"plus": plus, "cross": cross}


@register_model("easter_half_reparam", nfreqs=2, conversion_func=general_conversion_2)
def easter_half_reparam2(
    time,
    h_peak,
    h_2,
    logT_peak,
    logT_2,
    f_peak,
    f_2,
    a_peak,
    a_2,
    psi_peak,
    psi_2,
    t0
):
    """Defines an analytic model to describe the time domain strain (Easter et al. 2020)

    :returns: h_plus, h_cross
    :rtype: dict
    """
    h_peak = 10 ** h_peak
    h_2 = 10 ** h_2
    T_peak = 10 ** logT_peak
    T_2 = 10 ** logT_2
#    t0 = time[0]
    cross = np.zeros(len(time))
    a = np.zeros(len(time))
    b = np.zeros(len(time))
    c = np.zeros(len(time))

    tidx = time >= t0
    a[tidx] = (
        h_peak
        * np.exp(-(time[tidx] - t0) / T_peak)
        * np.cos(
            2 * np.pi * f_peak * (time[tidx] - t0) * (1 + a_peak * (time[tidx] - t0))
            + psi_peak
        )
    )

    c[tidx] = (
        h_2
        * np.exp(-(time[tidx] - t0) / T_2)
        * np.cos(
            2 * np.pi * f_2 * (time[tidx] - t0) * (1 + a_2 * (time[tidx] - t0)) + psi_2
        )
    )
    plus = a + b + c

    cross[tidx] = utils.generate_cross(plus[tidx])

    return {"plus": plus, "cross": cross}


@register_model("soultanis", conversion_func=soultanis_conversion)
def soultanis_model(
    time,
    t_star,
    A_peak,
    t_peak,
    f_peak0,
    zeta_drift,
    phi_peak,
    A_1,
    t_1,
    f_1,
    phi_1,
    A_2,
    t_2,
    f_2,
    phi_2,
    A_3,
    t_3,
    f_3,
    phi_3,
):
    """Defines an analytic model to describe the time domain strain (Soultanis et al. 2021)

    :returns: h_plus, h_cross
    :rtype: dict
    """
    # exR = 1000
    peak = np.zeros(len(time))
    a = np.zeros(len(time))
    b = np.zeros(len(time))
    c = np.zeros(len(time))

    # f_peak0 = f_peak0 * 1e3
    t_peak = t_peak * 1e-3
    t_1 = t_1 * 1e-3
    t_2 = t_2 * 1e-3
    t_3 = t_3 * 1e-3
    t_star = t_star * 1e-3
    # zeta_drift = zeta_drift * 1e6
    # A_peak = A_peak*exR / (40 * 8.35898516842975 * 1e20)
    # A_1 = A_1*exR /  (40 *8.35898516842975 * 1e20)
    # A_2 = A_2*exR /  (40 *8.35898516842975 * 1e20)
    # A_3 = A_3*exR / (40 *8.35898516842975 * 1e20)

    A_peak = 10 ** A_peak
    A_1 = 10 ** A_1
    A_2 = 10 ** A_2
    A_3 = 10 ** A_3
    # A_1 = A_1 * 1e-21
    # A_2 = A_2 * 1e-21
    # A_3 = A_3 * 1e-21
    # A_peak = A_peak * 1e-21

    time = time - time[0]
    t0 = time[0]
    tidx = time >= t0

    mask1 = time[tidx] <= t_star
    mask2 = time[tidx] > t_star

    f_peak_func = f_peak0 + zeta_drift * time[mask1]
    f_peak_star = f_peak_func[-1]

    phi1 = pd.DataFrame(
        2 * np.pi * (f_peak0 + zeta_drift / 2 * time[mask1]) * time[mask1] + phi_peak
    )
    phi2 = pd.DataFrame(
        2 * np.pi * f_peak_star * (time[mask2] - t_star) + phi1.iat[-1, 0]
    )

    phi_peak_func = pd.concat([phi1, phi2]).to_numpy().transpose()[0]

    peak[tidx] = A_peak * np.exp(-time[tidx] / t_peak) * np.sin(phi_peak_func[tidx])

    a[tidx] = (
        A_1 * np.exp(-time[tidx] / t_1) * np.sin(2 * np.pi * f_1 * time[tidx] + phi_1)
    )
    b[tidx] = (
        A_2 * np.exp(-time[tidx] / t_2) * np.sin(2 * np.pi * f_2 * time[tidx] + phi_2)
    )
    c[tidx] = (
        A_3 * np.exp(-time[tidx] / t_3) * np.sin(2 * np.pi * f_3 * time[tidx] + phi_3)
    )

    plus = peak + a + b + c

    cross = utils.generate_cross(plus)

    return {"plus": plus, "cross": cross}

@register_model("easter", nfreqs=2, conversion_func=easter_conversion_2)
def easter_model2(
    time, logH, w_peak, logT_peak, logT_2, f_peak, f_2, a_peak, a_2, psi_peak, psi_2,t0
):
    """Defines an analytic model to describe the time domain strain (Easter et al. 2020)

    :returns: h_plus, h_cross
    :rtype: dict
    """
    H = 10 ** logH
    T_peak = 10 ** logT_peak
    T_2 = 10 ** logT_2
    w_2 = 1 - w_peak
#    t0 = time[0]
    cross = np.zeros(len(time))
    a = np.zeros(len(time))
    c = np.zeros(len(time))

    tidx = time >= t0
    a[tidx] = (
        H
        * w_peak
        * np.exp(-(time[tidx] - t0) / T_peak)
        * np.cos(
            2 * np.pi * f_peak * (time[tidx] - t0) * (1 + a_peak * (time[tidx] - t0))
            + psi_peak
        )
    )
    c[tidx] = (
        H
        * w_2
        * np.exp(-(time[tidx] - t0) / T_2)
        * np.cos(
            2 * np.pi * f_2 * (time[tidx] - t0) * (1 + a_2 * (time[tidx] - t0)) + psi_2
        )
    )
    plus = a + c

    cross[tidx] = utils.generate_cross(plus[tidx])

    return {"plus": plus, "cross": cross}

@register_model("easter_reparam", nfreqs=3, conversion_func=easter_reparam_conversion_3)
def easter_model_reparam(
    time,
    h_peak_s,
    h_peak_c,
    h_1_s,
    h_1_c,
    h_2_s,
    h_2_c,
    logT_peak,
    logT_1,
    logT_2,
    f_peak,
    f_1,
    f_2,
    a_peak,
    a_1,
    a_2,
    t0
):
    """Defines an analytic model to describe the time domain strain (Easter et al. 2020)

    :returns: h_plus, h_cross
    :rtype: dict
    """

    h_peak_s = 10 ** h_peak_s
    h_peak_c = 10 ** h_peak_c
    h_1_s = 10 ** h_1_s
    h_1_c = 10 ** h_1_c
    h_2_s = 10 ** h_2_s
    h_2_c = 10 ** h_2_c

    T_peak = 10 ** logT_peak
    T_1 = 10 ** logT_1
    T_2 = 10 ** logT_2
    #t0 = time[0]
    cross = np.zeros(len(time))
    a = np.zeros(len(time))
    b = np.zeros(len(time))
    c = np.zeros(len(time))

    tidx = time >= t0
    a[tidx] = (
        np.exp(-(time[tidx] - t0) / T_peak)
        * (h_peak_c
        * np.cos(
            2 * np.pi * f_peak * (time[tidx] - t0) * (1 + a_peak * (time[tidx] - t0))
        )
        + h_peak_s
        * np.sin(
            2 * np.pi * f_peak * (time[tidx] - t0) * (1 + a_peak * (time[tidx] - t0))
        )
        )
    )

    b[tidx] = (
        np.exp(-(time[tidx] - t0) / T_1)
        * (h_1_c
        * np.cos(
            2 * np.pi * f_1 * (time[tidx] - t0) * (1 + a_1 * (time[tidx] - t0))
        )
        + h_1_s
        * np.sin(
            2 * np.pi * f_1 * (time[tidx] - t0) * (1 + a_1 * (time[tidx] - t0))
        )
        )
    )

    c[tidx] = (
        np.exp(-(time[tidx] - t0) / T_2)
        * (h_2_c
        * np.cos(
            2 * np.pi * f_2 * (time[tidx] - t0) * (1 + a_2 * (time[tidx] - t0))
        )
        + h_2_s
        * np.sin(
            2 * np.pi * f_2 * (time[tidx] - t0) * (1 + a_2 * (time[tidx] - t0))
        )
        )
    )

    plus = a + b + c

    cross[tidx] = utils.generate_cross(plus[tidx])

    return {"plus": plus, "cross": cross}

@register_model("easter_reparam", nfreqs=2, conversion_func=general_conversion_2)
def easter_model_reparam2(
    time,
    h_peak_s,
    h_peak_c,
    h_2_s,
    h_2_c,
    logT_peak,
    logT_2,
    f_peak,
    f_2,
    a_peak,
    a_2,
    t0
):
    """Defines an analytic model to describe the time domain strain (Easter et al. 2020)

    :returns: h_plus, h_cross
    :rtype: dict
    """

    h_peak_s = 10 ** h_peak_s
    h_peak_c = 10 ** h_peak_c
    h_2_s = 10 ** h_2_s
    h_2_c = 10 ** h_2_c

    T_peak = 10 ** logT_peak
    T_2 = 10 ** logT_2
    #t0 = time[0]
    cross = np.zeros(len(time))
    a = np.zeros(len(time))
    c = np.zeros(len(time))

    tidx = time >= t0
    a[tidx] = (
        np.exp(-(time[tidx] - t0) / T_peak)
        * (h_peak_c
        * np.cos(
            2 * np.pi * f_peak * (time[tidx] - t0) * (1 + a_peak * (time[tidx] - t0))
        )
        + h_peak_s
        * np.sin(
            2 * np.pi * f_peak * (time[tidx] - t0) * (1 + a_peak * (time[tidx] - t0))
        )
        )
    )

    c[tidx] = (
        np.exp(-(time[tidx] - t0) / T_2)
        * (h_2_c
        * np.cos(
            2 * np.pi * f_2 * (time[tidx] - t0) * (1 + a_2 * (time[tidx] - t0))
        )
        + h_2_s
        * np.sin(
            2 * np.pi * f_2 * (time[tidx] - t0) * (1 + a_2 * (time[tidx] - t0))
        )
        )
    )

    plus = a + c

    cross[tidx] = utils.generate_cross(plus[tidx])

    return {"plus": plus, "cross": cross}

def model_picker(model_name, nfreqs):
    model = ModelRegistry.get_model(model_name, nfreqs)
    if model is None:
        raise ValueError(f"Model '{model_name}' with nfreqs={nfreqs} not found in registry.")
    return model
