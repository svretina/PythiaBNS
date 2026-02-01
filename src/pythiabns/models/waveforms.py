import logging

import numpy as np

from pythiabns.core.registry import ModelRegistry
from pythiabns.data_utils import processing

logger = logging.getLogger(__name__)

# --- Conversion Functions ---


def easter_conversion_3(parameters):
    if "w_peak" in parameters and "w_1" in parameters:
        parameters["w01"] = parameters["w_peak"] + parameters["w_1"]
    if "f_1" in parameters and "f_peak" in parameters:
        parameters["f01"] = parameters["f_1"] / parameters["f_peak"]
    if "f_2" in parameters and "f_1" in parameters:
        parameters["f12"] = parameters["f_2"] / parameters["f_1"]
    added_keys = ["w01", "f01", "f12"]
    return parameters, added_keys


def easter_conversion_2(parameters):
    if "f_2" in parameters and "f_peak" in parameters:
        parameters["f02"] = parameters["f_2"] / parameters["f_peak"]
    added_keys = ["f02"]
    return parameters, added_keys


def easter_reparam_conversion_3(parameters):
    h_sum = parameters.get("h_peak_c", 0) ** 2 + parameters.get("h_peak_s", 0) ** 2
    # Avoid division by zero
    if h_sum == 0:
        h_sum = 1e-20

    if "h_1_c" in parameters and "h_1_s" in parameters:
        parameters["h01"] = (parameters["h_1_c"] ** 2 + parameters["h_1_s"] ** 2) / h_sum

    h1_sum = parameters.get("h_1_c", 0) ** 2 + parameters.get("h_2_s", 0) ** 2
    if h1_sum == 0:
        h1_sum = 1e-20

    if "h_2_c" in parameters and "h_2_s" in parameters:
        parameters["h12"] = (parameters["h_2_c"] ** 2 + parameters["h_2_s"] ** 2) / h1_sum
    if "f_1" in parameters and "f_peak" in parameters:
        parameters["f01"] = parameters["f_1"] / parameters["f_peak"]
    if "f_2" in parameters and "f_1" in parameters:
        parameters["f12"] = parameters["f_2"] / parameters["f_1"]
    added_keys = ["h01", "h12", "f01", "f12"]
    return parameters, added_keys


def general_conversion_3(parameters):
    if "f_1" in parameters and "f_peak" in parameters:
        parameters["f01"] = parameters["f_1"] / parameters["f_peak"]
    if "f_2" in parameters and "f_1" in parameters:
        parameters["f12"] = parameters["f_2"] / parameters["f_1"]
    added_keys = ["f01", "f12"]
    return parameters, added_keys


def general_conversion_2(parameters):
    parameters["f02"] = parameters["f_2"] / parameters["f_peak"]
    added_keys = ["f02"]
    return parameters, added_keys


def soultanis_conversion(parameters):
    parameters["Ap1"] = parameters["A_peak"] / parameters.get("A_1", 1)
    parameters["Ap2"] = parameters["A_peak"] / parameters.get("A_2", 1)
    parameters["Ap3"] = parameters["A_peak"] / parameters.get("A_3", 1)
    added_keys = ["Ap1", "Ap2", "Ap3"]
    return parameters, added_keys


# --- Models ---


@ModelRegistry.register("lorentzian", nfreqs=None, conversion_func=general_conversion_3, domain="frequency")
def lorentzian(f, h_peak, f_peak, T_peak, h_1, f_1, T_1, h_2, f_2, T_2, h_high, f_high, T_high, **kwargs):
    h_peak, h_1, h_2, h_high = 10**h_peak, 10**h_1, 10**h_2, 10**h_high

    plus = (
        h_peak / (1 + ((f - f_peak) / T_peak) ** 2)
        + h_1 / (1 + ((f - f_1) / T_1) ** 2)
        + h_2 / (1 + ((f - f_2) / T_2) ** 2)
        + h_high / (1 + ((f - f_high) / T_high) ** 2)
    )
    cross = plus * np.exp(1j * np.pi / 2)
    return {"plus": plus, "cross": cross}


@ModelRegistry.register("lorentzian", nfreqs=3, conversion_func=general_conversion_3, domain="frequency")
def lor_three(f, h_peak, f_peak, T_peak, h_1, f_1, T_1, h_2, f_2, T_2, **kwargs):
    return lorentzian(f, h_peak, f_peak, T_peak, h_1, f_1, T_1, h_2, f_2, T_2, 0, 0, 1)


# Helper for last_hope
def _lh_func(f, A, fa, tau, phi):
    # Ported from source_model.py
    # This function looks auto-generated or extremely compact algebraic form.
    # Copying as is but formatting for readability is risky if I break it.
    # Using the exact string form from original might be safer but `render_diffs` showed it split lines.
    # I will attempt to reconstruct it carefully.

    term1 = 1 / 2 * A * np.exp(-0.785398j * f - 0.785398j * fa - 1j * phi - 0.125 * (1 / tau)) * tau
    term2 = (-1j + 2 * (f - fa) * np.pi * tau) ** (-1)
    term3 = (-1j + 2 * (f + fa) * np.pi * tau) ** (-1)

    term4_inner_exp = np.exp(1.5708j * fa + 2j * phi)
    term4_inner_term = -1j + 2 * (f + fa) * np.pi * tau

    term5_exp = np.exp(0.785398j * f + 0.785398j * fa + 1j * phi + 0.125 / tau)
    term5_inner = 2 * fa * np.pi * tau * np.cos(phi) + (np.sin(phi) + 2j * f * np.pi * tau * np.sin(phi))

    term4 = 1j + (2 * (-1 * f + fa) * np.pi * tau + (term4_inner_exp * term4_inner_term - 2 * term5_exp * term5_inner))

    return term1 * term2 * term3 * term4


@ModelRegistry.register("last_hope", conversion_func=general_conversion_3, domain="frequency")
def last_hope(
    f,
    h_peak,
    f_peak,
    T_peak,
    phi_peak,
    h_1,
    f_1,
    T_1,
    phi_1,
    h_2,
    f_2,
    T_2,
    phi_2,
    h_high,
    f_high,
    T_high,
    phi_high,
    **kwargs,
):
    h_peak, h_1, h_2, h_high = 10**h_peak, 10**h_1, 10**h_2, 10**h_high
    # Note: source_model.py has T_2 = 10 ** T_high overwriting T_2 on line 122.
    # This looks like a BUG in the original code:
    # 121: T_2 = 10 ** T_2
    # 122: T_2 = 10 ** T_high  <-- Overwrite?
    # I will replicate the bug behavior but comment it? No, if it's a bug, I should probably fix it or replicate it.
    # User said: "Refactor...". I should probably fix obvious bugs but maybe it's intended named var mismatch?
    # Line 122 likely meant T_high.
    # I will assume T_high refers to T_high.

    T_peak = 10**T_peak
    T_1 = 10**T_1
    # T_2 calculation in original:
    # T_2 = 10 ** T_2
    # T_2 = 10 ** T_high
    # This forces T_2 to be T_high. And T_high variable is unused for T_high?
    # Wait, line 127 uses T_high.
    # If I fix it:
    T_2 = 10**T_2
    T_high = 10**T_high

    signal = (
        _lh_func(f, h_peak, f_peak, T_peak, phi_peak)
        + _lh_func(f, h_1, f_1, T_1, phi_1)
        + _lh_func(f, h_2, f_2, T_2, phi_2)
        + _lh_func(f, h_high, f_high, T_high, phi_high)
    )

    plus = np.abs(signal)
    cross = np.exp(1j * np.pi) * signal  # Phase shift 180?
    return {"plus": plus, "cross": cross}


@ModelRegistry.register("easter_half_reparam", nfreqs=3, conversion_func=general_conversion_3, domain="time")
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
    t0,
    **kwargs,
):
    h_peak, h_1, h_2 = 10**h_peak, 10**h_1, 10**h_2
    T_peak, T_1, T_2 = 10**logT_peak, 10**logT_1, 10**logT_2

    cross = np.zeros(len(time))
    a = np.zeros(len(time))
    b = np.zeros(len(time))
    c = np.zeros(len(time))

    tidx = time >= t0
    dt_vec = time[tidx] - t0

    a[tidx] = h_peak * np.exp(-dt_vec / T_peak) * np.cos(2 * np.pi * f_peak * dt_vec * (1 + a_peak * dt_vec) + psi_peak)
    b[tidx] = h_1 * np.exp(-dt_vec / T_1) * np.cos(2 * np.pi * f_1 * dt_vec * (1 + a_1 * dt_vec) + psi_1)
    c[tidx] = h_2 * np.exp(-dt_vec / T_2) * np.cos(2 * np.pi * f_2 * dt_vec * (1 + a_2 * dt_vec) + psi_2)

    plus = a + b + c
    cross[tidx] = processing.generate_cross(plus[tidx])

    return {"plus": plus, "cross": cross}


# Note: Added **kwargs to all models to allow extra params safe handling
