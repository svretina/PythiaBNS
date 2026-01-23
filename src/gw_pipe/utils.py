#!/usr/bin/env python3

# Copyright (C) 2022 Stamatis Vretinaris
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
import re
import numpy as np
import scipy.signal
import scipy.interpolate
from scipy import fft
import bilby
from gw_pipe import global_vars as glb

def get_bilby_noise_path(filename):
    """Get the absolute path to a bilby noise curve file."""
    bilby_dir = os.path.dirname(bilby.__file__)
    return os.path.abspath(os.path.join(bilby_dir, "gw", "detector", "noise_curves", filename))

def padding(time, h, samples):
    if len(time) >= samples:
        return time, h
    a = len(h)
    b = samples
    z = b - a
    dt = time[1] - time[0]
    zeroz = np.zeros(z)
    h = np.concatenate((h, zeroz), axis=0)
    x2add = np.arange(time[-1] + dt, time[-1] + (z + 1) * dt, dt)
    x2add = x2add[0 : len(zeroz)]
    time = np.concatenate((time, x2add), axis=0)
    return time, h


def windowing(signal, a=0.5):
    signal = scipy.signal.tukey(len(signal), a, sym=True) * signal
    return signal


def fourier(time, h):
    h = windowing(h)
    dt = time[2] - time[1]  # 1./16384
    # sampling_frequency = int(len(h) / time[-1])  # 16384 # 1.0/dt
    time, h = padding(time, h, 16384)
    N = len(h)
    yf = fft.fft(h)
    freqs = np.fft.fftfreq(N, d=dt)[: N // 2]
    yfft = (2.0 / N) * np.abs(yf[: N // 2])
    amplitude = yfft * np.sqrt(freqs)
    phase = np.angle(yf)[: N // 2]
    return freqs, amplitude, np.unwrap(phase)


def interpolate(x_old, y_old, x_new):
    interpolator = scipy.interpolate.PchipInterpolator(x_old, y_old)
    new_y = interpolator.__call__(x_new)
    return new_y


def get_filenames():
    return os.listdir(glb.strain_path)


# Stamatis' code to generate h_c given h_p
def generate_cross(signal):
    return shift_by_phase(signal, 90)  # np.pi/2)


def shift_by_phase(signal, phase):
    spec = np.fft.rfft(signal)
    spec *= np.exp(1j * np.deg2rad(phase))
    shifted_signal = np.fft.irfft(spec, n=len(signal))
    # shifted_signal =  np.exp(1j*phase)*signal
    # _, shifted_signal = normalize_amplitudes(signal,
    # shifted_signal)
    return shifted_signal


def my_arange(start, step, end):
    N = int((end - start) / step + 1)
    tmp = np.linspace(start, end, N)
    return np.round(tmp, 3)
