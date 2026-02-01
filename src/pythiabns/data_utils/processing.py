import numpy as np
import scipy.interpolate
import scipy.signal
from scipy import fft


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
    dt = time[2] - time[1]
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


def shift_by_phase(signal, phase):
    spec = np.fft.rfft(signal)
    spec *= np.exp(1j * np.deg2rad(phase))
    shifted_signal = np.fft.irfft(spec, n=len(signal))
    return shifted_signal


def generate_cross(signal):
    """Generate cross polarization from plus by shifting phase by 90 degrees."""
    return shift_by_phase(signal, 90)


def my_arange(start, step, end):
    N = int((end - start) / step + 1)
    tmp = np.linspace(start, end, N)
    return np.round(tmp, 3)
