#!/usr/bin/env python3

# Copyright (C) 2022 Stamatis Vretinaris, Christos Mermigkas, Georgios Vretinaris, Nikolaos Stergioulas
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

import matplotlib.pyplot as plt
import h5py
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, argrelmax
from astropy.constants import c, G, M_sun
from astropy import units as unit
import scipy
from gw_pipe import global_vars as glb
from gw_pipe import utils as u
#from bilby.gw.utils import noise_weighted_inner_product

def load_metadata(datapath):
    """Loads the metadata.txt file into a dictionary.

    :returns: metadata dictionary
    :rtype: dict
    """
    metadata_file = f"{datapath}/metadata.txt"

    metadata_dict = {}

    with open(metadata_file, "r") as mdf:
        for line in mdf:
            words = line.split()
            if len(words)==0:
                continue
            elif 'Evolution' in words:
                break
            if "id_" in words[0]:
                try:
                    value = float(words[-1])
                except:
                    value = words[-1]
                    
                metadata_dict[words[0]] = value
    return metadata_dict


def set_metadata_values(metadata_dict):
    """Returns the value of the keywords from metadata.

    :returns: m1, m2
    :rtype: float
    """
    m1 = metadata_dict["id_mass_starA"]
    m2 = metadata_dict["id_mass_starB"]

    return m1, m2


class NumericalData:
    def __init__(self, filename, sampling_frequency=None):


        if filename.split('/')[0] != 'Soultanis':
            # get datapath
            self.datapath = f"{glb.strain_path}/{filename}"
            self.sampling_frequency = sampling_frequency
            # construct useful metadata dictionary
            self.metadata_dict = load_metadata(self.datapath)

            # get metadata values
            self.m1, self.m2 = set_metadata_values(self.metadata_dict)

            # compute total mass of binary
            self.Mtot = (self.m1 + self.m2) * M_sun

            # obtain time, r*hplus and r*hcross from waveform data
            (
                self.rh_overmtot_p,
                self.rh_overmtot_c,
                self.time,
                self.extraction_radius,
            ) = self.load_NR_strains()

            # convert time to SI units
            self.time_to_SI()

            # set to 1 Mpc
            self.hp, self.hc = self.set_to_1Mpc()
        else:
            mass = float(filename.split('/')[-1])
            self.datapath = [i for i in os.listdir(f"{glb.strain_path}/{filename.split('/')[0]}") if filename.split('/')[-1] in i][0]
            self.datapath = f"{glb.strain_path}/{filename.split('/')[0]}/{self.datapath}"
            self.sampling_frequency = sampling_frequency
            self.metadata_dict = {'id_mass_starA': mass,'id_mass_starB': mass,'id_eos':'MPA1'}
            self.m1, self.m2 = mass, mass
            self.Mtot = (self.m1 + self.m2) * M_sun
            data = np.loadtxt(self.datapath)
            self.time = data.T[0]/1000 #ms to s
            self.hp = data.T[1]/8.35898e+20*40 #@ 1Mpc
            self.hc = data.T[2]/8.35898e+20*40 #@ 1Mpc

        # Resampling is mandatory every time!
        #self.resample()

    def check_even_sampled(self):
        """Checks if dt is constant.

        :returns: True or False
        :rtype: Boolean
        """

        # absolute tolerance
        abs_tol = 1e-10

        # largest difference between consecutive time values
        largest_difference = np.ptp(np.diff(self.time))

        if largest_difference > abs_tol:
            return False
        else:
            return True

    def resample(self):
        """Resamples h_plus and h_cross to be even sampled.

        :returns: resampled h_plus, h_cross, time
        :rtype: numpy arrays
        """

        # define median as new fixed time step
        if self.sampling_frequency is None:
            df = 8192
        else:
            df = self.sampling_frequency
        dt_fixed = 1 / df
        # new time array with fixed time step
        # np.arange is bad with fractional step!!!!!
        new_time = np.arange(self.time[0], self.time[-1], dt_fixed)

        # construct interpolated hplus and hcross for
        # fixed time step
        self.hp = u.interpolate(self.time, self.hp, new_time)
        self.hc = u.interpolate(self.time, self.hc, new_time)

        self.time = new_time

    def load_NR_strains(self):
        """Reads the hdf5 database and returns the time,
        hplus and hcross as numpy arrays.

        :returns: time, h_plus, h_cross
        :rtype: numpy arrays
        """

        h5py_data_file = h5py.File(f"{self.datapath}/data.h5")

        # get the list of names of all l=m=2 mode extractions
        l2m2_extraction_names = [
            x for x in list(h5py_data_file["/rh_22"]) if "l2_m2" in x
        ]

        #  select the extraction at the largest radius
        # (last one in list, unless last one is Inf)
        selection = l2m2_extraction_names[-1]
        if "Inf" in selection:
            selection = l2m2_extraction_names[-2]

        h5_data_selection_key = h5py_data_file[f"/rh_22/{selection}"]

        # read the data for the selected extraction radius
        # as a Pandas 2D data frame
        selected_data = pd.DataFrame(h5_data_selection_key)

        # extract time series for hplus and hcross polarizations
        time = selected_data.iloc[:, 0].values
        rhplus = selected_data.iloc[:, 1].values
        rhcross = selected_data.iloc[:, 2].values

        # get the extraction radius value in CORE database units
        extraction_radius = float(selection.split(".")[0].split("r")[1])

        return rhplus, rhcross, time, extraction_radius

    def tukey_window(self, a):
        """Applies a Tukey window to the time domain strain.

        :returns: h_plus, h_cross
        :rtype: numpy arrays
        """
        self.hp = self.hp * scipy.signal.tukey(len(self.hp), a)
        self.hc = self.hc * scipy.signal.tukey(len(self.hc), a)

    def hamming_window(self, a):
        """Applies a Tukey window to the time domain strain.

        :returns: h_plus, h_cross
        :rtype: numpy arrays
        """
        self.hp = self.hp * scipy.signal.hamming(len(self.hp))
        self.hc = self.hc * scipy.signal.hamming(len(self.hc))

    def get_post_merger(self, inplace=True):
        """Crops the NR strain to take the post merger part.
        We define the post merger as the first zero before
        the strain's maximum.

        :returns: time,  h_plus, h_cross
        :rtype: numpy arrays
        """

        self.resample()
        self.hc = u.generate_cross(self.hp)
        a = np.sqrt(self.hc * self.hc + self.hp * self.hp)

        #a = np.sqrt(self.hc * self.hc + self.hp * self.hp)
        # finding merger time
        indxmax = np.argmax(a)
        t = self.time[indxmax+1:]
        hp = self.hp[indxmax+1:]
        hc = self.hc[indxmax+1:]
        if len(hc) %2 !=0:
            hc = hc[:-1]
            hp = hp[:-1]
            t = t[:-1]



        ##### Get first zero before max
        #zeros = scipy.signal.find_peaks(-abs(self.hp))[0]
        #maxhp = np.argmax(abs(self.hp))
        #idx = np.where(zeros < maxhp)[0][-1]
        #idx = zeros[idx]
        #t = self.time[idx:]
        #hp = self.hp[idx:]
        #hc = self.hc[idx:]

        #self.time = self.time[indxmax:]
        #self.hp = self.hp[indxmax:]
        #self.hc = self.hc[indxmax:]

        #self.tukey_window(a=0.03)
        #t = self.time
        #hp = self.hp
        #hc = self.hc

        #t = self.time[indxmax:]
        #hp = self.hp[indxmax:]
        #hc = self.hc[indxmax:]



        #import bilby
        #duration = np.round(t[-1]-t[0],15)
        #ifos = bilby.gw.detector.InterferometerList(['H1','L1'])
        #for ifo in ifos:
            #ifo.minimum_frequency = 10
            #ifo.maximum_frequency = 5000
            #ifos.set_strain_data_from_zero_noise(
                #sampling_frequency=(2 * 8192),
                #duration=duration,
                #start_time=0,
            #)
        #psd = ifos[0].power_spectral_density_array
        #hc = np.hstack([np.zeros(hc.size//2),hc,np.zeros(2*hc.size//2)])
        #hctilde = np.fft.rfft(hc)
        #freqs = np.fft.rfftfreq(len(hc),t[1]-t[0])
        #plt.figure(figsize=(8,6),dpi=100)
        #hc = u.generate_cross(hp)
        #hctilde2 = np.fft.rfft(hc)
        #hctilde = hctilde[(freqs>10) &  (freqs<5000)]
        #hctilde2 = hctilde2[(freqs>10) &  (freqs<5000)]
        #try:
            #psd = psd[(freqs>10) &  (freqs<5000)]
        #except:
            #psd = psd[(freqs[:-1]>10) &  (freqs[:-1]<5000)]

        #freqs = freqs[(freqs>10) & (freqs<5000)]
        #plt.plot(freqs,np.abs(hctilde)*np.sqrt(freqs))
        #plt.plot(freqs,np.abs(hctilde2)*np.sqrt(freqs))
        #plt.yscale('log')
        #plt.show()
        #exit()


        #numerator =  noise_weighted_inner_product(hctilde, hctilde2, psd, duration)
        #denominator = np.sqrt(
            #noise_weighted_inner_product(hctilde, hctilde, psd, duration)*
            #noise_weighted_inner_product(hctilde2, hctilde2, psd, duration)
            #)
        #print(f"FF={np.real(numerator/denominator)}")
        #plt.title(f"FF={np.real(numerator/denominator)}")
        #plt.show()


        if inplace:
            self.time = t
            self.hp = hp
            self.hc = hc
        else:
            return t, hp, hc


    def rescale_to_SNR(self, current_snr, current_dist=1, prefered_snr=50):
        """Rescales the signal to a preferred SNR.
        Default preferred SNR is 50.
        The current SNR is calculated to distance = 1 Mpc

        :returns: h_plus, h_cross
        :rtype: numpy arrays
        """

        # dist in Mpc!
        dist = current_snr / prefered_snr * current_dist
        # print(dist)
        # exit()

        self.hp = self.hp / dist
        self.hc = self.hc / dist

    def rescale_to_dist(self, pref_dist, current_dist=1):
        """Rescales the signal to a preferred distancs (in Mpc).
        Default current_distance is 1Mpc

        :returns: h_plus, h_cross
        :rtype: numpy arrays
        """
        dist = pref_dist / current_dist

        self.hp = self.hp / dist
        self.hc = self.hc / dist

    def set_to_1Mpc(self):
        """Set the distance to 1 Mpc.

        :returns: rescaling plus and cross polarization
        :rtype: numpy arrays
        """
        # define 1 Mpc
        one_mpc = 1e6 * unit.parsec
        # Geometrized total mass c = G = 1 [SI]
        mtot_geom = G * self.Mtot / c ** 2

        hp = self.rh_overmtot_p * mtot_geom.value / one_mpc.to(unit.m).value
        hc = self.rh_overmtot_c * mtot_geom.value / one_mpc.to(unit.m).value

        return hp, hc

    def time_to_SI(self):
        """Converts time to SI units.

        :returns: time to seconds
        :rtype: numpy array
        """
        time_conversion_factor = G * self.Mtot / c ** 3
        self.time = self.time * time_conversion_factor.value


def get_filenames():
    """Gives a list with the Numerical Relativity simulations

    :returns: Numerical Relativity Simulations
    :rtype: list
    """
    return os.listdir(glb.strain_path)
