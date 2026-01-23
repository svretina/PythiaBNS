#!/usr/bin/env python3

import bilby
import scipy
import numpy as np
from bilby.core import utils
from astropy import units as unit
import matplotlib.pyplot as plt
from gw_pipe import NR_strains as nr
from gw_pipe import utils as u
from scipy.interpolate import interp1d

class InterferometerHandler:
    def __init__(
        self, data, injection_parameters, noise=False, ifo_list=["H1", "L1", "V1"],
    ):

        ##### Set the data #####
        self.data = data
        self.ifo_list = ifo_list
        self.injection_parameters = injection_parameters
        # No reason for arbitrary duration
        # (
        #     self.sampling_frequency,
        #     self.duration,
        # ) = utils.get_sampling_frequency_and_duration_from_time_array(data.time)
        self.sampling_frequency = 2*8192
        self.duration = 0.125

        self.start_time_wv = 0 #self.data.time[0]
        self.start_time_ifo = self.injection_parameters["geocent_time"]
        # the data's waveform generator
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            time_domain_source_model=self.time_domain_nr,
            start_time=self.start_time_wv,
            parameters=self.injection_parameters,
        )

        ##### Set the detectors #####
        self.ifos = self.prepare_ifo(ifo_list, noise)
        self.inject_data()
        # self.t0 = 0
        # self.t_end = self.data.time[-1]-self.t0
        #from gwpy.signal.filter_design import bandpass,lowpass,highpass
        #from gwpy.timeseries import TimeSeries
        #N = self.ifos[0].strain_data.time_domain_strain.size
        #dt = self.ifos[0].time_array[1]-self.ifos[0].time_array[0]
        #df = self.ifos[0].frequency_array[1]-self.ifos[0].frequency_array[0]
        #h = np.hstack([self.ifos[0].strain_data.time_domain_strain[N//2:],
                    #self.ifos[0].strain_data.time_domain_strain[:N//2]])

        #self.ifos[0].strain_data._time_domain_strain = h
        #self.ifos[0].strain_data._frequency_domain_strain = None

        #bp = bandpass(500,4000, self.sampling_frequency)
        #strain = TimeSeries(h, sample_rate=self.sampling_frequency)
        #strain = strain.filter(bp, filtfilt=True)
        #plt.figure(figsize=(6,4),dpi=100)
        ##plt.plot(h)
        ##plt.plot(strain.value)
        ##plt.show()
        ##exit()
        #plt.title(data.metadata_dict['id_eos'])
        #plt.plot(self.ifos[0].frequency_array,bilby.gw.utils.asd_from_freq_series(self.ifos[0].strain_data.frequency_domain_strain,df=df),label='$raw\quad strain$')
        #self.ifos[0].strain_data._time_domain_strain = strain.value
        #self.ifos[0].strain_data._frequency_domain_strain = None

        #plt.plot(self.ifos[0].frequency_array,bilby.gw.utils.asd_from_freq_series(self.ifos[0].strain_data.frequency_domain_strain,df=df),'--',label='$bandpassed\quad 500-4000$')
        #plt.plot(self.waveform_generator.frequency_array,
                 #bilby.gw.utils.asd_from_freq_series(self.waveform_generator.frequency_domain_strain()['plus'],df=df),label='$injection$')
        #plt.plot(self.ifos[0].frequency_array,self.ifos[0].amplitude_spectral_density_array,label='$detector ASD$')
        #plt.yscale('log')
        #plt.xlim(500,5000)
        #plt.ylim(5e-25,5e-22)
        #plt.legend()
        #plt.show()
        #exit()
        #self.ifos[0].frequency_domain_strain = utils.nfft(strain.value,self.sampling_frequency)[0]

        #for i in range(len(self.ifos)):
            #self.ifos[i].strain_data.low_pass_filter(filter_freq=50)
        ##### Get information about the injection #####
        # print(abs(self.ifos[0].meta_data['matched_filter_SNR']))
        # print(self.ifos[0].meta_data['optimal_SNR'])
        # exit()
        self.Lpsd = self.ifos[0].power_spectral_density_array
        # self.Vpsd = self.ifos[-1].power_spectral_density_array
        # self.Hsnr = self.ifos[1].meta_data["optimal_SNR"]
        self.Lsnr = self.ifos[0].meta_data["optimal_SNR"]
        # self.Vsnr = self.ifos[-1].meta_data["optimal_SNR"]

        self.fft_amplitude = [
            abs(self.ifos[i].frequency_domain_strain)
            * np.sqrt(self.ifos.frequency_array)
            for i in range(len(ifo_list))
        ]

######## EASTER METHOD #########
    def time_domain_nr(self, time):
        full_wave = (self.data.hp**2 + self.data.hp ** 2) ** 0.5
        postmerger_start_index = np.argmax(full_wave)
        hre = self.data.hp
        him = self.data.hc
        time_msun = self.data.time
        th = (time_msun - time_msun[postmerger_start_index]) # th = 0 @ merger
        tstartindex = np.argmax(th > 0.0)
        hrenew = hre[tstartindex:]
        himnew = him[tstartindex:]
        thnew = th[tstartindex:] + time_msun[postmerger_start_index] # postmerger started @ t=0 now @ t_0
        self.t0 = thnew[0]
        self.t_end = thnew[-1]
        #hrenew = self.data.hp
        #himnew = self.data.hc
        #thnew = self.data.time

        hplus_interp_func = interp1d(thnew,
                                    hrenew,
                                    bounds_error=False, fill_value=0)

        hcross_interp_func = interp1d(thnew,
                                    himnew,
                                    bounds_error=False, fill_value=0)

        #time = newtime - newtime[0]
        hplus = hplus_interp_func(time)
        hcross = hcross_interp_func(time)

        return {"plus": hplus, "cross": hcross}

    # def time_domain_nr(self, time):
    #     """Wrapper function needed for bilby."""
    #     return {"plus": self.data.hp, "cross": self.data.hc}

    def prepare_ifo(self, ifo_list, noise):
        """Prepares the detectors settings max and min frequency,
        sets the strain data's noise.

        :returns: Interferometer objects with Gaussian noise
        :rtype: bilby.gw.detector.interferometer.Interferometer
        """
        ifos = bilby.gw.detector.InterferometerList(ifo_list)
        for ifo in ifos:
            ifo.minimum_frequency = 10
            ifo.maximum_frequency = int(self.sampling_frequency//2)
            
        if noise:
            # for Gaussian noise
            ifos.set_strain_data_from_power_spectral_densities(
                sampling_frequency=self.sampling_frequency,
                duration=self.duration,
                start_time=self.start_time_ifo,
            )
        else:
            # for zero noise
            ifos.set_strain_data_from_zero_noise(
                sampling_frequency=self.sampling_frequency,
                duration=self.duration,
                start_time=self.start_time_ifo,
            )
        return ifos

    def inject_data(self):
        self.injected_data = self.ifos.inject_signal(
            #injection_polarizations = self.waveform_generator.frequency_domain_strain(),
            waveform_generator=self.waveform_generator,
            parameters=self.injection_parameters,
        )
        
