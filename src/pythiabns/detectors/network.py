import bilby
import numpy as np
from typing import List, Dict, Optional, Union
import logging

from pythiabns.data_utils.nr import NumericalWaveform

logger = logging.getLogger(__name__)

class DetectorNetwork:
    """Wrapper around bilby InterferometerList."""
    
    def __init__(self, ifo_names: List[str] = ["H1", "L1", "V1"], 
                 sampling_frequency: float = 4096,
                 duration: float = 1.0,
                 start_time: float = 0.0):
        
        self.ifo_names = ifo_names
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.start_time = start_time
        
        self.ifos = bilby.gw.detector.InterferometerList(ifo_names)
        self._configure_detectors()
        
    def _configure_detectors(self):
        for ifo in self.ifos:
            ifo.minimum_frequency = 20 # Configurable?
            ifo.maximum_frequency = self.sampling_frequency / 2.0
            
    def set_data(self, noise: bool = False):
        if noise:
            try:
                self.ifos.set_strain_data_from_power_spectral_densities(
                    sampling_frequency=self.sampling_frequency,
                    duration=self.duration,
                    start_time=self.start_time
                )
            except Exception as e:
                logger.warning(f"Failed to set noise from PSD: {e}. Fallback to zero noise + Gaussian?")
                # Logic from ifo.py line 150 used set_strain_data_from_power_spectral_densities
                # This generates Gaussian noise colored by PSD.
                raise e
        else:
            self.ifos.set_strain_data_from_zero_noise(
                sampling_frequency=self.sampling_frequency,
                duration=self.duration,
                start_time=self.start_time
            )

    def inject_signal(self, waveform_generator: bilby.gw.waveform_generator.WaveformGenerator, parameters: Dict):
        """Inject signal into detectors."""
        self.ifos.inject_signal(
            waveform_generator=waveform_generator,
            parameters=parameters,
            raise_error=False
        )

    @property
    def meta_data(self):
         # Helper to access SNR etc
         # bilby ifo.meta_data usually stores optimal_SNR after injection
         return {ifo.name: ifo.meta_data for ifo in self.ifos}
