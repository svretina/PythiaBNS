import bilby
from gw_pipe.models.interface import WaveformModel

class PostMergerLikelihood(bilby.gw.likelihood.GravitationalWaveTransient):
    """
    Likelihood class for Post-Merger PE.
    Inherits from GravitationalWaveTransient to use standard detector response logic.
    """
    def __init__(self, interferometers, waveform_generator):
        super().__init__(interferometers, waveform_generator)

    # Note: The standard GravitationalWaveTransient is usually sufficient 
    # IF the waveform_generator produces the correct mode (FD/TD).
    # However, we might want to override logic if we need custom handling meant for 3G detectors 
    # or specific noise models not in Bilby (though Bilby is quite complete).
    
    # For now, this is a thin wrapper but allows us to extend it later 
    # (e.g. for marginalization over calibration if needed).
    pass
