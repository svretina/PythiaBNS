from typing import Protocol, Dict, Union, Any,runtime_checkable
import numpy as np

@runtime_checkable
class WaveformModel(Protocol):
    """Protocol that all waveform models must adhere to."""
    
    def __call__(self, frequency_array: np.ndarray, **params: float) -> Dict[str, np.ndarray]:
        """
        Generate waveform polarizations.
        
        Args:
            frequency_array: Array of frequencies in Hz.
            **params: Model parameters (masses, spins, etc).
            
        Returns:
            Dict containing 'plus' and 'cross' keys with complex strain arrays.
        """
        ...
