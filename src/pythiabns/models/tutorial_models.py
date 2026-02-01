import numpy as np
from pythiabns.core.registry import ModelRegistry
from pythiabns.data_utils import processing

def tutorial_conversion(parameters):
    """Simple conversion function for tutorial model."""
    # No conversion needed for this simple model, but good to showcase
    return parameters, []

@ModelRegistry.register("three_sines", domain="time", conversion_func=tutorial_conversion)
def three_sines(time, 
                a1, f1, p1,
                a2, f2, p2,
                a3, f3, p3, 
                **kwargs):
    """
    Tutorial model: Sum of 3 sine waves.
    y(t) = a1*sin(2*pi*f1*t + p1) + a2*sin(2*pi*f2*t + p2) + a3*sin(2*pi*f3*t + p3)
    """
    plus = (
        a1 * np.sin(2 * np.pi * f1 * time + p1) +
        a2 * np.sin(2 * np.pi * f2 * time + p2) +
        a3 * np.sin(2 * np.pi * f3 * time + p3)
    )
    # Generate cross by 90 deg phase shift
    cross = processing.generate_cross(plus)
    
    return {"plus": plus, "cross": cross}
