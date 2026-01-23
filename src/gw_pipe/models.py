""" 
Models definition file for gw_pipe.
Users can define new gravitational wave models here using the @register_model decorator.
Registered models will be automatically available via the configuration file.
"""

import numpy as np
from gw_pipe.registry import register_model

# Example of a new user-defined model
# @register_model("my_custom_model")
# def my_custom_model(f, parameter1, parameter2, **kwargs):
#     # Logic to generate frequency domain strain (plus and cross)
#     plus = parameter1 * np.exp(-f/parameter2)
#     cross = 0 * plus
#     return {"plus": plus, "cross": cross}

# Example with specific nfreqs
# @register_model("my_custom_model", nfreqs=2)
# def my_custom_model_v2(f, parameter1, **kwargs):
#     # logic ...
#     return {"plus": ..., "cross": ...}
