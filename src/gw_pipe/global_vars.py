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

# Define project root relative to this file (src/gw_pipe/global_vars.py)
# Package structure check: if we are in src/gw_pipe/, root is ../../
_current_dir = os.path.abspath(os.path.dirname(__file__))
_root_dir = os.path.abspath(os.path.join(_current_dir, "..", ".."))

# Project paths with environment variable overrides
project_path = os.getenv("GW_PIPE_PROJECT_ROOT", _root_dir)
strain_path = os.getenv("GW_PIPE_STRAIN_PATH", os.path.join(_current_dir, "NR_strains"))
results_path = os.getenv("GW_PIPE_RESULTS_PATH", os.path.join(_root_dir, "results"))
config_path = os.getenv("GW_PIPE_CONFIG_PATH", os.path.join(_current_dir, "config.cfg"))
priors_path = os.getenv("GW_PIPE_PRIORS_PATH", _current_dir)

# Astronomy constants (SI units)
c = 299792458.0
G = 6.6743e-11
M_sun = 1.988409870698051e30
# bilby.constants has these too, but keeping them here for compatibility
