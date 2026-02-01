import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import os

from pythiabns.core import constants
from pythiabns.data_utils import processing

class NumericalWaveform:
    """Class to handle loading and processing of NR waveforms."""
    
    def __init__(self, filename: str, sampling_frequency: Optional[float] = None):
        self.filename = filename
        self.sampling_frequency = sampling_frequency
        
        # Determine path
        # Logic ported from NR_strains.py: 
        # If 'Soultanis' in name, special handling. Else scan STRAIN_PATH.
        
        if os.path.isabs(filename) or os.path.exists(filename):
            self._load_from_path(Path(filename))
        elif filename.startswith('Soultanis'):
            self._load_soultanis(filename)
        else:
            self._load_standard(filename)
            
        # Common initialization
        self.Mtot = (self.m1 + self.m2) * constants.MSUN_SI
        
        # Convert to SI
        self._time_to_SI()
        self.hp, self.hc = self._set_to_1Mpc()
        
        # Resample if needed
        # In original, resampling was mandatory?
        # self.resample() # TODO: Verify if mandatory

    def _load_from_path(self, path: Path):
        self.datapath = path
        # Check if it's an NR directory
        if (path / "metadata.txt").exists():
            self.metadata_dict = self._load_metadata(self.datapath)
            self.m1 = float(self.metadata_dict.get("id_mass_starA", 1.4))
            self.m2 = float(self.metadata_dict.get("id_mass_starB", 1.4))
            self.rh_overmtot_p, self.rh_overmtot_c, self.time, self.extraction_radius = \
                self._read_hdf5_data()
        else:
            # Simple file loading (txt/csv)
            self._load_simple_file(path)

    def _load_standard(self, filename: str):
        self.datapath = constants.STRAIN_PATH / filename
        self._load_from_path(self.datapath)

    def _load_simple_file(self, path: Path):
        # Assume 3 columns: time, hp, hc
        data = np.loadtxt(path)
        self.time = data[:, 0]
        self.rh_overmtot_p = data[:, 1]
        self.rh_overmtot_c = data[:, 2]
        self.metadata_dict = {"id_name": path.name}
        self.m1 = 1.4 # Defaults
        self.m2 = 1.4
        self.extraction_radius = 0
        
        # Flag to indicate its already SI and scaled to 1Mpc?
        # For simplicity, if loading a custom file, we assume it's h+ and hx at 1Mpc and in SI.
        # So we skip _time_to_SI and _set_to_1Mpc logic by setting special values.
        self._is_si = True

    def _load_soultanis(self, filename: str):
        # Soultanis/1.55
        mass = float(filename.split('/')[-1])
        base_dir = constants.STRAIN_PATH / filename.split('/')[0]
        
        # Find matching file?
        # Original: [i for i in os.listdir(...) if mass in i][0]
        # Assuming filename structure matches
        try:
             # This is a bit fragile but ports existing logic
             candidates = list(base_dir.glob(f"*{mass}*"))
             if not candidates:
                 raise FileNotFoundError(f"No file found for {filename}")
             self.datapath = candidates[0]
        except Exception as e:
            raise FileNotFoundError(f"Error finding Soultanis file: {e}")

        self.metadata_dict = {
            'id_mass_starA': mass, 
            'id_mass_starB': mass, 
            'id_eos': 'MPA1',
            'id_name': filename
        }
        self.m1 = mass
        self.m2 = mass
        self.extraction_radius = 0 # Not applicable/available?

        data = np.loadtxt(self.datapath)
        self.time = data.T[0] / 1000.0 # ms to s
        # Original scaling: data.T[1]/8.35898e+20*40
        # What is 8.358...? Likely unit conversion.
        # Original comment: #@ 1Mpc
        # Note: In _set_to_1Mpc we might re-scale. 
        # But Soultanis load seems to return hp already scaled?
        # Original NumericalData inits rh_overmtot for standard, but hp for Soultanis.
        
        # Here I will populate rh_overmtot_p/c assuming they are NOT scaled to Mtot yet?
        # Actually Soultanis loader in original SETS hp directly.
        # So I should handle that difference.
        
        self.rh_overmtot_p = data.T[1] / 8.35898e+20 * 40
        self.rh_overmtot_c = data.T[2] / 8.35898e+20 * 40
        
        # Hack: set a flag to skip SI conversion if already in SI?
        # Original: time_msun... 
        # Standard: load_NR_strains -> rh... then time_to_SI -> set_to_1Mpc
        # Soultanis: loads already converted time?
        # Original: time = data.T[0]/1000 (ms to s). So it IS in SI (seconds).
        # Standard loads geometric time?
        
    def _read_hdf5_data(self):
        h5_path = self.datapath / "data.h5"
        with h5py.File(h5_path, "r") as f:
             # List l=2, m=2 modes
             names = [x for x in f["/rh_22"] if "l2_m2" in x]
             # Select extraction at largest radius
             # Original logic: last one, check for Inf
             names.sort() # Ensure order?
             # Original used list(f[]) which is unordered in some h5py versions?
             # Assuming sorted by string works for radii r100, r200 etc.
             selection = names[-1]
             if "Inf" in selection and len(names) > 1:
                 selection = names[-2]
             
             dset = f[f"/rh_22/{selection}"]
             data = pd.DataFrame(dset[:]) # Read all
             
             time = data.iloc[:, 0].values
             rh_p = data.iloc[:, 1].values
             rh_c = data.iloc[:, 2].values
             
             # Extract radius from name "l2_m2_r400.txt" or similar
             # Original: float(selection.split(".")[0].split("r")[1])
             try:
                extraction_radius = float(selection.split("r")[-1].split(".")[0])
             except:
                extraction_radius = 0.0
                
             return rh_p, rh_c, time, extraction_radius

    def _load_metadata(self, path: Path) -> Dict[str, Any]:
        meta_file = path / "metadata.txt"
        meta = {}
        if not meta_file.exists():
            return meta
            
        with open(meta_file, "r") as f:
            for line in f:
                parts = line.split()
                if not parts: continue
                if 'Evolution' in parts: break # Stop reading
                if "id_" in parts[0]:
                    key = parts[0]
                    val = parts[-1]
                    try:
                        val = float(val)
                    except:
                        pass
                    meta[key] = val
        return meta

    def _time_to_SI(self):
        if hasattr(self, "_is_si") and self._is_si: return
        if self.filename.startswith("Soultanis"): return # Already SI
        # Convert geometric time to seconds
        # time_SI = time_geom * G * M / c^3
        factor = constants.G_SI * self.Mtot / (constants.C_SI**3)
        self.time = self.time * factor

    def _set_to_1Mpc(self):
        if hasattr(self, "_is_si") and self._is_si: 
            return self.rh_overmtot_p, self.rh_overmtot_c
        if self.filename.startswith("Soultanis"):
             # Already scaled? Original code set self.hp directly.
             # In my class I stored it in rh_overmtot for consistency of storage.
             # So just returns them.
             return self.rh_overmtot_p, self.rh_overmtot_c
             
        # Standard scaling
        # hp = (rh/M) * (G*M/c^2) * (1/dist)
        # rh_overmtot is actually r*h / Mtot ???
        # Original: rh_overmtot_p * mtot_geom / 1Mpc
        # mtot_geom = G * M / c^2 (Length)
        
        mtot_geom = constants.G_SI * self.Mtot / (constants.C_SI**2)
        one_mpc = 1e6 * 3.085677581e16 # Parsec to meters
        
        hp = self.rh_overmtot_p * mtot_geom / one_mpc
        hc = self.rh_overmtot_c * mtot_geom / one_mpc
        return hp, hc

    def get_post_merger(self, inplace=True):
        """Crop to post-merger signal."""
        # Find merger time (max amplitude)
        amp = np.sqrt(self.hp**2 + self.hc**2)
        idx = np.argmax(amp)
        
        if inplace:
            self.time = self.time[idx:]
            self.hp = self.hp[idx:]
            self.hc = self.hc[idx:]
            # Ensure odd/even length consistency? Original had some check
        else:
            return self.time[idx:], self.hp[idx:], self.hc[idx:]
            
    def resample(self, new_fs=None):
        if new_fs is None:
             if self.sampling_frequency: new_fs = self.sampling_frequency
             else: new_fs = 8192
             
        dt = 1.0/new_fs
        new_time = np.arange(self.time[0], self.time[-1], dt)
        self.hp = processing.interpolate(self.time, self.hp, new_time)
        self.hc = processing.interpolate(self.time, self.hc, new_time)
        self.time = new_time
