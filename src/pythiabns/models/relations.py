import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod
from scipy.interpolate import PchipInterpolator

from pythiabns.core import constants, registry

# Constants adapted from predictors.py
LENGTH_SCALE = constants.G_SI * constants.MSUN_SI / constants.C_SI**2 / 1e5 # km

class EmpiricalRelation(ABC):
    """Abstract base class for universal relations."""
    
    @abstractmethod
    def predict(self, m1: float, m2: float, eos_name: str) -> Dict[str, float]:
        """
        Predict f_peak and potentially other properties.
        """
        ...

@registry.RelationRegistry.register("Koutalios")
class KoutaliosRelation(EmpiricalRelation):
    def __init__(self, eos_dir: Path = constants.PACKAGE_ROOT / "EOS" / "from_Koutalios"):
        self.eos_dir = eos_dir

    def _load_eos_data(self, eos_name: str) -> Tuple[Any, Any, Any]:
        eos_name = eos_name.upper()
        file_mr = self.eos_dir / f"{eos_name}.npy"
        file_kl = self.eos_dir / f"K_L_{eos_name}.npy"
        
        if not file_mr.exists():
            raise FileNotFoundError(f"EOS data not found for {eos_name}")

        dataMR = np.load(file_mr)
        dataKL = np.load(file_kl)
        
        mass = dataMR[0]
        radius = dataMR[1] * LENGTH_SCALE 
        L = dataKL[1]
        
        mask = radius < 18
        return mass[mask], radius[mask], L[mask]

    def get_lambda_and_r(self, m: float, eos_name: str) -> Tuple[float, float]:
        mass, radius, L = self._load_eos_data(eos_name)
        
        sorted_indices = np.argsort(mass)
        mass = mass[sorted_indices]
        radius = radius[sorted_indices]
        L = L[sorted_indices]
        
        interp_L = PchipInterpolator(mass, L)
        interp_R = PchipInterpolator(mass, radius)
        
        return float(interp_L(m)), float(interp_R(m))

    def predict(self, m1: float, m2: float, eos_name: str) -> Dict[str, float]:
        L1, _ = self.get_lambda_and_r(m1, eos_name)
        L2, _ = self.get_lambda_and_r(m2, eos_name)
        
        temp = (m1 + 12*m2)*m1**4 * L1 + (m2 + 12*m1)*m2**4 * L2
        temp2 = (m1 + m2)**5
        lambda_tilde = 16.0/13.0 * (temp/temp2)
        
        return {"lambda_tilde": lambda_tilde}

@registry.RelationRegistry.register("VSB_R")
class VSB_R_Relation(KoutaliosRelation):
    def predict(self, m1: float, m2: float, eos_name: str) -> Dict[str, float]:
        Mchirp = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
        _, R18 = self.get_lambda_and_r(1.8, eos_name)
        
        fpeak = Mchirp*(10.942 - 0.369*Mchirp - 0.987*R18 + 1.095*Mchirp**2 - 0.201*Mchirp*R18 + 0.036*R18**2)*1e3
        f20 = Mchirp*(8.007 + 4.356*Mchirp - 1.241*R18 + 0.558*Mchirp**2 - 0.375*Mchirp*R18 + 0.054*R18**2)*1e3
        fspiral = Mchirp*(5.846 + 1.75*Mchirp - 0.555*R18 + 1.002*Mchirp**2 - 0.316*Mchirp*R18 + 0.026*R18**2)*1e3
        
        return {
            "f_peak": fpeak,
            "f_20": f20,
            "f_spiral": fspiral
        }

@registry.RelationRegistry.register("VSB_L")
class VSB_L_Relation(KoutaliosRelation):
    def predict(self, m1: float, m2: float, eos_name: str) -> Dict[str, float]:
        Mchirp = (m1*m2)**(3./5.)/(m1+m2)**(1./5.)
        
        L1, _ = self.get_lambda_and_r(m1, eos_name)
        L2, _ = self.get_lambda_and_r(m2, eos_name)
        temp = (m1 + 12*m2)*m1**4 * L1 + (m2 + 12*m1)*m2**4 * L2
        temp2 = (m1+m2)**5
        Lambda_tilde = 16.0/13.0 * (temp/temp2)
        
        fpeak = 1e3*(1.392 - 0.108*Mchirp + 51.7*Lambda_tilde**(-0.5))/Mchirp
        f20 = 1e3*(0.5577277 - 0.4064*Mchirp + 48.6962*Lambda_tilde**(-0.5))/Mchirp
        fspiral = 1e3*(1.1998558 - 0.4422*Mchirp + 45.8192*Lambda_tilde**(-0.5))/Mchirp
        
        return {
            "f_peak": fpeak,
            "f_20": f20,
            "f_spiral": fspiral
        }
