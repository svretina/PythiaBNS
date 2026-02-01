import os
from pathlib import Path
from astropy import constants as const

# Determine paths
_current_file = Path(__file__).resolve()
# src/pythiabns/core/constants.py -> src/pythiabns -> src -> root
PACKAGE_ROOT = _current_file.parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent 
if (PROJECT_ROOT / "src").exists():
    pass
else:
    # Fallback if installed in site-packages
    PROJECT_ROOT = Path.cwd()

# Configurable Paths via Env Vars
STRAIN_PATH = Path(os.getenv("PYTHIABNS_STRAIN_PATH", PACKAGE_ROOT / "NR_strains"))
RESULTS_PATH = Path(os.getenv("PYTHIABNS_RESULTS_PATH", PROJECT_ROOT / "results"))
PRIORS_PATH = Path(os.getenv("PYTHIABNS_PRIORS_PATH", PACKAGE_ROOT / "priors"))

# Physical Constants (SI)
C_SI = const.c.value
G_SI = const.G.value
MSUN_SI = const.M_sun.value
