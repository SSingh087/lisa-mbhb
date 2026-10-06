"""Physical and LISA constants (SI)."""
import numpy as np

C = 299792458.0
G = 6.67430e-11
MSUN = 1.988409870698051e30
MTSUN = G * MSUN / C**3          # solar mass in seconds
MPC = 3.085677581491367e22
AU = 1.495978707e11
YEAR = 365.25 * 86400.0

LISA_ARM = 2.5e9                 # m, arm length
F_STAR = C / (2 * np.pi * LISA_ARM)
E_ORBIT = LISA_ARM / (2 * np.sqrt(3) * AU)   # orbital eccentricity
