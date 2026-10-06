"""LISA noise: SciRDv1 (LISA-LCST-SGS-TN-001) single-link noise and TDI PSDs.

All PSDs are one-sided, in 1/Hz (fractional-frequency TDI units unless
the function says otherwise).
"""
import numpy as np
from .constants import C, LISA_ARM, F_STAR, YEAR

# SciRDv1 requirement-level noise
P_OMS_AMP = 15e-12     # m / sqrt(Hz)
P_ACC_AMP = 3e-15      # m s^-2 / sqrt(Hz)


def p_oms(f):
    """Optical metrology system noise, displacement PSD [m^2/Hz]."""
    f = np.asarray(f, float)
    return P_OMS_AMP**2 * (1 + (2e-3 / f) ** 4)


def p_acc(f):
    """Test-mass acceleration noise PSD [m^2 s^-4 /Hz]."""
    f = np.asarray(f, float)
    return P_ACC_AMP**2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)


def s_oms(f):
    """OMS noise in fractional frequency units [1/Hz]."""
    f = np.asarray(f, float)
    return p_oms(f) * (2 * np.pi * f / C) ** 2


def s_acc(f):
    """Acceleration noise in fractional frequency units [1/Hz]."""
    f = np.asarray(f, float)
    return p_acc(f) / ((2 * np.pi * f) ** 2 * C**2)


def galactic_confusion(f, t_obs_years=4.0):
    """Galactic binary confusion noise in strain units (SciRD fit, Robson+19 / LDC)."""
    f = np.asarray(f, float)
    # fit parameters per observation time (Babak+21 table)
    table = {0.5: (0.133, 243, 482, 917, 0.00258),
             1.0: (0.171, 292, 1020, 1680, 0.00215),
             2.0: (0.165, 299, 611, 1340, 0.00173),
             4.0: (0.138, -221, 521, 1680, 0.00113)}
    key = min(table, key=lambda k: abs(k - t_obs_years))
    alpha, beta, kappa, gamma, fk = table[key]
    amp = 9e-45
    return (amp * f ** (-7 / 3) * np.exp(-(f**alpha) + beta * f * np.sin(kappa * f))
            * (1 + np.tanh(gamma * (fk - f))))


def psd_tdi_A(f, tdi=1, confusion=False, t_obs_years=4.0):
    """PSD of the A (=E) channel, fractional-frequency units [1/Hz].

    tdi=1 uses the first-generation Michelson-based X (|1-D^2| factor);
    tdi=2 multiplies by the second-generation factor 4 sin^2(2x).
    """
    f = np.asarray(f, float)
    x = 2 * np.pi * f * LISA_ARM / C
    s = 8 * np.sin(x) ** 2 * ((2 + np.cos(x)) * s_oms(f)
                              + 2 * (3 + 2 * np.cos(x) + np.cos(2 * x)) * s_acc(f))
    if tdi == 2:
        s = s * 4 * np.sin(2 * x) ** 2
    if confusion:
        s = s + _confusion_tdi(f, t_obs_years, tdi)
    return s


def psd_tdi_T(f, tdi=1):
    """PSD of the null channel T, fractional-frequency units [1/Hz]."""
    f = np.asarray(f, float)
    x = 2 * np.pi * f * LISA_ARM / C
    s = 32 * np.sin(x) ** 2 * np.sin(x / 2) ** 2 * (
        4 * np.sin(x / 2) ** 2 * s_acc(f) + s_oms(f))
    if tdi == 2:
        s = s * 4 * np.sin(2 * x) ** 2
    return s


def _confusion_tdi(f, t_obs_years, tdi):
    x = 2 * np.pi * f * LISA_ARM / C
    # strain-PSD confusion -> A/E channel through the averaged response
    s = galactic_confusion(f, t_obs_years) * 4 * x**2 * np.sin(x) ** 2 * response_average_AE(f)
    if tdi == 2:
        s = s * 4 * np.sin(2 * x) ** 2
    return s


def response_average_AE(f):
    """Sky/polarisation/inclination averaged response R(f) of one AE channel
    (analytic fit of Cornish & Robson 2017, valid up to ~ 0.1 Hz)."""
    f = np.asarray(f, float)
    return 3.0 / 20.0 / (1 + 0.6 * (f / F_STAR) ** 2)


def sensitivity_curve(f, tdi=1, confusion=False, t_obs_years=4.0):
    """Sky-averaged sensitivity Sn(f) [1/Hz, strain] of the A+E network,
    i.e. the curve a signal's characteristic strain is compared with.
    """
    f = np.asarray(f, float)
    x = 2 * np.pi * f * LISA_ARM / C
    # PSD of one AE channel / (averaged response * transfer^2)
    norm = 4 * x**2 * np.sin(x) ** 2 * response_average_AE(f)
    if tdi == 2:
        norm = norm * 4 * np.sin(2 * x) ** 2
    s_inst = psd_tdi_A(f, tdi=tdi) / norm
    # two independent channels A, E
    s = s_inst / 2.0
    if confusion:
        s = s + galactic_confusion(f, t_obs_years) / 2.0
    return s
