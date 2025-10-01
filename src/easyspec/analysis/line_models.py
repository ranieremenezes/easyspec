"""
In this file we collect all line models accepted by easyspec.

"""


import numpy as np
from astropy.modeling.models import Voigt1D


def model_Gauss(theta, x):
    mean, amplitude, std = theta
    return amplitude * np.exp(-0.5 * ((x - mean) / std) ** 2)

def model_Lorentz(theta, x):
    """
    Single Lorentzian profile
    theta = [mean, amplitude, fwhm]
    """
    mean, amplitude, fwhm = theta
    gamma = fwhm / 2.0
    lorentzian = amplitude * (gamma**2) / ((x - mean) ** 2 + gamma**2)
    return lorentzian

def model_Voigt(theta,x):
    x_0, amplitude_L, fwhm_G, fwhm_L = theta
    a = Voigt1D(x_0, amplitude_L, fwhm_L, fwhm_G)
    return a(x)
