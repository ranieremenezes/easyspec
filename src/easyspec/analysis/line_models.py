"""
In this file we collect all line models accepted by easyspec.

"""


import numpy as np
from scipy.special import erf
from scipy.special import wofz
from scipy.optimize import brentq


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

def model_Voigt(theta, x):
    """
    Voigt profile - amplitude controls peak height directly
    theta = [x_0, amplitude, fwhm_G, fwhm_L]
    """
    x_0, amplitude, fwhm_G, fwhm_L = theta
    
    sigma = fwhm_G / (2 * np.sqrt(2 * np.log(2)))
    gamma = fwhm_L / 2.0
    
    z = (x - x_0 + 1j * gamma) / (sigma * np.sqrt(2))
    voigt = wofz(z).real
    
    Voigt_profile = amplitude * voigt
    
    return Voigt_profile


def xpeak_for_s(s, x_min, x_max):
    """
    Solve (s/sqrt(pi)) * exp(-s^2 x^2) * (1+x^2) - x*(1+erf(s x)) = 0
    for x, given s. Returns the root x_peak.
    """
    def f(x):
        return (s/np.sqrt(np.pi)) * np.exp(- (s**2)*(x**2)) * (1.0 + x**2) - x * (1.0 + erf(s*x))
    # bracket search if sign at ends is same -> expand bounds
    a, b = x_min, x_max
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    # expand bounds if needed
    while fa * fb > 0:
        a *= 2.0
        b *= 2.0
        fa, fb = f(a), f(b)
        # safety
        if abs(a) > 1e6 or abs(b) > 1e6:
            raise RuntimeError("Could not bracket root for xpeak")
    x_root = brentq(f, a, b, maxiter=200)
    return x_root

def model_skewed_lorentzian(theta, x):
    """
    Calculate a skewed Lorentzian profile.
    
    Parameters:
    -----------
    theta : numpy.array
        Array with the initial guesses for the skewed Lorentzian model.
    
    Returns:
    --------
    intensity : ndarray
        The intensity values of the skewed Lorentzian profile
    """

    lam_peak, amplitude, gamma, skewness = theta

    s = skewness / np.sqrt(2.0)
    # solve for x_peak numerically (dimensionless)
    x_peak = xpeak_for_s(s, x_min=-50.0, x_max=50.0)
    lam0 = lam_peak - x_peak * gamma
    x = (x - lam0) / gamma
    numer = 1.0 + erf(s * x)
    denom = 1.0 + x**2
    return amplitude * numer / denom
    


def xpeak_for_s_gaussian(s, x_min, x_max):
    """
    Solve for the peak position in a skewed Gaussian.
    For a skewed Gaussian: f(x) = (1/sqrt(2*pi)) * exp(-x**2/2) * [1 + erf(sx)]
    The peak satisfies: -x * (1 + erf(sx)) + (2s/sqrt(pi)) * exp(-s**2*x**2) = 0
    """
    def f(x):
        term1 = -x * (1.0 + erf(s * x))
        term2 = (2 * s / np.sqrt(np.pi)) * np.exp(-(s**2) * (x**2))
        return term1 + term2
    
    # bracket search
    a, b = x_min, x_max
    fa, fb = f(a), f(b)
    
    if fa == 0:
        return a
    if fb == 0:
        return b
    
    # expand bounds if needed
    while fa * fb > 0:
        a *= 2.0
        b *= 2.0
        fa, fb = f(a), f(b)
        if abs(a) > 1e6 or abs(b) > 1e6:
            raise RuntimeError("Could not bracket root for xpeak in Gaussian")
    
    x_root = brentq(f, a, b, maxiter=200)
    return x_root

def model_skewed_gaussian(theta, x):
    """
    Calculate a skewed Gaussian profile.
    
    Parameters:
    -----------
    theta : numpy.array
        Array with parameters [x_peak, amplitude, sigma, skewness]
        x_peak: position of the peak
        amplitude: maximum height
        sigma: width parameter  
        skewness: skew parameter (s > 0: right-skewed, s < 0: left-skewed)
    
    Returns:
    --------
    intensity : ndarray
        The intensity values of the skewed Gaussian profile
    """
    x_peak, amplitude, sigma, skewness = theta
    
    s = skewness / np.sqrt(2.0)
    
    # Solve for the dimensionless peak position shift
    x_peak_shift = xpeak_for_s_gaussian(s, x_min=-50.0, x_max=50.0)
    
    # Adjust the center to make the peak appear at x_peak
    x0 = x_peak - x_peak_shift * sigma
    
    # Dimensionless variable
    z = (x - x0) / sigma
    
    # Skewed Gaussian formula
    gaussian = np.exp(-0.5 * z**2)
    skew_term = 1.0 + erf(s * z)
    
    return amplitude * gaussian * skew_term
