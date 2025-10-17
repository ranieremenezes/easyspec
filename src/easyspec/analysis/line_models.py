"""
In this file we collect all line models accepted by easyspec.

"""


import numpy as np
from scipy.special import erf
from scipy.special import wofz
from scipy.optimize import brentq, minimize_scalar


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


def xpeak_for_s(s, x_min=-50.0, x_max=50.0):
    """
    Find x_peak solving f(x)=0 for given s, or fallback to maximum of g(x).
    """

    if abs(s) < 1e-10:  # symmetric case
        return 0.0
    
    def f(x):
        return (s/np.sqrt(np.pi)) * np.exp(-(s*x)**2) * (1+x**2) - x*(1+erf(s*x))

    # sample on grid to find sign change
    xs = np.linspace(x_min, x_max, 5001)
    ys = f(xs)
    signs = np.sign(ys)
    flip = np.where(signs[:-1]*signs[1:] < 0)[0]

    if flip.size > 0:
        # use brentq between first sign change
        i = flip[0]
        return brentq(f, xs[i], xs[i+1])
    else:
        # fallback: directly maximize profile function
        g = lambda x: - (1 + erf(s*x)) / (1 + x**2)
        res = minimize_scalar(g, bounds=(x_min, x_max), method='bounded')
        if res.success:
            return res.x
        return 0.0

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
    


def xpeak_for_s_gaussian(s, x_min=-50.0, x_max=50.0):
    """
    Find the peak position x for a skewed Gaussian defined by:
        f(x) = exp(-x^2/2) * (1 + erf(sx))
    The derivative f'(x) = 0 is solved analytically or numerically.
    If no sign change is found in the derivative, a direct maximization is used.
    """

    if abs(s) < 1e-10:  # symmetric case
        return 0.0
    
    def fprime(x):
        # derivative of f(x)
        term1 = -x * (1.0 + erf(s * x))
        term2 = (2 * s / np.sqrt(np.pi)) * np.exp(-(s * x) ** 2)
        return term1 + term2

    # sample to detect zero crossings
    xs = np.linspace(x_min, x_max, 4001)
    ys = fprime(xs)
    signs = np.sign(ys)
    flip = np.where(signs[:-1] * signs[1:] < 0)[0]

    if flip.size > 0:
        # root exists, use brentq between first sign change
        i = flip[0]
        return brentq(fprime, xs[i], xs[i + 1])
    else:
        # fallback: numerically maximize f(x)
        f = lambda x: - np.exp(-0.5 * x**2) * (1 + erf(s * x))
        res = minimize_scalar(f, bounds=(x_min, x_max), method='bounded')
        return res.x if res.success else 0.0

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
