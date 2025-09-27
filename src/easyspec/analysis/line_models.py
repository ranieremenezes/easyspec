"""
In this file we collect all line models accepted by easyspec.

"""


import numpy as np
from astropy.modeling.models import Gaussian1D, Voigt1D, Lorentz1D
    

def model_Gauss(theta, x):
    mean, amplitude, std = theta
    return amplitude * np.exp(-0.5 * ((x - mean) / std) ** 2)

def model_Gauss_Gauss(theta, x):
    """
    Sum of 2 Gaussian profiles
    theta = [mean1, amp1, std1, mean2, amp2, std2]
    """
    mean1, amp1, std1, mean2, amp2, std2 = theta
    g1 = amp1 * np.exp(-0.5 * ((x - mean1) / std1) ** 2)
    g2 = amp2 * np.exp(-0.5 * ((x - mean2) / std2) ** 2)
    return g1 + g2

def model_Gauss_Lorentz(theta, x):
    """
    Sum of Gaussian and Lorentzian profiles
    theta = [mean_G, amp_G, std_G, mean_L, amp_L, fwhm_L]
    """
    mean_G, amp_G, std_G, mean_L, amp_L, fwhm_L = theta
    gaussian = amp_G * np.exp(-0.5 * ((x - mean_G) / std_G) ** 2)
    gamma = fwhm_L / 2.0
    lorentzian = amp_L * (gamma**2) / ((x - mean_L) ** 2 + gamma**2)
    return gaussian + lorentzian

def model_Gauss_Voigt(theta,x):
    mean, amplitude, std, x_0, amplitude_L, fwhm_G, fwhm_L = theta
    a = Gaussian1D(amplitude, mean, std) + Voigt1D(x_0, amplitude_L, fwhm_L, fwhm_G)
    return a(x)

def model_Gauss_Gauss_Gauss(theta, x):
    """
    Sum of 3 Gaussian profiles
    theta = [mean1, amp1, std1, mean2, amp2, std2, mean3, amp3, std3]
    """
    mean1, amp1, std1, mean2, amp2, std2, mean3, amp3, std3 = theta
    g1 = amp1 * np.exp(-0.5 * ((x - mean1) / std1) ** 2)
    g2 = amp2 * np.exp(-0.5 * ((x - mean2) / std2) ** 2)
    g3 = amp3 * np.exp(-0.5 * ((x - mean3) / std3) ** 2)
    return g1 + g2 + g3

def model_Gauss_Lorentz_Gauss(theta, x):
    """
    Sum of Gaussian + Lorentzian + Gaussian profiles
    theta = [mean_G1, amp_G1, std_G1, mean_L, amp_L, fwhm_L, mean_G2, amp_G2, std_G2]
    """
    mean_G1, amp_G1, std_G1, mean_L, amp_L, fwhm_L, mean_G2, amp_G2, std_G2 = theta
    # First Gaussian component
    gaussian1 = amp_G1 * np.exp(-0.5 * ((x - mean_G1) / std_G1) ** 2)
    # Lorentzian component
    gamma = fwhm_L / 2.0
    lorentzian = amp_L * (gamma**2) / ((x - mean_L) ** 2 + gamma**2)
    # Second Gaussian component
    gaussian2 = amp_G2 * np.exp(-0.5 * ((x - mean_G2) / std_G2) ** 2)
    return gaussian1 + lorentzian + gaussian2

def model_Gauss_Gauss_Lorentz(theta, x):
    """
    Sum of Gaussian + Gaussian + Lorentzian profiles
    theta = [mean_G1, amp_G1, std_G1, mean_G2, amp_G2, std_G2, mean_L, amp_L, fwhm_L]
    """
    mean_G1, amp_G1, std_G1, mean_G2, amp_G2, std_G2, mean_L, amp_L, fwhm_L = theta
    # First Gaussian component
    gaussian1 = amp_G1 * np.exp(-0.5 * ((x - mean_G1) / std_G1) ** 2)
    # Second Gaussian component
    gaussian2 = amp_G2 * np.exp(-0.5 * ((x - mean_G2) / std_G2) ** 2)
    # Lorentzian component
    gamma = fwhm_L / 2.0
    lorentzian = amp_L * (gamma**2) / ((x - mean_L) ** 2 + gamma**2)
    return gaussian1 + gaussian2 + lorentzian

def model_Gauss_Lorentz_Lorentz(theta, x):
    """
    Sum of Gaussian + Lorentzian + Lorentzian profiles
    theta = [mean_G, amp_G, std_G, mean_L1, amp_L1, fwhm_L1, mean_L2, amp_L2, fwhm_L2]
    """
    mean_G, amp_G, std_G, mean_L1, amp_L1, fwhm_L1, mean_L2, amp_L2, fwhm_L2 = theta
    # Gaussian component
    gaussian = amp_G * np.exp(-0.5 * ((x - mean_G) / std_G) ** 2)
    # First Lorentzian component
    gamma1 = fwhm_L1 / 2.0
    lorentzian1 = amp_L1 * (gamma1**2) / ((x - mean_L1) ** 2 + gamma1**2)
    # Second Lorentzian component
    gamma2 = fwhm_L2 / 2.0
    lorentzian2 = amp_L2 * (gamma2**2) / ((x - mean_L2) ** 2 + gamma2**2)
    return gaussian + lorentzian1 + lorentzian2

def model_Gauss_Voigt_Voigt(theta,x):
    mean, amplitude, std, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt, x_02, amplitude_Voigt2, fwhm_G2, fwhm_L_Voigt2 = theta
    a = Gaussian1D(amplitude, mean, std) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G) + Voigt1D(x_02, amplitude_Voigt2, fwhm_L_Voigt2, fwhm_G2)
    return a(x)

def model_Gauss_Voigt_Lorentz(theta,x):
    mean, amplitude, std, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt, mean_L, amplitude_L, fwhm_L = theta
    a = Gaussian1D(amplitude, mean, std) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G) + Lorentz1D(amplitude_L, mean_L, fwhm_L)
    return a(x)

def model_Gauss_Lorentz_Voigt(theta,x):
    mean, amplitude, std, mean_L, amplitude_L, fwhm_L, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt = theta
    a = Gaussian1D(amplitude, mean, std) + Lorentz1D(amplitude_L, mean_L, fwhm_L) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G)
    return a(x)

def model_Gauss_Gauss_Voigt(theta,x):
    mean, amplitude, std, mean2, amplitude2, std2, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt = theta
    a = Gaussian1D(amplitude, mean, std) + Gaussian1D(amplitude2, mean2, std2) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G)
    return a(x)

def model_Gauss_Voigt_Gauss(theta,x):
    mean, amplitude, std, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt, mean2, amplitude2, std2 = theta
    a = Gaussian1D(amplitude, mean, std) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G) + Gaussian1D(amplitude2, mean2, std2)
    return a(x)

def model_Gauss_Gauss_Gauss_Gauss(theta, x):
    """
    Sum of 4 Gaussian profiles
    theta = [mean1, amp1, std1, mean2, amp2, std2, mean3, amp3, std3, mean4, amp4, std4]
    """
    mean1, amp1, std1, mean2, amp2, std2, mean3, amp3, std3, mean4, amp4, std4 = theta
    g1 = amp1 * np.exp(-0.5 * ((x - mean1) / std1) ** 2)
    g2 = amp2 * np.exp(-0.5 * ((x - mean2) / std2) ** 2)
    g3 = amp3 * np.exp(-0.5 * ((x - mean3) / std3) ** 2)
    g4 = amp4 * np.exp(-0.5 * ((x - mean4) / std4) ** 2)
    return g1 + g2 + g3 + g4

def model_Gauss_Gauss_Gauss_Gauss_Gauss(theta, x):
    """
    Sum of 5 Gaussian profiles
    theta = [mean1, amp1, std1, mean2, amp2, std2, mean3, amp3, std3, mean4, amp4, std4, mean5, amp5, std5]
    """
    mean1, amp1, std1, mean2, amp2, std2, mean3, amp3, std3, mean4, amp4, std4, mean5, amp5, std5 = theta
    g1 = amp1 * np.exp(-0.5 * ((x - mean1) / std1) ** 2)
    g2 = amp2 * np.exp(-0.5 * ((x - mean2) / std2) ** 2)
    g3 = amp3 * np.exp(-0.5 * ((x - mean3) / std3) ** 2)
    g4 = amp4 * np.exp(-0.5 * ((x - mean4) / std4) ** 2)
    g5 = amp5 * np.exp(-0.5 * ((x - mean5) / std5) ** 2)
    return g1 + g2 + g3 + g4 + g5

def model_Lorentz(theta, x):
    """
    Single Lorentzian profile
    theta = [mean, amplitude, fwhm]
    """
    mean, amplitude, fwhm = theta
    gamma = fwhm / 2.0
    lorentzian = amplitude * (gamma**2) / ((x - mean) ** 2 + gamma**2)
    return lorentzian

def model_Lorentz_Lorentz(theta, x):
    """
    Sum of 2 Lorentzian profiles
    theta = [mean1, amplitude1, fwhm1, mean2, amplitude2, fwhm2]
    """
    mean1, amplitude1, fwhm1, mean2, amplitude2, fwhm2 = theta
    gamma1 = fwhm1 / 2.0
    gamma2 = fwhm2 / 2.0
    l1 = amplitude1 * (gamma1**2) / ((x - mean1) ** 2 + gamma1**2)
    l2 = amplitude2 * (gamma2**2) / ((x - mean2) ** 2 + gamma2**2)
    return l1 + l2

def model_Lorentz_Gauss(theta, x):
    """
    Sum of Lorentzian and Gaussian profiles
    theta = [mean_L, amplitude_L, fwhm_L, mean_G, amplitude_G, std_G]
    """
    mean_L, amplitude_L, fwhm_L, mean_G, amplitude_G, std_G = theta
    # Lorentzian component
    gamma = fwhm_L / 2.0
    lorentzian = amplitude_L * (gamma**2) / ((x - mean_L) ** 2 + gamma**2)
    # Gaussian component
    gaussian = amplitude_G * np.exp(-0.5 * ((x - mean_G) / std_G) ** 2)
    return lorentzian + gaussian

def model_Lorentz_Voigt(theta,x):
    mean, amplitude, fwhm, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt = theta
    a = Lorentz1D(amplitude, mean, fwhm) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G)
    return a(x)

def model_Lorentz_Gauss_Gauss(theta, x):
    """
    Sum of Lorentzian + Gaussian + Gaussian profiles
    theta = [mean_L, amplitude_L, fwhm_L, mean_G1, amplitude_G1, std_G1, mean_G2, amplitude_G2, std_G2]
    """
    mean_L, amplitude_L, fwhm_L, mean_G1, amplitude_G1, std_G1, mean_G2, amplitude_G2, std_G2 = theta
    # Lorentzian component
    gamma = fwhm_L / 2.0
    lorentzian = amplitude_L * (gamma**2) / ((x - mean_L) ** 2 + gamma**2)
    # First Gaussian component
    gaussian1 = amplitude_G1 * np.exp(-0.5 * ((x - mean_G1) / std_G1) ** 2)
    # Second Gaussian component
    gaussian2 = amplitude_G2 * np.exp(-0.5 * ((x - mean_G2) / std_G2) ** 2)
    return lorentzian + gaussian1 + gaussian2

def model_Lorentz_Lorentz_Lorentz(theta, x):
    """
    Sum of 3 Lorentzian profiles
    theta = [mean_L1, amplitude_L1, fwhm_L1, mean_L2, amplitude_L2, fwhm_L2, mean_L3, amplitude_L3, fwhm_L3]
    """
    mean_L1, amplitude_L1, fwhm_L1, mean_L2, amplitude_L2, fwhm_L2, mean_L3, amplitude_L3, fwhm_L3 = theta
    # First Lorentzian component
    gamma1 = fwhm_L1 / 2.0
    lorentzian1 = amplitude_L1 * (gamma1**2) / ((x - mean_L1) ** 2 + gamma1**2)
    # Second Lorentzian component
    gamma2 = fwhm_L2 / 2.0
    lorentzian2 = amplitude_L2 * (gamma2**2) / ((x - mean_L2) ** 2 + gamma2**2)
    # Third Lorentzian component
    gamma3 = fwhm_L3 / 2.0
    lorentzian3 = amplitude_L3 * (gamma3**2) / ((x - mean_L3) ** 2 + gamma3**2)
    return lorentzian1 + lorentzian2 + lorentzian3

def model_Lorentz_Voigt_Voigt(theta,x):
    mean_L0, amplitude_L0, fwhm_L0, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt, x_02, amplitude_Voigt2, fwhm_G2, fwhm_L_Voigt2 = theta
    a = Lorentz1D(amplitude_L0, mean_L0, fwhm_L0) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G) + Voigt1D(x_02, amplitude_Voigt2, fwhm_L_Voigt2, fwhm_G2)
    return a(x)

def model_Lorentz_Lorentz_Gauss(theta, x):
    """
    Sum of Lorentzian + Lorentzian + Gaussian profiles
    theta = [mean_L1, amplitude_L1, fwhm_L1, mean_L2, amplitude_L2, fwhm_L2, mean_G, amplitude_G, std_G]
    """
    mean_L1, amplitude_L1, fwhm_L1, mean_L2, amplitude_L2, fwhm_L2, mean_G, amplitude_G, std_G = theta
    # First Lorentzian component
    gamma1 = fwhm_L1 / 2.0
    lorentzian1 = amplitude_L1 * (gamma1**2) / ((x - mean_L1) ** 2 + gamma1**2)
    # Second Lorentzian component
    gamma2 = fwhm_L2 / 2.0
    lorentzian2 = amplitude_L2 * (gamma2**2) / ((x - mean_L2) ** 2 + gamma2**2)
    # Gaussian component
    gaussian = amplitude_G * np.exp(-0.5 * ((x - mean_G) / std_G) ** 2)
    return lorentzian1 + lorentzian2 + gaussian

def model_Lorentz_Gauss_Lorentz(theta, x):
    """
    Sum of Lorentzian + Gaussian + Lorentzian profiles
    theta = [mean_L1, amplitude_L1, fwhm_L1, mean_G, amplitude_G, std_G, mean_L2, amplitude_L2, fwhm_L2]
    """
    mean_L1, amplitude_L1, fwhm_L1, mean_G, amplitude_G, std_G, mean_L2, amplitude_L2, fwhm_L2 = theta
    # First Lorentzian component
    gamma1 = fwhm_L1 / 2.0
    lorentzian1 = amplitude_L1 * (gamma1**2) / ((x - mean_L1) ** 2 + gamma1**2)
    # Gaussian component
    gaussian = amplitude_G * np.exp(-0.5 * ((x - mean_G) / std_G) ** 2)
    # Second Lorentzian component
    gamma2 = fwhm_L2 / 2.0
    lorentzian2 = amplitude_L2 * (gamma2**2) / ((x - mean_L2) ** 2 + gamma2**2)
    return lorentzian1 + gaussian + lorentzian2

def model_Lorentz_Voigt_Lorentz(theta,x):
    mean_L0, amplitude_L0, fwhm_L0, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt, mean_L, amplitude_L, fwhm_L = theta
    a = Lorentz1D(amplitude_L0, mean_L0, fwhm_L0) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G) + Lorentz1D(amplitude_L, mean_L, fwhm_L)
    return a(x)

def model_Lorentz_Lorentz_Voigt(theta,x):
    mean_L0, amplitude_L0, fwhm_L0, mean_L, amplitude_L, fwhm_L, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt = theta
    a = Lorentz1D(amplitude_L0, mean_L0, fwhm_L0) + Lorentz1D(amplitude_L, mean_L, fwhm_L) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G)
    return a(x)

def model_Lorentz_Gauss_Voigt(theta,x):
    mean_L0, amplitude_L0, fwhm_L0, mean2, amplitude2, std2, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt = theta
    a = Lorentz1D(amplitude_L0, mean_L0, fwhm_L0) + Gaussian1D(amplitude2, mean2, std2) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G)
    return a(x)

def model_Lorentz_Voigt_Gauss(theta,x):
    mean_L0, amplitude_L0, fwhm_L0, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt, mean2, amplitude2, std2 = theta
    a = Lorentz1D(amplitude_L0, mean_L0, fwhm_L0) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G) + Gaussian1D(amplitude2, mean2, std2)
    return a(x)

def model_Lorentz_Lorentz_Lorentz_Lorentz(theta, x):
    """
    Sum of 4 Lorentzian profiles
    theta = [mean_L1, amplitude_L1, fwhm_L1, mean_L2, amplitude_L2, fwhm_L2, mean_L3, amplitude_L3, fwhm_L3, mean_L4, amplitude_L4, fwhm_L4]
    """
    mean_L1, amplitude_L1, fwhm_L1, mean_L2, amplitude_L2, fwhm_L2, mean_L3, amplitude_L3, fwhm_L3, mean_L4, amplitude_L4, fwhm_L4 = theta
    # First Lorentzian component
    gamma1 = fwhm_L1 / 2.0
    lorentzian1 = amplitude_L1 * (gamma1**2) / ((x - mean_L1) ** 2 + gamma1**2)
    # Second Lorentzian component
    gamma2 = fwhm_L2 / 2.0
    lorentzian2 = amplitude_L2 * (gamma2**2) / ((x - mean_L2) ** 2 + gamma2**2)
    # Third Lorentzian component
    gamma3 = fwhm_L3 / 2.0
    lorentzian3 = amplitude_L3 * (gamma3**2) / ((x - mean_L3) ** 2 + gamma3**2)
    # Fourth Lorentzian component
    gamma4 = fwhm_L4 / 2.0
    lorentzian4 = amplitude_L4 * (gamma4**2) / ((x - mean_L4) ** 2 + gamma4**2)
    return lorentzian1 + lorentzian2 + lorentzian3 + lorentzian4

def model_Lorentz_Lorentz_Lorentz_Lorentz_Lorentz(theta, x):
    """
    Sum of 5 Lorentzian profiles
    theta = [mean_L1, amplitude_L1, fwhm_L1, mean_L2, amplitude_L2, fwhm_L2, mean_L3, amplitude_L3, fwhm_L3, mean_L4, amplitude_L4, fwhm_L4, mean_L5, amplitude_L5, fwhm_L5]
    """
    mean_L1, amplitude_L1, fwhm_L1, mean_L2, amplitude_L2, fwhm_L2, mean_L3, amplitude_L3, fwhm_L3, mean_L4, amplitude_L4, fwhm_L4, mean_L5, amplitude_L5, fwhm_L5 = theta
    # First Lorentzian component
    gamma1 = fwhm_L1 / 2.0
    lorentzian1 = amplitude_L1 * (gamma1**2) / ((x - mean_L1) ** 2 + gamma1**2)
    # Second Lorentzian component
    gamma2 = fwhm_L2 / 2.0
    lorentzian2 = amplitude_L2 * (gamma2**2) / ((x - mean_L2) ** 2 + gamma2**2)
    # Third Lorentzian component
    gamma3 = fwhm_L3 / 2.0
    lorentzian3 = amplitude_L3 * (gamma3**2) / ((x - mean_L3) ** 2 + gamma3**2)
    # Fourth Lorentzian component
    gamma4 = fwhm_L4 / 2.0
    lorentzian4 = amplitude_L4 * (gamma4**2) / ((x - mean_L4) ** 2 + gamma4**2)
    # Fifth Lorentzian component
    gamma5 = fwhm_L5 / 2.0
    lorentzian5 = amplitude_L5 * (gamma5**2) / ((x - mean_L5) ** 2 + gamma5**2)
    return lorentzian1 + lorentzian2 + lorentzian3 + lorentzian4 + lorentzian5

def model_Voigt(theta,x):
    x_0, amplitude_L, fwhm_G, fwhm_L = theta
    a = Voigt1D(x_0, amplitude_L, fwhm_L, fwhm_G)
    return a(x)

def model_Voigt_Voigt(theta,x):
    x_00, amplitude_L0, fwhm_G0, fwhm_L0, x_0, amplitude_L, fwhm_G, fwhm_L = theta
    a = Voigt1D(x_00, amplitude_L0, fwhm_L0, fwhm_G0) + Voigt1D(x_0, amplitude_L, fwhm_L, fwhm_G)
    return a(x)

def model_Voigt_Gauss(theta,x):
    x_00, amplitude_L0, fwhm_G0, fwhm_L0, mean2, amplitude2, std2 = theta
    a = Voigt1D(x_00, amplitude_L0, fwhm_L0, fwhm_G0) + Gaussian1D(amplitude2, mean2, std2)
    return a(x)

def model_Voigt_Lorentz(theta,x):
    x_00, amplitude_L0, fwhm_G0, fwhm_L0, mean_L, amplitude_L, fwhm_L = theta
    a = Voigt1D(x_00, amplitude_L0, fwhm_L0, fwhm_G0) + Lorentz1D(amplitude_L, mean_L, fwhm_L)
    return a(x)

def model_Voigt_Gauss_Gauss(theta,x):
    x_00, amplitude_L0, fwhm_G0, fwhm_L0, mean, amplitude, std, mean2, amplitude2, std2 = theta
    a = Voigt1D(x_00, amplitude_L0, fwhm_L0, fwhm_G0) + Gaussian1D(amplitude, mean, std) + Gaussian1D(amplitude2, mean2, std2)
    return a(x)

def model_Voigt_Lorentz_Lorentz(theta,x):
    x_00, amplitude_L0, fwhm_G0, fwhm_L0, mean_L, amplitude_L, fwhm_L, mean_L2, amplitude_L2, fwhm_L2 = theta
    a = Voigt1D(x_00, amplitude_L0, fwhm_L0, fwhm_G0) + Lorentz1D(amplitude_L, mean_L, fwhm_L) + Lorentz1D(amplitude_L2, mean_L2, fwhm_L2)
    return a(x)

def model_Voigt_Voigt_Voigt(theta,x):
    x_00, amplitude_L0, fwhm_G0, fwhm_L0, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt, x_02, amplitude_Voigt2, fwhm_G2, fwhm_L_Voigt2 = theta
    a = Voigt1D(x_00, amplitude_L0, fwhm_L0, fwhm_G0) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G) + Voigt1D(x_02, amplitude_Voigt2, fwhm_L_Voigt2, fwhm_G2)
    return a(x)

def model_Voigt_Lorentz_Gauss(theta,x):
    x_00, amplitude_L0, fwhm_G0, fwhm_L0, mean_L, amplitude_L, fwhm_L, mean2, amplitude2, std2 = theta
    a = Voigt1D(x_00, amplitude_L0, fwhm_L0, fwhm_G0) + Lorentz1D(amplitude_L, mean_L, fwhm_L) + Gaussian1D(amplitude2, mean2, std2)
    return a(x)

def model_Voigt_Gauss_Lorentz(theta,x):
    x_00, amplitude_L0, fwhm_G0, fwhm_L0, mean2, amplitude2, std2, mean_L, amplitude_L, fwhm_L = theta
    a = Voigt1D(x_00, amplitude_L0, fwhm_L0, fwhm_G0) + Gaussian1D(amplitude2, mean2, std2) + Lorentz1D(amplitude_L, mean_L, fwhm_L)
    return a(x)

def model_Voigt_Voigt_Lorentz(theta,x):
    x_00, amplitude_L0, fwhm_G0, fwhm_L0, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt, mean_L, amplitude_L, fwhm_L = theta
    a = Voigt1D(x_00, amplitude_L0, fwhm_L0, fwhm_G0) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G) + Lorentz1D(amplitude_L, mean_L, fwhm_L)
    return a(x)

def model_Voigt_Lorentz_Voigt(theta,x):
    x_00, amplitude_L0, fwhm_G0, fwhm_L0, mean_L, amplitude_L, fwhm_L, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt = theta
    a = Voigt1D(x_00, amplitude_L0, fwhm_L0, fwhm_G0) + Lorentz1D(amplitude_L, mean_L, fwhm_L) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G)
    return a(x)

def model_Voigt_Gauss_Voigt(theta,x):
    x_00, amplitude_L0, fwhm_G0, fwhm_L0, mean2, amplitude2, std2, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt = theta
    a = Voigt1D(x_00, amplitude_L0, fwhm_L0, fwhm_G0) + Gaussian1D(amplitude2, mean2, std2) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G)
    return a(x)

def model_Voigt_Voigt_Gauss(theta,x):
    x_00, amplitude_L0, fwhm_G0, fwhm_L0, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt, mean2, amplitude2, std2 = theta
    a = Voigt1D(x_00, amplitude_L0, fwhm_L0, fwhm_G0) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G) + Gaussian1D(amplitude2, mean2, std2)
    return a(x)

def model_Voigt_Voigt_Voigt_Voigt(theta,x):
    x_00, amplitude_L0, fwhm_G0, fwhm_L0, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt, x_02, amplitude_Voigt2, fwhm_G2, fwhm_L_Voigt2, x_03, amplitude_Voigt3, fwhm_G3, fwhm_L_Voigt3 = theta
    a = Voigt1D(x_00, amplitude_L0, fwhm_L0, fwhm_G0) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G) + Voigt1D(x_02, amplitude_Voigt2, fwhm_L_Voigt2, fwhm_G2) + Voigt1D(x_03, amplitude_Voigt3, fwhm_L_Voigt3, fwhm_G3)
    return a(x)

def model_Voigt_Voigt_Voigt_Voigt_Voigt(theta,x):
    x_00, amplitude_L0, fwhm_G0, fwhm_L0, x_0, amplitude_Voigt, fwhm_G, fwhm_L_Voigt, x_02, amplitude_Voigt2, fwhm_G2, fwhm_L_Voigt2, x_03, amplitude_Voigt3, fwhm_G3, fwhm_L_Voigt3, x_04, amplitude_Voigt4, fwhm_G4, fwhm_L_Voigt4 = theta
    a = Voigt1D(x_00, amplitude_L0, fwhm_L0, fwhm_G0) + Voigt1D(x_0, amplitude_Voigt, fwhm_L_Voigt, fwhm_G) + Voigt1D(x_02, amplitude_Voigt2, fwhm_L_Voigt2, fwhm_G2) + Voigt1D(x_03, amplitude_Voigt3, fwhm_L_Voigt3, fwhm_G3) + Voigt1D(x_04, amplitude_Voigt4, fwhm_L_Voigt4, fwhm_G4)
    return a(x)