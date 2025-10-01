<p align="center" width="100%">
 <img width="100%" height="100%" src="/docs/source/logo.png">
</p> 

# easyspec

The easiest way to do long-slit spectroscopy.

`easyspec` is a tool designed to streamline long-slit spectroscopy, offering an intuitive framework for reducing, extracting, and analyzing astrophysical spectra.

If you'd like to support the maintenance of `easyspec` (or [easyfermi](https://github.com/ranieremenezes/easyfermi)), consider buying us a coffee:

<a href="https://www.buymeacoffee.com/easyfermi" target="_blank"><img src="https://github.com/ranieremenezes/ranieremenezes/blob/main/bmc-button.png" alt="Buy Me A Coffee" style="height: 58px !important;width: 208px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

### Current release 
[![DOI](https://zenodo.org/badge/775611173.svg)](https://doi.org/10.5281/zenodo.15732788)

[![CI](https://github.com/ranieremenezes/easyspec/actions/workflows/ci.yml/badge.svg)](https://github.com/ranieremenezes/easyspec/actions/workflows/ci.yml)

# Requirements

- Linux OS / Mac OS / Windows

### Optional but recommended:
- [Miniconda 3](https://docs.conda.io/projects/miniconda/en/latest/),
  [Anaconda 3](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Miniforge](https://github.com/conda-forge/miniforge) (recommended) distribution.

# Installation

### Mamba-based installation 

The following instructions assume an installation of `conda` or `mamba` (i.e. a faster version of `conda`).

In the terminal, run (substitute *mamba* by *conda* if it is the case for you):
<pre><code>mamba create --name easyspec -c conda-forge python=3.9 "scipy=1.9.1" "astropy=5.1" "emcee=3.1.4" "corner=2.2.2" "ccdproc=2.4.0" "matplotlib=3.5.2" "numpy=1.21.5" "dust_extinction=1.2" "notebook=6.4.4" "tqdm=4.64.1" </code></pre>

This will create the virtual environment and install all dependencies. Then activate the environment and install _easyspec_:
<pre><code>mamba activate easyspec
pip install easyspec</code></pre>

### Direct installation

If you don't have mamba or conda, you can install easyspec and its dependecies directly with pip:

<pre><code>pip install scipy==1.9.1 astropy==5.1 emcee==3.1.4 corner==2.2.2 ccdproc==2.4.0 matplotlib==3.5.2 numpy==1.21.5 dust_extinction==1.2 notebook==6.4.4 tqdm==4.64.1</code></pre>


# Upgrading

You can check your currently installed version of `easyspec` with _pip show_:
<pre><code>pip show easyspec</code></pre>
   
If it is not the latest version, you can upgrade your installation by running the following command in the _easyspec_ environment:
<pre><code>pip install easyspec --upgrade --no-deps</code></pre>

# Uninstalling

In the terminal, run:
<pre><code>mamba deactivate</code></pre>
<pre><code>mamba env remove --name easyspec</code></pre>

# Tutorials and Documentation

The instructions on how to use `easyspec` can be found in the GitHub directory "Tutorials".

The main tutorials are:
* [Image\_cleaning\_easyspec](https://github.com/ranieremenezes/easyspec/blob/main/tutorial/Image_cleaning_easyspec.ipynb): here we will guide you on how to reduce raw long-slit spectroscopic data, i.e., we will show you how to trim, debias, dedark, flatten, remove the cosmic rays, and stack the data.
* [spectroscopy\_tracing\_easyspec](https://github.com/ranieremenezes/easyspec/blob/main/tutorial/spectroscopy_tracing_easyspec.ipynb): here we will show you how to extract your spectra and calibrate them in wavelegnth and flux.
* [spectral\_analysis\_easyspec](https://github.com/ranieremenezes/easyspec/blob/main/tutorial/spectral_analysis_easyspec.ipynb): finally, we show you how to fit a model to each line of your spectrum with a MCMC approach and recover physical quantities such as redshift, dispersion velocity, FWHM, line flux and many more.

For a more advanced analysis, we recommend the tutorial
* [spectral\_analysis\_fitting\_a\_line\_with\_two\_Gaussians](https://github.com/ranieremenezes/easyspec/blob/main/tutorial/spectral_analysis_fitting_a_line_with_two_Gaussians.ipynb): here we fit the Hbeta line with two Gaussian components and also explore the MCMC posterior distributions.


The documentation of `easyspec` can be found in the header of all the functions, and at [this link](https://easyspec.readthedocs.io/en/latest/index.html).


# Acknowledgments

To acknowledge `easyspec` in a publication, please cite de Menezes et al., 2025 (https://iopscience.iop.org/article/10.3847/1538-3881/adf220) and the DOI in [Zenodo](https://zenodo.org/records/17211465).

