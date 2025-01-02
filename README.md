# easyspec

<p align="left" width="100%">
 <img width="100%" height="250" src="https://github.com/clodoN1109/easyFermi/assets/104923248/d1a25a66-0fc6-4484-93fa-aaa8717f4276">
 <h1>easyfermi</h1>
</p>

<a href="https://easyfermi.readthedocs.io/en/latest/">
 
![easyfermiDocsBadge](https://img.shields.io/badge/docs-easyfermi-green?style=for-the-badge&logo=googledocs&logoColor=white&labelColor=gray&color=blue)

</a>


The easiest way to do long-slit spectroscopy.


`easyspec` is a tool designed to streamline long-slit spectroscopy, offering an intuitive framework for reducing, extracting, and analyzing astrophysical spectra.

"If you'd like to support the maintenance of `easyspec` (or <a href="https://github.com/ranieremenezes/easyfermi">easyfermi), consider buying us a coffee:"

<a href="https://www.buymeacoffee.com/easyfermi" target="_blank"><img src="https://github.com/ranieremenezes/ranieremenezes/blob/main/bmc-button.png" alt="Buy Me A Coffee" style="height: 58px !important;width: 208px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>


# Requirements

- Linux OS / Mac OS / Windows
- [Miniconda 3](https://docs.conda.io/projects/miniconda/en/latest/),
  [Anaconda 3](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) or [Miniforge](https://github.com/conda-forge/miniforge) (recommended) distribution.

# Installation

The following instructions assume an installation of `conda` or `mamba` (i.e. a faster version of `conda`).

### Mamba-based installation 

In the terminal, run (substitute *mamba* by *conda* if it is the case for you):
<pre><code>mamba create --name easyspec -c conda-forge python=3.9 "scipy=1.9.1" "astropy=5.1" "emcee=3.1.4" "corner=2.2.2" "matplotlib=3.5.2" "numpy=1.21.5" "dust_extinction=1.2" </code></pre>

This will create the virtual environment and install all dependencies. Then activate the environment and install _easyspec_:
<pre><code>mamba activate easyspec
pip install easyspec</code></pre>

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
* Image\_cleaning\_easyspec: here we will guide you on how to reduce raw long-slit spectroscopic data, i.e., we will show you how to trim, debias, dedark, flatten, remove the cosmic rays, and stack the data.
* spectroscopy\_tracing\_easyspec.ipynb: here we will show you how to extract your spectra and calibrate them in wavelegnth and flux.
* spectral\_analysis\_easyspec: finally, we show you how to fit a model to each line of your spectrum with a MCMC approach and recover physical quantities such as redshift, dispersion velocity, FWHM, line flux and many more.


The documentation of `easyspec` can be found at TBD.


# Acknowledgments

To acknowledge `easyspec` in a publication, please cite de Menezes, R (2025) (work in progress).

