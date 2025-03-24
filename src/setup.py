from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.0.0.7'
DESCRIPTION = 'The easiest way to do long-slit spectroscopy'
LONG_DESCRIPTION = 'A package that enables the reduction, extraction, and analysis of long-slit astrophysical spectra.'

# Setting up
setup(
    name="easyspec",
    version=VERSION,
    author="Raniere de Menezes",
    author_email="<easyfermi@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    #install_requires=['emcee', 'astropy'],
    keywords=['python', 'optical spectra', 'long-slit', 'spectroscopy', 'infrared', 'infrared spectroscopy', 'ultraviolet'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
    ],
    include_package_data=True,
    package_data={'': ['analysis/lines/astro_lines.dat','extraction/airmass/*.txt','extraction/standards/blackbody/*.*',
                       'extraction/standards/bstdscal/*.*','extraction/standards/ctiocal/*.*','extraction/standards/ctionewcal/*.*',
                       'extraction/standards/iidscal/*.*','extraction/standards/irscal/*.*','extraction/standards/oke1990/*.*',
                       'extraction/standards/redcal/*.*','extraction/standards/spec16cal/*.*','extraction/standards/spec50cal/*.*',
                       'extraction/standards/spechayescal/*.*']},
)
