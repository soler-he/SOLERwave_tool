[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20644556-blue)](https://doi.org/10.5281/zenodo.20644556)

# SOLERwave

SOLERwave is a python tool developed to facilitate the investigation of large-scale coronal waves and Moreton waves.
It consists of the wave tracing tool itself and accompanying functions for file loading and handeling.

## Setup
SOLERwave requires a Python installation of 3.10 or newer. 

Further required packages are

* jupyter >= 1.1.1 
* numpy >= 2.2.3 
* astropy >=7.0.2
* sunpy >=6.1.1  => pip install sunpy[all]
* numba >= 0.61.0 
* moviepy >=2.1.2

the tool has been tested with version up to 

* jupyter 1.1.1 
* numpy 2.4.4
* astropy 7.2.0
* sunpy 7.1.2 
* numba 0.65.1 
* moviepy 2.2.1

Alternatively the Jupyter notebook "Install_packages.ipynb" can be run after the installation of Python and Jupyter. 
It will check the availability of the packages required and download any missing. Note, it will not check the version
of packages already installed. 

## Download Observations

To Download new Observations, follow the Load_new_event Jupyter Notebook. ** !! DO NOT EXECUTE the whole Notebook at once, but cell by cell !!**.
It contains multiple examples, you might not want to download and preprocess all of them at the same time. 

Currently the following Instruments are supported/ have been tested:
- SDO/AIA
- STEREO/SECHII
- SOLO/EUI/FSI
- GONG H-alpha (download is observatory specific)
- Kanzelhoehe H-alpha


## Wave Tracing Tool
In order to use the tool on the downloaded data, one of the pre-programmed Jupyter notebooks for 
events should be used. Currently one is included for the EUV observations based on a wave observed
on the 6th September 2011 by SDO/AIA in 211 Angstrom.

## Citation

If you use the tool, please cite "Baumgartner-Steinleitner et al. 2026 "**[ADS](https://ui.adsabs.harvard.edu/abs/2026arXiv260523599B)**, (doi = 10.48550/arXiv.2605.23599) 

Release V1.0 (SOLERwave tool with Single-Sector Analysis) of the tool can be also cited via zenodo: [10.5281/zenodo.20644556](https://doi.org/10.5281/zenodo.20644556) 

**Note that this git repo holds a newer, unreleased version!**

## Miscellaneous
This project has received funding from the European Union's Horizon Europe research and innovation program under grant agreement No 101134999.  As part of the grant agreement the tool will be made public.
