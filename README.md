[![DOI](https://zenodo.org/badge/807570243.svg)](https://doi.org/10.5281/zenodo.14051948)

This module is a work in progress.

This module was made to ensure the work from Azad Khan et al., 2024 [![DOI](https://zenodo.org/badge/807570243.svg)](https://doi.org/10.22541/au.173359504.42353416/v1) is open access and reproducible.

Currently this module contains functions to:
- importing, processing, and standardising earthquake source parameter data
- statistically analyse seismicity (source parameters e.g. time, location, and magnitude)
- selecting mainshocks using Fixed Window and Magnitude-Dependent Window aftershock exclusion methods
- identifying foreshocks using the Background Poisson, Gamma Inter-Event Time, and Empirical Seismicity Rate methods

Many functions require the renaming of earthquake catalog dataframe columns to: ID, MAGNITUDE, DATETIME, LON, LAT, DEPTH.

This module contains methods for a Magnitude-Dependent Window mainshock selection method, as described in Trugman and Ross (2019), and a Fixed Window method as described in Moutote et al. (2021). It also integrates code from van den Ende and Ampuero (2020) for a Gamma Inter-Event Time foreshock identification method, and from Herrmann and Marzocchi (2021) for estimating the magnitude of completeness using the Lilliefors test.

Implementations of functions from this module can be found at: [![DOI](https://zenodo.org/badge/807570243.svg)](https://doi.org/10.5281/zenodo.14055539).