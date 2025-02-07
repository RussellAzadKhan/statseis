[![DOI](https://zenodo.org/badge/807570243.svg)](https://doi.org/10.5281/zenodo.14051948)

https://pypi.org/project/statseis/  

This module is a work in progress.
This module contains functions to statistically analyse seismicity (source parameters e.g. time, location, and magnitude).
Package for importing, processing, and standardising earthquake source parameter data;
selecting mainshocks using the FET, MDET, and DDET methods;
identifying foreshocks using the BP, G-IET, and ESR methods.
Many functions require the renaming of earthquake catalog dataframe columns to: ID, MAGNITUDE, DATETIME, LON, LAT, DEPTH.

This module contains methods for a Magnitude-Dependent exclusion threshold mainshock selection method, as described in Trugman and Ross (2019), and a Fixed Exclusion Threshold method as described in Moutote et al. (2021). It also integrates code from van den Ende and Ampuero (2020) for a Gamma Inter-Event Time foreshock identification method, and from Herrmann and Marzocchi (2021) for estimating the magnitude of completeness using the Lilliefors test.