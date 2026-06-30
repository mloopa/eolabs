# SAR Lab 7 Report: Xinmo Landslide Analysis

Event: Xinmo landslide, Sichuan Province, China

Event date: 2017-06-24

AOI: 103.62 to 103.68 E, 32.04 to 32.09 N

LiCSAR frame: 062D_05831_131313

## 1. Introduction and Study Objective

This report recreates the pre-failure deformation analysis requested in SAR Lab 7
using public COMET-LiCSAR interferograms and MintPy SBAS processing. It then
compares event-spanning LiCSAR coherence, Sentinel-1 GRD VV backscatter change,
and Sentinel-2 optical changes for mapping the 2017 Xinmo landslide.

The goal is not to exactly reproduce the supplied PS-based study. Instead, the
goal is to evaluate what can be recovered from publicly available pre-processed
LiCSAR interferograms and open Sentinel data.

## 2. Data and Study Area

The study area is centered near 103.6506 E, 32.0661 N. The working AOI is
103.62, 32.04, 103.68, 32.09. The SAR time-series analysis uses LiCSAR frame
062D_05831_131313. Event mapping uses one event-spanning LiCSAR coherence pair,
Sentinel-1 GRD VV images before and after the failure, and Sentinel-2 imagery
before and after the failure.

## 3. LiCSAR and MintPy SBAS Methodology

Pre-event LiCSAR unwrapped phase and coherence products were downloaded for
interferograms before 2017-06-24. A 72-day temporal-baseline selection produced
a disconnected network, so the maximum temporal baseline was expanded to 108
days. The final MintPy input stack contains 42 acquisitions and 127
interferograms from 2014-10-09 to 2017-06-07. The perpendicular baseline range
is approximately -178 m to +189 m.

The LiCSAR products were converted to MintPy HDF5 inputs, subset to the AOI, and
processed with `smallbaselineApp.py`. Network inversion used coherence weighting.

## 4. Reference-Point Selection

The automatic reference point was used for the first run. The final run used a
manual local reference point at Y/X = 28,31, lat/lon = 32.06233, 103.65083. This
point was selected because it is close to the landslide but outside the
interpreted affected area and has high temporal coherence in the MintPy result.

The manual reference reduces sensitivity to long-wavelength atmospheric,
seasonal, or topographic differences between a distant automatic reference and
the landslide area. The tradeoff is that all displacement is interpreted
relative to this selected pixel.

## 5. Pre-Failure Time-Series Results

The representative landslide pixel at Y/X = 29,29, lat/lon = 32.06133,
103.64883 shows a broad negative LOS trend of about -1.07 +/- 0.08 cm/year.
This supports pre-event motion at the landslide, but the available stack does
not show a clear late-stage acceleration.

The key limitation is acquisition timing. The MintPy stack ends on 2017-06-07,
which is 17 days before the 2017-06-24 failure. The assignment notes that the
last few points in the supplied PS-based paper's figure were not processed by
COMET-LiCSAR. Therefore, this SBAS recreation can support pre-event motion but
cannot independently confirm the final acceleration reported by the published
PS-based analysis.

## 6. Coherence Analysis

The selected event-spanning LiCSAR pair is 20170607_20170725, with a 48-day
temporal baseline. The mean coherence in the landslide-centered sample is
0.046. The mean coherence at the manual-reference sample is
0.053. Both local samples are very low coherence, so this pair does
not cleanly separate the landslide from nearby low-coherence terrain.

Coherence loss is therefore weak supporting evidence in this workflow. It is
not sufficient as a standalone landslide boundary because the mountainous,
vegetated AOI has broad temporal and geometric decorrelation.

## 7. Sentinel-1 VV Analysis

The Sentinel-1 GRD comparison uses descending relative orbit
62. The pre-event acquisition is
2017-06-19, and the post-event acquisition is
2017-07-13. VV change is computed as post-event VV minus
pre-event VV in dB.

The landslide-centered sample has mean VV change of -0.49 dB. The
stable comparison sample has mean VV change of -0.89 dB. The local
contrast is small, and the map is affected by speckle, moisture, and terrain
geometry. VV change is therefore weak for mapping the landslide in this case.

## 8. Sentinel-2 and Spectral-Index Analysis

Sentinel-2 surface-reflectance imagery did not provide a usable 2017 pre/post
pair for this AOI, so the workflow used `COPERNICUS/S2_HARMONIZED` Level-1C
top-of-atmosphere imagery. The pre-event image date is
2017-02-19; the post-event image date is
2017-08-13. Both have AOI clear fraction 1.000 after the
QA60 cloud/cirrus mask.

The final affected-area mask uses a post-event optical-scar rule:
post brightness >= 0.115, post NDVI <= 0.0, and NDVI change <= -0.10. The
largest connected component inside a 1.8 km landslide-centered search radius is
retained. The resulting affected area is 80.04
ha, or 0.8004 km2.

Mean NDVI change in the affected area is -0.229. The
stable reference sample has mean NDVI change of 0.333.
This contrast supports vegetation loss or burial by fresh debris in the mapped
scar. BSI is less useful for this pair: the affected area has mean BSI change
of -0.108, so a simple positive-BSI-change threshold
would under-detect the scar.

## 9. Comparison and Discussion

SBAS is useful for pre-failure deformation, but not for mapping the event
extent. The inspected landslide pixel shows a negative pre-event LOS trend, yet
the available LiCSAR stack ends before the final 17 days.

Sentinel-2 true color plus NDVI decrease gives the clearest landslide extent.
The bright post-event scar is visually distinct, and the refined optical mask
captures the source and runout area. Coherence and Sentinel-1 VV change are
weaker in this workflow. The coherence pair has very low coherence in both the
landslide and manual-reference samples, while the Sentinel-1 VV change has only
small local contrast.

## 10. Conclusions

The main conclusions are:

- MintPy SBAS supports pre-event motion at the landslide, with an inspected
  landslide-pixel trend of about -1.07 +/- 0.08 cm/year.
- The manually selected reference point gives a local relative basis for the
  time series and is preferable to relying only on an automatic reference.
- The available LiCSAR stack does not include the final days before failure, so
  the late acceleration from the supplied PS-based study cannot be reproduced.
- Coherence loss is weak for this event-spanning pair because both local samples
  have very low coherence.
- Sentinel-1 VV change is also weak because local backscatter contrast is small.
- Sentinel-2 true color plus NDVI decrease provides the clearest spatial map,
  with a refined affected-area estimate of 80.04
  ha.
- NDVI change supports vegetation loss or burial; BSI change is not a reliable
  positive-threshold discriminator for this image pair.

## 11. References

- SAR Lab 7 assignment handout, `SAR_Lab_7.pdf`.
- Supplied Xinmo landslide PS-based study: DOI 10.1007/s10346-017-0915-7.
- COMET-LiCSAR public interferogram products, frame 062D_05831_131313.
- Yunjun, Z., Fattahi, H., and Amelung, F. MintPy: a Python package for
  InSAR time series analysis.
- ESA Copernicus Sentinel-1 GRD imagery.
- ESA Copernicus Sentinel-2 imagery.
- Google Earth Engine cloud-processing platform.
