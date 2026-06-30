# Phase 10 Interpretation and Method Comparison

## SBAS deformation signal

The manually referenced MintPy SBAS result shows a broad negative pre-failure
LOS displacement trend at the inspected landslide pixel. The representative
time-series point at Y/X = 29,29, lat/lon = 32.06133, 103.64883 has a fitted
trend of about -1.07 +/- 0.08 cm/year. The signal is meaningful as evidence of
pre-event motion, but the current stack does not show a clear late-stage
acceleration by itself.

The main reason is temporal coverage. The LiCSAR/MintPy stack ends on
2017-06-07, while the Xinmo landslide occurred on 2017-06-24. The assignment
notes that the last few points in the published PS-based figure were not
processed by COMET-LiCSAR, so this SBAS recreation cannot reproduce the final
pre-failure acceleration reported in the paper.

## Reference point effect

The manual reference point is Y/X = 28,31, lat/lon = 32.06233, 103.65083. It is
near the landslide but outside the interpreted affected area, has high temporal
coherence, and was stable before rereferencing. This is preferable to relying
only on MintPy's automatic reference because the local reference reduces
long-wavelength seasonal, atmospheric, and topographic differences between the
reference and the landslide area.

The tradeoff is that all displacement is relative to this chosen point. After
rereferencing, the reference pixel is fixed to zero velocity, so the landslide
time series should be interpreted as relative LOS motion, not an absolute
ground-motion measurement.

## Published PS-based study comparison

The published study linked in the assignment used a PS/SqueeSAR-style time
series approach, while this lab uses SBAS on publicly available LiCSAR
interferograms. The qualitative agreement is that both approaches point to
pre-event deformation at the Xinmo landslide. The difference is temporal
resolution and late-event sensitivity: the published analysis can discuss the
final pre-failure behavior, while the LiCSAR stack available here stops before
the final 17 days.

Therefore, the correct comparison is limited: this SBAS result supports the
existence of pre-event motion, but it cannot independently confirm the
published late acceleration immediately before failure.

## Mapping comparison

| Method | Main signal | Result | Interpretation |
| --- | --- | --- | --- |
| MintPy SBAS | Pre-failure LOS motion | About -1.07 +/- 0.08 cm/year at inspected landslide pixel | Useful for pre-event deformation, not event extent |
| LiCSAR coherence | Event disturbance | Mean coherence 0.046 in landslide sample vs 0.053 at the manual reference sample | Weak separator because both local samples are very low coherence |
| Sentinel-1 VV | Backscatter change | Mean VV change -0.49 dB in landslide sample vs -0.89 dB in stable sample | Weak contrast; worse than coherence here |
| Sentinel-2 true color | Visible optical scar | Refined affected area 80.04 ha / 0.8004 km2 | Clearest spatial mapping evidence |
| NDVI change | Vegetation loss | Affected mean -0.229 vs stable sample +0.333 | Strong support for vegetation removal or burial |
| BSI change | Bare soil/debris exposure | Affected mean -0.108 vs stable sample -0.221 | Not reliable as a simple positive-change threshold for this pair |

## Best method for mapping the landslide

Sentinel-2 true color combined with NDVI decrease provides the clearest map of
the landslide extent. The post-event scar is visually distinct, and the refined
mask follows the source and runout area better than the initial strict NDVI plus
BSI rule. Coherence loss is spatially ambiguous because both the landslide and
manual-reference samples have very low coherence in the event-spanning pair.
Sentinel-1 VV change is also weak because the landslide and stable samples have
similar noisy backscatter changes.

## Main limitations

- Sparse LiCSAR acquisitions and missing late pre-failure points.
- Stack ending on 2017-06-07, before the 2017-06-24 failure.
- Atmospheric and seasonal phase signals in SBAS.
- Reference-point sensitivity in relative displacement products.
- Low SAR coherence in vegetated, steep terrain.
- Layover, shadow, and terrain geometry effects.
- Sentinel-1 speckle and moisture sensitivity.
- Sentinel-2 cloud, illumination, and seasonal vegetation effects.
- Different acquisition dates and spatial resolutions across methods.
