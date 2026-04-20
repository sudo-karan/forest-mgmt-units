"""
Module 4: Static Structural & Topographic Layers.

Produces 3 static feature bands:
  - Canopy_Height : Lang et al. 2023 global 10m canopy height (2020 snapshot)
  - Elevation     : NASADEM elevation (30m, resampled at aggregation stage)
  - Slope         : derived from NASADEM elevation
"""

import ee


def compute_features(config, valid_mask, verbose=True):
    """Return the 3-band static feature stack."""
    roi = config['roi']

    # ---- Lang et al. 2023 Canopy Height (10m, 2020 snapshot) -----------
    canopy_h = (
        ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1")
        .select("b1")
        .rename("Canopy_Height")
        .clip(roi)
        .unmask(0)
        .updateMask(valid_mask)
        .toFloat()
    )
    if verbose:
        print("Lang et al. 2023 canopy height loaded (10m).")

    # ---- NASADEM — Elevation, Slope ------------------------------------
    dem  = ee.Image("NASA/NASADEM_HGT/001").clip(roi)
    elev = dem.select("elevation").rename("Elevation").updateMask(valid_mask).toFloat()
    slp  = ee.Terrain.slope(elev).rename("Slope").updateMask(valid_mask).toFloat()
    if verbose:
        print("NASADEM elevation + slope loaded (30m).")

    static_features = ee.Image.cat([canopy_h, elev, slp])

    if verbose:
        print("\nStatic features (3 bands): Canopy_Height, Elevation, Slope.")

    return static_features, {}
