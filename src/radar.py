"""
Module 3: Sentinel-1 Radar Features.

Produces 3 radar feature bands at 10m:
  - S1_Ratio_P90    : 90th-percentile VH/VV ratio (peak canopy/volume)
  - S1_Ratio_P10    : 10th-percentile VH/VV ratio (peak ground/stem)
  - S1_Ratio_StdDev : stdDev of VH/VV ratio (structural plasticity)

Uses VH - VV in dB space (cancels incidence-angle effects that dominate raw VH).
"""

import ee


def compute_features(config, valid_mask, verbose=True):
    """Compute the 3-band Sentinel-1 radar feature stack."""
    roi         = config['roi']
    radar_start = config['radar_start']
    radar_end   = config['radar_end']

    s1_archive = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(roi)
        .filterDate(radar_start, radar_end)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
    )

    n_s1 = s1_archive.size().getInfo()
    if verbose:
        print(f"S1 observations loaded: {n_s1} "
              f"(window: {radar_start} -> {radar_end})")

    # VH - VV in dB space (ratio in linear units)
    def add_ratio(img):
        ratio = img.select("VH").subtract(img.select("VV")).rename("VH_VV_Ratio")
        return img.addBands(ratio)

    s1_with_ratio = s1_archive.map(add_ratio)

    # Three percentile-derived features
    s1_percentiles = (
        s1_with_ratio.select("VH_VV_Ratio")
        .reduce(ee.Reducer.percentile([10, 90]))
    )

    s1_max_ratio = s1_percentiles.select("VH_VV_Ratio_p90").rename("S1_Ratio_P90")
    s1_min_ratio = s1_percentiles.select("VH_VV_Ratio_p10").rename("S1_Ratio_P10")
    s1_variance  = (
        s1_with_ratio.select("VH_VV_Ratio")
        .reduce(ee.Reducer.stdDev())
        .rename("S1_Ratio_StdDev")
    )

    radar_features = (
        ee.Image.cat([s1_max_ratio, s1_min_ratio, s1_variance])
        .updateMask(valid_mask)
        .toFloat()
    )

    if verbose:
        print("Radar features (all at 10m, masked):")
        print("  S1_Ratio_P90    - peak canopy/volume (leaf-on)")
        print("  S1_Ratio_P10    - peak ground/stem (leaf-off)")
        print("  S1_Ratio_StdDev - structural plasticity")

    meta = {"n_s1_obs": n_s1}
    return radar_features, meta
