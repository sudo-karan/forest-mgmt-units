"""
Module 1: Pre-processing & Masking.

Combines three global products into a forest suitability mask:
  - ESA WorldCover v200 : vegetation classes (trees/shrubland/grassland)
  - JRC GSW             : surface water occurrence
  - VIIRS DNB           : night lights, thresholded at 95th percentile
                          within a 5 km buffer of the ROI (auto-adaptive)

Returns an ee.Image with a single band 'Suitability_Mask' (1 = valid forest).
"""

import ee


def build_mask(config, verbose=True):
    """Build the suitability mask. Returns (valid_mask, mask_stats_dict)."""
    roi     = config['roi']
    roi_buf = config['roi_buffered']
    scale   = config['analysis_scale']

    # ---- 1. ESA WorldCover v200 ------------------------------------------
    worldcover = (
        ee.ImageCollection("ESA/WorldCover/v200")
        .filterBounds(roi)
        .first()
    )
    lc_map = worldcover.select('Map')
    is_veg = lc_map.eq(10).Or(lc_map.eq(20)).Or(lc_map.eq(30))
    if verbose:
        print("WorldCover: Trees(10), Shrubland(20), Grassland(30).")

    # ---- 2. JRC Global Surface Water -------------------------------------
    water = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select('occurrence')
        .clip(roi)
        .unmask(0)
    )
    is_dry = water.eq(0)
    if verbose:
        print("JRC Surface Water mask ready.")

    # ---- 3. VIIRS Night Lights — percentile-based threshold --------------
    viirs_mean = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
        .filterBounds(roi_buf)
        .filterDate(config['pheno_start'], config['pheno_end'])
        .select("avg_rad")
        .mean()
        .clip(roi_buf)
    )

    viirs_stats = viirs_mean.reduceRegion(
        reducer   = ee.Reducer.percentile([95]),
        geometry  = roi_buf,
        scale     = 1000,
        maxPixels = int(1e9),
    ).getInfo()

    viirs_threshold_val = list(viirs_stats.values())[0] if viirs_stats else 30.0
    viirs_threshold = ee.Number(max(float(viirs_threshold_val), 1.5))

    is_dark = viirs_mean.clip(roi).lt(viirs_threshold)

    viirs_thresh_value = viirs_threshold.getInfo()
    if verbose:
        print(f"VIIRS threshold (p95 of buffered ROI): "
              f"{viirs_thresh_value:.2f} nW/cm^2/sr")
        print("Pixels above this are flagged as urban/lit and excluded.")

    # ---- 4. Combine into suitability mask --------------------------------
    valid_mask = (
        is_veg.And(is_dry).And(is_dark)
        .rename("Suitability_Mask")
    )

    # ---- 5. Report survival rate -----------------------------------------
    total_px = ee.Image.constant(1).reduceRegion(
        reducer=ee.Reducer.count(), geometry=roi, scale=scale, maxPixels=int(1e9)
    ).get("constant").getInfo()

    valid_px = valid_mask.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=roi, scale=scale, maxPixels=int(1e9)
    ).get("Suitability_Mask").getInfo()

    survival_rate = 100 * valid_px / total_px if total_px else 0.0
    stats = {
        "total_px":          int(total_px),
        "valid_px":          int(valid_px),
        "survival_rate_pct": survival_rate,
        "viirs_threshold":   viirs_thresh_value,
    }

    if verbose:
        print(f"\n--- Masking Report ---")
        print(f"Total ROI pixels ({scale}m): {stats['total_px']}")
        print(f"Valid forest pixels        : {stats['valid_px']}")
        print(f"Survival rate              : {stats['survival_rate_pct']:.1f}%")

    return valid_mask, stats
