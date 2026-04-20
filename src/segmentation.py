"""
Segmentation stage: Modules 6a + 7a + 8.

Three responsibilities packed together because they form one pipeline step:

  6a: Assemble the high-res stack (S2 composites + S1 radar + canopy height).
      Only 10m-native features. Phenology (30m) is EXCLUDED here by design
      to avoid smearing stand boundaries.

  7a: Pixel-level Z-score normalization (different from stand-level norm
      in aggregation.py — see that file for the full discussion).

  8:  SNIC superpixel segmentation. seed_spacing auto-scales so we target
      ~N stands regardless of ROI size.

Exposed functions:
  - build_highres_stack(config, s2_stack, radar_features, canopy_h, valid_mask)
      -> (highres_stack_norm, hr_band_names, seed_spacing_px)
  - run_snic(config, highres_stack_norm, seed_spacing_px)
      -> snic_result (has 'clusters' band + '<feature>_mean' bands)
"""

import math
import ee


def build_highres_stack(config, s2_stack, radar_features, canopy_h, valid_mask,
                        verbose=True):
    """Assemble + pixel-normalize the high-res stack. Also compute SNIC seed spacing.

    Returns:
      highres_stack_norm : normalized multi-band image at 10m
      hr_band_names      : ee.List of band names (same order as input)
      seed_spacing_px    : int, pixels between SNIC seeds
    """
    roi   = config['roi']
    scale = config['analysis_scale']

    # ---- Assemble the high-res stack (10m only) -------------------------
    highres_stack_raw = (
        ee.Image.cat([s2_stack, radar_features, canopy_h])
        .updateMask(valid_mask)
        .toFloat()
    )

    hr_band_names = highres_stack_raw.bandNames()
    if verbose:
        n = hr_band_names.size().getInfo()
        print(f"High-res stack: {n} bands at {scale}m "
              f"(S2 composites + S1 + canopy height)")

    # ---- Pixel-level Z-score normalization ------------------------------
    if verbose:
        print("\nNormalization #1: pixel-level Z-score ...")

    hr_stats = highres_stack_raw.reduceRegion(
        reducer    = ee.Reducer.mean().combine(ee.Reducer.stdDev(),
                                               sharedInputs=True),
        geometry   = roi,
        scale      = scale,
        maxPixels  = int(1e9),
        bestEffort = True,
    )

    def _normalize(b):
        b    = ee.String(b)
        mean = ee.Number(hr_stats.get(b.cat("_mean")))
        std  = ee.Number(hr_stats.get(b.cat("_stdDev"))).max(1e-6)
        return highres_stack_raw.select(b).subtract(mean).divide(std).rename(b)

    highres_stack_norm = (
        ee.ImageCollection(hr_band_names.map(_normalize))
        .toBands()
        .rename(hr_band_names)
        .updateMask(valid_mask)
        .toFloat()
    )
    if verbose:
        print("High-res stack normalized (pixel Z-scores). Ready for SNIC.")

    # ---- Auto-scale SNIC seed spacing -----------------------------------
    target_stands = config['snic_target_stands']
    roi_area_m2   = roi.area(maxError=1).getInfo()
    roi_area_km2  = roi_area_m2 / 1e6

    stand_side_m    = math.sqrt(roi_area_m2 / target_stands)
    seed_spacing_px = max(2, round(stand_side_m / scale))

    if verbose:
        print(f"\nROI area : {roi_area_km2:.2f} km^2")
        print(f"Target   : {target_stands} stands")
        print(f"Stand side (target): {stand_side_m:.0f} m "
              f"-> seed spacing: {seed_spacing_px} px at {scale}m")

    return highres_stack_norm, hr_band_names, seed_spacing_px


def run_snic(config, highres_stack_norm, seed_spacing_px, verbose=True):
    """Run SNIC superpixel segmentation on the normalized high-res stack."""
    compactness     = config['snic_compactness']
    connectivity    = config['snic_connectivity']
    neighborhood_sz = config['snic_neighborhood']
    scale           = config['analysis_scale']

    if verbose:
        print(f"Running SNIC with seed_spacing={seed_spacing_px}px, "
              f"compactness={compactness} ...")

    snic_seeds = ee.Algorithms.Image.Segmentation.seedGrid(seed_spacing_px)

    snic = ee.Algorithms.Image.Segmentation.SNIC(
        image            = highres_stack_norm,
        compactness      = compactness,
        connectivity     = connectivity,
        neighborhoodSize = neighborhood_sz,
        seeds            = snic_seeds,
    ).reproject(crs="EPSG:4326", scale=scale)

    if verbose:
        print("SNIC superpixels generated.")

    return snic
