"""
Aggregation stage: Modules 6b + 9 + 10.

This is the big architectural step from the baseline notebook:

  6b: Build the FULL feature stack including 30m phenology (upsampled to 10m
      via bilinear) and topography. Upsampling is bookkeeping only — no new
      spatial information is created. It just puts everything on the same grid.

  9:  For each SNIC stand, compute the mean of every feature in the full stack
      (zonal statistics). This collapses pixel-level -> stand-level.

  10: NORMALIZATION #2 — stand-level Z-score.
      This is STATISTICALLY DIFFERENT from the pixel-level normalization in
      segmentation.py. Averaging dampens variance, so the mean/std of per-stand
      values differ from the mean/std of per-pixel values. Both normalizations
      are correct for their stage; neither is redundant.

Exposed functions:
  - build_full_stack(config, pheno_features, radar_features, canopy_h, static_features, valid_mask)
      -> (full_stack_raw, full_band_names)
  - aggregate_per_stand(config, full_stack_raw, snic_clusters, valid_mask, full_band_names)
      -> stand_means image
  - normalize_stands(config, stand_means, full_band_names, valid_mask)
      -> normalized_stands image
"""

import ee


def build_full_stack(config, pheno_features, radar_features, canopy_h,
                     static_features, valid_mask, verbose=True):
    """Assemble the FULL feature stack at `analysis_scale` (10m)."""
    scale = config['analysis_scale']

    # Upsample 30m -> 10m via bilinear. reproject() forces evaluation.
    pheno_10m = (
        pheno_features
        .reproject(crs="EPSG:4326", scale=scale)
        .resample("bilinear")
    )

    static_10m = (
        static_features
        .reproject(crs="EPSG:4326", scale=scale)
        .resample("bilinear")
    )

    full_stack_raw = (
        ee.Image.cat([
            pheno_10m,                                  # 6 bands, upsampled 30m
            radar_features,                             # 3 bands, 10m native
            canopy_h,                                   # 1 band,  10m native
            static_10m.select(["Elevation", "Slope"]),  # 2 bands, upsampled 30m
        ])
        .updateMask(valid_mask)
        .toFloat()
    )

    full_band_names = full_stack_raw.bandNames()

    if verbose:
        n_features = full_band_names.size().getInfo()
        print(f"Full feature stack: {n_features} features at {scale}m")
        print("Features:", full_band_names.getInfo())

    return full_stack_raw, full_band_names


def aggregate_per_stand(config, full_stack_raw, snic_clusters, valid_mask,
                        full_band_names, verbose=True):
    """Compute per-stand means via reduceConnectedComponents on SNIC cluster IDs.

    `snic_clusters` is the single-band 'clusters' image from segmentation.run_snic.
    """
    full_stack_with_clusters = full_stack_raw.addBands(snic_clusters)

    stand_means = (
        full_stack_with_clusters
        .reduceConnectedComponents(
            reducer   = ee.Reducer.mean(),
            labelBand = "clusters",
        )
        .rename(full_band_names)
        .updateMask(valid_mask)
        .toFloat()
    )

    if verbose:
        print("Per-stand aggregation complete (stand_means image).")

    return stand_means


def normalize_stands(config, stand_means, full_band_names, valid_mask,
                     verbose=True):
    """Stand-level Z-score normalization (Normalization #2)."""
    roi   = config['roi']
    scale = config['analysis_scale']

    if verbose:
        print("\nNormalization #2: stand-level Z-score ...")

    stand_stats = stand_means.reduceRegion(
        reducer    = ee.Reducer.mean().combine(ee.Reducer.stdDev(),
                                               sharedInputs=True),
        geometry   = roi,
        scale      = scale,
        maxPixels  = int(1e9),
        bestEffort = True,
    )

    def _normalize(b):
        b    = ee.String(b)
        mean = ee.Number(stand_stats.get(b.cat("_mean")))
        std  = ee.Number(stand_stats.get(b.cat("_stdDev"))).max(1e-6)
        return stand_means.select(b).subtract(mean).divide(std).rename(b)

    normalized_stands = (
        ee.ImageCollection(full_band_names.map(_normalize))
        .toBands()
        .rename(full_band_names)
        .updateMask(valid_mask)
        .toFloat()
    )

    if verbose:
        print("Stand-level normalization complete.")
        print("normalized_stands ready for K-means clustering.")

    return normalized_stands
