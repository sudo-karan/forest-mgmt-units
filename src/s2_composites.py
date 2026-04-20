"""
Module 5: Sentinel-2 Seasonal Composites.

Produces a 24-band (4 seasons * 6 bands) composite stack at 10m.
Used ONLY as SNIC input (Module 8). NOT used for phenology (HLS does that).

Bands per season: B2, B3, B4, B8, B11, B12.
Seasons are defined in config['s2_seasons'] as [(name, start_MMDD, end_MMDD), ...].

Cloud masking uses SCL (the v1 bug fix — v1 had this defined but not applied).
"""

import ee


def _mask_s2_scl(img):
    """Per-pixel SCL cloud/shadow mask.

    SCL class codes:
      1 = saturated/defective  2 = dark shadow  3 = cloud shadow
      4 = vegetation           5 = bare soil    6 = water
      7 = unclassified         8 = cloud medium prob
      9 = cloud high prob     10 = thin cirrus 11 = snow/ice

    Drop: 3, 8, 9, 10. Keep the rest.
    """
    scl = img.select("SCL")
    valid = (
        scl.neq(3)
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
    )
    return img.updateMask(valid)


def compute_features(config, valid_mask, verbose=True):
    """Build the per-season S2 composite stack."""
    roi       = config['roi']
    s2_year   = config['s2_year']
    seasons   = config['s2_seasons']
    cloud_pct = config['cloud_pct_max']

    composites     = []
    output_bands   = []

    for (season_name, start_mmdd, end_mmdd) in seasons:
        # Winter wraps the year boundary (Nov -> Jan)
        start_yr = s2_year - 1 if season_name == "winter" else s2_year
        end_yr   = s2_year     if season_name == "winter" else s2_year

        start_date = f"{start_yr}-{start_mmdd}"
        end_date   = f"{end_yr}-{end_mmdd}"

        composite = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
            .map(_mask_s2_scl)
            .select(["B2", "B3", "B4", "B8", "B11", "B12"])
            .median()
            .clip(roi)
        )

        for b in ["B2", "B3", "B4", "B8", "B11", "B12"]:
            output_bands.append(f"{season_name}_{b}")

        composites.append(composite)
        if verbose:
            print(f"  {season_name}: {start_date} -> {end_date}")

    s2_stack = (
        ee.Image.cat(composites)
        .rename(output_bands)
        .updateMask(valid_mask)
        .toFloat()
    )

    if verbose:
        print(f"\nS2 seasonal composite stack: {len(output_bands)} bands at 10m.")

    return s2_stack, {"band_names": output_bands}
