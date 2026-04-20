"""
Module 2: HLS Phenology (dual harmonic + PCA).

Produces 6 phenology feature bands at 30m:
  - Pheno_PC1..Pheno_PC5 : top 5 PCs of the 24-sample fitted curve
  - Pheno_Trend          : beta1 (linear trend coefficient)

The curve is fit as:
  NIRv(t) = b0 + b1*t + b2*cos(2*pi*t) + b3*sin(2*pi*t)
                       + b4*cos(4*pi*t) + b5*sin(4*pi*t)

NIRv is sampled from HLS L30 + S30 merged (NASA Harmonized Landsat-Sentinel).

Exposed functions:
  - compute_coefficients(config, valid_mask)
      -> (hls_coeffs_image, hls_merged_collection, n_obs)
      The 6 raw harmonic coefficients. Useful for diagnostics (Cell 4b).
  - compute_features(config, valid_mask)
      -> (pheno_features, meta_dict)
      The 6-band phenology feature image (5 PCs + trend) for downstream use.
"""

import math
import ee


# ---- HLS helpers (module-level so they can be reused in diagnostics) ----

def _mask_hls_clouds(img):
    """Keep pixels where Fmask bits 1 (shadow) and 5 (cloud) are both 0."""
    fmask = img.select("Fmask")
    no_cloud  = fmask.bitwiseAnd(1 << 5).eq(0)
    no_shadow = fmask.bitwiseAnd(1 << 1).eq(0)
    return img.updateMask(no_cloud.And(no_shadow))


def _rename_s30(img):
    """Rename S30 bands to match L30 naming.
    L30: B4=Red, B5=NIR. S30: B4=Red, B8A=NIR(narrow) -> rename to B5.
    Select only B4/B5 to avoid clash with the existing S30 B5 (SWIR).
    """
    return img.select(['B4', 'B8A']).rename(['B4', 'B5'])


def _add_nirv_and_time(img, pheno_start):
    """Add NIRv plus the 6 regressor bands needed for dual harmonic fit."""
    nir  = img.select("B5")
    red  = img.select("B4")
    ndvi = nir.subtract(red).divide(nir.add(red))
    nirv = ndvi.multiply(nir).rename("NIRv")

    t = img.date().difference(ee.Date(pheno_start), "year")
    t_img = ee.Image.constant(t).toFloat()

    two_pi_t  = t_img.multiply(2 * math.pi)
    four_pi_t = t_img.multiply(4 * math.pi)

    return img.addBands([
        nirv,
        t_img.rename("t"),
        ee.Image.constant(1).rename("constant"),
        t_img.rename("trend"),
        two_pi_t.cos().rename("cos1"),
        two_pi_t.sin().rename("sin1"),
        four_pi_t.cos().rename("cos2"),
        four_pi_t.sin().rename("sin2"),
    ])


# ---- Main entry points --------------------------------------------------

def compute_coefficients(config, valid_mask, verbose=True):
    """Fit dual harmonic regression on HLS NIRv. Returns (coeffs, merged, n_obs)."""
    roi         = config['roi']
    pheno_start = config['pheno_start']
    pheno_end   = config['pheno_end']

    # Closures to bake in pheno_start
    def _with_nirv(img):
        return _add_nirv_and_time(img, pheno_start)

    # ---- Load & merge HLS L30 + S30 ------------------------------------
    if verbose:
        print("Loading HLS L30 (Landsat) ...")
    hls_l30 = (
        ee.ImageCollection("NASA/HLS/HLSL30/v002")
        .filterBounds(roi)
        .filterDate(pheno_start, pheno_end)
        .map(_mask_hls_clouds)
        .map(_with_nirv)
    )

    if verbose:
        print("Loading HLS S30 (Sentinel-2) ...")
    hls_s30 = (
        ee.ImageCollection("NASA/HLS/HLSS30/v002")
        .filterBounds(roi)
        .filterDate(pheno_start, pheno_end)
        .map(_mask_hls_clouds)
        .map(_rename_s30)
        .map(_with_nirv)
    )

    hls_merged = hls_l30.merge(hls_s30)
    n_obs = hls_merged.size().getInfo()
    if verbose:
        print(f"\nTotal HLS observations (cloud-masked, merged): {n_obs}")

    # ---- Fit dual harmonic regression ----------------------------------
    if verbose:
        print("\nFitting dual harmonic regression on NIRv time series ...")

    regressors = ["constant", "trend", "cos1", "sin1", "cos2", "sin2"]
    regression = (
        hls_merged
        .select(regressors + ["NIRv"])
        .reduce(ee.Reducer.linearRegression(numX=6, numY=1))
    )

    coeffs = (
        regression.select("coefficients")
        .arrayProject([0])
        .arrayFlatten([["beta0", "beta1", "beta2", "beta3", "beta4", "beta5"]])
    )

    hls_coeffs = coeffs.updateMask(
        valid_mask.reproject("EPSG:4326", None, 30)
    )

    if verbose:
        print("Dual harmonic coefficients computed:")
        print("  beta0 = intercept (NIRv baseline)")
        print("  beta1 = linear trend (greening/browning over 10 years)")
        print("  beta2,beta3 = annual cosine/sine")
        print("  beta4,beta5 = semi-annual cosine/sine (monsoon bimodality)")

    return hls_coeffs, hls_merged, n_obs


def compute_features(config, valid_mask, verbose=True):
    """Run the full Module 2: coefficients -> 24-sample curve -> PCA -> features.

    Returns:
      pheno_features : ee.Image, 6 bands at 30m
                       (Pheno_PC1..PC5 + Pheno_Trend)
      meta : dict with n_obs, eigenvectors used, etc.
    """
    roi   = config['roi']
    N     = config['n_curve_samples']
    N_PCA = config['n_pca_components']

    # Step 0: coefficients
    hls_coeffs, _, n_obs = compute_coefficients(config, valid_mask, verbose=verbose)

    # ---- Step 1: sample fitted curve at N evenly-spaced points ---------
    # Seasonal component only (no beta1 trend); trend is added separately.
    sampled_bands = []
    for k in range(N):
        t_frac = k / N
        t_ann  = 2 * math.pi * t_frac
        t_semi = 4 * math.pi * t_frac

        sample = (
            hls_coeffs.select("beta0")
            .add(hls_coeffs.select("beta2").multiply(math.cos(t_ann)))
            .add(hls_coeffs.select("beta3").multiply(math.sin(t_ann)))
            .add(hls_coeffs.select("beta4").multiply(math.cos(t_semi)))
            .add(hls_coeffs.select("beta5").multiply(math.sin(t_semi)))
            .rename(f"NIRv_t{k:02d}")
        )
        sampled_bands.append(sample)

    curve_24 = (
        ee.Image.cat(sampled_bands)
        .updateMask(valid_mask.reproject("EPSG:4326", None, 30))
    )
    if verbose:
        print(f"Sampled curve: {N} bands at 30m.")

    # ---- Step 2: PCA — 24 bands -> N_PCA components --------------------
    if verbose:
        print(f"Computing PCA: {N} bands -> {N_PCA} components ...")

    band_names_24 = curve_24.bandNames()
    means_dict = curve_24.reduceRegion(
        reducer    = ee.Reducer.mean(),
        geometry   = roi,
        scale      = 30,
        maxPixels  = int(1e9),
        bestEffort = True,
    )
    means_img   = ee.Image.constant(means_dict.values(band_names_24))
    centered_24 = curve_24.subtract(means_img)

    array_img = centered_24.toArray()
    covar_dict = array_img.reduceRegion(
        reducer    = ee.Reducer.centeredCovariance(),
        geometry   = roi,
        scale      = 30,
        maxPixels  = int(1e9),
        bestEffort = True,
    )
    covar_arr = ee.Array(covar_dict.get("array"))

    eigens      = covar_arr.eigen()
    top_vectors = eigens.slice(1, 1).slice(0, 0, N_PCA)

    eigvec_list = top_vectors.toList().getInfo()  # small, fine to fetch
    band_list   = centered_24.bandNames().getInfo()

    pc_images = []
    for i in range(N_PCA):
        weights = eigvec_list[i]
        weighted_img = ee.Image.cat([
            centered_24.select(band_list[j]).multiply(weights[j]).rename(f"w{j}")
            for j in range(N)
        ])
        pc_k = weighted_img.reduce(ee.Reducer.sum()).rename(f"Pheno_PC{i+1}")
        pc_images.append(pc_k)

    pheno_pcs = ee.Image.cat(pc_images)

    # ---- Step 3: add beta1 trend as explicit feature --------------------
    pheno_trend = hls_coeffs.select("beta1").rename("Pheno_Trend")

    pheno_features = ee.Image.cat([pheno_pcs, pheno_trend])

    if verbose:
        print(f"\nPhenology features ready: {N_PCA} PCs + 1 trend band "
              f"= {N_PCA + 1} features at 30m.")

    meta = {
        "n_obs":         n_obs,
        "n_pca":         N_PCA,
        "n_samples":     N,
        "eigenvectors":  eigvec_list,
    }
    return pheno_features, meta
