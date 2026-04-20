# Forest Management Units Pipeline

Modular GEE pipeline for data-driven forest management unit delineation.

## Structure

```
forest-mgmt-units/
├── notebooks/
│   ├── 00_baseline.ipynb               # FROZEN reference (original Untitled11.ipynb)
│   ├── scratch_verify_refactor.ipynb   # Run once to confirm src/ == baseline
│   ├── 01_build_assets.ipynb           # (next) Run modules, export EE assets
│   └── 02_explore_assets.ipynb         # (next) Interactive clustering on loaded assets
├── src/
│   ├── config.py              # YAML loader + derived geometry
│   ├── masking.py             # Module 1: suitability mask
│   ├── phenology.py           # Module 2: HLS dual harmonic + PCA
│   ├── radar.py               # Module 3: S1 VH/VV features
│   ├── static_features.py     # Module 4: canopy height + topo
│   ├── s2_composites.py       # Module 5: S2 seasonal medians (SNIC input)
│   ├── segmentation.py        # Modules 6a + 7a + 8: HR stack, pixel norm, SNIC
│   ├── aggregation.py         # Modules 6b + 9 + 10: full stack, zonal, stand norm
│   ├── clustering.py          # Module 11: K-sweep + final wekaKMeans
│   └── assets.py              # Export/load helpers with metadata sidecars
├── configs/
│   └── sanjay_van.yaml        # ROI + all parameters
└── experiments/
    ├── log.md                 # Running log of findings
    └── asset_metadata/        # JSON sidecars per exported asset
```

## Workflow

1. **Verify refactor**: Run `notebooks/scratch_verify_refactor.ipynb` end to end and confirm outputs match `00_baseline.ipynb`.
2. **Build assets**: Run `notebooks/01_build_assets.ipynb` once per ROI. This exports each feature group as its own EE asset. Takes a while (SNIC and the final normalized-stands export are the slow parts).
3. **Explore**: Use `notebooks/02_explore_assets.ipynb` to load pre-computed assets and run clustering experiments. Each experiment → entry in `experiments/log.md`.
4. **New ROI**: copy `configs/sanjay_van.yaml` → `configs/new_roi.yaml`, edit, re-run step 2.

## Design principles

- **CONFIG is the only knob when switching ROIs.** Every `src/` module reads from the config dict; nothing is hard-coded.
- **Every module exposes `compute_*(config, ...) -> (ee.Image, meta_dict)`.** This uniform interface is what makes modules swappable.
- **Assets are versioned, not overwritten.** When you change how phenology is computed, export as `pheno_..._v2`, keep `v1` for comparison.
- **The notebook is orchestration, the science is in `src/`.** If you're tempted to write logic in a notebook cell, it belongs in a function in `src/`.

## Notes

The architecture has two distinct normalization steps:
- `segmentation.py` does pixel-level Z-score (for SNIC's Euclidean distance).
- `aggregation.py` does stand-level Z-score (for K-means on stand vectors).
These are statistically different distributions; both are needed.

Phenology features are 30m (HLS native). They are EXCLUDED from SNIC (to avoid smearing boundaries) and only join the stack at the per-stand aggregation stage where they're bilinearly upsampled for grid alignment.
