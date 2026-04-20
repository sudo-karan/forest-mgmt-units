"""
Config loader.

Loads a YAML config file and returns a dict that matches the CONFIG dict
from the baseline notebook, with two derived geometry fields added:
  - config['roi']          : ee.Geometry.Rectangle
  - config['roi_buffered'] : ee.Geometry.Rectangle (5 km buffer for VIIRS)

Also adds a 'roi_slug' field derived from roi_name, used for asset naming.
"""

from pathlib import Path
import yaml
import ee


def load_config(path):
    """Load a YAML config and enrich it with derived fields."""
    path = Path(path)
    with path.open() as f:
        config = yaml.safe_load(f)

    # Derived: slug for asset naming (lowercase, no spaces/commas)
    config['roi_slug'] = (
        config['roi_name']
        .lower()
        .replace(',', '')
        .replace(' ', '_')
    )

    # Derived: ROI geometry
    b = config['roi_bounds']  # [west, south, east, north]
    config['roi'] = ee.Geometry.Rectangle(b)

    # Derived: 5 km buffer around ROI (for VIIRS threshold calculation)
    config['roi_buffered'] = ee.Geometry.Rectangle([
        b[0] - 0.05, b[1] - 0.05,
        b[2] + 0.05, b[3] + 0.05,
    ])

    return config


def print_config_summary(config):
    """Print a human-readable summary — mirrors the baseline notebook's prints."""
    print(f"Config loaded for ROI: {config['roi_name']}")
    print(f"  Phenology window : {config['pheno_start']} → {config['pheno_end']}")
    print(f"  Analysis scale   : {config['analysis_scale']} m")
    print(f"  SNIC compactness : {config['snic_compactness']}")
    print(f"  Curve samples    : {config['n_curve_samples']}  →  "
          f"PCA to {config['n_pca_components']} components")
