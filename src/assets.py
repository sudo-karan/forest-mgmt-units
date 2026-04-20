"""
Asset export/load helpers.

Centralizes:
  - Asset ID construction (consistent naming)
  - Export with wait-until-complete polling
  - Metadata JSON sidecar (so we don't forget what each asset's CONFIG was)
  - Asset loading

Asset naming convention:
  {export_asset_base}{kind}_{roi_slug}_v{version}

Examples:
  projects/replicating-paper/assets/mask_sanjay_van_delhi_v1
  projects/replicating-paper/assets/pheno_sanjay_van_delhi_v1
  projects/replicating-paper/assets/radar_sanjay_van_delhi_v1
  projects/replicating-paper/assets/normalized_stands_sanjay_van_delhi_v1
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
import ee


# Where to write local metadata sidecars (relative to project root).
METADATA_DIR = Path("experiments/asset_metadata")


def asset_id(config, kind, version=1):
    """Construct an asset ID for a given feature group."""
    base = config['export_asset_base']
    slug = config['roi_slug']
    return f"{base}{kind}_{slug}_v{version}"


# ---- Export helpers -----------------------------------------------------

def export_image_and_wait(image, asset_id_str, config, description=None,
                          extra_meta=None, poll_seconds=15, verbose=True):
    """Export an image to an EE asset and block until complete.

    Writes a metadata JSON sidecar to experiments/asset_metadata/ with:
      - asset_id, kind, roi_slug, timestamp
      - full CONFIG dict used
      - any `extra_meta` provided by the caller (e.g. eigenvectors, counts)
    """
    roi   = config['roi']
    scale = config['analysis_scale']

    if description is None:
        # EE description can't contain slashes or be too long
        description = asset_id_str.split('/')[-1][:100]

    # Cancel any pre-existing asset at this ID? No — safer to fail loudly.
    # The user should delete assets manually if they want to re-export.

    task = ee.batch.Export.image.toAsset(
        image       = image,
        description = description,
        assetId     = asset_id_str,
        region      = roi,
        scale       = scale,
        maxPixels   = int(1e13),
    )
    task.start()
    if verbose:
        print(f"Export started: {asset_id_str}")
        print(f"  Task description: {description}")

    # Poll
    while True:
        status = task.status()
        state  = status.get('state', 'UNKNOWN')
        if state in ('COMPLETED',):
            if verbose:
                print(f"  -> COMPLETED: {asset_id_str}")
            break
        if state in ('FAILED', 'CANCELLED', 'CANCEL_REQUESTED'):
            raise RuntimeError(
                f"Export {state}: {asset_id_str}\n"
                f"Full status: {status}"
            )
        if verbose:
            print(f"  ... {state}", end='\r')
        time.sleep(poll_seconds)

    _write_metadata(asset_id_str, config, extra_meta=extra_meta)
    return asset_id_str


def _write_metadata(asset_id_str, config, extra_meta=None):
    """Write a local JSON sidecar with the config + timestamp for this asset."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Strip non-serializable ee.Geometry objects from config copy
    config_serializable = {
        k: v for k, v in config.items()
        if not isinstance(v, ee.ComputedObject)
    }

    meta = {
        "asset_id":  asset_id_str,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config":    config_serializable,
    }
    if extra_meta:
        meta["extra"] = extra_meta

    fname = asset_id_str.split('/')[-1] + ".json"
    path  = METADATA_DIR / fname
    with path.open("w") as f:
        json.dump(meta, f, indent=2, default=str)


# ---- Load helpers -------------------------------------------------------

def load_image(config, kind, version=1):
    """Load an exported asset by (kind, version) for this config's ROI."""
    return ee.Image(asset_id(config, kind, version))


def asset_exists(config, kind, version=1):
    """Check if an asset exists. Returns bool."""
    try:
        ee.data.getAsset(asset_id(config, kind, version))
        return True
    except ee.EEException:
        return False
    except Exception:
        return False
