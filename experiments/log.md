# Experiments Log

A running record of pipeline changes and findings. Every meaningful run or refactor gets an entry. Reference this in mentor meetings.

---

## 2026-04-21 — Initial refactor

**What changed:**
- Migrated baseline `Untitled11.ipynb` → modular `src/` package.
- Each feature group (masking, phenology, radar, static, s2, segmentation, aggregation, clustering) is now its own Python module with a uniform `compute_*(config, ...) -> (ee.Image, meta)` interface.
- CONFIG externalized to `configs/sanjay_van.yaml`.
- Added `src/assets.py` with export + metadata sidecar helpers (not yet used).

**Verification status:**
- [ ] `scratch_verify_refactor.ipynb` run end to end in Colab
- [ ] Masking stats match baseline
- [ ] Phenology n_obs and test-point coeffs match baseline
- [ ] Radar n_obs matches baseline
- [ ] S2 band count = 24
- [ ] SNIC map visually matches baseline

**Next:**
- Build `01_build_assets.ipynb` to export intermediate assets.
- Once assets exist, port K-sweep to load from asset (fixes the "User memory limit exceeded" error).

**Open questions for mentor:**
1. S2 harmonization: did he mean `S2_SR_HARMONIZED` (already used) or something else? Building both phenology variants (HLS vs S2-only) for comparison.
2. Process modelling: version A (features), B (post-hoc), or C (per-stand fit)?
3. Study site: Sanjay Van as dev sandbox; main validation on managed forest with FSI data?
