# Q2 2026 Archive

This directory contains deprecated scripts, data loaders, and failed experiments from the Q2 2026 research push. 
They are retained for historical reference but are no longer part of the active MITRE-CORE pipeline.

## Contents
- **OpTC Loaders**: Early iterations of the DARPA OpTC loader before they were consolidated into `darpa_optc_loader_unified.py` (which was subsequently renamed to `darpa_optc_loader.py` in production).
- **Sweep Scripts**: One-off parameter sweeps (e.g., ZCA, epsilon, spectral clustering) that have been integrated into the main `run_gate_tuning.py` and `analyse_gate_tuning.py` scripts.
- **Entity Collapse Fixes**: Temporary scripts used during the investigation of the SQTK_SIEM embedding collapse issue before the robust `siem_supcon_v4` solution was implemented.
