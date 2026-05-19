# Documentation Cleanup Report

## Objective
To analyze the `docs/` directory and recommend which files to keep as authoritative references for the current HGNN-based MITRE-CORE architecture, and which files to remove as they are obsolete, redundant, or point-in-time artifacts.

## 1. Required Documentation (Keep)
These documents represent the current state, architecture, research, and valid methodologies for MITRE-CORE.

**Architecture & Core Systems**
* `docs/architecture/SYSTEM_ARCHITECTURE_DETAILED.md` - Authoritative guide on the 6-stage pipeline (HGNN & Union-Find).
* `docs/architecture/ARCHITECTURE.md` - High-level overview.
* `docs/architecture/ARCHITECTURE_AND_DATASETS.md` - Dataset integration details.

**Research & Modern Integration**
* `docs/research/IEEE_Research_Paper_MITRE_CORE.md` - Draft paper outlining the primary HGNN and hybrid framework contributions.
* `docs/reports/TESTING_MODERN_DATASETS.md` - Instructions on the `ModernDatasetLoader` and new dataset integration.
* `docs/analysis/OPTC_ARI_INVESTIGATION_REPORT.md` - Recent investigation relevant to the OpTC dataset and bridge edges.

**Project Summaries & Guides**
* `docs/planning/PROJECT_SUMMARY_UPDATED.md` - Current summary of the multi-algorithm correlation engine.
* `docs/security/SECURITY_AUDIT.md` - Baseline security assessment.
* `docs/visualization/IMPLEMENTATION_GUIDE.md`
* `docs/visualization/VISUALIZATION_STRATEGY.md`
* `docs/DATA_PROVENANCE.md` - Tracing data lineage.

## 2. Recommended for Removal (Obsolete, Redundant, or Point-in-Time)
These files clutter the repository and often reflect past states (e.g., deprecated models, completed phases, raw LLM outputs).

**Point-in-Time Automated Reports & Analyses**
* `docs/reports/code_analysis_*.md`
* `docs/reports/comprehensive_evaluation_*.md`
* `docs/reports/engine_capability_check_*.md`
* `docs/reports/industry_comparison_*.md`
* `docs/analysis/CODEBASE_STRUCTURE.md`
* `docs/analysis/DUPLICATION_REPORT.md`

**Completed Phase Reports & Summaries**
* `docs/reports/FINAL_PHASE1_COMPLETION_REPORT.md`
* `docs/reports/IMPLEMENTATION_SUMMARY.md`
* `docs/reports/EXECUTION_SUMMARY.md`
* `docs/reports/SELF_EVALUATION_REPORT.md`

**Old Plans & Refactoring Artifacts**
* `docs/planning/FIX_PLAN.md`
* `docs/planning/PENDING_CHANGES.md`
* `docs/planning/PROJECT_SUMMARY.md` (Replaced by `PROJECT_SUMMARY_UPDATED.md`)
* `docs/planning/technical_improvements.md`
* `docs/planning/uf_temporal_gap_analysis.md`
* `docs/planning/DATASETS.md`
* `docs/reports/code_refactoring_plan.md`
* `docs/REFACTORING_PLAN.md`
* `docs/POSITIONING_STRATEGY.md`
* `docs/TEST_VERIFICATION_METHODOLOGY.md`
* `docs/ARCHITECTURE_CLARIFICATION.md`
* `docs/ARCHITECTURE_REORGANIZATION.md`
* `docs/architecture/foundation_model_split.md`

**Old Research Files**
* `docs/research/literature_review_plan.md`
* `docs/research/research_roadmap.md`

**Miscellaneous**
* `docs/misc/prompt_v3_0_comprehensive.md` - Raw prompt file.
* `docs/misc/RESUME_POINTS.md` - Personal artifact.

## Action Plan
1. Review the "Recommended for Removal" list.
2. Confirm if any intermediate reports hold archival value (if so, move them to a `docs/archive/` folder).
3. Delete the confirmed obsolete files to streamline the `docs/` tree.
