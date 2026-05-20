"""
Asserts that docs/plans/MASTER_PLAN_v1.0.md has not been modified since it was frozen.
Any edit to the file fails this test — changes must produce MASTER_PLAN_v1.1.md instead.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PLAN_PATH = REPO_ROOT / "docs" / "plans" / "MASTER_PLAN_v1.0.md"
SHA256_PATH = REPO_ROOT / "docs" / "plans" / "MASTER_PLAN_v1.0.sha256"

FROZEN_SHA256 = "a779e478496b427591e79b2f1808aec48a68c7459d28c0738e61650756a98b54"


def test_master_plan_v1_0_unchanged():
    assert PLAN_PATH.exists(), f"Master plan missing: {PLAN_PATH}"
    assert SHA256_PATH.exists(), f"Master plan SHA256 file missing: {SHA256_PATH}"

    h = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
    assert h == FROZEN_SHA256, (
        f"MASTER_PLAN_v1.0.md has been modified (current SHA256: {h}).\n"
        "Do not edit v1.0 in place. Instead:\n"
        "  1. Create docs/plans/MASTER_PLAN_v1.1.md with a diff summary.\n"
        "  2. Update docs/plans/README.md to point at v1.1 as active.\n"
        "  3. Update the FROZEN_SHA256 constant in this test to match v1.1."
    )
