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

PLAN_v1_1_PATH = REPO_ROOT / "docs" / "plans" / "MASTER_PLAN_v1.1.md"
SHA256_v1_1_PATH = REPO_ROOT / "docs" / "plans" / "MASTER_PLAN_v1.1.sha256"

PLAN_v1_2_PATH = REPO_ROOT / "docs" / "plans" / "MASTER_PLAN_v1.2.md"
SHA256_v1_2_PATH = REPO_ROOT / "docs" / "plans" / "MASTER_PLAN_v1.2.sha256"

FROZEN_SHA256 = "a779e478496b427591e79b2f1808aec48a68c7459d28c0738e61650756a98b54"
FROZEN_SHA256_v1_1 = "950ba7bfaf4ece0389ea663fc9f7568995641dc461a8aec08cc2ba99cd820234"
FROZEN_SHA256_v1_2 = "DB1A12EE88AAFF96310D954559B520E60EEE0D9D0411E955510E454530E412DB"


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


def test_master_plan_v1_1_unchanged():
    assert PLAN_v1_1_PATH.exists(), f"Master plan v1.1 missing: {PLAN_v1_1_PATH}"
    assert SHA256_v1_1_PATH.exists(), f"Master plan v1.1 SHA256 file missing: {SHA256_v1_1_PATH}"

    h = hashlib.sha256(PLAN_v1_1_PATH.read_bytes()).hexdigest()
    assert h == FROZEN_SHA256_v1_1, (
        f"MASTER_PLAN_v1.1.md has been modified (current SHA256: {h}).\n"
        "Do not edit v1.1 in place. Instead:\n"
        "  1. Create docs/plans/MASTER_PLAN_v1.2.md with a diff summary.\n"
        "  2. Update docs/plans/README.md to point at v1.2 as active.\n"
        "  3. Update the FROZEN_SHA256_v1_1 constant in this test to match v1.2."
    )


def test_master_plan_v1_2_unchanged():
    """v1.2 must match its SHA256."""
    import hashlib
    from pathlib import Path
    plan_path = Path(__file__).parent.parent / "docs" / "plans" / "MASTER_PLAN_v1.2.md"
    sha_path = Path(__file__).parent.parent / "docs" / "plans" / "MASTER_PLAN_v1.2.sha256"
    assert plan_path.exists(), f"Master plan v1.2 missing: {plan_path}"
    assert sha_path.exists(), f"SHA file missing: {sha_path}"
    expected = sha_path.read_text().strip().upper()
    actual = hashlib.sha256(plan_path.read_bytes()).hexdigest().upper()
    assert actual == expected, (
        f"MASTER_PLAN_v1.2.md SHA mismatch.\n"
        f"  Expected: {expected}\n"
        f"  Actual:   {actual}\n"
        f"v1.2 is frozen. If you intended to update it, write v1.3 instead."
    )
