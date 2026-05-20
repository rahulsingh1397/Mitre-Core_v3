# UNSW-NB15 — Learnings

---

## L1 — Zero-shot transfer from NSL-KDD holds on UNSW-NB15

V3 ARI on UNSW attack_cat = 0.564, using the exact same checkpoint and engine config as NSL-KDD. No retuning. The `network_v9_v3` checkpoint trained on network IDS data generalizes across the two most popular network-IDS benchmarks without any dataset-specific adaptation.

**Carry-forward:** Zero-shot claim is now supported by two datasets. Use this explicitly in the paper. The CICIDS2017 run will be the third confirmation (or refutation) of this pattern.

---

## L2 — attack_cat is better than tactic for UNSW

NSL-KDD used `tactic` as primary because it was the most natural MITRE label. On UNSW, `attack_cat` (the original UNSW label) is more fine-grained, has 0 nulls, and is what published UNSW papers cite. For UNSW, `attack_cat` should be the headline number.

**Carry-forward:** For future datasets, check if there's an original native label that's more semantically precise than the MITRE-converted `tactic`. Prefer the native label as primary when available.

---

## L3 — dominant_confusion_accuracy is degenerate on UNSW (second dataset to demote a metric)

On NSL-KDD we demoted `attack_f1`; on UNSW we demoted `dominant_confusion_accuracy`. Two consecutive datasets with one degenerate metric each. This pattern suggests the 12-metric set has some redundancy and that certain metrics only become informative under specific clustering failure modes.

**Carry-forward:** For future datasets, always run the degenerate metric check (Stage 5) before citing any metric. Consider whether `dominant_confusion_accuracy` is structurally degenerate (always 1.0 on well-separated datasets) or UNSW-specific.

---

## L4 — Spectral (raw) has high variance on UNSW (std=0.159 on attack_cat)

On NSL-KDD, Spectral (raw) was surprisingly strong. On UNSW, it's also relatively strong (mean ARI=0.321) but with very high seed variance (seed 43 ARI=0.479 vs seed 44 ARI=0.320 vs seed 42 ARI=0.162). This makes Spectral (raw) unreliable as a published baseline number — its ARI depends heavily on random initialization.

**Carry-forward:** Always report mean ± std for all baselines. Single-seed numbers for Spectral (raw) are not representative.

---

## L5 — Worms class (~130 rows) is effectively unmeasurable at 10K sample

With 130 Worms rows in 175K, a 10K random sample contains ~7 Worms rows. These are too sparse for reliable cluster recall. The class is present but its per-class metrics are noise. Document this in each dataset's baseline doc as a known limitation rather than trying to handle it by oversampling.

**Carry-forward:** For TON_IoT and others, flag any class with <200 rows in the full corpus as "evaluability limited" in the audit doc.

---

## L6 — The six-stage lifecycle took approximately 1 working day for UNSW-NB15

Estimated effort in master plan was "1–2 days". Actual: ~1 day. The audit script, split generation, baseline roster run, and freeze were straightforward once the NSL-KDD template was in place.

**Carry-forward:** The lifecycle template is validated and reusable. TON_IoT should be 1–2 days (the loader is the heavy part, per master plan). CICIDS2017 should also be ~1 day.
