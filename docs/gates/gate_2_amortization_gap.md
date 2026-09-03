# Gate 2 — Amortization gap

Status: **NOT STARTED; depends on Gates 0–1**.

## Exit criteria

- Authoritative local inference parameters and per-image objective documented.
- All global/model weights demonstrably frozen during refinement.
- Simplex validity maintained at every step.
- Encoder-only and approved refinement depths use identical objective accounting.
- Automated check confirms diagnosis labels do not enter optimization, tuning, or stopping.
- Runtime measurement includes warm-up/synchronization policy.
- Absolute/relative gaps, reconstruction, coordinate displacement, prediction changes, and failures reported on held-out data across approved seeds.

Any label-guided test-time experiment must be separately named and cannot satisfy this gate.

