# Conference baseline inventory

## Identity and confidence

- Canonical project root: `C:\Users\UserPc\Desktop\ai\Deep-Archetypal-Analysis-on-HAM10000`
- Remote: `https://github.com/Amy007-ali/Deep-Archetypal-Analysis-on-HAM10000.git`
- Branch: `main`
- Working baseline commit: `29acec505161530f9ea811534b05dd52d8df9899`
- Commit time: `2026-05-09T15:15:47+03:00`
- Commit subject: `adding results for K=9`
- Tags: none
- Confidence: high, not formal certainty

Git history adds the matching May result bundles at `661cc7f` (K=5), `6dd8154` (K=3 and K=7), and `29acec5` (K=9). The source and README are byte-identical from `661cc7f` through HEAD. Exact artifact and checkpoint reproduction provides strong numerical provenance. A conference tag/release, submitted-code archive, exact invocation log, and author confirmation remain absent.

## Authoritative file hashes

| File | SHA-256 |
|---|---|
| `daa_ham10000_interpretability_fixed_v2.py` | `6dbbc22f79d2f1242ae384b5468fdacaf019527114fbc248bdbaf796b245ef4d` |
| `README.md` | `8a59774720b8ccebdd0973380c85b5c934129aa8812eb490a7fd53e7e7fba6a5` |

The authoritative conference source was not modified during reproduction.

## Result bundles and checkpoints

Run timestamps below are inferred from directory names and have no encoded timezone. Git commit times are authoritative repository metadata.

| K | Result directory | Inferred run time | Added by commit (commit time) | Files | Bytes | Checkpoint SHA-256 |
|---:|---|---|---|---:|---:|---|
| 3 | `Results_HAM_DAA/20260508_002312` | 2026-05-08 00:23:12 | `6dd8154` (2026-05-08T21:30:52+03:00) | 37 | 16,540,208 | `80014e53fe89862ab936362f54636f841b0e16b61143daa4d1d4a5862004ec7e` |
| 5 | `Results_HAM_DAA/20260507_131506` | 2026-05-07 13:15:06 | `661cc7f` (2026-05-07T17:08:09+03:00) | 49 | 21,552,948 | `7e7b2054ff2ec413aabcf5fc93effe7915d9bc40b77362b7f311541f81cec7e0` |
| 7 | `Results_HAM_DAA/20260508_104147` | 2026-05-08 10:41:47 | `6dd8154` (2026-05-08T21:30:52+03:00) | 63 | 27,764,376 | `205b09fd32f5d1548c21c06275d924ecc5e65655a7c49a0a269cea37e01bdb01` |
| 9 | `Results_HAM_DAA/20260509_112437` | 2026-05-09 11:24:37 | `29acec5` (2026-05-09T15:15:47+03:00) | 77 | 34,083,071 | `b3f30ad6acc45ed9ba9c8a6bbd1c4c088f8e467cacd5fd8e5d76faa097f18d20` |

Every checkpoint filename is `daa_ham_weights.weights.h5` inside its run directory. Checkpoint byte sizes are respectively 5,783,136 (K=3), 5,854,552 (K=5), 5,927,256 (K=7), and 5,999,960 (K=9).

`baseline_artifact_hashes.csv` records relative path, K, inferred run timestamp, byte size, SHA-256, and checkpoint flag for all **226 files / 99,940,603 bytes**. Its own SHA-256 at generation was `45ace52b552e6a7de14db8e13f21973a00b2659a2372392c60f52a63e2ba7166`.

## Drift-detection boundary

Baseline preservation covers the commit identity, authoritative source and README bytes, every committed May result artifact, reconstructed test order, and checkpoint behavior. New material belongs under `repro/` or repository-local `docs/`. Historical `Results_HAM_DAA` files must remain untouched.
