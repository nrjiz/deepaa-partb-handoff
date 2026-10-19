# Accepted HAM10000 study provenance (read-only)

The working manuscript checkout was on `journal-research-local` at
`2eecc13643bf56cdcd9d0c11f91293f46d87d024` with26 modified/untracked files.
It was not used as if its current working tree were the scientific source.
Those edits remain in place and are not part of the handoff commits. No
`AGENTS.md` was found in the workspace ancestors or either repository's tree.

The handoff uses a separate checkout on `jaffe-partb`, based on the accepted K7
ST-RECOVERY-06 commit **`2cdcc565bef776543346b95a5dbd23b53000a85f`**. A Git bundle
of that commit and its ancestry was created locally and Git object integrity was
checked. No historical branch, tag, commit or scientific artifact was rewritten.

| Accepted training run | Implementation revision recorded in configuration |
|---|---|
| K5/113 | `0e32c8d469be141d30e627219aeac7ee253a34fa` |
| K5/211 | `f097f2f9e6ad72464685dacca59ba95ba273ab33` |
| K5/307 | `1a7214d89ac2d5d52f6161b19f86db71ed144ab7` |
| K5/401 | `e9ef9071453a2d4021470d0586e315463868b5bb` (authorized recovered final export; preserve recovery identity separately) |
| K5/503 | `2eecc13643bf56cdcd9d0c11f91293f46d87d024` |
| All five accepted K7 runs | `2cdcc565bef776543346b95a5dbd23b53000a85f` |

Protected model: `daa_ham10000_interpretability_fixed_v2.py`, original baseline
commit `29acec505161530f9ea811534b05dd52d8df9899`, accepted file SHA256
`6dbbc22f79d2f1242ae384b5468fdacaf019527114fbc248bdbaf796b245ef4d`.
The original stochastic implementation, not Route B, produced accepted results.
The handoff imports this model rather than copying or rewriting its architecture.

`HAM10000_EVIDENCE_INDEX.json` records configurations and artifact hash maps from
the completed registered aggregate. Every listed run and pair file was rehashed
read-only against that aggregate during this task. This is a source/evidence
index, not a new scientific validation or an assertion that all original models
were trained at the same revision. The original ten independent validations and
eight pair receipts remain authoritative and external to this code handoff.

K7/401 LOCAL01 remains the interrupted infrastructure failure; LOCAL02 is the
authorized completed result. K5/401's recovered export has a separate history.
Do not replace either history or omit collapsed results. Reference113 remains
fixed. Colab history/global uniqueness and off-device preservation are unverified.
Local SHA256 consistency does not provide independent backup.

## Necessary checkout fix beyond the accepted revision

The protected model's accepted file hash includes CRLF line endings, whereas its
Git blob is normalized LF. A plain clone with `core.autocrlf=false` therefore
failed the existing hash guard in the first engineering check. `.gitattributes`
now explicitly checks this one file out as CRLF and the three frozen manifests
as LF on every platform. The model's Git blob, Python logic and accepted SHA256
are unchanged. No integrity check or constant is relaxed. Fresh-checkout testing
is required and recorded in CHECKS.md.

All other new work is in `jaffe_partb/` plus ignore rules for external/generated
data. The retained HAM training/evaluation tools, protocol, splits and historical
files are not modified. New HAM K values use the separate experiment adapter,
not edits to `PRIMARY_K` or accepted authorization records.

Proposed preservation tag (not created or pushed):
`ham10000-partb-k7-source-2cdcc565` →
`2cdcc565bef776543346b95a5dbd23b53000a85f`.
Its name deliberately describes the K7 source, not all ten training revisions
or the external evidence archive. Preserve the full revision table with any tag.
