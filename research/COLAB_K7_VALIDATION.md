# ST-04A: K7 reference validation correction

Review candidate only; no primary launch is authorized. The scope remains exactly
K7/latent6/seed113 under the frozen protocol. Actual target readiness is pending.

The `colab_gpu_worker.py validate` branch dispatches only `(7,113)` to
`colab_k7_validation.validate_k7_reference`. It checks configuration identity and
refuses recovered-export paths, then calls the unchanged hash-bound shared
validator. Only empty reason cells in the exact run semantic CSV become None;
all nonempty strings, numeric parsing and scientific tolerances remain unchanged.
The reader override is limited to a dedicated validation worker and restored on
every exit. Historical K5 worker validation keeps its original dispatch.

The defect is CSV serialization: defined semantic reasons are None, serialized
as empty cells, but pandas default reload produces NaN. This does not authorize
normalizing numeric missingness or undefined semantic distributions. D-019 bound
source bytes and receipts remain untouched. Synthetic complete-worker tests
exercise the actual shared validator with explicitly labeled test-double models
and checkpoints; those results are not primary K7 scientific outcomes.

After source review, the user must commit approved changes and build a new release
from that exact clean HEAD. Dirty-source packaging without approved HEAD yields a
disabled review draft. Do not change its release flag/manifest/hash to execute it.
Use a new matching notebook/ZIP/receipt and fresh target readiness after release
approval. Retain all old packages and all permanent attempt claims.
