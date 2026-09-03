# Gate 3 — Simplex relaxation

Status: **BLOCKED ON DESIGN APPROVAL**.

## Pre-implementation approval required

- Exact `Z_delta` orientation and scale parameterization.
- Mathematical treatment of the existing archetype regularizer.
- Any scale prior/penalty and identifiability constraint.
- Interaction with variational mean/variance and decoder inputs.
- Proof/test specification that `delta = 0` recovers the approved baseline.

## Exit criteria after approval

- Equivalence, bound, gradient, serialization, and reload tests pass.
- `delta = {0,0.1,0.25,0.5}` compared on fixed splits/seeds.
- Reconstruction, classification, usage, semantics, decoded interpretation, and aligned stability reported.
- Learned scales and boundary saturation reported.

No model implementation begins before the decision log records an ACCEPTED formulation.
