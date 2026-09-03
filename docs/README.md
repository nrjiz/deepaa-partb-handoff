# DeepAA HAM10000 journal project knowledge base

This directory is the long-term source of truth for the extended journal project following the DATA 2026 conference paper, *Probabilistic Variational Archetypal Learning on the Simplex for Interpretable Medical Image Analysis*.

## Purpose and phase

The project asks whether DeepAA's simplex-constrained archetypal representation for medical images can be made reliable and robust while preserving interpretability and fast inference.

Current phase: **repository and dataset audit; Gate 0 preparation**. The journal investigations are proposals and protocols, not completed findings. A trusted conference-baseline implementation has not yet been located in the writable workspace.

## Deadlines

- Publisher decision deadline: **2026-09-09**.
- Journal final-version deadline communicated by Prof. Mageda: **2026-11-18**.
- Publisher requirements: cite the conference version; add at least 30% new material; cite the source of reused conference figures/tables; avoid large verbatim reuse; use a new title, abstract, and conclusion; explicitly state differences from the conference paper.

## Navigation

- [Current project state](project_state.md)
- [Conference paper analysis](conference_paper_analysis.md)
- [Coherent journal extension plan](journal_extension_plan.md)
- [Research questions and hypotheses](research_questions.md)
- Research sources:
  - [Delaney et al. (2023), counterfactual explanations](research_sources/counterfactual_explanations_delaney_2023.md)
  - [ParTIpy (2026)](research_sources/partipy_2026.md)
- Experiment protocols:
  - [Experiment documentation conventions](experiments/README.md)
  - [Archetype stability](experiments/archetype_stability.md)
  - [Amortization gap](experiments/amortization_gap.md)
  - [Controlled simplex relaxation](experiments/simplex_relaxation.md)
  - [Simplex-coordinate counterfactuals](experiments/simplex_counterfactuals.md)
  - [Experiment registry](experiments/experiment_registry.md)
- [Decision log](decisions/decision_log.md)
- Stage gates:
  - [Gate 0: scope and reproduction](gates/gate_0_scope_and_reproduction.md)
  - [Gate 1: stability infrastructure](gates/gate_1_stability_infrastructure.md)
  - [Gate 2: amortization gap](gates/gate_2_amortization_gap.md)
  - [Gate 3: simplex relaxation](gates/gate_3_simplex_relaxation.md)
  - [Gate 4: counterfactuals](gates/gate_4_counterfactuals.md)
  - [Gate 5: journal claim freeze](gates/gate_5_journal_claim_freeze.md)
- [Lessons learned](lessons_learned.md)

## Documentation rules

- Separate established facts, proposals, hypotheses, decisions, and results.
- Never turn an intended metric or expected outcome into a finding.
- Record dataset split IDs, seeds, manually supplied commit hashes, checkpoints, and outputs for every run.
- Treat medical counterfactuals as explanations of model behavior, not clinical or causal claims.
- Update `project_state.md`, the experiment registry, and the decision log when project state changes.

