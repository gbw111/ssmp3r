# code_review.md

# Code Review Rules for CUT3R-Based Stable Dual-State Project

## 0. Purpose

This document defines code review rules for all changes in this repository.

The project is a research implementation built on top of CUT3R.
The review standard is not only whether the code runs, but whether the code remains aligned with the project’s scientific objective:

- stable state propagation,
- dual-state factorization,
- residual-to-trend consolidation,
- optional release,
- later spectral refinement.

Any patch that drifts away from this narrative should be treated as suspicious, even if it appears technically functional.

This document is complementary to `AGENTS.md`.
If there is any conflict, `AGENTS.md` defines the project direction, and this file defines the review discipline.

---

## 1. Primary Review Principle

Every code change must be reviewed against three axes:

### 1.1 Scientific alignment
Does the patch serve the intended method?

The intended method is:
- CUT3R as baseline,
- minimum-invasive modification,
- state-path redesign rather than full architecture rewrite,
- stable propagation first,
- dual-state second,
- consolidation third,
- release optional,
- spectral refinement later.

Reject or request revision if the patch:
- changes the research direction,
- shifts the main innovation away from state dynamics,
- introduces unrelated memory paradigms,
- or makes the code harder to ablate scientifically.

### 1.2 Baseline preservation
Does the patch preserve CUT3R behavior where preservation is expected?

The baseline must remain intact as much as possible in:
- encoder,
- decoder skeleton,
- output heads,
- training interface,
- inference interface,
- script usability,
- checkpoint compatibility where feasible.

Reject or request revision if the patch:
- rewrites major baseline components without necessity,
- silently breaks model I/O contracts,
- or changes baseline semantics during an early phase that should be conservative.

### 1.3 Experimental cleanliness
Does the patch preserve clean ablation boundaries?

Every major module must be:
- explicit,
- modular,
- independently switchable,
- and attributable in experiments.

Reject or request revision if the patch:
- entangles multiple contributions in one patch,
- hides method changes inside unrelated utilities,
- or makes it difficult to isolate what caused performance changes.

---

## 2. Review Priorities by Phase

## Phase 0 — State API Refactor

### Expected nature of change
This phase is a structural refactor only.
Behavior should remain baseline-equivalent as much as possible.

### Review focus
Check:
- whether the state representation is now explicit and structured,
- whether state packing and unpacking are centralized,
- whether train/inference/demo state passing still works,
- whether backward compatibility is preserved where practical.

### Reject if
- the refactor changes learning behavior unnecessarily,
- state semantics are altered without documentation,
- random logic is added “for future use” without necessity,
- or scripts break due to state format drift.

### Review questions
- Is the new state container clearly defined?
- Are field names explicit and non-ambiguous?
- Is state lifecycle easy to trace?
- Are there still fragile positional tuple assumptions scattered around?

---

## Phase 1 — Stable Propagation

### Expected nature of change
This phase adds a conservative trend-side stable propagation mechanism.

### Review focus
Check:
- whether propagation is inserted at the correct point in the recurrent state update path,
- whether stability is enforced explicitly,
- whether the implementation is lightweight and local,
- whether it preserves baseline interaction structure.

### Preferred properties
- contraction-like update,
- clear control signals,
- minimal architectural disturbance,
- easy config gating,
- easy inspection of update magnitudes.

### Reject if
- the patch redesigns the entire backbone,
- propagation logic is mathematically unstable,
- the implementation hides state update in opaque monolithic code,
- or trend propagation is mixed with unrelated major changes.

### Review questions
- Is stability explicit in code, or only implied?
- Can the new trend propagation be disabled cleanly?
- Is the update path understandable from code alone?
- Are control signals documented and minimally justified?

---

## Phase 2 — Dual-State Factorization

### Expected nature of change
This phase introduces trend and residual states.

### Review focus
Check:
- whether trend/residual roles are clearly separated,
- whether the first implementation preserves a fused memory view for baseline compatibility,
- whether state factorization is lightweight and controlled,
- whether residual behavior is fast but not uncontrolled.

### Preferred properties
- factorization after candidate update,
- fused memory for baseline interaction in early versions,
- token-wise or coarse-grained gates first,
- modular factorizer and residual updater,
- explicit state fusion interface.

### Reject if
- the patch immediately forces the whole model to attend two independent memory banks without justification,
- trend/residual semantics are not documented,
- factorization becomes a hidden uncontrolled transformation,
- or the code effectively implements a different memory research direction.

### Review questions
- Is dual-state introduced as temporal role separation, not frequency split?
- Is the code still ablatable?
- Is fused state clearly defined?
- Can one inspect trend-only and residual-only behavior separately?

---

## Phase 3 — Consolidation

### Expected nature of change
This phase implements residual-to-trend consolidation as the main memory formation mechanism.

### Review focus
Check:
- whether new information first enters residual,
- whether eligibility or persistence is accumulated explicitly,
- whether the gate injects into trend gradually,
- whether residual is attenuated rather than hard-transferred.

### Preferred properties
- explicit eligibility state,
- interpretable gate inputs,
- gradual and sparse consolidation,
- copy-plus-attenuation rather than move-and-delete,
- config-controlled activation.

### Reject if
- consolidation becomes a hard overwrite,
- residual is emptied aggressively,
- the gate is undocumented,
- or the first version depends on heavy geometric subsystems that were not requested.

### Review questions
- Is this really consolidation, or just another write rule?
- Is the transfer progressive?
- Is eligibility state persistent and observable?
- Are proxy inputs to consolidation understandable?

---

## Phase 4 — Release

### Expected nature of change
This phase is optional and secondary.

### Review focus
Check:
- whether release is disabled by default,
- whether it is sparse, local, and conditional,
- whether it is documented as an emergency valve rather than a symmetric main mechanism.

### Reject if
- release is enabled by default without explicit reason,
- trend becomes unstable due to frequent release,
- release is presented as equally important as consolidation,
- or release logic is broad and destructive.

### Review questions
- Is release local?
- Is it optional?
- Is it clearly weaker than consolidation?
- Can experiments disable it completely?

---

## Phase 5 — Spectral Refinement

### Expected nature of change
This phase is a later enhancement.

### Review focus
Check:
- whether spectral logic is attached to a controlled insertion point,
- whether the implementation does not redefine the core state split,
- whether it is independently switchable,
- whether readout-side insertion is preserved unless explicitly changed.

### Reject if
- frequency decomposition becomes the primary state definition in early versions,
- spectral code entangles all output heads at once,
- or the patch makes attribution impossible.

### Review questions
- Is spectral refinement auxiliary rather than foundational?
- Is the insertion point justified?
- Is world-branch-first refinement preserved where intended?
- Can spectral effects be ablated cleanly?

---

## 3. Hard Rejection Conditions

Immediately request revision if any patch does one or more of the following without explicit authorization:

1. Replaces CUT3R’s baseline architecture wholesale.
2. Converts persistent state into a different memory paradigm.
3. Breaks forward return contracts silently.
4. Changes train/inference scripts in incompatible ways without documentation.
5. Bundles unrelated cleanup with method-critical patches.
6. Adds many new losses simultaneously without necessity.
7. Removes ablation switches or hardcodes method features permanently.
8. Introduces undocumented tensor shape assumptions.
9. Hides complex state logic inside anonymous utility code.
10. Makes scientific attribution harder rather than easier.

---

## 4. Required Review Checklist

Every substantial patch should be reviewed with this checklist.

### 4.1 Scope
- What exact problem is this patch solving?
- Which implementation phase does it belong to?
- Does it stay within that phase’s intended scope?

### 4.2 Scientific role
- Which part of the project narrative does this patch implement?
- stable propagation?
- dual-state?
- consolidation?
- release?
- spectral refinement?

### 4.3 Boundaries
- Does it accidentally implement a nearby but different method?
- Does it introduce a non-goal?
- Does it shift the project away from the intended contribution?

### 4.4 Interface safety
- Are input and output interfaces preserved?
- Is state passing explicit and safe?
- Are config defaults conservative?

### 4.5 Modularity
- Can the new component be turned off?
- Can it be ablated independently?
- Is it implemented as a coherent module rather than tangled inline logic?

### 4.6 Observability
- Can the new mechanism be inspected through logging or debug stats?
- Are trend/residual/gate statistics available if relevant?
- Is the behavior falsifiable during debugging?

### 4.7 Minimality
- Is the patch the smallest meaningful implementation?
- Is there unnecessary complexity?
- Could the same goal be reached with less disturbance to the baseline?

---

## 5. Required Reviewer Output Format

When reviewing a patch, structure comments using the following sections.

## Summary
State in 2–5 sentences:
- what the patch changes,
- whether it is directionally correct,
- and whether it should be accepted, revised, or split.

## Alignment
Comment on:
- whether the patch matches the project’s intended scientific direction,
- whether it implements the correct phase objective,
- whether it introduces conceptual drift.

## Baseline Safety
Comment on:
- preserved interfaces,
- compatibility risk,
- architectural invasiveness,
- train/inference safety.

## Modularity and Ablation
Comment on:
- config switches,
- module boundaries,
- testability,
- clarity of attribution.

## Risks
List concrete risks only:
- runtime,
- shape mismatch,
- instability,
- accidental semantic change,
- logging blind spots,
- checkpoint incompatibility.

## Requested Revisions
When revisions are needed, be specific.
Do not say “needs cleanup”.
Instead say exactly:
- what to move,
- what to rename,
- what to split,
- what to document,
- what to make switchable,
- what to remove.

---

## 6. Review Preferences for Code Structure

Prefer patches that:
- add new modules/classes with clear names,
- keep state logic centralized,
- add explicit docstrings,
- annotate tensor shapes where useful,
- preserve script-level compatibility,
- and isolate research logic from utility logic.

Be cautious with patches that:
- spread one concept across too many files,
- hide research logic in helper functions,
- add magic constants without config entries,
- or mutate existing baseline code in place without explanation.

---

## 7. Logging and Debug Review Requirements

Whenever a patch introduces:
- state propagation,
- state factorization,
- gating,
- consolidation,
- release,

review whether the patch also exposes enough observability.

At minimum, encourage logging of:
- trend feature norm,
- residual feature norm,
- trend update magnitude,
- residual update magnitude,
- consolidation gate mean/sparsity,
- eligibility statistics,
- release sparsity if enabled.

A method that cannot be inspected is harder to debug and harder to defend scientifically.

---

## 8. Patch Splitting Policy

If a patch implements more than one major phase at once, request a split unless there is a strong reason not to.

Examples:
- Do not combine state refactor + dual-state + consolidation in one patch.
- Do not combine spectral refinement + release + loss redesign in one patch.
- Do not combine large refactors with broad formatting cleanup.

One patch should correspond to one main scientific step whenever possible.

---

## 9. Final Review Standard

A patch is ready to merge when all of the following are true:

1. It is aligned with the intended method.
2. It stays within the current phase scope.
3. It preserves baseline compatibility where expected.
4. It is modular and ablatable.
5. It is sufficiently documented.
6. It has a minimal validation path.
7. It does not introduce major conceptual drift.
8. It makes the research codebase clearer, not harder to reason about.

If any of these are not true, request revision before merge.