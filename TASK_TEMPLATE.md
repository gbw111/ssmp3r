# TASK_TEMPLATE.md

# Task Template for Codex on CUT3R-Based Stable Dual-State Project

## 0. Purpose

Use this template for every substantial Codex task in this repository.

This template ensures that every task:
- stays aligned with the project method,
- has a bounded scope,
- preserves ablation clarity,
- and produces reviewable output.

Each task should be written as an implementation contract, not as a vague request.

---

## 1. How to use this template

When creating a task for Codex:

1. Fill in all required sections.
2. Keep the task limited to one main phase or one narrow sub-problem.
3. Explicitly state what must not be changed.
4. Require a structured completion report.
5. Require minimal validation.
6. Require explanation of compatibility and next step.

Do not ask Codex to “implement the full method” in one task.
Do not leave scientific intent implicit.

---

## 2. Task Template

Copy from the next line onward for each new task.

---

# Task Title
`[Insert a short, concrete task title]`

## Objective
Implement a bounded change for the CUT3R-based stable dual-state project.

This task belongs to:

- [ ] Phase 0 — State API refactor
- [ ] Phase 1 — Stable propagation
- [ ] Phase 2 — Dual-state factorization
- [ ] Phase 3 — Consolidation
- [ ] Phase 4 — Optional release
- [ ] Phase 5 — Spectral refinement
- [ ] Other narrowly scoped support task: `[describe]`

## Scientific Intent
Explain which exact method idea this task serves.

Use explicit language such as:
- stable state propagation,
- trend/residual separation,
- residual-to-trend consolidation,
- optional release,
- spectral refinement.

State clearly what this task is **not** trying to do.

Example:
This task is implementing stable propagation for the slow memory branch.
It is not redesigning the CUT3R backbone, not changing the loss framework, and not introducing spectral decomposition.

## Background Context
Summarize the project context that Codex must remember while doing this task.

Required reminders:
- CUT3R is the baseline.
- The project uses minimum-invasive modification.
- The primary contribution is state-path redesign.
- The main scientific order is:
  1. stable propagation,
  2. dual-state,
  3. consolidation,
  4. optional release,
  5. spectral refinement.

Add any task-specific context here:
`[insert detailed context]`

## Exact Goal
Describe exactly what the code should do after this task is complete.

Be concrete.
Good examples:
- replace tuple-based state passing with a structured DualState container while preserving baseline behavior,
- add a TrendPropagator module with explicit contraction-based update,
- split candidate delta into trend-like and residual-like components with a token-wise factorizer,
- add eligibility state and a consolidation gate for residual-to-trend injection.

Bad examples:
- improve memory,
- optimize recurrent reconstruction,
- make the model more stable.

## In-Scope Changes
List the files, modules, or logic that Codex is allowed to modify.

Examples:
- model-side state structures,
- recurrent forward logic,
- inference-time state passing,
- config definitions for new module switches,
- narrow logging additions for trend/residual/consolidation diagnostics.

Explicitly list likely relevant files if known:
- `[file path 1]`
- `[file path 2]`
- `[file path 3]`

## Out-of-Scope Changes
List what Codex must not modify in this task unless absolutely necessary.

Examples:
- encoder backbone,
- decoder architecture beyond the narrow state insertion point,
- output head semantics,
- unrelated training scripts,
- demo visualization behavior unless required for compatibility,
- global alignment utilities,
- unrelated cleanup or formatting-only edits.

This section is mandatory.

## Required Design Constraints
List non-negotiable design rules for this task.

Examples:
- preserve baseline-compatible forward return structure,
- keep the patch minimally invasive,
- expose new behavior through config flags,
- make every new module independently switchable,
- keep the implementation modular and readable,
- prefer token-wise gate over full free-form tensor factorization,
- do not add new losses in this task,
- do not hardcode always-on behavior.

Fill here:
1. `[constraint]`
2. `[constraint]`
3. `[constraint]`

## Method-Specific Requirements
Describe the exact method rules that apply to this task.

Examples for different phases:

### For stable propagation
- use explicit stability enforcement,
- keep the propagation local to the state update path,
- do not rewrite the whole recurrent stack.

### For dual-state
- keep trend/residual as temporal roles,
- do not implement four-way frequency split,
- allow fused state for baseline interaction in early versions.

### For consolidation
- residual receives new evidence first,
- consolidation must be gradual,
- use copy-plus-attenuation, not hard transfer,
- use lightweight proxy inputs before geometric consistency machinery.

### For release
- default disabled,
- local and sparse,
- not symmetric with consolidation.

### For spectral refinement
- do not redefine the core state split,
- prefer readout-side insertion first,
- keep it independently switchable.

Fill here:
`[insert detailed method-specific rules]`

## Expected New Modules or Interfaces
List the exact modules, classes, functions, or config flags that should exist after this task.

Examples:
- `DualState`
- `TrendPropagator`
- `ResidualUpdater`
- `TemporalFactorizer`
- `ConsolidationGate`
- `use_dual_state`
- `use_consolidation`

Fill here:
- `[module or flag 1]`
- `[module or flag 2]`
- `[module or flag 3]`

## Shape / Interface Expectations
If known, specify expected tensor or interface behavior.

Examples:
- trend/residual feature shape should initially match baseline state feature shape,
- state container should preserve token count,
- eligibility should align with token dimension,
- forward outputs should preserve existing keys,
- inference scripts should continue to accept and return state cleanly.

Fill here:
`[insert tensor/interface expectations]`

## Validation Requirements
Specify the minimum validation Codex must perform.

Examples:
- import check,
- unit-level shape check,
- dry run for recurrent forward,
- one inference step,
- config toggling test,
- baseline-compatibility smoke test.

Mandatory fields:
1. Minimal command or test to run: `[insert]`
2. What must pass: `[insert]`
3. What can remain unverified at this stage: `[insert]`

## Required Completion Report
When the task is complete, Codex must respond using the following structure:

### 1. What changed
A concise technical summary.

### 2. Modified files
A complete file list.

### 3. New modules / interfaces
Each with purpose and I/O summary.

### 4. Config additions
List new flags and defaults.

### 5. Validation performed
Exact commands/tests run and outcome.

### 6. Compatibility notes
What remains baseline-compatible and what changed.

### 7. Risks / caveats
Concrete risks only.

### 8. Suggested next step
Exactly one recommended next task.

## Success Criteria
The task is successful only if all of the following are true:

- the implementation matches the intended method role,
- the patch stays within scope,
- baseline compatibility is preserved where expected,
- new behavior is modular and switchable,
- the code is reviewable and testable,
- validation was performed,
- the completion report is structured and complete.

## Additional Notes
Add any extra repository-specific instruction here.

Examples:
- preserve checkpoint field names where possible,
- avoid breaking old configs,
- comment tensor shapes in new state modules,
- keep logging additions lightweight,
- do not mix this task with formatting cleanup.

Fill here:
`[insert additional notes]`

---

## 3. Example Filled Task — Phase 0 State API Refactor

# Task Title
`Refactor persistent state into a structured DualState container`

## Objective
Implement a Phase 0 structural refactor for the CUT3R-based stable dual-state project.

This task belongs to:

- [x] Phase 0 — State API refactor
- [ ] Phase 1 — Stable propagation
- [ ] Phase 2 — Dual-state factorization
- [ ] Phase 3 — Consolidation
- [ ] Phase 4 — Optional release
- [ ] Phase 5 — Spectral refinement

## Scientific Intent
This task prepares the repository for later implementation of stable propagation, dual-state factorization, and consolidation by replacing fragile single-state passing with a structured state container.

This task is not:
- changing model behavior,
- introducing new learning dynamics,
- altering baseline heads,
- adding spectral logic,
- or implementing consolidation yet.

## Background Context
CUT3R is the baseline.
The project uses minimum-invasive modification.
The method target is a stable dual-state memory system, but this task only prepares the state API.

All changes should preserve baseline-equivalent behavior as much as possible.

## Exact Goal
Replace tuple-like or ad hoc persistent state passing with a structured state container that is dual-state-ready, while preserving current model behavior and keeping train/inference/script-level flows runnable.

## In-Scope Changes
- model-side state creation and return logic,
- recurrent state passing,
- inference-time state_args packing/unpacking,
- narrow compatibility changes in scripts that consume returned state.

Likely files:
- `src/.../model.py`
- `src/.../inference.py`
- relevant demo/inference wrappers if they directly unpack state

## Out-of-Scope Changes
- encoder architecture,
- decoder architecture,
- output heads,
- loss logic,
- training objective,
- spectral modules,
- consolidation logic,
- release logic,
- unrelated refactoring or cleanup.

## Required Design Constraints
1. Preserve baseline-equivalent behavior.
2. Use a named state structure instead of fragile positional tuples where possible.
3. Keep compatibility wrappers if needed.
4. Do not change token count or embedding dimensions.
5. Do not add new learning behavior in this task.

## Method-Specific Requirements
- The new structure should be ready for future fields such as trend_feat, residual_feat, trend_hidden, residual_hidden, eligibility, and aux stats.
- If full dual-state is not yet active, the container may internally mirror baseline state.
- State lifecycle must become easier to reason about after this task, not harder.

## Expected New Modules or Interfaces
- `DualState` or equivalent state container
- centralized pack/unpack helpers if needed
- compatible `state_args` handling
- config-neutral behavior

## Shape / Interface Expectations
- existing state-facing call sites should still run,
- state token count must remain unchanged,
- forward outputs should remain compatible unless unavoidable and documented,
- returned state should be easy to pass into the next recurrent step.

## Validation Requirements
1. Minimal command or test to run: import test and one minimal recurrent forward/inference smoke test
2. What must pass: model import, state creation, state return, second-step state reuse
3. What can remain unverified: full training stability across long runs

## Required Completion Report
Use the standard 8-section completion report exactly.

## Success Criteria
- structured state container exists,
- behavior is effectively baseline-equivalent,
- inference state passing still works,
- patch is limited in scope,
- validation is reported,
- next step is clearly identified.

## Additional Notes
Do not combine this task with unrelated cleanup.
If backward compatibility cannot be preserved in one place, document it precisely.

---

## 4. Example Filled Task — Phase 1 Stable Propagation

# Task Title
`Add a stable trend propagator to the recurrent state update path`

## Objective
Implement Phase 1 stable propagation for the slow memory branch.

## Scientific Intent
This task introduces the project’s primary conceptual change: state update becomes stable propagation.

This task is not:
- full dual-state yet,
- not consolidation,
- not spectral refinement,
- not a backbone rewrite.

## Background Context
The project’s highest-priority innovation is to stop treating persistent state as a passive cache and instead evolve it with an explicitly stable propagation rule.

## Exact Goal
Insert a lightweight stable propagator into the recurrent state update path so that trend-like state is updated through contraction-style hidden dynamics rather than direct overwrite.

## In-Scope Changes
- recurrent update path,
- model-side state update logic,
- addition of TrendPropagator module,
- lightweight control-signal extraction if needed,
- config switch for stable propagation.

## Out-of-Scope Changes
- full dual-state split,
- consolidation,
- release,
- spectral readout,
- large loss redesign,
- major decoder refactor.

## Required Design Constraints
1. Stability must be explicit in code.
2. The feature must be switchable.
3. The implementation must remain lightweight and local.
4. Existing forward outputs must remain usable.
5. Avoid broad architectural disturbance.

## Method-Specific Requirements
- Prefer contraction-style update such as `exp(-softplus(...))` for decay/stability.
- Hidden state should evolve before write-back.
- Control inputs should stay lightweight in this phase.
- Do not redesign attention blocks wholesale.

## Expected New Modules or Interfaces
- `TrendPropagator`
- `use_stable_propagation`
- optional control feature helper
- trend update logging hooks if practical

## Shape / Interface Expectations
- trend state shape should match baseline state feature shape in early implementation,
- no token-count change,
- no head contract change.

## Validation Requirements
1. Minimal command or test to run: one recurrent forward/inference smoke test with the flag both off and on
2. What must pass: import, runtime, state reuse, no shape mismatch
3. What can remain unverified: long training stability

## Required Completion Report
Use the standard 8-section completion report exactly.

## Success Criteria
- stable propagator exists,
- it is inserted at the intended update point,
- it is switchable,
- runtime works,
- scope remains controlled.

## Additional Notes
Do not combine this task with dual-state factorization unless explicitly asked.

---

## 5. Minimal Short Task Form

Use this compact form only for very small follow-up tasks.

# Task Title
`[title]`

## Phase
`[phase]`

## Goal
`[exact goal]`

## In Scope
- `[item]`
- `[item]`

## Out of Scope
- `[item]`
- `[item]`

## Constraints
- `[constraint]`
- `[constraint]`

## Validation
- Run: `[command/test]`
- Must pass: `[requirement]`

## Report Required
- changed files
- modules/interfaces
- config additions
- validation
- compatibility
- risks
- next step

---

## 6. Final Reminder for Task Authors

A good Codex task in this repository must do all of the following:
- identify the implementation phase,
- name the exact scientific intent,
- define scope tightly,
- forbid likely drift directions,
- specify constraints,
- demand validation,
- demand a structured report.

If any of these are missing, the task is underspecified.