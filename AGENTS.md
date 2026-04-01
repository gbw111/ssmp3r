# AGENTS.md

## 0. Purpose of this file

This repository uses Codex as a coding agent for research-grade model development on top of CUT3R.

This file defines:
- the scientific goal of the project,
- the exact method direction,
- the non-goals and boundaries,
- the implementation strategy,
- the stage-by-stage workflow,
- the acceptance criteria for every coding task,
- the coding and experimentation discipline that Codex must follow.

Read this file as the primary project contract before making any code changes.

The goal is not to produce arbitrary “working code”.
The goal is to implement a research method faithfully, incrementally, and in a way that preserves the CUT3R baseline wherever possible.

---

## 1. Project identity

### 1.1 Baseline
The baseline is **CUT3R**.

Treat CUT3R as:
- a recurrent / streaming 3D reconstruction baseline,
- with persistent state,
- with state-update and state-readout logic,
- and with existing encoder / decoder / head / training interfaces that should be preserved as much as possible.

Do **not** rewrite the whole model from scratch.
Do **not** casually replace the backbone.
Do **not** convert the project into a different memory paradigm unless explicitly instructed.

### 1.2 Project objective
The project objective is to redesign the **internal memory dynamics** of CUT3R.

This project is **not** primarily about:
- a smarter patch write rule,
- a larger memory bank,
- sliding window engineering,
- external explicit memory routing,
- test-time parameter optimization,
- or a completely new architecture family.

Instead, this project is about:

> turning CUT3R’s single persistent state into a **stable dual-state propagation system**, where new evidence is first absorbed by a fast residual branch, then gradually consolidated into a slow trend branch, with optional frequency-aware refinement added later.

### 1.3 One-sentence method summary
The method to implement is:

> Replace CUT3R’s single persistent state update with a **stable state propagation mechanism over dual states** (trend and residual), where residual captures fast local corrections, trend captures slow global stable structure, and residual-to-trend consolidation gradually forms long-term memory; frequency-aware refinement is a later enhancement rather than the first implementation target.

---

## 2. Scientific motivation and method interpretation

### 2.1 Core problem statement
Existing streaming 3D reconstruction methods often improve “how to write memory”, but this project is based on a higher-level view:

**memory should be treated as a dynamic system, not a passive cache.**

The central scientific idea is:
- memory should evolve with stability constraints,
- memory should be factorized by temporal role,
- stable structures should be consolidated rather than instantly written,
- and different spectral bands should eventually be read and refined differently.

### 2.2 What problem this method is solving
The method aims to solve the following issues in recurrent streaming reconstruction:

1. **State corruption by direct overwrite**
   - naive updates can let current noisy observations disturb long-term scene structure.

2. **Mixing long-term stable information with short-term spiky corrections**
   - a single homogeneous state causes interference between stable geometry and local transient errors.

3. **Lack of memory formation**
   - new information is often either written immediately or forgotten immediately, without a principled mechanism for gradual long-term consolidation.

4. **Poor separation of global structure vs local detail**
   - global geometric layout and high-frequency corrective evidence are often mixed in one update channel.

### 2.3 The method philosophy
The philosophy of this project is:

- **state update should become state propagation**
- **single state should become dual state**
- **instant write should become staged consolidation**
- **uniform feature usage should eventually become frequency-aware evidence routing**

This hierarchy matters.

The priority order is:
1. stable propagation,
2. dual-state factorization,
3. consolidation,
4. optional release,
5. spectral refinement.

Do not invert this priority.

---

## 3. Method definition

## 3.1 The three primary components

### A. Stable state propagation
We do **not** want direct algebraic overwrite of persistent state.

Instead:
- derive a candidate update from current observation and previous fused memory,
- use control signals to propagate internal hidden state stably,
- then write back to external state.

Interpretation:
- memory has inertia,
- memory has decay,
- memory absorbs evidence selectively,
- memory evolves before it is written back.

This is the main conceptual contribution.

### B. Dual-state factorization
The persistent state is split into two functional states:

#### Trend state
Role:
- slow-changing,
- global,
- stable,
- long-term,
- supports scene-level structure and trajectory consistency.

Trend state should be:
- conservative,
- stable,
- not easily disturbed by short-term noise,
- the main carrier of long-term geometry anchors.

#### Residual state
Role:
- fast-changing,
- local,
- corrective,
- short-term,
- absorbs new detail and transient corrections.

Residual state should:
- react quickly,
- carry local residual evidence,
- handle new edges / textures / newly visible structure,
- absorb short-term disturbances without corrupting trend.

Important:
**trend/residual is a temporal-dynamics factorization, not a frequency factorization.**

Do not confuse:
- trend/residual with low/high frequency,
- or force them into a four-way state split in early implementations.

### C. Residual-to-trend consolidation
This mechanism is essential.

New evidence should **not** go directly into trend by default.

Correct logic:
1. new evidence enters residual,
2. residual accumulates eligibility / persistence,
3. stable and repeatedly useful residual information is gradually consolidated into trend.

This mechanism is what turns “two branches” into an actual memory formation system.

Without consolidation, dual-state is only a split buffer.
With consolidation, dual-state becomes a genuine dynamic memory architecture.

---

## 3.2 Optional but secondary components

### D. Trend-to-residual release
This is optional in the first strong implementation.

Interpretation:
- release is an emergency valve,
- not the main memory mechanism,
- used only when trend persistently fails to explain current evidence.

Release should be:
- local,
- weak,
- sparse,
- disabled by default in early versions unless explicitly required.

Do **not** make release symmetric with consolidation.
They are not equally important.

### E. Spectral refinement
This is a later enhancement.

Interpretation:
- low-frequency information supports global structure,
- high-frequency information supports local edges / corners / texture discontinuities.

But:
- frequency decomposition should **not** be used to define the core state partition in v1,
- spectral design should be applied later at readout / refinement stages,
- not immediately as a four-state memory split.

---

## 4. Hard boundaries and non-goals

These are strict boundaries. Codex must not drift outside them.

### 4.1 What this project is NOT
This project is **not**:
- explicit pointer memory,
- sliding-window memory replacement,
- camera token pool research,
- patch routing / least-aligned patch selection research,
- test-time optimizer-style training,
- external large memory bank design,
- dynamic-object suppression as the primary theme,
- a full architecture rewrite,
- a “replace everything” project.

### 4.2 What Codex must avoid
Do not introduce these directions unless explicitly requested:
- explicit 3D pointerized memory units,
- replacing persistent state with a window-only system,
- multi-bank patch routing as the main idea,
- full-blown test-time training loops,
- geometry graph optimization pipelines,
- large-scale new auxiliary objective suites,
- four-way state split in the first implementation,
- aggressive release/demotion logic in the initial mainline.

### 4.3 Why this matters
The scientific novelty of this project must stay anchored on:
- **stable propagation**
- **dual-state memory factorization**
- **consolidation**

Anything that shifts the center of gravity away from these three is undesirable.

---

## 5. Global implementation strategy

## 5.1 Development principle
Use **minimum-invasive modification** on top of CUT3R.

Preserve as much of the baseline as possible:
- encoder,
- decoder skeleton,
- output heads,
- training pipeline,
- inference interface,
- data interface,
- checkpoint logic,
- existing losses, unless explicitly extended.

The method should first appear as a **state-path redesign**, not as a total architecture replacement.

## 5.2 Where to cut into the code
Focus changes primarily on:
- model-side persistent state representation,
- state packing / unpacking,
- state update logic,
- recurrent forward path,
- inference-time state passing.

Do **not** start by modifying:
- visualization-only scripts,
- demo-only logic,
- global alignment utilities,
- unrelated training plumbing,
unless compatibility requires it.

## 5.3 Workflow principle
Never attempt to implement the entire method in one step.

Instead, use staged development:
1. refactor state interface,
2. add stable propagation,
3. add dual-state factorization,
4. add consolidation,
5. add optional release,
6. add spectral refinement.

Each stage must be individually runnable and testable.

---

````markdown
## 6. Canonical internal representation

Codex should refactor the existing single persistent state into a structured representation.

### Suggested conceptual structure

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class DualState:
    state_pos: Any                     # state token positions / layout compatible with baseline
    trend_feat: Any                    # external trend state
    residual_feat: Any                 # external residual state
    trend_hidden: Any                  # internal hidden dynamics for trend propagation
    residual_hidden: Any               # internal hidden dynamics for residual update
    eligibility: Any                   # residual-to-trend consolidation buffer
    aux: Optional[Dict[str, Any]] = field(default_factory=dict)  # confidence, motion, persistence, etc.
````

### Requirements

* `trend_feat` and `residual_feat` should initially have shapes compatible with the baseline state shape.
* Avoid changing token count in early versions.
* Avoid changing embedding dimension in early versions.
* Compatibility and stability matter more than theoretical elegance in the first implementation.

### Important

The first stage of this refactor should preserve baseline-equivalent behavior as much as possible.

That means:

* dual containers may exist,
* but they can initially mirror baseline state behavior,
* until propagation and factorization modules are introduced.

---

## 7. Phase-by-phase implementation plan

### Phase 0 — State API refactor

#### Goal

Replace the baseline single-state interface with a structured state container **without changing effective behavior**.

#### Required outcomes

* All baseline calls still work.
* Training and inference remain runnable.
* State packing / unpacking is explicit and centralized.
* Future modules can be inserted cleanly.

#### Required tasks

1. Identify all places where persistent state is:

   * created,
   * updated,
   * passed between functions,
   * serialized or checkpointed,
   * returned to inference/demo code.

2. Replace ad-hoc tuple passing with a named structure:

   * `dataclass`,
   * simple class,
   * or disciplined dict wrapper.

3. Keep backward compatibility where practical.

4. Ensure existing recurrent forward logic still runs.

#### Constraints

* No behavior change intended.
* No new learning behavior yet.
* No spectral module.
* No release module.
* No altered losses.

#### Deliverables

* Modified file list.
* New state data structure.
* Compatibility notes.
* Minimal run test.
* Explanation of unchanged behavior.

---

### Phase 1 — Stable propagation

#### Goal

Introduce a stable propagator for the slow branch logic, while keeping the rest of the system as conservative as possible.

#### Core idea

We do **not** directly overwrite trend state.

Instead:

* candidate update is computed,
* controls are extracted,
* hidden dynamics evolve with contraction,
* external trend state is updated from hidden dynamics.

#### Preferred lightweight formulation

Use a stable contraction-style update first, not a complicated full SSM implementation.

Conceptually:

```python
a = exp(-softplus(a_head(ctrl)))   # stable contraction coefficient
b = sigmoid(b_head(ctrl))          # input injection strength
c = sigmoid(c_head(ctrl))          # write-back strength

trend_hidden = a * trend_hidden + b * proj(delta_tr)
trend_feat   = trend_feat + c * out(trend_hidden)
```

#### Why this is preferred

* stable,
* easy to inspect,
* minimally invasive,
* easy to ablate,
* directly matches the project’s theoretical goal.

#### Inputs for control signal

First implementation may use lightweight control statistics such as:

* confidence statistics,
* candidate-vs-state discrepancy,
* motion magnitude proxy,
* update/reset flags,
* pooled feature statistics.

Do not over-engineer control signals initially.

#### Constraints

* Keep propagation simple.
* Keep it local to state update path.
* Do not redesign all attention blocks.

#### Deliverables

* Trend propagator module.
* Explicit config flag.
* Shape documentation.
* Test that confirms trend propagation runs.
* Explanation of how stability is enforced.

---

### Phase 2 — Dual-state factorization

#### Goal

Split the memory update into trend-like and residual-like components.

#### Important implementation strategy

In the first main implementation:

* the decoder / interaction path may still use a fused memory view,
* do not force the backbone to natively attend to two separate banks from day one,
* first change write-back law, then later consider richer read logic if necessary.

#### Recommended design

1. Create fused previous state:

   * weighted sum or gated fusion of trend and residual.

2. Let baseline-like interaction produce a candidate state.

3. Compute delta between candidate and fused previous state.

4. Factorize delta into:

   * `delta_tr`,
   * `delta_res`.

#### Factorizer recommendation

Use a light gating module first.

Example conceptual behavior:

* pooled stats from candidate delta,
* optional confidence/motion inputs,
* predict token-wise or coarse gate `alpha`,
* `delta_tr = alpha * delta`,
* `delta_res = (1 - alpha) * delta`.

#### Strong recommendation

Prefer:

* token-wise gates,

over:

* full token-channel free-form factorization,

in the first version.

#### Residual update behavior

Residual branch should be:

* faster,
* leakier,
* more reactive,
* but not uncontrolled.

Conceptually:

```python
g = sigmoid(g_head(ctrl))
lam = sigmoid(lam_head(ctrl))

residual_hidden = lam * residual_hidden + proj_res(delta_res)
residual_feat   = g * residual_feat + out_res(residual_hidden)
```

#### Constraints

* No trend/residual four-way frequency split.
* No external memory bank.
* No fancy routing.
* No heavy new loss dependency.

#### Deliverables

* Factorizer module.
* Residual updater module.
* Fused state definition.
* Dual-state recurrent flow diagram in comments/docs.
* Config flags.
* Test for shape and run consistency.

---

### Phase 3 — Residual-to-trend consolidation

#### Goal

Turn dual-state into a true memory system by implementing gradual formation of stable long-term memory.

#### Required interpretation

Consolidation is the main mechanism.

It must be treated as:

* progressive,
* gated,
* evidence-based,
* not an instantaneous hard transfer.

#### Required memory logic

New information path:

1. current evidence influences residual,
2. residual accumulates eligibility,
3. stable residual content is projected into trend,
4. residual is not fully cleared.

#### Preferred first implementation

Use lightweight proxy signals instead of full geometric consistency machinery.

Candidate proxy inputs may include:

* confidence,
* residual persistence,
* update magnitude stability,
* motion proxy,
* temporal re-occurrence score.

#### Eligibility update

Conceptually:

```python
eligibility = rho * eligibility + score(...)
```

#### Consolidation gate

Conceptually:

```python
m = sigmoid(consolidate_head(features))
trend_feat    = trend_feat + m * proj_r2t(residual_feat)
residual_feat = (1 - gamma * m) * residual_feat + residual_new
```

#### Required principle

Consolidation should behave like:

* copy + attenuation,

not:

* hard move + delete.

This is extremely important.

#### What not to do in Phase 3

Do **not** require:

* full reprojection-based geometric validation,
* complex view graph reasoning,
* dynamic-object heavy filtering,
* expensive optimization loops,

unless explicitly instructed later.

#### Deliverables

* Eligibility state.
* Consolidation gate.
* Residual-to-trend projection.
* Config switches.
* Simple ablation path.
* Explanation of why this counts as memory consolidation.

---

### Phase 4 — Optional trend-to-residual release

#### Goal

Provide an optional mechanism for local reallocation of explanatory capacity when trend persistently mismatches current evidence.

#### Status

This is **not** the mainline innovation.
It is an optional emergency valve.

#### Default policy

* Implement interface.
* Default off.
* Document behavior.
* Only enable via config.

#### Desired behavior

Release should be:

* sparse,
* local,
* weak,
* conditional on persistent surprise or mismatch.

#### What release is **not**

Release is not:

* symmetric demotion,
* routine memory shuffling,
* a second main branch of equal importance.

#### Deliverables

* Optional release module.
* Disabled-by-default config.
* Clear explanation of when it should be used.
* Ablation-only usage path unless explicitly promoted later.

---

### Phase 5 — Spectral refinement

#### Goal

Add frequency-aware refinement as a later enhancement, not as the initial core mechanism.

#### Interpretation

Low-frequency information should support:

* global structure,
* coarse geometry,
* stable layout.

High-frequency information should support:

* edges,
* corners,
* texture discontinuities,
* local corrective detail.

#### Strong rule

Do **not** make spectral decomposition define the primary state split in v1.

The preferred insertion point is:

* readout refinement side,
* especially world-branch refinement,
* before changing the whole state structure.

#### Preferred implementation order

1. Introduce a spectral refinement module.
2. Attach it to world readout features or pre-head features.
3. Keep self branch and pose branch minimally disturbed.
4. Only later consider high-frequency revisit specialization.

#### Constraints

* Not required for first successful version.
* Must be independently switchable.
* Should not entangle all heads at once.

#### Deliverables

* Spectral module.
* Insertion-point explanation.
* Config switch.
* Independent ablation path.
* Comments on low/high responsibilities.

---

## 8. Configuration policy

All new functionality must be controlled by explicit config flags.

### At minimum, expose

* `use_stable_propagation`
* `use_dual_state`
* `use_consolidation`
* `use_release`
* `use_spectral`

### Optional finer-grained flags

* `use_tokenwise_factorizer`
* `use_residual_hidden`
* `use_trend_hidden`
* `use_proxy_consolidation`
* `use_highfreq_revisit`
* `freeze_release_by_default`

### Config discipline

* Every feature should be independently switchable.
* Default configs should preserve or approximate baseline behavior before the full method is enabled.
* Ablation configs should be easy to construct.

---

## 9. Coding discipline

### 9.1 General code style

* Keep changes modular.
* Avoid giant monolithic patches.
* Prefer small new classes/modules over tangled inline logic.
* Use explicit names.
* Add docstrings for all new major modules.
* Record input/output tensor shapes in comments where helpful.

### 9.2 State-related coding style

* Centralize state packing/unpacking.
* Do not spread state-format assumptions across many files.
* Use a named container instead of fragile positional tuples whenever possible.
* Keep backward compatibility wrappers if necessary.

### 9.3 Training/inference discipline

* Avoid changing both training and inference semantics at the same time without necessity.
* Whenever possible, preserve forward return keys.
* Preserve existing script usability.

### 9.4 Logging and observability

When adding new modules, expose useful diagnostics where practical:

* trend/residual norm statistics,
* consolidation gate statistics,
* eligibility statistics,
* residual/trend update magnitudes,
* release sparsity if enabled.

These diagnostics help validate whether the intended memory behavior is actually happening.

---

## 10. Experimental and ablation discipline

### 10.1 Required ablation path

The intended experiment ladder is:

1. baseline CUT3R
2. CUT3R + stable propagation
3. CUT3R + stable propagation + dual-state
4. CUT3R + stable propagation + dual-state + consolidation
5. optional release
6. spectral refinement

Do not skip this order unless explicitly requested.

### 10.2 Why this matters

This order reflects the scientific logic of the method:

* dynamics first,
* factorization second,
* memory formation third,
* refinement later.

This is not just an engineering preference.
It is part of the scientific claim.

### 10.3 Loss discipline

Do not add many auxiliary losses at once.

If regularization is needed, prefer minimal and interpretable forms first, for example:

* trend/residual decorrelation penalty,
* gate sparsity penalty,
* release sparsity penalty.

Only add new losses when necessary and explain why.

---

## 11. Acceptance criteria for every coding task

Every task completion must include:

### 1. Summary of what changed

* concise but technically clear.

### 2. Modified file list

* all edited files.

### 3. New module list

* class/function names,
* purpose,
* shape summary.

### 4. Behavioral impact

* what baseline behavior is preserved,
* what new behavior is introduced.

### 5. Config additions

* new flags,
* defaults,
* effect.

### 6. Minimal validation

* what command or test was run,
* what passed,
* what remains unverified.

### 7. Risks / caveats

* training stability,
* compatibility concerns,
* expected failure modes.

### 8. Suggested next step

* exactly one most sensible next implementation step.

Do not just dump a diff.
Explain the reasoning.

---

## 12. What Codex should do when uncertain

If there is ambiguity:

* prefer the more conservative implementation,
* preserve baseline compatibility,
* ask whether the uncertainty concerns scientific intent or only minor engineering detail,
* avoid silently changing the research direction.

When choosing between:

* elegant but invasive,

versus:

* simpler and more baseline-compatible,

prefer:

* simpler and more baseline-compatible,

unless explicitly instructed otherwise.

---

## 13. Repository operating procedure for Codex

For any substantial task, follow this order:

### Step 1 — Inspect

* identify baseline state flow,
* locate recurrent forward path,
* map model/inference/demo dependencies.

### Step 2 — Plan

* define the smallest meaningful patch,
* state what will and will not change,
* identify risk points.

### Step 3 — Implement

* make modular changes,
* keep them localized,
* avoid unrelated cleanup.

### Step 4 — Validate

* run the narrowest meaningful test first,
* confirm import/runtime compatibility,
* confirm shapes and state passing.

### Step 5 — Report

Return:

* what changed,
* why,
* what passed,
* what is next.

---

## 14. First-task priority queue

Unless the user explicitly overrides, the implementation priority is:

### Priority 1

State API refactor.

### Priority 2

Stable trend propagator.

### Priority 3

Dual-state factorizer + residual updater.

### Priority 4

Consolidation.

### Priority 5

Optional release.

### Priority 6

Spectral refinement.

Do not implement priority 4–6 before priority 1–3 are operational.

---

## 15. Concrete first-task instruction

If no more specific task is provided, the default first task is:

> Refactor CUT3R’s single persistent state into a structured dual-state-ready container, while preserving baseline-equivalent behavior and compatibility across model forward, recurrent inference, and script-level state passing.

### Expected output

* changed files,
* new state structure,
* compatibility notes,
* minimal run validation,
* next-step recommendation.

---

## 16. Final reminder

This repository is a research implementation project.
The main objective is not code novelty by itself.

The objective is to faithfully realize the following scientific narrative:

* state update becomes stable propagation,
* single memory becomes dual-state memory,
* temporary evidence becomes long-term memory through consolidation,
* spectral refinement is layered on top later.

Stay aligned with this narrative.
Do not drift into nearby but different ideas.
Implement incrementally, preserve the baseline where possible, and make every step ablatable.

```
```
