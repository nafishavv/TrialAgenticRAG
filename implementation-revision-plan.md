# Implementation Revision Plan — Naive / Enhanced / Agentic RAG

> **Status:** Architecture-freeze document. Design phase only — no code is changed by this file.
> **Reviewer stance:** Software architect + RAG researcher + thesis examiner + senior code reviewer.
> My job is to challenge, not rubber-stamp. Every claim is tied to code in `src/ragtrial/…` and to the
> measured results already committed (`docs/EVAL_REPORT.md`, `eval/results/`) and the two prior reviews
> (`capabilities-review.md`, `architecture_review.md`).
> **Effort tags:** Low (hours) · Medium (1–3 days) · High (week+).

---

## 1. Executive Summary

**Overall: this revision is a clear scientific improvement over the current implementation, on one axis,
and a lateral move on the other.** It should be adopted — with one confound named honestly in the thesis.

**Why it is stronger.** The single worst methodological flaw in the current system (flagged in both prior
reviews) is that **Naive vs Enhanced changes 3–5 variables at once** (routing + rewrite + retrieval stack).
Your revision *fixes exactly this on the Naive↔Enhanced axis*:

- **Naive** = fan-out + dense + no rerank.
- **Enhanced** = fan-out + **hybrid + cross-encoder rerank** + intent gate.

Everything else (fan-out, fixed control flow, same corpus/embedder/LLM) is now held constant, so
**Naive→Enhanced isolates a single scientific question: "does a better retrieval *stack* (hybrid +
rerank) pay for itself?"** That is a clean, defensible, publishable comparison. This is the most valuable
thing in the whole revision, and it directly implements `capabilities-review.md` §8.1.

**Why it is only lateral on the other axis.** Enhanced→Agentic still bundles **three** simultaneous
changes: control flow (fixed→LLM), domain routing (fan-out→routed), and query rewriting
(none→adaptive). So "Agentic beats Enhanced" will **still** not be attributable to "agency" — it could be
routing alone, or rewriting alone. This is the same confounding the prior reviews flagged, merely moved
one tier up. It is *acceptable for an undergraduate thesis* if you frame Agentic honestly as an
**adaptive bundle vs a fixed bundle**, and log enough per-query trace to decompose it post-hoc. It is
**not** acceptable if you claim a single-variable causal result.

**Feasibility: high.** ~80% of this is configuration and reuse, not new architecture:

- Enhanced is already fully config-assembled (`EnhancedRAGConfig` + `build_enhanced`), so re-defining it =
  changing defaults, not rewriting a pipeline.
- Hybrid retrieval is already built and already the capability default (so Agentic already uses it).
- Agentic already does implicit domain routing (tool = domain) and implicit adaptive rewrite (per-tool
  `query` arg).
- The two new builds are: **(1)** the **cross-encoder reranker** (currently a `NotImplementedError`
  stub) — on the critical path for **both** Enhanced and Agentic; and **(2)** a **unified execution log**
  (a normalized decision-trace emitted identically by all three modes, for visualization + debugging —
  §5A). Both are small.

**Scope confirmed with author (this revision):**
- **Semantic router = intent gate ONLY.** Domain routing is fully removed from Enhanced, so there is *no*
  semantic router doing collection selection anywhere in Enhanced — the only semantic-router usage is the
  VALID/INVALID intent gate. (§2.4)
- **Keep the retrieval-stack ablation presets** in Enhanced (low effort, high analytical value). (§8.3)
- **Reranker testing starts small:** it is time/token-heavy, so first validation runs on a **10–20
  question subset** of the test set before any full re-run. (§6, §7)
- **Deferred (revisit after seeing first results, not now):** the `agentic-fanout` ablation and the
  routing fan-out fallback (§8.1, §8.2); deeper confound-decomposition (§7). These are parked, not
  rejected.

**Remaining concerns (ranked):**

1. **The Enhanced↔Agentic comparison remains confounded** (control flow + routing + rewrite). Name it;
   don't hide it. (§2.3, §7)
2. **Predicted non-monotone result:** because Enhanced now *fan-outs* (high recall) while Agentic
   *routes* (recall-lossy when wrong), you may measure **Enhanced ≥ Agentic on raw retrieval recall**,
   with Agentic winning on answer-quality/efficiency instead. Your data already shows fan-out > routing.
   Be ready to *report* this as the finding, not treat it as a bug. (§7)
3. **Reranker is now load-bearing for two tiers** — a heavy model (`bge-reranker-v2-m3`,
   `sentence-transformers`/torch) with real first-call latency, especially on CPU/free-tier. (§7)
4. **Fairness of the reranked context window** must be standardized (same final top-N into the generator
   across tiers), or Enhanced/Agentic get an uncontrolled context-size advantage over Naive. (§7)
5. **Retrieval metrics are still generator-coupled** (unchanged from prior reviews) — all recall numbers
   remain optimistic upper bounds. Out of scope for this revision, but do not let the revision imply it
   was fixed. (§7)

**Verdict: adopt the revision. Build the reranker first (validate on a 10–20 Q subset), add the unified
execution log so all tiers are inspectable in one format, keep the residual Enhanced↔Agentic confound
explicit in Bab metodologi, and pre-register the expectation that the tiers may be non-monotone.**

---

## 1.5 STORAGE DECISION — Unified single index (supersedes per-domain collections)

**Decided 2026-07-10 (author):** move from 4 per-domain Chroma collections to **ONE unified collection**
with a `domain` metadata field on every chunk. This is the shared substrate for all three tiers.

**Rationale.** Canonical RAG (naive → advanced) uses a single index; multi-collection routing is a
*Modular-RAG-era* elaboration. The per-domain design forced fan-out + cross-collection merge + a
per-source candidate quota (the exact thing we got stuck tuning), and made domain routing
*catastrophic-when-wrong* (route to the wrong collection → 0 recall). Unified fixes all of this:

- **Naive** = one index, dense top-k. Textbook-clean (no fan-out/merge).
- **Enhanced** = retrieve global top-N (dense|hybrid) → rerank → top-5. **No fan-out, no per-source quota,
  no pool arithmetic.** The candidate pool is simply the global top-N.
- **Agentic** = domain routing becomes a **metadata filter** (`where={"domain": X}`) — safe/additive, not a
  hard collection gate. Still a real, studyable capability; just the production-grade form.
- **Hybrid BM25** = one index over the whole corpus (simpler + correct) instead of 4 per-collection.
- **Controlled variable** = literally one identical index across all tiers (stronger than "4 identical
  collections"). Matches `capabilities-review.md` §11's recommendation.

**What is kept.** Domain-specific ingestion (chunkers/preprocess) is untouched. Per-domain `Capability`
objects are kept for **formatting / gold-id / citation / agentic-tool metadata**, dispatched by the
chunk's `domain` (== `_source`) metadata via `format_context`. They stop being the retrieval backend.

**Build cost: Low.** Copy existing per-domain vectors into one collection, stamping `domain` — **no
re-embed** (mirrors the retired `vectorstore/unified.py build_unified()` deleted in commit #9). Chunking,
gold IDs, and the test set are unaffected (gold IDs are per-chunk, storage-agnostic).

**What shifts.** Agentic tools change from "one tool per *collection*" to "search + `domain` filter" (or N
filter-tools over one index). Hybrid + domain-filter needs care on the BM25 side (filter the doc set before
BM25); Enhanced doesn't filter (global) so it's unaffected. "Routing accuracy" in eval re-interprets from
collection-choice to filter-choice (same computation).

---

## 2. Critical Review of Every Decision

### 2.1 Naive RAG — dense fan-out, non-modular, monolithic

**Summary.** Simplest baseline: dense retrieval only, fan-out over all collections, no intent / rewrite /
routing / rerank / reflection / iteration. Intentionally **not** folded into the modular pipeline.

**Opinion: keep exactly as proposed. This is correct and already matches the code.** `ask_naive`
([rag/naive.py](src/ragtrial/rag/naive.py)) is already a single self-contained function: fan-out
`search_with_scores` across `SEARCHABLE_CAPABILITIES`, merge by L2 distance, stuff, one LLM call.

**Strengths.**
- A monolithic baseline is *methodologically correct*, not a shortcut. The baseline must be the "honest
  floor" with zero decision machinery, so every gain in the other tiers is attributable to something.
- Keeping it out of the modular pipeline **prevents accidental capability leakage** (e.g. an intent gate
  silently applying to "Naive" because it shares a Stage list). Physical separation enforces the
  scientific boundary. This is a *research-grade* reason to resist DRY here.
- It already produces the most interesting result in your data (best raw recall precisely *because* it
  never routes). That result depends on Naive staying dumb.

**Weaknesses / objections.**
- A reviewer may ask: *"Is dense-only a fair baseline, or a strawman, now that Enhanced gets hybrid+rerank?"*
  Defensible answer: Naive is defined as the **canonical minimal RAG** (dense similarity + stuff), which
  is the standard baseline in the literature. Say so explicitly and cite it.
- "Non-modular" must mean **pipeline-non-modular**, not **corpus-independent**. Naive must still read the
  *same* Chroma collections / embedder as the others (the controlled variable). The current code does this
  correctly via the shared registry. Do not "purify" it into its own store — that would break the
  controlled comparison. Flag this in the doc so no one refactors it the wrong way.

**Implementation risk:** ~none. No change required beyond confirming it stays dense fan-out.

**Recommendation: KEEP as-is. Zero code change** (beyond the execution-log field, §5A).

---

### 2.2 Enhanced RAG — fixed pipeline: intent gate + hybrid + rerank + fan-out; NO routing, NO rewrite

**Summary.** Deterministic fixed pipeline. Adds binary **intent handling** (semantic-router VALID/INVALID,
*not* collection routing), **hybrid retrieval**, **cross-encoder rerank**, over **fan-out** across all
collections. Explicitly excludes domain routing, query rewriting, reflection, iteration, planning.

**Opinion: this is the strongest single decision in the revision. Adopt it.** It converts Enhanced from
"a crippled bundle measured with its levers off" into "a clean retrieval-stack treatment on top of Naive."

**Strengths.**
- **It isolates the retrieval stack.** Naive and Enhanced now differ *only* in {dense→hybrid, +rerank,
  +intent gate}. That is the clean ablation `capabilities-review.md` §8.1 and §13 argued for.
- **It kills the two things your data says were hurting Enhanced:** HyDE (worst latency 14.6s, no recall
  gain) and semantic domain routing (0.72 routing accuracy dragging recall from 0.81→0.62). Removing both
  is *evidence-driven*, not aesthetic. Your own "Notes Aku" instinct (routing should be retrieve/don't,
  not domain selection) is vindicated here.
- **It finishes the tier honestly.** Turning on hybrid + building the reranker means Enhanced is finally
  measured *complete*, so any Naive-vs-Enhanced claim is fair.
- Intent handling as the *only* branch keeps the pipeline deterministic: every VALID query takes an
  identical path. That is exactly what "fixed pipeline" should mean.

**Weaknesses / objections.**
- **Naive→Enhanced still bundles two retrieval changes** (dense→hybrid **and** +rerank) plus the intent
  gate. So "Enhanced beats Naive" won't tell you *whether hybrid or rerank* did the work. **Mitigation:**
  keep the `PRESETS` ablations (`hybrid-only`, `rerank-only`, `dense+rerank`) available for a within-tier
  factorial (`capabilities-review.md` §13.3). The headline tier bundles them; the ablation decomposes them.
  This is cheap because the pipeline is already config-driven.
- **Intent gate is a confound vs Naive too** (Naive can't refuse; Enhanced can). But this one is *cleanly
  measurable already* via the intent eval (recall_invalid 0.00 vs 1.00), and it is cheap and safe. Keep it,
  and report its effect separately rather than letting it muddy the retrieval comparison.
- A reviewer may object that **"Enhanced with no query rewriting isn't really 'advanced RAG'."** Rebuttal:
  your corpus is legal/permit text where *exact lexical terms* (NIK, pasal/Perda numbers, DPMPTSP) dominate;
  hybrid (BM25) + cross-encoder rerank is the *right* enhancement for lexical-heavy text, and HyDE
  measurably didn't help. This is a defensible, corpus-specific design justification — put it in the doc.

**Implementation risk:** Low. The pipeline already supports every needed stage; the only new code is the
reranker. Main risk is reranker model weight/latency (§7).

**Recommendation: KEEP. This is the centerpiece. New canonical `EnhancedRAGConfig` =
`intent="semantic", rewriter="passthrough", router="none", retrieval="hybrid", reranker="cross_encoder"`.**

---

### 2.3 Agentic RAG — single agent: LLM decides retrieve?/domain?/rewrite?; hybrid + rerank; no reflection/iteration yet

**Summary.** LLM tool-calling loop. Agent decides *whether* to retrieve (implicit intent), *which* domain
(domain routing via tool selection), and *whether/how* to rewrite (adaptive per-tool query). Adds hybrid +
cross-encoder rerank. Explicitly **no** reflection, iterative retrieval, planning, or multi-agent yet.

**Opinion: keep, but with eyes open — this tier is where the science gets soft, and you have already been
warned twice.** Both prior reviews concluded that your current Agentic ≈ "LLM routing + optional retry"
with *no* reflection/grading node — i.e. it lacks the capability that *defines* Agentic RAG in the
literature (CRAG/Self-RAG). Your revision **explicitly postpones** reflection/iteration. That is a
legitimate scoping decision, **but it means you must not market this tier as "agentic reasoning."**

**Strengths.**
- Most of it already exists: implicit intent (call a tool or not), implicit domain routing (tool = domain),
  implicit adaptive rewrite (the `query` arg the LLM writes). Adding hybrid is free (already the capability
  default). So the *incremental* build is small.
- The control-flow distinction from Enhanced is real and mechanically honest: **who decides, and when.**
  Enhanced decides at design time (fixed); Agentic decides at inference time (LLM). That is a genuine
  architectural difference, not cosmetic.
- LLM routing (0.92) genuinely beats semantic centroid routing (0.72) in your data — so Agentic's routing
  is a real, measured advantage over what Enhanced *used* to do.

**Weaknesses / objections (this is the critical one).**
- **Triple confound vs Enhanced.** Enhanced→Agentic changes control flow **+** routing (fan-out→routed)
  **+** rewrite (none→adaptive) at once. You cannot attribute an Agentic win to "agency." A thesis examiner
  who read `capabilities-review.md` §7 will ask precisely this. **You must pre-empt it.**
- **Predicted retrieval regression.** Enhanced now *fan-outs* (your highest-recall configuration); Agentic
  *routes*. Your data says routing *loses* recall vs fan-out. So Agentic may well score **lower retrieval
  recall than Enhanced**, winning instead on answer-quality/faithfulness/latency. That is a *fine* and
  *interesting* result — but only if you *expect and frame* it. If you assume "Agentic > Enhanced
  everywhere," the data will embarrass you.
- **"Agentic without reflection/iteration"** invites the objection *"then it's just function-calling
  routing."* Your defense must be: (a) this is a deliberately scoped single-agent tier; (b) reflection/
  iteration are a documented next phase; (c) the contribution here is the *controlled comparison*, not the
  novelty of the agent. Say this plainly in Bab I/III. Do **not** cite a capability table that shows
  Agentic ✅ Reflection / ✅ Planning — your code has neither.

**Implementation risk:** Medium — the reranker must be wired into the agent's retrieval path (the agent is
a LangGraph loop, *not* a `Pipeline` of `Stage`s), so the reranker logic must be **factored into a shared
function** callable from both the Enhanced `Stage` and the Agentic tools node. Risk = duplicated/divergent
rerank logic if not shared. (§5, §6)

**Recommendation: KEEP, with one required guardrail:** log per-query trace (tool calls, chosen domain,
rewrite, docs) so the bundle can be decomposed post-hoc. (The fan-out fallback for low-confidence routing is
**deferred** per author — §8.2.)

---

### 2.4 Cross-cutting decision — Intent Handling ≠ Domain Routing

**Summary.** Intent Handling = "need retrieval?" (binary). Domain Routing = "which collection?". Enhanced
gets intent handling only; Agentic gets both; Naive gets neither. These must never be conflated.

**Opinion: correct, important, and already correctly separated in the code.** The intent gate
([pipeline/intent.py](src/ragtrial/pipeline/intent.py)) uses the `semantic-router` *library* for a
VALID/INVALID decision; the domain router ([pipeline/route.py](src/ragtrial/pipeline/route.py)) is a
separate in-house `SemanticRouter`/`LLMRouter`. They already live in different modules.

**Author's scoping decision (confirmed): the semantic router is used for the INTENT GATE ONLY.** Because
domain routing is removed from Enhanced entirely (Enhanced fan-outs), there is **no semantic router doing
collection selection anywhere in Enhanced.** The in-house `SemanticRouter`/`LLMRouter` in `route.py` stay
in the tree only as *optional ablation levers for a possible separate routing-axis study* — they are **not
part of any default tier**. Enhanced's `router` field is `"none"` (fan-out). This removes the terminological
trap at the architecture level, not just the naming level.

**The remaining risk is purely terminological in writing:** both the intent gate and the (parked) domain
router share the phrase "semantic router." In the thesis, name them distinctly — **Intent Gate** vs
**Domain Router** — so an examiner never conflates them.

**Recommendation: KEEP. Semantic router → intent gate only; domain routing absent from Enhanced. This
separation is a strength — surface it explicitly in the thesis taxonomy.**

---

## 3. Capability Mapping

| Capability | Purpose | Naive | Enhanced | Agentic | Implementation notes |
|---|---|---|---|---|---|
| **Dense retrieval** | Semantic similarity search | ✅ only | ✅ (within hybrid) | ✅ (within hybrid) | `VectorSourceCapability.invoke(strategy="dense")`; naive uses `search_with_scores` for global merge |
| **Hybrid retrieval (BM25+dense, RRF)** | Recover exact legal/permit terms dense misses | ❌ | ✅ **(turn ON)** | ✅ (already capability default) | Built: [vector_source.py:55-71](src/ragtrial/capabilities/vector_source.py#L55-L71). Enhanced default currently forces `dense` → flip to `hybrid` |
| **Cross-encoder rerank** | Filter "similar-but-wrong" (sosial preamble noise) | ❌ | ✅ **(BUILD)** | ✅ **(BUILD + wire in)** | **STUB** `NotImplementedError` [rerank.py:46](src/ragtrial/pipeline/rerank.py#L46). `bge-reranker-v2-m3`. Critical-path for both tiers |
| **Fan-out (all collections)** | Max recall, no routing risk | ✅ | ✅ | ⚠️ only as fallback | Naive: distance-merge. Enhanced: `router="none"` → `RetrieveStage` fan-out. Agentic: routes instead |
| **Intent handling (retrieve y/n)** | Refuse OOS / chit-chat without retrieval | ❌ (always retrieves) | ✅ semantic-router binary | ✅ implicit (LLM chooses to call a tool) | Enhanced: [intent.py](src/ragtrial/pipeline/intent.py) (measured recall_invalid 1.00). Naive recall_invalid 0.00 |
| **Domain routing (which collection)** | Select source(s) | ❌ | ❌ **(removed)** | ✅ implicit (tool = domain) | Agentic only. `route.py` routers stay available for a *separate* routing-axis study, not in Enhanced |
| **Adaptive query rewriting** | Reshape query when useful | ❌ | ❌ **(removed; no HyDE)** | ✅ implicit (LLM writes `query` arg) | HyDE demoted out of Enhanced default (was worst-latency, no gain). Agentic rewrite is emergent, not a stage |
| **Control flow** | Who decides the path | fixed/monolithic | fixed/deterministic pipeline | LLM-adaptive (LangGraph loop ≤5) | The true differentiator (§4) |
| **Iterative / recursive retrieval** | Re-search after seeing results | ❌ | ❌ | ⚠️ weak (loop can re-issue) — **postponed** | Out of scope this revision |
| **Reflection / doc grading** | Judge context sufficiency | ❌ | ❌ | ❌ **postponed** | Absent; do NOT claim it |
| **Planning / decomposition** | Multi-step query plans | ❌ | ❌ | ❌ | Out of scope; corpus is single-hop |
| **Unified execution log** | Uniform decision-trace for viz + debugging | ✅ **(BUILD)** | ✅ **(BUILD)** | ✅ **(BUILD)** | Normalized `decisions` dict on `RagResult`, identical schema across modes (§5A) |

Legend: ✅ present/planned · ❌ absent by design · ⚠️ partial/fallback. **Bold** = requires code work in
this revision.

**The whole revision reduces to four build actions:** (1) build the cross-encoder reranker as a shared
utility; (2) flip Enhanced's config defaults (incl. keeping retrieval-stack ablation presets); (3) wire the
reranker into the Agentic loop; (4) add the unified execution log to `RagResult` and populate it in all
three modes. Everything else already exists or is intentionally absent.

---

## 4. Architectural Difference (philosophy, not feature count)

The three tiers are **not** "small / medium / large bags of features." They are three points on a single
conceptual axis: **where decision authority lives — the locus of control.**

- **Naive — Zero-decision retrieval.** The system makes *no* decisions. It always retrieves, always
  fan-outs, always stuffs, answers once. Its identity is *absence of control*: a pure function from query
  to answer with no branch points. This is the scientific "floor" — any behavior above it is attributable
  to a decision mechanism the other tiers add.

- **Enhanced — Decisions fixed at design time.** The *developer* chose the path once, and every query
  follows it deterministically. There is exactly **one** branch — the intent gate (retrieve or answer
  directly) — and even that is a fixed classifier, not reasoning. Its identity is *predetermination*: the
  intelligence is in the **retrieval stack** (hybrid + rerank), engineered ahead of time, not in any
  runtime choice. Given the same input it always does the same thing. This is why it is "enhanced," not
  "agentic": more capable retrieval, zero autonomy.

- **Agentic — Decisions made at inference time.** An *LLM* decides, per query, whether to retrieve, which
  domain(s) to search, and how to phrase the search. Its identity is *delegated control*: authority over
  the pipeline shape moves from author-time to run-time. The same input can produce different paths on
  different runs. What makes it fundamentally different from Enhanced is **not** that it "has routing and
  rewriting" — Enhanced *could* have those as fixed stages — but that **an LLM chooses them dynamically.**

So the spine is: **no control → author-time control → inference-time control.** Framed this way, the three
tiers answer three escalating questions:

1. Naive: *How far does raw retrieval get us?*
2. Enhanced: *How much does a better, fixed retrieval stack add?*
3. Agentic: *Does moving control from the developer to an LLM add anything beyond the fixed stack?*

That framing is defensible, avoids "more features = better," and — crucially — makes a **non-monotone
result publishable** ("inference-time control did not beat author-time control on this corpus" is a real
finding, not a failure).

---

## 5A. Unified Execution Log (new deliverable)

**Goal (author's request):** every architecture emits an execution log in **one identical format**, so
visualization and debugging are uniform. Example shape the author gave (Agentic):

```json
{ "intent": "retrieve", "rewrite": true, "routing": "dukcapil", "retrieval": "hybrid", "rerank": true }
```

**Best home: a normalized `decisions` field on the shared `RagResult` contract**
([result.py](src/ragtrial/result.py)) — not scattered in each mode's `meta`. `RagResult` is already the
single contract all consumers (eval, app, session) read, and it already keeps mode-specific extras in
`meta`. Adding one fixed-schema `decisions` dict gives a uniform, mode-agnostic trace *without* disturbing
`meta` (which stays for rich mode-specific detail like `route_scores`, `rerank_scores`, agent `steps`).

**Proposed fixed schema (same keys, every mode; constants where a stage doesn't exist):**

| Key | Type | Meaning | Naive | Enhanced | Agentic |
|---|---|---|---|---|---|
| `intent` | `"retrieve"` \| `"direct"` | Did the system retrieve, or answer directly? | always `"retrieve"` | intent gate result | LLM: called a tool? |
| `rewrite` | `bool` | Was the query reshaped before retrieval? | `false` | `false` (no rewrite) | `true` iff a tool `query` ≠ question |
| `routing` | `"fanout"` \| domain \| `[domains]` \| `"none"` | Which collection(s) searched | `"fanout"` | `"fanout"` | domain(s) the agent picked |
| `retrieval` | `"dense"` \| `"hybrid"` | Retrieval strategy used | `"dense"` | `"hybrid"` | `"hybrid"` |
| `rerank` | `bool` | Was a cross-encoder applied? | `false` | `true` | `true` |
| `iterations` | `int` | Agent loop count (1 for fixed pipelines) | `1` | `1` | `n` (≤5) |

**Design principles:**
- **Fixed key set, uniform types.** A field that doesn't apply to a mode is filled with a truthful constant
  (e.g. Naive `rerank=false`), never omitted — so a visualizer can render all three side by side without
  special-casing. This is the whole point of "same format."
- **Derived, not intrusive.** Each mode already *has* this information (intent in `meta["intent"]`, route in
  `RagResult.route`, agent `steps`, config in `meta["config"]`). The log is assembled from existing state at
  the end of each `ask_*`, so the pipelines themselves don't change — only a small `_build_decisions(...)`
  helper per mode (or one shared helper reading `RagResult` + `meta`).
- **`decisions` is the normalized layer; `meta` stays the detailed layer.** e.g. `decisions.routing =
  "dukcapil"` while `meta.route_scores = {…}` and `meta.steps = [...]` keep the full trace. Visualization
  reads `decisions`; deep debugging drops into `meta`.
- **Surface it in `to_dict()`** so `eval/` JSON dumps and `app.py` get it for free.

**Effort: Low.** One field on `RagResult`, one small assembler per mode (or a shared one), one line in
`to_dict()`. No pipeline logic changes. High payoff for debugging/thesis figures (you can literally
tabulate "what did each architecture decide per question").

---

## 5. Codebase Impact Analysis

**Must change:**

| File | Change | Effort |
|---|---|---|
| [pipeline/rerank.py](src/ragtrial/pipeline/rerank.py) | Implement `CrossEncoderReranker.run` (replace `NotImplementedError`). Factor the scoring into a **shared, framework-agnostic function** (query × docs → ranked docs) that the `Stage` wraps. | Med |
| [rag/enhanced.py](src/ragtrial/rag/enhanced.py) | Change `EnhancedRAGConfig` defaults → `rewriter="passthrough"`, `router="none"`, `retrieval="hybrid"`, `reranker="cross_encoder"` (intent stays `"semantic"`). Update `PRESETS`/docstring: new canonical **+ keep** retrieval-stack ablation presets (`dense_only`, `hybrid_no_rerank`, `rerank_no_hybrid`). | Low |
| [rag/agentic.py](src/ragtrial/rag/agentic.py) | Call the shared reranker on retrieved docs before generation (in/after `_node_tools`, or once before END). Confirm hybrid is used (it is, via capability default). Ensure final top-N matches Enhanced for fairness. | Med |
| [result.py](src/ragtrial/result.py) | Add normalized `decisions: Dict` field + surface it in `to_dict()` (unified execution log, §5A). | Low |
| [rag/naive.py](src/ragtrial/rag/naive.py), [rag/enhanced.py](src/ragtrial/rag/enhanced.py), [rag/agentic.py](src/ragtrial/rag/agentic.py) | Each populates `decisions` from existing state (small assembler; no pipeline logic change). | Low |

**Can remain unchanged (contract holds):**

- [rag/naive.py](src/ragtrial/rag/naive.py) — stays a dense fan-out monolith. **Only** addition is
  populating the `decisions` log (a few constant fields); its retrieval/generation logic is untouched.
- [chat/session.py](src/ragtrial/chat/session.py), [app.py](app.py) — dispatch is by mode name via
  `_load_mode_fn`; the `ask_*(question, verbose) -> RagResult` signatures do not change.
- [eval/run_eval.py](eval/run_eval.py), [eval/run_intent_eval.py](eval/run_intent_eval.py),
  [eval/eval_core.py](eval/eval_core.py) — import the three `ask_*` functions unchanged; only the
  *results* move.
- [capabilities/vector_source.py](src/ragtrial/capabilities/vector_source.py) — hybrid already
  implemented; no change (just exercised more).
- [pipeline/route.py](src/ragtrial/pipeline/route.py), [pipeline/rewrite.py](src/ragtrial/pipeline/rewrite.py)
  — routers/HyDE stay in the codebase as **ablation switches** (Enhanced simply stops selecting them by
  default). MultiQuery stub can stay stubbed or be deleted from the narrative.

**Should become modular (shared across tiers):**
- The **cross-encoder scoring** — one function, two callers (Enhanced `Stage`, Agentic tools node). This is
  the one place the "modular RAG" idea genuinely pays off: without sharing, the rerank logic will drift
  between tiers and silently break comparison fairness.
- The **execution-log schema** — a single normalized `decisions` dict on `RagResult` with a fixed key set,
  assembled per mode from existing state. Uniformity *is* the requirement, so the schema must be defined
  once and reused, not redefined per mode.

**Should stay architecture-specific (do NOT modularize):**
- Naive's fan-out/merge/stuff — keep monolithic (per the research decision).
- Agentic's LangGraph loop — do not force it into the `Stage`/`Pipeline` abstraction; its control flow is
  fundamentally different (that difference *is* the tier's identity).

**Complexity estimate:** overall **Low–Medium**. One real new component (reranker, Med), one config flip
(Low), one wiring change (Med), one small contract field (Low). No schema/eval-harness changes. The heaviest
*hidden* cost is operational: the reranker model download, memory, and CPU latency (§7), not code volume.

---

## 6. Proposed Refactoring Order

Order is dependency-driven: the reranker is required by both upgraded tiers, so it comes first; the config
flip must not precede a working reranker (else the default pipeline raises `NotImplementedError`).

**Execution discipline (author-confirmed): implement ONE checkpoint at a time → smoke test it → PAUSE for
review → fix if needed → only then proceed.** No barrelling through multiple checkpoints in one go. Each
step below is a review gate.

Revised for the unified-storage decision (§1.5): CP-A (unified index) is now the true foundation; the
reranker (CP1) is already in hand.

**CP1 — Cross-encoder reranker (shared utility). ✅ DONE.** `rerank_documents()` + `CrossEncoderReranker`
in [pipeline/rerank.py](src/ragtrial/pipeline/rerank.py); `bge-reranker-v2-m3`; smoke test passed (~2s/5
docs cached, CPU; sharp discrimination on ID legal text).

**CP-A — Unified index (FOUNDATION; blocks CP-B…CP-E).** `UNIFIED_VECTOR_STORE`/`UNIFIED_COLLECTION` in
[config.py](src/ragtrial/config.py); `vectorstore/unified.py build_unified()` COPIES per-domain vectors
into one collection stamping `domain` (== `_source`), no re-embed; build-script entry. *Smoke:* count ==
2,678; `where={"domain":...}` isolates a domain; a dense query returns cross-domain hits.

**CP-B — Unified retrieval layer.** One `UnifiedStore` (dense + hybrid + optional `domain` filter) as the
sole searchable retriever; keep per-domain `Capability` for formatting/gold-id/citation/agentic-tool
metadata (dispatch by `domain`). *Smoke:* dense/hybrid/filtered retrieval + `format_context`/`gold_id`
dispatch all correct.

**CP-C — Naive on unified + execution-log scaffold.** Naive = global dense top-k from the unified store
(drop fan-out/merge). Add `decisions: Dict` to [result.py](src/ragtrial/result.py) (+`to_dict()`); populate
naive. *Smoke:* KTP question answers correctly; `decisions` present.

**CP-D — Enhanced on unified + retrieval-stack ablation presets.** intent → retrieve(global, hybrid) →
rerank → generate; no router/HyDE. New canonical `EnhancedRAGConfig` + **4 ablation presets** (`dense_only`
/ `hybrid_no_rerank` / `rerank_no_hybrid` / `default`) sharing one global candidate pool. Populate
`decisions`. *Smoke + the hybrid-vs-rerank comparison the author asked about:* run the 4 presets on a few
queries, eyeball deltas.

**CP-E — Agentic on unified.** Domain routing = `domain` metadata filter on the unified store; wire the
shared reranker; standardize final top-N to Enhanced; populate `decisions` (+ `iterations`); keep
`meta.steps`. *Smoke:* single-domain, multi-domain, and OOS-skip behave; reranked docs reach the answer.

**CP-F — Fairness pass + small-subset smoke eval.** Verify identical unified index/embedder/LLM + matched
final top-N; run naive/enhanced/agentic **+ the 4 Enhanced ablation cells** on a **10–20 Q subset** before
any full re-run + `docs/EVAL_REPORT.md` rewrite (post-freeze).

**Critical path:** CP1 ✅ → CP-A → CP-B → CP-C → CP-D → CP-E → CP-F.

---

## 7. Potential Risks

**Scientific risks.**
- **Residual Enhanced↔Agentic confound** (control flow + routing + rewrite bundled). Any single-cause
  Agentic claim is unsupported. *Mitigation:* frame as "adaptive bundle vs fixed bundle"; log traces to
  decompose; (deferred) add an `agentic-fanout` ablation to isolate control flow.
- **Non-monotone tiers.** Enhanced fan-outs (high recall), Agentic routes (recall-lossy when wrong) → you
  may get Enhanced ≥ Agentic on recall. *Mitigation:* pre-register this expectation; report retrieval and
  answer-quality separately; treat non-monotonicity as the finding.
- **Postponed reflection/iteration** means "Agentic" here is really "LLM-orchestrated routing+rewrite."
  *Mitigation:* say so explicitly; never cite a capability grid that claims reflection/planning your code
  lacks (`capabilities-review.md` §15, `architecture_review.md` §9).

**Implementation risks.**
- **Reranker weight/latency/token cost.** `bge-reranker-v2-m3` + torch is a heavy dependency; first-call
  load + CPU inference can dominate latency, and reranking many candidates is time/token-heavy on free-tier.
  *Mitigation (author-confirmed):* **validate on a 10–20 question subset before any full run**; lazy
  singleton; cap reranked candidate count; document the added latency as a cost dimension (a legitimate
  efficiency finding).
- **Rerank logic drift** between the Enhanced Stage and the Agentic node if not shared. *Mitigation:*
  single shared scoring function (step 1).
- **BM25 rebuild on first hybrid call** materializes the whole collection in-process
  ([vector_source.py:55-71](src/ragtrial/capabilities/vector_source.py#L55-L71)); now hit by *both*
  Enhanced and Agentic (esp. sosial, 2,433 docs). *Mitigation:* accept at current scale; note as a
  scalability caveat.

**Evaluation risks.**
- **Re-running costs quota/time** (free-tier; answer-quality already limited to ~60/198). *Mitigation:*
  design the ablation matrix before running; don't iterate live. Start on the 10–20 Q subset.
- **Generator→gold coupling persists** — retrieval recall stays an optimistic upper bound. This revision
  does **not** fix it; do not imply it does.
- **Difficulty skew (hard=1/198)** means Agentic's theoretical edge (iteration/reflection) is
  un-provable — which is *fine* because you postponed those, but don't claim adaptivity helps on hard
  cases you didn't test.

**Comparison-fairness risks.**
- **Context-window asymmetry.** Reranked tiers could feed the generator a different number/quality of
  chunks than Naive. *Mitigation:* standardize final top-N; report it.
- **New model only in two tiers.** The reranker is intentionally part of Enhanced/Agentic, absent from
  Naive — that is *by design* (it's a tier capability), but state it so it isn't read as an unfair extra.

**Technical debt.**
- Reranker singleton + torch import increases cold-start and memory footprint for every entry point.
- Leftover stubs (`MultiQueryRewriter`, unused routers/HyDE) remain in the tree as ablation levers — keep
  them *documented as ablations*, or an examiner reading the code will think they're dead code.
- Committed binary vector stores keep churning in `git status` (noted in `architecture_review.md` §6.1);
  unrelated to this revision but will noise up the diff — consider gitignoring before the implementation
  commits land.

---

## 8. Suggestions (where I disagree or would redesign)

**8.1 DEFERRED (author's call) — `agentic-fanout` de-confounding ablation.** I still consider this the
single most valuable *evaluation* addition (it separates "LLM control flow" from "domain routing," which is
otherwise uninterpretable). **Parked until first results are in**, then reconsider. Recorded here so it is a
conscious deferral, not an oversight. Cheap when revisited (tell the agent to search all domains, or
post-hoc compare against fan-out Enhanced).

**8.2 DEFERRED (author's call) — fan-out fallback when Agentic routing fails.** Protects Agentic from the
catastrophic single-domain miss that sank the old Enhanced. **Parked**; revisit only if first results show
Agentic losing recall to routing errors. Trade-off when added: it blurs "pure routing," so log when it fires.

**8.3 KEEP the retrieval-stack ablation presets in Enhanced (author confirmed — do it).** The headline tiers
bundle hybrid+rerank, but you *must* be able to answer "was it hybrid or rerank?" Keep `dense_only`,
`hybrid_no_rerank`, `rerank_no_hybrid` presets and run the 4-cell factorial within Enhanced. This is where
your cleanest publishable numbers come from (`capabilities-review.md` §13). Cost: near-zero, already
supported.

**8.4 DO NOT re-add HyDE or domain routing to Enhanced.** Your evidence (worst latency, no recall gain;
routing 0.72 dragging recall) justifies removing both. Resist the temptation to "make Enhanced look
advanced" by re-adding them — that reintroduces the confound and contradicts your data. If you want to
study routing, study it as a **separate 3-way axis** (none/semantic/llm) using the existing routers, *not*
as a silent Enhanced default.

**8.5 DO NOT modularize Naive or Agentic-into-Pipeline.** Keep Naive monolithic (research decision, agreed)
and keep Agentic on LangGraph. Forcing all three into one `Stage` pipeline would *destroy* the architectural
distinction that is the whole point of the comparison. The only things to share are the reranker scorer and
the execution-log schema.

**8.6 Out of scope but flag for the thesis (not this revision):** the generator→gold retrieval coupling and
the single-judge monoculture remain the biggest *validity* threats (both prior reviews). This revision
improves *internal* fairness (finishing Enhanced) but not *external* validity. Say so; don't oversell.

**8.7 Naming hygiene:** in the thesis, call the two "semantic router" things **Intent Gate** and **Domain
Router** to prevent examiner confusion — the code already separates them, but the shared library name is a
trap.

---

## Appendix — where each capability lives today

| Capability | State | Evidence |
|---|---|---|
| Naive dense fan-out | ✅ built (monolith) | [rag/naive.py:37-71](src/ragtrial/rag/naive.py#L37-L71) |
| Enhanced config-assembled pipeline | ✅ built | [rag/enhanced.py:104-127](src/ragtrial/rag/enhanced.py#L104-L127) |
| Intent gate (VALID/INVALID) | ✅ built | [pipeline/intent.py:105-150](src/ragtrial/pipeline/intent.py#L105-L150) |
| Hybrid BM25+dense | ✅ built (default in capability; off in current Enhanced) | [vector_source.py:55-85](src/ragtrial/capabilities/vector_source.py#L55-L85) |
| Cross-encoder rerank | ❌ **stub** | [pipeline/rerank.py:30-48](src/ragtrial/pipeline/rerank.py#L30-L48) |
| Domain routers (none/semantic/llm) | ✅ built (ablation levers) | [pipeline/route.py:29-142](src/ragtrial/pipeline/route.py#L29-L142) |
| HyDE rewrite | ✅ built (to be demoted from default) | [pipeline/rewrite.py:26-48](src/ragtrial/pipeline/rewrite.py#L26-L48) |
| Agentic loop (implicit intent/routing/rewrite) | ✅ built | [rag/agentic.py:166-182](src/ragtrial/rag/agentic.py#L166-L182) |
| Reflection / planning / iteration | ❌ absent (postponed) | — |
| Uniform dispatch (`ask_*`→RagResult) | ✅ built | [chat/session.py:27-36](src/ragtrial/chat/session.py#L27-L36) |
| Unified execution log (`decisions`) | ❌ to build (§5A) | [result.py](src/ragtrial/result.py) |

*End of revision plan.*
