# Phase 3 Implementation Design: The Model Selection Pilot

*Executable companion to `08_implementation_roadmap.md` Phase 3 ("The Model Selection Pilot — The Decider"), translating `05_task5_model_selection_plan.md` into an actual package addition on top of the merged Phase 1 pipeline and the Phase 2 split artifacts (`10_phase2_implementation_design.md`). Like docs 09 and 10, this is a design document, not a finished implementation — every section with a real design choice presents options with a recommendation. This doc assumes Phase 2's fold artifacts (`data/splits/folds/*.parquet`) already exist; it does not re-derive splitting logic.*

---

## 0. Scope and non-goals

**In scope** (mirrors doc 08's Phase 3 bullets and doc 05's decision rule):
- A physicochemical-descriptor baseline regressor, run on the Phase 2 splits, that every deep-learning candidate must clear.
- A pilot grid over `hybrid-small`, `mtr-small`, `mlm-small`, `mlm-large` (doc 05 §4.1's four core candidates), plus the `PeptideCLM-23M-all` (v1) fallback priced in per doc 05's "worth pricing in now rather than after committing to v2."
- Two evaluation passes per candidate: a cheap frozen-encoder probe and a short (≤10-epoch) full fine-tune, on one designated pilot split.
- A documented decision — which encoder variant Phase 4 builds on — recorded with its rationale, not just implied by whichever script was run last.

**Explicitly out of scope for this doc:**
- The actual shared-encoder multitask architecture (mask-aware pooling → shared projection → per-task 2-layer MLP heads, the two-stage freeze/unfreeze training protocol, the masked multitask loss) — that is Phase 4, doc 04. This phase's fine-tuning heads are deliberately minimal, single-task, throwaway scaffolding built only to isolate encoder quality; they must not be mistaken for, or reused as, the Phase 4 architecture.
- Full multi-fold LOCO evaluation of the winning candidate — the pilot runs on **one** designated fold (§8.3); the full k-fold LOCO sweep happens naturally once Phase 4's real training loop exists and is not duplicated here just to double-check the pilot's winner.
- Any of Phase 2's embedding/clustering/splitting logic — this doc consumes `data/splits/folds/*.parquet` and `data/processed/dataset.parquet` as read-only inputs.
- The warm-start (PAMPA-fine-tuned checkpoint) and therapeutic-index ablations — those are Phase 5 (docs 03 and the TI ablation), explicitly out of scope here.

**Deliberate namespace choice:** the new package below is named `clamp.model_selection`, not `clamp.modeling` or `clamp.train`. Phase 4 will need a real training/modeling namespace for the shared architecture, and this phase's code — throwaway single-task heads whose only job is ranking encoders — should not occupy or imply ownership of that name. Whoever implements Phase 4 should feel free to introduce `clamp.modeling`/`clamp.train` fresh, importing only the narrow pieces from `clamp.model_selection` that are explicitly designed for reuse (§4, §9).

---

## 1. Relationship to Phase 1/2 outputs

This doc reads, and never writes to:
- `data/processed/dataset.parquet` (Phase 1) — for labels (`hc50_log_uM`, `mic_log_uM`), `smiles`, `peptide_uid`.
- `data/splits/folds/fold_{i}.parquet` (Phase 2) — for `peptide_uid`, `role` (train/val/test), `cluster_id`.
- `data/embeddings/peptide_embeddings.parquet` (Phase 2) — reusable **only** for the `hybrid-small` candidate's frozen-probe pass, since that's the one model Phase 2 already embedded everything with; every other candidate needs its own embedding pass (§4).

**One designated pilot fold, not the full LOCO sweep** (doc 08: "on a subset of your split data, run a short pilot"): use **fold 0** (the first LOCO rotation Phase 2's `build_loco_folds` produced) as the fixed pilot split for the entire candidate grid. This keeps the grid's cost to (candidates × eval modes × tasks), not (candidates × eval modes × tasks × k folds) — the full per-fold sweep is exactly what Phase 4's actual training run will produce for the winning architecture anyway, so re-running it here for every losing candidate would be pure waste. Record which fold was used in the decision report (§10) so this choice is auditable, not assumed.

---

## 2. Package layout

**Recommendation: a new `src/clamp/model_selection/` subpackage**, sibling to `clamp.data` and `clamp.splitting`.

```
src/clamp/
├── config.py                       # extended, not replaced — see §5
└── model_selection/                 # NEW (this doc)
    ├── __init__.py
    ├── cli.py                       # typer app, entry point `clamp-pilot`
    ├── schema.py                    # CandidateModel, PilotResult, ProbeResult, FineTuneResult (§4)
    ├── descriptors.py                # RDKit physicochemical feature extraction (§6)
    ├── baseline.py                   # the P1 baseline regressor (§6)
    ├── candidates.py                 # candidate registry + loaders, incl. v1 adapter (§7)
    ├── probe.py                       # frozen-encoder probe (§8)
    ├── finetune.py                    # short full fine-tune (§9)
    ├── metrics.py                     # MAE, Spearman rho, Pearson r (§10)
    ├── decision.py                    # ranking rule + escalation logic (§10)
    ├── report.py                      # pilot report, mirrors data/datasheet.py's split (§11)
    └── pipeline.py                    # orchestrator (§12)
```

```
tests/
└── model_selection/
    ├── __init__.py
    ├── fixtures/                    # tiny synthetic SMILES + labels, a fake tiny encoder
    ├── test_descriptors.py
    ├── test_baseline.py
    ├── test_candidates.py
    ├── test_probe.py
    ├── test_finetune.py
    ├── test_metrics.py
    └── test_decision.py
```

---

## 3. New dependencies

```toml
dependencies = [
    # ... existing Phase 1/2 deps unchanged (includes torch, transformers, scikit-learn from doc 10 §3) ...
    "huggingface-hub>=0.23",   # HfApi().model_info(...).sha, for revision pinning (§7.4)
]

[dependency-groups]
dev = [
    # ... existing dev deps unchanged ...
]
```

No new heavyweight dependencies beyond what doc 10 §3 already introduced (`torch`, `transformers`, `scikit-learn`) — this phase reuses that stack directly rather than adding a training-framework dependency (e.g. `lightning`/`accelerate`); at pilot scale (≤2,000 rows, ≤10 epochs, `*-small`/`*-base`/`*-large` single-GPU jobs per doc 05 §4.3) a plain PyTorch training loop is enough, and pulling in a framework now would be exactly the kind of premature machinery doc 09's HELM-stub reasoning (§6) warns against.

**PeptideCLM v1 fallback's own dependency, isolated:** `SmilesPE` (the package `SMILES_SPE_Tokenizer` needs) and the cloned `github.com/AaronFeller/PeptideCLM` repo itself are **not** `pyproject.toml` dependencies — per doc 05 §2.2, v1's tokenizer isn't pip-installable, and per §6.5 v1 is an escalation-only path, not part of the default candidate set that runs on every pilot invocation. `candidates.py`'s v1 loader (§7.3) documents the manual `git clone` + `sys.path` step as a runbook prerequisite (§13) gated behind a `--include-v1` flag, not a hard dependency of `clamp-pilot`.

---

## 4. Data model (`model_selection/schema.py`)

```python
from enum import StrEnum
from pydantic import BaseModel

class Objective(StrEnum):
    MLM = "mlm"
    MTR = "mtr"
    HYBRID = "hybrid"

class LoaderKind(StrEnum):
    V2_AUTO = "v2_auto"        # AutoTokenizer/AutoModel, trust_remote_code for the model only
    V1_CUSTOM = "v1_custom"     # SMILES_SPE_Tokenizer + AutoModelForMaskedLM, doc 05 §2.2

class CandidateModel(BaseModel):
    name: str                   # short id used throughout, e.g. "hybrid-small"
    hf_repo: str
    revision: str                # pinned commit sha — "" is invalid at runtime, same rule as doc 10 §6.4
    objective: Objective | None  # None for v1 (single pretraining objective, no MTR/hybrid variant)
    params_millions: float
    loader: LoaderKind
    is_fallback: bool = False    # True only for the v1 entry — excluded from the default grid (§7.2)


class EvalMode(StrEnum):
    FROZEN_PROBE = "frozen_probe"
    FULL_FINETUNE = "full_finetune"

class Task(StrEnum):
    HC50 = "hc50"
    MIC = "mic"


class ProbeResult(BaseModel):
    candidate: str
    task: Task
    mae: float
    spearman_rho: float | None    # None iff compute_metrics saw a non-finite value (§10) — a
                                    # degenerate (near-constant) prediction, not a crash
    pearson_r: float | None       # same None convention as spearman_rho
    n_train: int
    n_test: int
    wall_clock_s: float

class FineTuneResult(BaseModel):
    candidate: str
    task: Task
    seed: int
    mae: float
    spearman_rho: float | None    # None iff compute_metrics saw a non-finite value (§10) — a
                                    # degenerate (near-constant) prediction, not a crash
    pearson_r: float | None       # same None convention as spearman_rho
    best_epoch: int
    n_train: int
    n_test: int
    wall_clock_s: float
    peak_vram_gb: float | None = None


class BaselineResult(BaseModel):
    task: Task
    mae: float
    spearman_rho: float | None    # None iff compute_metrics saw a non-finite value (§10) — a
                                    # degenerate (near-constant) prediction, not a crash
    pearson_r: float | None       # same None convention as spearman_rho
    n_train: int
    n_test: int


class PilotResult(BaseModel):
    """One row of the full pilot grid — the unit the report (§11) tabulates."""
    candidate: str
    eval_mode: EvalMode
    task: Task
    mae: float
    spearman_rho: float | None    # None iff compute_metrics saw a non-finite value (§10) — a
                                    # degenerate (near-constant) prediction, not a crash
    pearson_r: float | None       # same None convention as spearman_rho
    wall_clock_s: float
```

---

## 5. Config additions

Extend `clamp.config.Settings` again (not a third parallel settings object):

```python
class Settings(BaseSettings):
    # ... existing Phase 1/2 fields unchanged ...

    pilot_fold_id: int = 0                  # which Phase 2 LOCO rotation the whole grid runs on (§1)
    pilot_finetune_lr: float = 1e-5          # doc 05 §3's paper-reported defaults, reused as pilot starting point
    pilot_finetune_batch_size: int = 16
    pilot_finetune_dropout: float = 0.1
    pilot_finetune_max_epochs: int = 10
    pilot_finetune_early_stopping_patience: int = 2
    pilot_finetune_seeds: list[int] = [0, 1]   # 1-2 seeds for the pilot; 3 for the eventual winner (doc 05 §4.2)
    pilot_include_v1: bool = False             # default for --include-v1 (§7.3/§12) — the CLI flag
                                                 # always wins if passed explicitly, this is just its default
    model_revisions: dict[str, str] = {}        # candidate name -> pinned sha (§7.1/§7.4), sourced from
                                                 # CLAMP_MODEL_REVISIONS_JSON — empty until §7.4's runbook
                                                 # step has actually been run

    @property
    def model_selection_dir(self) -> Path:
        return self.data_root / "model_selection"
```

---

## 6. Modules: `descriptors.py` and `baseline.py` — the physicochemical baseline

```python
# descriptors.py
from rdkit import Chem
from rdkit.Chem import Descriptors

def compute_descriptors(smiles: list[str]) -> pd.DataFrame:
    """RDKit's full built-in descriptor list (Descriptors.CalcMolDescriptors,
    ~200 physicochemical features per molecule — MW, LogP, TPSA, H-bond
    donor/acceptor counts, rotatable bonds, etc.) — the same RDKit call
    already a project dependency since Phase 1's normalize.py (doc 09 §8),
    no new chemistry library needed. Returns one row per input SMILES,
    NaN-filled where RDKit's descriptor calculators themselves fail (rare,
    but do not raise — a baseline model should degrade gracefully on a
    handful of bad rows rather than aborting the whole pilot)."""
    ...
```

```python
# baseline.py
from sklearn.linear_model import ElasticNetCV

def train_baseline(
    train_df: pd.DataFrame, test_df: pd.DataFrame, task: Task
) -> BaselineResult:
    """Descriptors -> impute -> ElasticNetCV (built-in cross-validated
    regularization search, avoids a separate hyperparameter-sweep harness
    for what's meant to be a minimal reference bar) -> predict on test_df
    -> metrics.py. This is the P1 baseline doc 05 §4.2 step 2 and doc 07's
    source material both refer to — every deep-learning candidate below
    must clear it.

    NaN handling: compute_descriptors documents that it NaN-fills rather
    than raises on a bad row (§6 above), but ElasticNetCV itself rejects
    NaN input outright — a single bad descriptor would otherwise abort the
    whole pilot. Fit a `sklearn.impute.SimpleImputer` (median strategy) on
    `train_df`'s descriptors ONLY, then apply that same fitted imputer to
    `test_df`'s descriptors before predicting — never fit (or re-fit) the
    imputer on test_df, which would leak test-set statistics into the
    training-derived fill values."""
    ...
```

**Why `ElasticNetCV` over a tree ensemble for the baseline:** doc 05's own threshold-sensitivity reference point (§3, citing the paper's frozen-embedding probes) and doc 07 §5.3's leakage-diagnostic probe both use a **linear** model on descriptors/embeddings as "the standard cheap baseline" — reusing the same model family for the P1 baseline keeps this doc's baseline directly comparable to those other linear-probe reference points rather than introducing a fourth model class. If `ElasticNetCV` clearly underperforms a `RandomForestRegressor`/`HistGradientBoostingRegressor` on the real pilot data, swapping is a one-line change in `baseline.py`, not a design change — log both if there's any doubt, but don't block the pilot on picking the "best possible" baseline; per doc 05 §4.2 its job is to be a floor, not the star of the analysis.

---

## 7. Module: `candidates.py` — the registry and loaders

### 7.1 The registry

```python
# Registry template committed to source control — note the empty revisions.
# This is NOT what candidates.py hands to the rest of the pipeline; see
# resolve_registry() below.
_CANDIDATE_TEMPLATES: list[CandidateModel] = [
    CandidateModel(name="hybrid-small", hf_repo="aaronfeller/peptideclm-2-hybrid-small",
                   revision="", objective=Objective.HYBRID, params_millions=31.7, loader=LoaderKind.V2_AUTO),
    CandidateModel(name="mtr-small", hf_repo="aaronfeller/peptideclm-2-mtr-small",
                   revision="", objective=Objective.MTR, params_millions=31.7, loader=LoaderKind.V2_AUTO),
    CandidateModel(name="mlm-small", hf_repo="aaronfeller/peptideclm-2-mlm-small",
                   revision="", objective=Objective.MLM, params_millions=31.7, loader=LoaderKind.V2_AUTO),
    CandidateModel(name="mlm-large", hf_repo="aaronfeller/peptideclm-2-mlm-large",
                   revision="", objective=Objective.MLM, params_millions=337.0, loader=LoaderKind.V2_AUTO),
    CandidateModel(name="peptideclm-v1", hf_repo="aaronfeller/PeptideCLM-23M-all",
                   revision="", objective=None, params_millions=23.0, loader=LoaderKind.V1_CUSTOM,
                   is_fallback=True),
]

def resolve_registry(revisions: dict[str, str]) -> list[CandidateModel]:
    """Returns _CANDIDATE_TEMPLATES with each entry's revision filled in
    from `revisions` (candidate name -> pinned sha, sourced from
    `settings.model_revisions` — see §5 — which is itself populated from
    `.env`/`CLAMP_MODEL_REVISIONS_JSON` once §7.4's runbook step has
    actually been run). Raises ValueError naming every candidate still
    missing a revision, pointing at §7.4, rather than silently returning a
    registry with empty-string revisions for the rest of the pipeline to
    trip over one candidate at a time."""
    missing = [c.name for c in _CANDIDATE_TEMPLATES if not revisions.get(c.name)]
    if missing:
        raise ValueError(
            f"No pinned revision for: {', '.join(missing)} — resolve via §7.4's "
            "resolve_revision() and set settings.model_revisions before running the pilot."
        )
    return [c.model_copy(update={"revision": revisions[c.name]}) for c in _CANDIDATE_TEMPLATES]

CANDIDATES: list[CandidateModel] = resolve_registry(settings.model_revisions)
```

This is the operational split CodeRabbit's review asked for: `_CANDIDATE_TEMPLATES` (committed, `revision=""` placeholders — there is no way to commit a real sha for a model that hasn't been pinned yet) is deliberately **not** what the rest of the pipeline imports. `CANDIDATES` is what `§7.2`/`§7.3`/everything downstream actually uses, and it either contains real pinned revisions or `resolve_registry` raises immediately at import time — the same fail-loud-don't-float principle as doc 10 §6.4, just enforced at construction instead of scattered across every loader. This also means the registry-validation test (§14) can be a normal, always-green test of `resolve_registry`'s own logic (feed it a fixture `revisions` dict, confirm it fills in correctly; feed it an incomplete one, confirm it raises naming the right candidates) rather than a test that's permanently red against the real committed templates until a human intervenes — see §14's updated row.

This registry is the literal reproduction of doc 05 §4.1's candidate table — four `hybrid-small`/`mtr-small`/`mlm-small`/`mlm-large` core candidates plus v1, no `base`-tier candidates in the default grid (doc 05 §4.1's note that the full 3×3 v2 grid is affordable if compute allows — see §7.5 for how to opt into it).

### 7.2 v2 loader — reuses Phase 2's `embed.py`, doesn't duplicate it

```python
def load_v2(candidate: CandidateModel, device: str):
    """Identical mechanics to clamp.splitting.embed.load_encoder (doc 10
    §6.1) — AutoTokenizer without trust_remote_code, AutoModel with it,
    both pinned to candidate.revision. Reuse that function directly,
    parameterized by candidate.hf_repo/candidate.revision, rather than
    reimplementing the same six lines a second time in this package."""
    from clamp.splitting.embed import load_encoder
    return load_encoder(candidate.hf_repo, candidate.revision, device)
```

This is the one piece of `clamp.model_selection` explicitly designed to depend on `clamp.splitting` (§0's namespace note flags this as intentional, not an exception) — encoder-loading mechanics are genuinely the same operation in both phases, just applied to different candidate models. Keeping one implementation avoids the two packages' loading logic silently drifting (e.g. one place remembering to pin `revision` and the other forgetting).

### 7.3 v1 loader — isolated, escalation-only

```python
def load_v1(candidate: CandidateModel, peptideclm_v1_repo_path: Path):
    """Doc 05 §2.2's exact de-risking checklist: sys.path.append the cloned
    github.com/AaronFeller/PeptideCLM checkout, load SMILES_SPE_Tokenizer
    from its tokenizer/new_vocab.txt + tokenizer/new_splits.txt, load the
    encoder via plain AutoModelForMaskedLM.from_pretrained (no
    trust_remote_code needed for v1's model — only its tokenizer is
    nonstandard). Raises a clear error if peptideclm_v1_repo_path doesn't
    exist, pointing at the runbook step (§13) rather than a bare
    FileNotFoundError."""
    ...
```

Only invoked when the `--include-v1` CLI flag (§12) is set — that flag is the **single, authoritative** gate; `settings.pilot_include_v1` is merely its default value (so automation can set `CLAMP_PILOT_INCLUDE_V1=true` to always run v1, without requiring the flag on every invocation) but passing `--include-v1` explicitly on the command line always enables v1 regardless of the settings value. There is deliberately no scenario where both a settings flag AND a CLI flag must independently agree before v1 runs — doc 05 §6.5's "rerun with `--include-v1`" runbook instruction (§13 step 7) must reliably enable v1 by itself. This enforces doc 05 §6.5's explicit warning not to pay v1's setup tax "merely because it's the known quantity" as an opt-in, not a default grid member, even though doc 05 §4.1 also says "worth pricing in now rather than after committing to v2." Both are honored: v1 is in the registry (priced in, ready to run) but off by default (not paid for unless a real trigger condition from doc 05 §6.5 applies — v2 loading breaks entirely, or v2 underperforms the baseline).

### 7.4 Resolving revisions — the same blocking runbook step as Phase 2

```python
from huggingface_hub import HfApi

def resolve_revision(hf_repo: str) -> str:
    return HfApi().model_info(hf_repo).sha

def resolve_all_revisions() -> dict[str, str]:
    """Runs resolve_revision for every _CANDIDATE_TEMPLATES entry, returning
    the {name: sha} mapping that goes into CLAMP_MODEL_REVISIONS_JSON /
    settings.model_revisions (§5) — a small CLI-runnable helper (`clamp-pilot
    resolve-revisions`, §12) so this is a one-command runbook step, not five
    manual copy-pastes."""
    return {c.name: resolve_revision(c.hf_repo) for c in _CANDIDATE_TEMPLATES}
```

Run once as part of the pilot runbook (§13, step 1) — same reasoning as doc 10 §6.4, restated because this phase introduces four *more* `trust_remote_code=True` loads (one per v2 candidate) that each need their own pinned sha, not just the one Phase 2 already pinned for `hybrid-small`. The resulting mapping is set via `CLAMP_MODEL_REVISIONS_JSON` (or written directly into `.env`), never hand-edited into `_CANDIDATE_TEMPLATES` itself — keeping the resolved shas out of the committed registry is exactly what lets `resolve_registry` (§7.1) fail loudly instead of silently running against a stale or guessed sha.

### 7.5 Opting into the full grid

`CANDIDATES` above is doc 05 §4.1's floor (5 entries, 4 v2 + v1). Doc 05 explicitly notes "if pilot compute is genuinely cheap... running the full 3×3 v2 grid instead of this subset of 4 costs little extra." Represent this as a second, opt-in registry constant (`FULL_GRID_CANDIDATES`, adding the three `*-base` variants) rather than a config flag that silently changes `CANDIDATES` — keeps the default grid's membership visible and stable in code review, with the expanded grid a deliberate, visible choice (`clamp-pilot run-grid --full`) rather than an easy-to-miss settings toggle.

---

## 8. Module: `probe.py` — frozen-encoder probe

```python
def frozen_probe(
    embeddings: np.ndarray, labels: np.ndarray, train_mask: np.ndarray, test_mask: np.ndarray,
    head: Literal["elasticnet", "mlp"] = "elasticnet",
) -> ProbeResult:
    """Mean-pooled embeddings (already computed, or computed fresh per
    candidate via candidates.load_v2/load_v1 + splitting.embed.embed_smiles)
    -> ElasticNetCV or a small 2-layer MLP -> metrics.py. Doc 05 §4.2 step 3
    explicitly expects this to be weak in absolute terms ('R²<0.30
    regardless of model scale' per the paper's own frozen-probe finding,
    §3) — this is a fast filter, not the deciding signal; don't eliminate a
    candidate on a bad probe score alone."""
    ...
```

**Reuse note, forward-referenced from doc 10 §10:** this is the same shape of computation as Phase 2's `validate.threshold_sensitivity_probe`. If Phase 2 is implemented first, that function should import and call `model_selection.probe.frozen_probe` rather than maintaining a second ElasticNet-on-embeddings implementation; if Phase 3 is implemented first, `frozen_probe` should be written generically enough (embeddings + labels + train/test masks in, metrics out) that doc 10's validation step can call it later without modification. Either doc can be built first — the dependency is designed to be satisfiable in either order, with a plain function signature as the seam.

---

## 9. Module: `finetune.py` — short full fine-tune

### 9.1 A deliberately minimal head — not the Phase 4 architecture

```python
class PilotRegressionHead(nn.Module):
    """mean_pool(encoder output) -> Dropout(p) -> Linear(embed_dim, 1).
    Single task, single head, no shared projection layer, no multitask
    masking — doc 04's shared architecture is explicitly Phase 4's job.
    This head exists only to let each candidate's encoder be fine-tuned
    long enough to measure whether IT (not a particular head design) is
    good at this task."""
    def __init__(self, encoder: nn.Module, embed_dim: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, input_ids, attention_mask):
        from clamp.splitting.embed import mean_pool
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        return self.head(self.dropout(pooled)).squeeze(-1)
```

Reuses `clamp.splitting.embed.mean_pool` directly (§0 already flags this cross-package dependency as intentional) rather than a third copy of the same pooling logic — and unlike Phase 2's usage, this call sites needs gradients to flow through, which `mean_pool`'s plain-tensor-ops implementation already supports (doc 10 §6.2 flagged this compatibility explicitly, not by accident).

### 9.2 Training loop

```python
def finetune_candidate(
    candidate: CandidateModel, task: Task,
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    config: "FineTuneConfig", seed: int,
) -> FineTuneResult:
    """LR/batch/dropout/max_epochs/early_stopping from settings (§5),
    seeded per doc 05 §4.2's 'start from the paper's own hyperparameters as
    defaults' guidance. Checkpoint selection by lowest validation MAE (not
    loss directly, to keep the selection criterion identical to the
    reported metric). Records wall_clock_s and peak_vram_gb (via
    torch.cuda.max_memory_allocated(), None on CPU) for the compute-cost
    side of the §10 decision rule."""
    ...
```

**Early-stopping logic as an independently testable unit:** factor the "has validation metric stopped improving for `patience` epochs" check into its own small pure function (`should_stop(history: list[float], patience: int) -> bool`) rather than inlining it in the training loop — this is the one piece of `finetune_candidate` testable without a real model or GPU (§14), by feeding it a synthetic loss history.

---

## 10. Modules: `metrics.py` and `decision.py`

```python
# metrics.py
import math
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Finite-metric policy: pearsonr/spearmanr return NaN for a constant
    (or too-small) y_pred — e.g. a candidate whose fine-tune collapsed to
    predicting the mean. Rather than let a NaN silently propagate into
    ranking/sorting (where NaN comparisons are not just wrong but
    order-dependent), store an explicit None for that metric. mae is never
    None — mean_absolute_error is always finite for finite inputs, so a
    degenerate correlation doesn't erase the one metric that still measures
    something."""
    spearman = spearmanr(y_true, y_pred).statistic
    pearson = pearsonr(y_true, y_pred)[0]
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "spearman_rho": spearman if math.isfinite(spearman) else None,
        "pearson_r": pearson if math.isfinite(pearson) else None,
    }
```

```python
# decision.py
class CandidateStatus(StrEnum):
    SUCCEEDED = "succeeded"          # has PilotResult rows for every required task/eval_mode
    LOAD_FAILED = "load_failed"      # candidates.load_v2/load_v1 raised
    SKIPPED = "skipped"              # not run this invocation (e.g. v1 without --include-v1)
    INCOMPLETE = "incomplete"        # started but missing a required task/eval_mode result

class CandidateOutcome(BaseModel):
    """One entry per candidate actually considered this run — the unit
    rank_candidates operates on, instead of raw PilotResult rows, so it can
    tell a candidate that was never attempted (SKIPPED) apart from one that
    was attempted and broke (LOAD_FAILED) apart from one that's missing a
    task's result for some other reason (INCOMPLETE). This is what makes
    doc 05 §6.5's escalation conditions ("all v2 loads failed") actually
    checkable — inferring it from an absent PilotResult can't distinguish
    those three cases."""
    candidate: str
    status: CandidateStatus
    error: str | None = None                  # set when status == LOAD_FAILED
    pilot_results: list[PilotResult] = []      # populated only when status == SUCCEEDED

class DecisionResult(BaseModel):
    winner: str
    task_scores: dict[Task, dict[str, float]]     # winner's own metrics per task
    baseline_scores: dict[Task, BaselineResult]
    rationale: str
    escalated_to_v1: bool
    escalation_reason: str | None
    candidate_statuses: dict[str, CandidateStatus]   # every considered candidate's final status,
                                                       # so the report (§11) can show why a
                                                       # candidate is absent from the ranking

def rank_candidates(
    outcomes: list[CandidateOutcome], baseline: list[BaselineResult], v2_candidate_names: list[str]
) -> DecisionResult:
    """Doc 05 §4.2 step 5's decision gate, made concrete:
    0. Require an outcome for every name in v2_candidate_names (the active
       v2 registry for this run) before proceeding — a missing outcome is a
       bug (a candidate that was neither run nor explicitly marked SKIPPED),
       not a silent 'treat as failed.'
    1. Among outcomes with status == SUCCEEDED: drop any candidate whose
       full-fine-tune spearman_rho is None on either task (an unrankable,
       degenerate result counts as NOT clearing the baseline — never
       compared as if it were a real number), or that does not clear the
       baseline on BOTH tasks by a non-trivial margin (a fixed, documented
       epsilon — not zero, since a marginal win within noise isn't a real
       clearance).
    2. Among survivors, score = (mean task Spearman rho improvement over
       baseline) / (relative compute cost, using wall_clock_s as the proxy
       — params_millions is a poor proxy per doc 05 §4.3's finding that
       compute is not actually the constraint at this scale, so don't rank
       on param count).
    3. Highest score wins. If no candidate survives step 1, `winner` is set
       to the string 'none' and `rationale` states this explicitly — doc 05
       §4.2 step 5's 'if nothing clears the baseline, that's a real
       finding' is a valid, reportable outcome, not a pipeline failure.
    4. escalated_to_v1 is True only if triggered by doc 05 §6.5's exact,
       now-checkable conditions: either every v2_candidate_names outcome
       has status == LOAD_FAILED, or every SUCCEEDED v2 candidate failed
       step 1's baseline clearance while a separately-run v1
       CandidateOutcome (status == SUCCEEDED) does clear it — never merely
       because v1 was included in `outcomes`."""
    ...
```

---

## 11. Module: `report.py`

Same shape as doc 09's `datasheet.py` and doc 10's `report.py`: emits `model_selection/pilot_report.json` and `model_selection/pilot_report.md` from one underlying data structure. Contents:
- The full pilot grid table (every `PilotResult` row: candidate × eval mode × task).
- Every candidate's final status (`DecisionResult.candidate_statuses`) — including SKIPPED/LOAD_FAILED/INCOMPLETE ones, so a reader can see *why* a candidate is missing from the ranking rather than just its absence.
- The baseline's own scores per task.
- `DecisionResult` — the winner, its margin over baseline, and the rationale.
- Which pilot fold (`pilot_fold_id`) was used, and the `transformers`/`torch` versions the run executed against (doc 05 §7's "first concrete task... log the transformers version it worked against").
- Per-candidate wall-clock time and peak VRAM, to make the compute-cost side of the ranking auditable rather than asserted.
- Whether `--include-v1`/`--full` were used for this run.

Per doc 05 §6.6: **this report is the persistent record of the decision** — log it once, here, so whoever starts Phase 4 doesn't have to re-run the pilot to know why a particular variant was picked.

---

## 12. CLI (`model_selection/cli.py`, entry point `clamp-pilot`)

```toml
[project.scripts]
clamp-data = "clamp.cli:app"
clamp-split = "clamp.splitting.cli:app"
clamp-pilot = "clamp.model_selection.cli:app"
```

```python
app = typer.Typer()

@app.command(name="resolve-revisions")
def resolve_revisions_cmd() -> None:
    """Runs §7.4's resolve_all_revisions() and prints the {name: sha}
    mapping as JSON, ready to paste into CLAMP_MODEL_REVISIONS_JSON — the
    one-command form of runbook step 1 (§13)."""
    ...

@app.command()
def baseline() -> None: ...

@app.command()
def probe(candidate: str, full: bool = False, include_v1: bool = False) -> None:
    """Run the frozen probe for one candidate, or every candidate in the
    default grid (or the full grid if --full) if candidate == 'all'.
    include_v1 defaults from settings.pilot_include_v1 but --include-v1
    passed here always wins (§7.3)."""
    ...

@app.command()
def finetune(candidate: str, include_v1: bool = False) -> None: ...

@app.command(name="run-grid")
def run_grid(full: bool = False, include_v1: bool = False) -> None:
    """baseline + probe + finetune for every candidate in the selected
    registry (§7.5), writing pilot_results.parquet incrementally so a
    partial run (e.g. interrupted mid-grid) isn't fully lost — mirrors
    Phase 1's resumable-pull philosophy (doc 09 §5) applied to a compute
    grid instead of an API pull. Also writes candidate_outcomes.json (§10,
    §15): every candidate not selected this run (v1 without --include-v1)
    gets a CandidateOutcome(status=SKIPPED); a candidate whose load_v2/
    load_v1 call raises gets LOAD_FAILED with the exception recorded in
    `error`, and run_grid continues with the remaining candidates rather
    than aborting the whole grid; a candidate missing a required task/
    eval_mode result at the end of the run gets INCOMPLETE. Only a
    candidate with every required PilotResult row gets SUCCEEDED."""
    ...

@app.command()
def report() -> None: ...
```

---

## 13. Runbook (execution order for Phase 3)

1. **Resolve and pin every candidate's revision** (§7.4) — run `clamp-pilot resolve-revisions` (or `resolve_all_revisions()` directly), confirm `AutoModel.from_pretrained(..., trust_remote_code=True, revision=<sha>)` actually loads for each, and log the `transformers` version that worked (doc 05 §7's literal first concrete task). Set the resulting mapping via `CLAMP_MODEL_REVISIONS_JSON` in `.env` — never hand-edit a sha into `_CANDIDATE_TEMPLATES` (§7.1) itself.
2. Confirm Phase 2's `data/splits/folds/fold_0.parquet` (or whichever `pilot_fold_id`) exists — this doc has no fallback if Phase 2 hasn't produced real folds yet.
3. `clamp-pilot baseline` — establishes the floor every candidate must clear.
4. `clamp-pilot run-grid` — by default, runs both the frozen-probe pass and the full-fine-tune pass for the **4 v2 candidates only**. `CANDIDATES` has 5 entries (registry membership includes v1), but v1 is excluded from every default-run stage — probe and fine-tune alike — unless `--include-v1` is passed. "5 candidates" refers to what's registered, not what executes by default.
5. **Inspect the frozen-probe numbers**, but per doc 05 §4.2 don't eliminate anything based on them alone — expect them to be uniformly weak.
6. **Inspect the full-fine-tune numbers** — this is the real signal (doc 05 §3's "frozen-embedding probing... was found weak" finding from the paper itself).
7. If no v2 candidate clears the baseline, or a v2 load broke entirely and couldn't be fixed quickly, re-run step 4 with `--include-v1` per doc 05 §6.5's exact escalation conditions — first completing the v1 setup (`git clone github.com/AaronFeller/PeptideCLM`, confirm `sys.path` wiring per §7.3) if it hasn't been done yet.
8. `clamp-pilot report` — writes and prints `pilot_report.{json,md}` including `DecisionResult`.
9. **Read `pilot_report.md` by hand** and treat the recorded winner as the one Phase 4 builds on — don't let Phase 4 quietly re-guess a different variant later without updating this report.

---

## 14. Testing plan

| Target | Approach | Why this one |
|---|---|---|
| `descriptors.compute_descriptors` | Unit tests against a handful of real, simple SMILES (glycine, a short synthetic peptide) with hand-checked MW/LogP sanity bounds | Cheap, RDKit does the real work; confirms the wiring, not RDKit's own correctness. |
| `baseline.train_baseline` | Unit test on synthetic descriptor/label data with a known linear relationship — confirm `ElasticNetCV` recovers it and `BaselineResult`'s fields are populated correctly | Pure sklearn logic; no need for real peptide data to validate the plumbing. |
| `candidates.resolve_registry` | A normal (always-run, always-expected-to-pass) test asserting: every `_CANDIDATE_TEMPLATES` entry has a non-empty `name`/`hf_repo`; given a fixture `revisions` dict covering every template name, `resolve_registry` returns entries with the matching non-empty `revision`; given an incomplete `revisions` dict, it raises `ValueError` naming exactly the missing candidates | Tests the *validation logic* deterministically — unlike a test asserting the real, unresolved `_CANDIDATE_TEMPLATES` have non-empty revisions (which would be permanently red until a human runs §7.4, conflicting with "pytest passes for the full suite," §17), this passes in CI immediately while still enforcing doc 10/doc 05's "don't run against an unpinned revision" rule at the registry level. |
| `candidates.load_v2`/`load_v1` | Mocked — monkeypatch `clamp.splitting.embed.load_encoder` (for v2) and the `sys.path`/`SMILES_SPE_Tokenizer` import (for v1) to return a tiny fake tokenizer/model pair; confirm the right loader is dispatched per `CandidateModel.loader` | Never load a real multi-hundred-MB checkpoint in CI; confirms dispatch logic only. |
| `probe.frozen_probe` | Unit test on synthetic embeddings + labels with a known separable structure, both `train_mask`/`test_mask` variants | Pure sklearn logic on synthetic data — same philosophy as doc 10 §14's cluster tests. |
| `finetune.should_stop` (early-stopping helper, §9.2) | Parametrized `pytest` cases: a strictly-improving history never stops, a flat/degrading history for exactly `patience` epochs stops, an improve-then-plateau history stops at the right epoch | This is the one piece of the real training loop that's meaningfully testable without a GPU or a real model — isolate it precisely so it can be. |
| `finetune.PilotRegressionHead.forward` | A forward-pass shape/gradient-flow test using a tiny fake encoder (a `nn.Embedding` + `nn.Linear` stand-in returning a `last_hidden_state`-shaped tensor) — confirm output shape is `(batch,)` and gradients reach the fake encoder's parameters | Confirms the mean-pool-with-gradients wiring (doc 10 §6.2's forward-compat claim) actually holds, without needing PeptideCLM-2 itself. |
| `metrics.compute_metrics` | Unit tests against small arrays with hand-computed MAE/Spearman/Pearson | Cheap, scipy/sklearn do the real math; confirms correct field mapping. |
| `decision.rank_candidates` | Unit tests covering: a clear winner clears baseline on both tasks; nothing clears baseline (confirm `winner == "none"`, not an exception); a tie-breaking-by-compute-cost case; an escalation-triggered case built from explicit `CandidateOutcome(status=LOAD_FAILED)` entries (not inferred from missing `PilotResult`s); a `SKIPPED` v1 outcome correctly excluded from ranking without affecting escalation; a `None` `spearman_rho` (degenerate prediction) correctly treated as non-clearing rather than crashing or sorting as if it were a number; a missing outcome for a `v2_candidate_names` entry raising rather than being silently skipped | This is the actual decision logic doc 05 §6 spells out as a numbered rule list, now including the status-distinction and finite-metric fixes — worth a regression test per condition, not just a happy-path test. |
| `report.py` | Snapshot-style test: fixed synthetic `PilotResult`/`DecisionResult` inputs produce `pilot_report.json`/`.md` whose numbers agree | Same drift-prevention rationale as docs 09/10. |

**Fixtures:** a tiny synthetic "fake encoder" (a couple of `nn.Module` layers producing embed-dim-shaped output from token ids) reused across `test_candidates.py`, `test_probe.py`, and `test_finetune.py` — this stands in for every real PeptideCLM-2 checkpoint in the test suite, exactly as doc 10 §14 uses a fake encoder for `embed_dataset`'s tests.

**Explicitly not covered by this test plan:** an actual GPU fine-tuning run against a real PeptideCLM-2 checkpoint, and the real pilot grid's actual numbers. Those need `clamp-pilot run-grid` executed once for real (§13) — no unit test substitutes for seeing whether `mlm-large` actually beats `hybrid-small` on this project's own HC50/MIC data, which is the entire empirical question this phase exists to answer.

---

## 15. Storage layout

```
data/
└── model_selection/
    ├── baseline_results.json          # BaselineResult per task
    ├── pilot_results.parquet          # every successful PilotResult row (candidate x eval_mode x task)
    ├── candidate_outcomes.json         # one CandidateOutcome per candidate considered this run
                                         # (§10) — SUCCEEDED/LOAD_FAILED/SKIPPED/INCOMPLETE, the
                                         # decision stage's actual input, not pilot_results.parquet directly
    ├── finetune_runs/
    │   └── {candidate}_{task}_seed{n}.json   # full FineTuneResult, one file per run, written incrementally
    ├── decision.json                   # DecisionResult
    └── pilot_report.{json,md}          # §11
```

Same gitignored convention as docs 09/10. `finetune_runs/` written incrementally (one file per run as it completes, not one big file at the end) specifically so `run-grid` can be interrupted and resumed without re-running already-completed candidate/task/seed combinations — mirrors Phase 1's resumable-pull philosophy (doc 09 §5) applied to compute jobs instead of network pulls.

---

## 16. Orchestration (`model_selection/pipeline.py`)

Same manifest pattern as Phase 1/2, via the shared `clamp.pipeline_utils` (doc 10 §2):

```python
STAGES: list[Stage] = [
    Stage("baseline", _baseline, inputs=["splits/folds/", "processed/dataset.parquet"],
          output="model_selection/baseline_results.json"),
    Stage("run_grid", _run_grid, inputs=["splits/folds/", "processed/dataset.parquet"],
          outputs=["model_selection/pilot_results.parquet", "model_selection/candidate_outcomes.json"]),
    Stage("decision", _decision, inputs=["model_selection/candidate_outcomes.json", "model_selection/baseline_results.json"],
          output="model_selection/decision.json"),
    Stage("report", _report, inputs=["model_selection/decision.json"],
          output="model_selection/pilot_report.json"),
]
```

`run_grid`'s own internal resumability (§15) is a level below this manifest's stage-level caching — the manifest decides whether to re-run `run_grid` **at all** given unchanged inputs; `run_grid`'s own per-run file checks decide which individual candidate/task/seed combinations to skip **within** a run. Both matter: the outer manifest avoids re-running the whole grid after an unrelated Phase 2 rerun with identical fold contents, the inner check avoids losing partial progress within one grid invocation.

---

## 17. Definition of done (Phase 3)

- [ ] `baseline_results.json` contains `BaselineResult` for both HC50 and MIC, computed on the designated pilot fold.
- [ ] Every v2 candidate in the default grid (`hybrid-small`, `mtr-small`, `mlm-small`, `mlm-large`) has a `ProbeResult` and at least one `FineTuneResult` per task, using a **pinned, resolved** revision (not `main`).
- [ ] The `transformers`/`torch` versions the pilot actually ran against are recorded in `pilot_report.md` (doc 05 §7's first concrete task, closed out).
- [ ] `decision.json` contains a `DecisionResult` with a non-placeholder `winner` (either a real candidate name or the explicit string `"none"` with rationale) — never silently empty.
- [ ] If the escalation conditions (doc 05 §6.5) were triggered, the v1 fallback was actually run (`--include-v1`) and its results are included in `pilot_report.md`, not just noted as "would run if needed."
- [ ] `pilot_report.md` has been **read** by whoever runs this, and the chosen winner is the one Phase 4 is actually built against — not a stale or re-guessed variant.
- [ ] `pytest` passes for the full `tests/model_selection/` suite in §14.

---

## 18. Open risks carried into implementation

Restated from doc 05 §7 as concrete things that can break this specific scaffold:

1. **No one has actually executed a live `trust_remote_code=True` load of any PeptideCLM-2 variant end-to-end yet** (doc 05 §7) — §13 runbook step 1 is the first time this happens for real; budget time for `ChemPepMTR.py`/installed-`transformers`-version incompatibility as a real possibility, not a hypothetical.
2. **Whether `AutoTokenizer.from_pretrained` truly works without `trust_remote_code=True` for v2 is unverified** (doc 05 §2.1/§7) — confirm during the same first real load, and fall back to passing the flag for the tokenizer too if it turns out required, updating `candidates.load_v2` accordingly.
3. **Paper Table 1 numbers (§3's MCC/AUROC figures) were read from a parsed summary, not the raw table** (doc 05 §7) — treat any comparison of this project's own pilot numbers against the paper's reported AmpHGT/PepMSND figures as directional, not exact, until re-verified against the primary PDF/PMC table.
4. **Whether v2's pretraining corpus inherited v1's cyclization ring-closure bug is unverified** (doc 05 §1.2/§7) — if the pilot's cyclic-peptide subset (identifiable via Phase 2's `is_cyclic`/`cyclization_type` fields) shows anomalously poor fine-tune performance across every v2 candidate, this is a concrete hypothesis worth checking before assuming it's a modeling-capacity issue instead.
5. **Tokenizer vocab coverage for modified/non-canonical-residue SMILES is untested for both v1 and v2** (doc 05 §7) — if the pilot's fine-tune metrics look unexpectedly bad specifically on the `PARTIAL`/`BACKBONE_ONLY` fidelity-tier subset of peptides (per Phase 1's `chemical_fidelity_tier` field), check for excessive `[UNK]`/fallback tokenization before concluding the encoder itself is a poor fit.
6. **v1's released checkpoint was pretrained on ring-closure-buggy cyclic SMILES data, with no retrained checkpoint available** (doc 05 §1.2) — a standing, unfixable-without-repretraining caveat specifically on any v1 fallback run; log it in `pilot_report.md` if v1 is ever actually run, not just in this doc.
7. **Compute is not expected to be the pilot's limiting factor** (doc 05 §4.3) — if a real run instead finds `mlm-large` fine-tuning is surprisingly slow or VRAM-constrained on the available hardware, that's new information worth feeding back into the §10 decision rule's compute-cost term, not a sign the estimate methodology was wrong.
