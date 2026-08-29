# Phase 5 Implementation Design: Ablations & Final Validation

*Executable companion to `08_implementation_roadmap.md` Phase 5 ("Ablations & Final Validation"), translating `03_task5_warmstart_encoder_ablation_plan.md` (the warm-start ablation) and the roadmap's Therapeutic Index bullet into an actual module layout, class/function signatures, script structure, and test plan — the way `09_phase1_implementation_design.md` did for Phase 1 and `12_phase4_implementation_design.md` did for Phase 4. This phase has two independent workstreams (warm-start-encoder ablation; derived-vs-learned TI ablation) that share no data dependency on each other but do share tooling built in Phase 4 (`clamp.models`, `clamp.training.checkpoint`, `clamp.training.metrics`) — this doc reuses that tooling rather than re-deriving it.*

---

## 0. Scope and non-goals

**In scope** (mirrors doc 03 and doc 08's Phase 5 bullets):
- **Workstream A — warm-start ablation.** Reproduce the PAMPA-fine-tuned checkpoint from the PeptideCLM-2 authors' own training script (since the weights themselves were never published, per doc 03 §1) faithfully enough to serve as arm (b) of a controlled comparison, then run that checkpoint through the exact P2 protocol (doc 12 §7) alongside the base-checkpoint arm (a), and report whether PAMPA-derived permeability features transfer to HC50/MIC.
- **Workstream B — Therapeutic Index ablation.** Compare the default derived TI (`log(HC50) - log(MIC)`, already implemented in Phase 4's `LysisMultitaskModel.encode`/marginal heads, doc 04 §4.3) against a third, learned TI head trained only on the dual-labeled (both HC50 and MIC present) subset.
- The statistical protocol both workstreams need (paired significance testing across folds/seeds, per-metric/per-stage reporting discipline) — built once, shared.

**Explicitly out of scope for this doc:**
- Anything from Phase 1-4 (data pipeline, splitting, model selection pilot, the core P2/P3 multitask model itself) — this phase consumes all of that as already-built.
- Any change to the *default* production training path decided in Phase 4 (§10 of doc 12) — both ablations here are read-only consumers of Phase 4 checkpoints/splits, plus one new checkpoint-construction path (the reproduced PAMPA checkpoint, Workstream A) that exists solely to feed the ablation, not to become a new default.
- GradNorm-style dynamic task weighting, LoRA-across-all-blocks as a *default* fine-tuning protocol — doc 03 §4 raises LoRA as an alternative to freeze/unfreeze staging; this doc picks freeze/unfreeze (§2.3 below) and explains why, rather than building both.

---

## 1. Package layout

**Recommendation: a new `clamp.ablations` package**, sibling to `clamp.models`/`clamp.training`, with one subpackage per workstream plus a shared `common.py` — mirrors doc 12's "package per lifecycle concern" pattern (data vs. models vs. training), applied one level further for "one-off controlled experiments that consume the training stack but aren't part of the default pipeline."

```
CLAMP/
├── src/
│   └── clamp/
│       ├── data/                      # Phase 1
│       ├── models/                    # Phase 4
│       ├── training/                  # Phase 4
│       └── ablations/
│           ├── __init__.py
│           ├── common.py              # paired-run bookkeeping, seed pairing (§4)
│           ├── warmstart/
│           │   ├── __init__.py
│           │   ├── pampa_data.py      # load PAMPA_clusters.csv, LOCO folds (§2.1)
│           │   ├── pampa_finetune.py  # reproduce finetune_ensemble.py protocol (§2.2)
│           │   ├── ckpt_convert.py    # HF safetensors <-> reproduction-run state dict parity check (§2.2)
│           │   └── run_ablation.py    # arm (a) vs (b) through the P2 protocol (§2.3-2.5)
│           └── ti/
│               ├── __init__.py
│               ├── dual_labeled.py    # dual-labeled subset construction (§3.1)
│               ├── learned_head.py    # LearnedTIHead module (§3.2)
│               ├── train.py           # train the learned head on the dual-labeled subset (§3.3)
│               └── compare.py         # derived vs. learned TI comparison (§3.4)
├── tests/
│   ├── training/                       # Phase 4, unchanged
│   └── ablations/
│       ├── test_pampa_data.py
│       ├── test_ckpt_convert.py
│       ├── test_common_pairing.py
│       ├── test_learned_ti_head.py
│       └── test_dual_labeled_subset.py
└── data/
    └── runs/
        ├── p2_hc50/, p3_multitask/     # Phase 4, unchanged
        ├── pampa_reproduction/{replicate_id}/     # Workstream A intermediate checkpoints (§2.2)
        ├── warmstart_ablation/{arm}_{fold}_{seed}/ # Workstream A P2-protocol runs (§2.5)
        └── ti_ablation/{fold}_{seed}/               # Workstream B runs (§3.3)
```

No new top-level `models/`-style module is needed for Workstream A: it reuses `clamp.models.multitask.LysisMultitaskModel` and `clamp.training.freezing`/`losses`/`sampler`/`loop`/`checkpoint` unchanged, constructing the *encoder's initial weights* differently (from the reproduced PAMPA checkpoint instead of the base HF checkpoint) but running the identical P2 training code afterward. This is the concrete form of doc 03 §2's "everything else identical" requirement: reusing the exact same P2 code path for both arms is what makes "identical" true by construction rather than by careful duplication.

---

## 2. Workstream A: warm-start-encoder ablation

### 2.1 PAMPA data and folds (`ablations/warmstart/pampa_data.py`)

Per doc 03 §1, the reproduction source data is `PeptideCLM-2/data/PAMPA_clusters.csv` (6,702 rows: `SMILES, PAMPA, cluster`, 6 pre-computed k-means clusters). This is a **one-time external download**, not something Phase 1's pipeline produces — treat it as a new, narrowly-scoped raw input, cached the same way Phase 1 caches bulk downloads (doc 09 §5's `BulkDownloader` pattern, reused conceptually, not literally — this isn't an AMP label source so it doesn't belong in `clamp.data.sources`).

```python
# ablations/warmstart/pampa_data.py
import pandas as pd
from pathlib import Path

def load_pampa_clusters(path: Path) -> pd.DataFrame:
    """Loads the authors' PAMPA_clusters.csv (SMILES, PAMPA, cluster).
    Source: github.com/AaronFeller/PeptideCLM-2, data/PAMPA_clusters.csv
    (doc 03 §1) -- cache under data/raw/pampa/PAMPA_clusters.csv, fetched
    once, not re-derived."""
    df = pd.read_csv(path)
    required = {"SMILES", "PAMPA", "cluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"unexpected PAMPA_clusters.csv schema, missing {missing}")
    return df

class PampaFold:
    def __init__(self, train_df, val_df, test_cluster: int):
        self.train_df, self.val_df, self.test_cluster = train_df, val_df, test_cluster

def make_loco_folds(df: pd.DataFrame, cluster_col: str = "cluster") -> list[PampaFold]:
    """doc 03 §4's simplified single-validation-cluster variant of Feller &
    Wilke's nested scheme (not the full 6x5=30-run nested CV) -- one fold
    per cluster acting as the held-out test set, with one other cluster
    fixed as validation for early stopping, the rest as train. This keeps
    Workstream A's own compute bounded (this is *reproducing a checkpoint
    that feeds an ablation*, not the primary experiment) while preserving
    the leakage-safe LOCO property."""
    ...
```

**Note on why full nested 6x5 CV is not the default here:** doc 03 §3 recommends reproducing at least the ~15-30-run scale for the *HC50/MIC ablation itself* (arm a/b through the P2 protocol, §2.5). The *PAMPA-checkpoint-construction* step (this section) is one layer upstream of that — it only needs to produce a small number of independently-reproduced PAMPA checkpoints (§2.2's replicate count), not a full nested CV grid, since its own held-out PAMPA metric is not the thing being reported; only the resulting checkpoint's *downstream* HC50/MIC performance is.

### 2.2 Reproducing the PAMPA fine-tune (`ablations/warmstart/pampa_finetune.py`)

Concrete hyperparameters, taken directly from doc 03 §1 (already verified against the authors' live script, not re-guessed here):

```python
# ablations/warmstart/pampa_finetune.py
from pydantic import BaseModel

class PampaFinetuneConfig(BaseModel):
    backbone: str                     # same HF repo id as configs/locked_backbone.json (doc 12 §2.2) --
                                       # arm (a) and arm (b) MUST be the same architecture/size
                                       # (doc 03 §6 checklist item 1), or the comparison is meaningless
    lr: float                         # size-dependent: small=3e-4, base=1e-4, large=5e-5 (doc 03 §1)
    max_epochs: int = 10
    val_check_every_frac_epoch: float = 0.2     # validate 5x/epoch, matching the authors' script
    early_stopping_patience: int = 4
    grad_clip_val: float = 0.1
    precision: str = "bf16-mixed"
    batch_size_train: int = 16
    batch_size_val: int = 128
    resample_bins: int = 5            # pd.cut(train_df['PAMPA'], bins=5) + upsample-to-max-bin-size,
                                       # doc 03 §1 -- the authors' own value-distribution rebalancing
    transfer_learning: bool = False    # False = full unfrozen fine-tune, matching the authors'
                                       # ACTUAL production runs (doc 03 §1 point 3 -- the
                                       # --transfer_learning flag is OMITTED in their real PAMPA
                                       # study runs, so the real published checkpoint-equivalent
                                       # is full fine-tune, not a frozen-encoder probe)
    seed: int = 0


def train_pampa_finetune(cfg: PampaFinetuneConfig, fold: "PampaFold", out_dir: str) -> str:
    """Reimplements finetune_ensemble.py's protocol using this project's
    own training primitives (clamp.training.loop, clamp.training.freezing)
    rather than adding a pytorch-lightning dependency -- see the explicit
    tradeoff note below. Returns the path to the saved reproduction
    checkpoint (same clamp.training.checkpoint format as Phase 4, so it
    can be loaded by the exact same load_checkpoint() call).
    """
    ...
```

**Design choice, stated explicitly (doc 09-style options-and-recommendation):**

| | Reimplement in `clamp.training` primitives | Depend on `pytorch-lightning` to match the original script line-for-line |
|---|---|---|
| New dependency | none (reuses Phase 4's `loop.py`/`freezing.py`) | adds `lightning` + its `Trainer` API surface |
| Fidelity to the authors' exact code | high on hyperparameters (LR, batch size, clip, precision, patience, resampling), reimplemented mechanically for the *training loop shape* (epoch loop, `val_check_interval` as "validate 5x per epoch") | exact for loop mechanics too, at the cost of a second training-loop idiom in this codebase |
| Consistency with Phase 4's tooling | high — same checkpoint format, same `EarlyStopper`, same metrics module | low — would need its own checkpoint bridge back into `clamp.training.checkpoint` |

**Recommendation: reimplement using `clamp.training` primitives**, matching every stated hyperparameter exactly (§ above) but not adopting Lightning as a second training-loop framework in this codebase — consistent with doc 12 §0's decision not to add a training-framework dependency for the core P2/P3 loop either. `val_check_interval=0.2` becomes "run a validation pass after every `ceil(0.2 * steps_per_epoch)` optimizer steps" inside `run_stage`-equivalent code, a mechanical, low-risk translation.

**Critical detail carried forward, not re-derived:** doc 03 §1 point 3 confirms the authors' actual production PAMPA runs used **full, unfrozen fine-tuning** (`transfer_learning=False` above) — not the frozen-encoder-probe path their own script *also* supports. `PampaFinetuneConfig.transfer_learning` defaults to `False` for exactly this reason: matching the real published-result recipe is the point, not matching a path they built but didn't use for the actual PAMPA study.

### 2.3 One fine-tuning protocol for everything, chosen once

Doc 03 §4 explicitly requires **one** fine-tuning protocol to be used identically across (i) the arm-(b) PAMPA reproduction, (ii) arm (a)'s HC50/MIC fine-tune, and (iii) arm (b)'s HC50/MIC fine-tune — mixing freeze/unfreeze staging for one arm and LoRA for another would itself be a confound.

**Recommendation: freeze -> top-k-unfreeze (Phase 4's P2 two-stage protocol, doc 12 §6-7), not LoRA-across-all-blocks**, for two concrete reasons specific to this ablation:
1. It is **already built** (doc 12 §6-7) and shared verbatim by both arms of the HC50/MIC comparison (§2.5) — no new fine-tuning code path is needed for (ii)/(iii) above, only for the upstream PAMPA reproduction itself (§2.2), which is a one-time cost regardless of protocol choice.
2. It is **more interpretable for this specific ablation's purpose**: doc 03's Risk B/§2's mitigation for "is a difference attributable to genuine feature transfer vs. narrow full-fine-tune overfitting on PAMPA" is to compare the **stage-1 frozen-probe** result between arms before any HC50/MIC-specific adaptation happens. Freeze/unfreeze staging produces that frozen-probe number as a first-class, already-reported intermediate (doc 12 §7.2); LoRA-across-all-blocks does not have an equivalent "representation-only, zero task-adaptation" checkpoint to compare, since LoRA adapters and the head train jointly from step one.

This is a case where the roadmap's own phrasing ("Execute 03...") plus doc 03 §4's explicit flag ("pick one protocol explicitly... If engineering time is tight, LoRA-all-blocks is the better-tested path... if the cleaner mechanistic story matters more than convenience, freeze->top-k-unfreeze is more interpretable") is resolved here in favor of the mechanistic-story option, specifically because Phase 4 already paid the engineering cost of building it.

### 2.4 Checkpoint parity: HF safetensors vs. reproduction-run state dict (`ablations/warmstart/ckpt_convert.py`)

Doc 03 §1 point 4 flags a real gotcha: the authors' original script loads a Lightning `.ckpt` in their own `MTR_model` state-dict shape, not the HF `AutoModel` this project's `LysisMultitaskModel` uses. Since this reproduction is built entirely on `clamp.models`/`clamp.training` (§2.2's recommendation), **no format bridge to the authors' own Lightning checkpoint format is needed** — the reproduction starts from the same `AutoModel.from_pretrained(backbone, ...)` call `LysisMultitaskModel` already uses (doc 12 §5), sidestepping the safetensors<->Lightning-state-dict conversion problem entirely, at the cost of not being a byte-for-byte reproduction of the authors' exact loading code path.

```python
# ablations/warmstart/ckpt_convert.py
def verify_tokenizer_parity(backbone_a: str, backbone_b: str, probe_smiles: list[str]) -> bool:
    """doc 03 §1 point 4's flagged risk: the tokenizer used in the authors'
    original script (aaronfeller/PeptideMTR) may not be identical to the
    tokenizer shipped with whichever peptideclm-2-hybrid-* repo is locked
    (doc 12 §2.2). Tokenize the same probe SMILES with both and assert
    identical token-id sequences before trusting that arm (a) and arm (b)
    share a tokenizer -- if they diverge, the ablation's 'only the encoder
    differs' premise (doc 03 §2.2) is already broken before training starts.
    """
    ...
```

This function is the one piece of real, load-bearing verification work this section needs — everything else in §2.2-2.3 is training-protocol reuse, but tokenizer parity is exactly the kind of "easy to silently get wrong" gap doc 03 flags explicitly and that would invalidate the whole ablation if missed.

### 2.5 Running arms (a) and (b) through the P2 protocol

```python
# ablations/warmstart/run_ablation.py
from clamp.models.multitask import LysisMultitaskModel
from clamp.training import freezing, loop, checkpoint

def build_arm_model(backbone: str, arm: str, pampa_checkpoint_dir: str | None, head_init_seed: int):
    """arm 'a': base pretrained encoder, unmodified.
    arm 'b': base architecture, encoder weights loaded from the reproduced
    PAMPA checkpoint (§2.2/§2.4).
    Both arms: head freshly initialized using head_init_seed -- the ONLY
    thing that differs between (a) run i and (b) run i is the encoder's
    starting state_dict (doc 03 §2.2)."""
    torch.manual_seed(head_init_seed)
    model = LysisMultitaskModel(backbone=backbone, tasks=("hc50",))
    if arm == "b":
        pampa_model, _ = checkpoint.load_checkpoint(pampa_checkpoint_dir)
        model.encoder.load_state_dict(pampa_model.encoder.state_dict())
    return model


def run_arm(arm: str, fold: int, seed: int, cfg, pampa_checkpoint_dir: str | None) -> "ArmResult":
    """Runs the exact P2 two-stage protocol (doc 12 §6-7) for one arm, one
    fold, one seed. Reports stage-1 (frozen-probe) AND stage-2 metrics
    separately (doc 03 §4's 'report frozen-probe result for both arms as
    a primary diagnostic'), for both HC50 and MIC as independent runs
    (doc 03 §2.5 -- 'do not blend into one score first')."""
    ...
```

**Step-count-vs-early-stopping subtlety (doc 03 §2.4):** log `steps_at_stop` for every run of both arms in `training_meta.json` (reusing Phase 4's `CheckpointMeta`, doc 12 §10, with an added `pampa_ablation_arm: str | None` field). If arm (a) and arm (b) diverge substantially in realized step count under the shared early-stopping criterion, additionally run the **fixed-step-budget** variant (same total optimizer steps for both arms, no early stopping) as a robustness check, per doc 03 §2.4 — implement this as a second `StageConfig` (doc 12 §7.1) with `early_stopping_patience` set high enough to be effectively disabled and a hard step cap instead, rather than a new mechanism.

### 2.6 Seeds, folds, and paired design

Direct implementation of doc 03 §2.2/§3:

```python
# ablations/common.py
from dataclasses import dataclass

@dataclass(frozen=True)
class PairedRunSpec:
    fold: int
    head_init_seed: int    # same value used for BOTH arm (a) run i and arm (b) run i (doc 03 §2.2) --
                            # this is what licenses the paired Wilcoxon test (doc 12 §9's paired_wilcoxon)

def paired_run_grid(n_folds: int, n_seeds: int = 3) -> list[PairedRunSpec]:
    """doc 03 §3: >=5 folds x 3 seeds = >=15 runs per arm, 30 total, for
    the single-task HC50-first ablation. Do not go below 3 seeds (doc 03
    §3: a single seed cannot distinguish a genuine arm-(b)-is-worse result
    from an unlucky head initialization draw)."""
    return [PairedRunSpec(fold=f, head_init_seed=s) for f in range(n_folds) for s in range(n_seeds)]
```

Also budget, per doc 03 §3's last bullet: if compute allows, reproduce arm (b)'s **construction** (§2.2) itself 2-3x with different seeds before ever touching HC50/MIC data, treating "which PAMPA-checkpoint replicate" as a nested random factor (doc 03 Risk C) rather than silently using one draw. This doc's `data/runs/pampa_reproduction/{replicate_id}/` layout (§1) already anticipates multiple replicates for exactly this reason — `replicate_id` should range over at least 1-3 if this budget is taken.

### 2.7 Reporting and statistics

Per doc 03 §2.5/§5 (Risk E/F): report HC50 and MIC **separately**, stage-1 (frozen-probe) and stage-2 (unfrozen) **separately**, step-count-at-stop for both arms, and run the **paired Wilcoxon signed-rank test** (`clamp.training.metrics.paired_wilcoxon`, doc 12 §9) on the per-fold `(arm_a_metric - arm_b_metric)` deltas as the pre-registered primary comparison — decided and written down **before** arm (b) is ever trained, per doc 03 §5 Risk F's explicit warning against post-hoc metric/stage cherry-picking.

```python
# ablations/warmstart/run_ablation.py (cont.)
def summarize(results_a: list["ArmResult"], results_b: list["ArmResult"]) -> "pd.DataFrame":
    """One row per (task, stage) x fold: metric_a, metric_b, delta,
    steps_at_stop_a, steps_at_stop_b. Feeds paired_wilcoxon per (task, stage)."""
    ...
```

**Pre-registration, stated here rather than left to whoever runs this:** the primary comparison is **Pearson r on the locked held-out folds, stage-2 (unfrozen), reported separately for HC50 and MIC**, tested via paired Wilcoxon on per-fold deltas — matching doc 03 §2.5's rationale (PCC matches QMAP's own reporting convention). Stage-1 frozen-probe results and step-count diagnostics are secondary/diagnostic, reported alongside but not substituted as the headline number after the fact.

---

## 3. Workstream B: Therapeutic Index ablation

### 3.1 Dual-labeled subset construction (`ablations/ti/dual_labeled.py`)

```python
# ablations/ti/dual_labeled.py
import pandas as pd

def build_dual_labeled_subset(dataset_path: str, split_table: "SplitTable") -> pd.DataFrame:
    """Filters processed/dataset.parquet (Phase 1 output, one row per
    peptide-construct assay-record, doc 09 §3) down to peptide_uids that
    have BOTH an hc50_log_uM and an mic_log_uM value on at least one row
    each -- doc 04 §3.1's 'both-labeled' subset, the only slice that can
    supervise a learned TI head directly. Joins the Phase 4 split table
    (doc 12 §2.1) so this subset inherits the SAME train/val/test roles
    already assigned to those peptide_uids -- no new clustering/splitting,
    reusing Phase 2's joint split is required here for the same leakage
    reason doc 07 §3.3 gives for the marginal heads.
    """
    ...

def dual_labeled_ti_actual(df: pd.DataFrame) -> pd.Series:
    """TI_actual = hc50_log_uM - mic_log_uM, computed from the RAW labels
    (not model predictions) for peptides in the dual-labeled subset --
    this is the learned head's supervision target."""
    return df["hc50_log_uM"] - df["mic_log_uM"]
```

**Open item flagged rather than resolved (carried from doc 04 §7 item 6):** this subset's size is unknown until measured against the real dataset — doc 04 explicitly flags that no source consulted gives an exact count of peptides with directly comparable both-labeled measurements. **Do not commit engineering time to Workstream B's training step (§3.3) until this count is actually measured** (a one-line addition to Phase 1's datasheet, doc 09 §9, or a standalone check against `processed/dataset.parquet`) — if it's small enough that a train/val/test split within it is statistically meaningless (say, well under ~100 peptides total), report that as the finding and fall back to a qualitative case-study comparison rather than a formal held-out evaluation, exactly the same discipline doc 07 §4.2 recommends for the cyclic-holdout ablation when its subset turns out too small.

### 3.2 Learned TI head (`ablations/ti/learned_head.py`)

Direct implementation of doc 04 §4.3's `→ OPEN (P4 ablation)` item:

```python
# ablations/ti/learned_head.py
import torch
import torch.nn as nn
from typing import Literal

class LearnedTIHead(nn.Module):
    """Takes [shared_projection_output, pred_hc50, pred_mic] (concatenated,
    dim d+2) and predicts either TI_actual directly, or the residual
    (TI_actual - TI_derived) -- doc 04 §4.3's exact framing of what this
    ablation isolates: does a learned head capture selectivity signal not
    already present in the two marginal heads' predictions?
    """

    def __init__(self, d: int, mode: Literal["direct", "residual"] = "residual", dropout: float = 0.2):
        super().__init__()
        self.mode = mode
        self.net = nn.Sequential(
            nn.Linear(d + 2, d), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d, 1),
        )

    def forward(self, shared_embedding: torch.Tensor, pred_hc50: torch.Tensor, pred_mic: torch.Tensor) -> torch.Tensor:
        x = torch.cat([shared_embedding, pred_hc50.unsqueeze(-1), pred_mic.unsqueeze(-1)], dim=-1)
        out = self.net(x).squeeze(-1)
        if self.mode == "residual":
            derived = pred_hc50 - pred_mic
            return derived + out   # predicted TI = derived + learned correction
        return out                  # predicted TI = learned head's direct output
```

**Recommendation: `mode="residual"` as the default arm to report**, with `mode="direct"` as a secondary variant — predicting a residual against an already-reasonable derived baseline is a strictly easier learning problem than predicting the full quantity from scratch on a small subset, and it makes the comparison's null hypothesis explicit (`out == 0` everywhere means the learned head found nothing beyond the marginals, which is a clean, interpretable failure mode rather than an ambiguous "the head just learned something different but not necessarily better" result).

### 3.3 Training the learned head (`ablations/ti/train.py`)

```python
# ablations/ti/train.py
from clamp.training import checkpoint, loop

def train_ti_head(
    p3_checkpoint_dir: str,     # a completed Phase 4 P3 (multitask) run, doc 12 §10
    dual_labeled_df: "pd.DataFrame",
    mode: str = "residual",
    freeze_marginals: bool = True,
) -> str:
    """Loads a trained P3 LysisMultitaskModel (encoder + shared_projection +
    head_hc50 + head_mic all already fit on their FULL respective label
    sets, per doc 04 §4.3's reasoning for why derived TI uses strictly
    more data than a from-scratch learned head could). Freezes everything
    except a newly-constructed LearnedTIHead, trains only that head on the
    dual-labeled subset's train/val split, early-stopping on val loss.
    Returns the path to the saved run (LearnedTIHead state dict + its own
    small CheckpointMeta-equivalent, not a full LysisMultitaskModel
    checkpoint since only one new module was trained).
    """
    model, meta = checkpoint.load_checkpoint(p3_checkpoint_dir)
    if freeze_marginals:
        for p in model.parameters():
            p.requires_grad_(False)
    ti_head = LearnedTIHead(d=model.encoder.config.hidden_size, mode=mode)
    ...
```

`freeze_marginals=True` is the default and the primary comparison arm — the point of this ablation (doc 04 §4.3) is whether a *third* head sees selectivity signal the *already-trained* marginals don't expose, not whether jointly re-tuning everything on the (small) dual-labeled subset helps, which would be a different, noisier question given how much smaller this subset is than either marginal task's full label set.

### 3.4 Comparison: derived vs. learned (`ablations/ti/compare.py`)

```python
# ablations/ti/compare.py
from clamp.training.metrics import spearman, paired_wilcoxon

def derived_ti(pred_hc50, pred_mic):
    """Same formula as clamp.models.multitask -- TI = log(HC50) - log(MIC),
    larger = more selective. Reproduced here (not imported) only because
    this module deliberately has no import-time dependency on a live
    LysisMultitaskModel instance; recompute from stored predictions."""
    return pred_hc50 - pred_mic

def compare_derived_vs_learned(
    ti_actual, pred_hc50, pred_mic, learned_ti_pred, cluster_ids,
) -> "pd.DataFrame":
    """Per held-out fold: Spearman rho(derived_ti, ti_actual) vs.
    Spearman rho(learned_ti_pred, ti_actual). Returns per-fold deltas,
    feeds paired_wilcoxon exactly as Workstream A does (§2.7) -- same
    shared statistical tool, doc 12 §9, reused rather than reimplemented.
    """
    ...
```

**Primary comparison, pre-registered:** Spearman ρ (ranking correlation is the operative quantity for TI — a therapeutic index is used to *rank* candidate peptides by selectivity, not to hit an exact numeric value) between each method's TI prediction and `TI_actual` on the held-out fold(s) of the dual-labeled subset, tested for significance via the same paired Wilcoxon machinery as Workstream A. This mirrors doc 04 §4.3's own framing of the question ("does a learned head capture selectivity signal not in the marginals") directly, rather than substituting a different metric after the fact.

---

## 4. Shared statistical/reporting utilities (`ablations/common.py`, cont.)

Both workstreams need the same three things: paired seed bookkeeping (§2.6), a paired significance test (reused from doc 12 §9's `paired_wilcoxon`, not duplicated), and a reporting convention that keeps per-metric/per-stage results separate rather than collapsed. Rather than each workstream inventing its own summary-table shape, both `run_ablation.py` (§2.7) and `compare.py` (§3.4) should emit a common tidy long-format table:

```python
# ablations/common.py (cont.)
import pandas as pd

class AblationResultRow(BaseModel):
    workstream: str            # "warmstart" | "ti"
    arm: str                   # "a"/"b" for warmstart; "derived"/"learned_direct"/"learned_residual" for TI
    task: str                  # "hc50" | "mic" | "ti"
    stage: str                 # "frozen" | "unfrozen" for warmstart; "n/a" for TI
    fold: int
    seed: int
    metric_name: str           # "spearman" | "pearson" | "rmse" | "mae"
    metric_value: float
    steps_at_stop: int | None = None

def to_long_table(rows: list[AblationResultRow]) -> pd.DataFrame: ...
def write_ablation_report(rows: list[AblationResultRow], out_path: str) -> None:
    """Writes both a machine-readable ablation_results.json and a
    human-readable ablation_report.md from the same underlying rows --
    same 'don't hand-author the markdown separately from the JSON'
    discipline as Phase 1's datasheet.py (doc 09 §9)."""
    ...
```

---

## 5. CLI

Extends the `clamp-train` entry point (doc 12 §11) with an `ablate` subcommand group, rather than a third top-level script — both ablations are, mechanically, specialized consumers of the same training stack, not a new lifecycle.

```python
# ablations/cli.py (registered as a Typer sub-app under clamp.training.cli, or its own
# clamp-ablate entry point -- either is fine; recommend a separate entry point since
# these are one-off experiments, not part of the routine train/evaluate loop)
import typer

app = typer.Typer()

@app.command()
def pampa_reproduce(config_path: str, replicate_id: int = 0) -> None:
    """Workstream A step 1: reproduce one PAMPA fine-tune checkpoint (§2.2)."""
    ...

@app.command()
def warmstart_run(fold: int, seed: int, arm: str, pampa_checkpoint_dir: str = "") -> None:
    """Workstream A step 2: one paired (fold, seed) run of one arm through P2 (§2.5)."""
    ...

@app.command()
def warmstart_report(runs_dir: str = "data/runs/warmstart_ablation") -> None:
    """Workstream A step 3: aggregate all completed runs, paired Wilcoxon, write report (§2.7, §4)."""
    ...

@app.command()
def ti_measure_subset(dataset_path: str, split_path: str) -> None:
    """Workstream B step 0: measure dual-labeled subset size before committing to training it (§3.1)."""
    ...

@app.command()
def ti_train(p3_checkpoint_dir: str, fold: int, seed: int, mode: str = "residual") -> None:
    """Workstream B step 1: train the learned TI head (§3.3)."""
    ...

@app.command()
def ti_report(runs_dir: str = "data/runs/ti_ablation") -> None:
    """Workstream B step 2: aggregate, compare derived vs. learned, write report (§3.4, §4)."""
    ...
```

```toml
[project.scripts]
clamp-data = "clamp.cli:app"
clamp-train = "clamp.training.cli:app"
clamp-ablate = "clamp.ablations.cli:app"
```

---

## 6. Testing plan

Same discipline as doc 09/doc 12: pure-logic unit tests in CI, real training runs exercised manually/once per protocol change.

| Target | Approach | Why this one |
|---|---|---|
| `warmstart/pampa_data.py` (`load_pampa_clusters`, `make_loco_folds`) | Small synthetic CSV fixture matching the real 3-column schema; assert `make_loco_folds` produces exactly one fold per unique cluster value, each fold's train/val/test partition is disjoint | Cheap schema-shape test; the real `PAMPA_clusters.csv` download is a one-time manual step (§2.1), not something CI re-fetches. |
| `warmstart/ckpt_convert.py` (`verify_tokenizer_parity`) | Mock two tokenizer objects (or use two tiny real fast tokenizers with deliberately different vocabularies) and assert the function correctly returns `False` on a known-mismatched pair and `True` on an identical pair | This is the one real correctness-critical check in Workstream A's setup (§2.4) — a false "parity confirmed" would silently invalidate the whole ablation. |
| `common.py` (`paired_run_grid`) | Assert `paired_run_grid(n_folds=5, n_seeds=3)` returns exactly 15 `PairedRunSpec`s, with every `(fold, seed)` combination present exactly once, and that calling it twice with the same arguments is deterministic | Directly verifies doc 03 §3's ">=5 folds x 3 seeds = 15 runs/arm" requirement is what the code actually produces, not just documented. |
| `ti/dual_labeled.py` (`build_dual_labeled_subset`, `dual_labeled_ti_actual`) | Synthetic `dataset.parquet`-shaped DataFrame with a mix of HC50-only, MIC-only, and dual-labeled rows sharing various `peptide_uid`s; assert the subset includes only true dual-labeled peptides and `dual_labeled_ti_actual` matches a hand-computed value | This is the filter the whole workstream's validity depends on — an off-by-one join bug here (e.g. including a peptide_uid where the *dataset* has both labels but not on rows that both pass QC) would quietly corrupt the ablation's target variable. |
| `ti/learned_head.py` (`LearnedTIHead`) | Forward-pass shape test for both `mode="direct"` and `mode="residual"`; for `mode="residual"`, assert that zeroing `self.net`'s final-layer weights makes the head's output exactly equal `pred_hc50 - pred_mic` (i.e., confirms the residual-add wiring, not just that it runs) | Directly tests the one subtle wiring point (residual-vs-direct) that determines what the comparison in §3.4 actually measures. |
| `ablations/common.py` (`to_long_table`, `write_ablation_report`) | Round-trip a handful of synthetic `AblationResultRow`s through both functions; assert the JSON and Markdown outputs are derived from the same rows (e.g. count of rows in JSON matches count of table rows rendered in the Markdown) | Same "don't let the human-readable and machine-readable outputs drift" concern doc 09 §9 flags for the Phase 1 datasheet, applied here. |

**Explicitly not covered by this test plan** (same discipline as doc 09 §12 / doc 12 §12): the actual PAMPA reproduction's fidelity to the authors' real results (only checkable by comparing reproduced-checkpoint PAMPA-holdout metrics against the authors' own published `_study_{1,2,3}.csv` numbers, a one-time manual comparison, not a unit test), the real magnitude of the warm-start effect on real HC50/MIC data, the real dual-labeled subset size, and whether the learned TI head actually beats the derived baseline — these are the actual scientific questions Phase 5 exists to answer, not things a test suite can pre-verify.

---

## 7. Runbook (execution order for this phase)

**Workstream A (warm-start ablation):**
1. Confirm `configs/locked_backbone.json` (doc 12 §2.2) and a completed Phase 4 P2 model (arm (a) is literally the base checkpoint used unmodified — no separate "build arm (a)" step needed beyond what Phase 4 already produced).
2. Download `PeptideCLM-2/data/PAMPA_clusters.csv` once; cache under `data/raw/pampa/`.
3. Implement and unit-test `pampa_data.py`, `ckpt_convert.py` (§6); run `verify_tokenizer_parity` against the actual locked backbone's tokenizer vs. `aaronfeller/PeptideMTR` (doc 03 §1 point 4's flagged discrepancy) **before** any training — if they diverge, resolve which tokenizer to standardize on before proceeding, don't train around a known mismatch.
4. `clamp-ablate pampa-reproduce` — produces one (or, budget permitting, 2-3 replicate, doc 03 §3 last bullet) reproduced PAMPA checkpoint(s). Sanity-check the reproduction's own held-out PAMPA metric against the authors' published `_study_{1,2,3}.csv` numbers as a one-time fidelity check (not automated, per §6) before trusting it as arm (b)'s starting point.
5. Pre-register the primary comparison (§2.7: Pearson r, stage-2, HC50 and MIC separately, paired Wilcoxon) in writing before step 6 — per doc 03 §5 Risk F.
6. `clamp-ablate warmstart-run` across the full paired grid (§2.6: >=5 folds x 3 seeds x 2 arms = >=30 runs), logging `steps_at_stop` for every run.
7. `clamp-ablate warmstart-report` — aggregate, run the paired Wilcoxon test, check whether step counts diverged substantially between arms (if so, add the fixed-step-budget robustness run per §2.5).
8. Report HC50 and MIC results separately; do not average into one "(b) <= (a)" verdict (doc 03 Risk E).

**Workstream B (TI ablation):**
1. `clamp-ablate ti-measure-subset` against the real `processed/dataset.parquet` and Phase 4's split table — **stop here and report the count** if it's too small for a formal held-out comparison (§3.1); don't proceed to training on an underpowered subset without flagging that explicitly.
2. If the subset is large enough: implement and unit-test `dual_labeled.py`, `learned_head.py` (§6).
3. `clamp-ablate ti-train` for both `mode="residual"` (primary) and `mode="direct"` (secondary), across whatever fold/seed grid the subset size supports (likely fewer than Workstream A's 15/arm, given the subset is smaller than either marginal task's full set — report the actual grid size used, don't force doc 03's 15-run convention onto a dataset that can't support it).
4. `clamp-ablate ti-report` — Spearman ρ of derived vs. learned TI against `TI_actual`, paired Wilcoxon across folds/seeds.
5. Record the decision: does the default architecture gain a learned TI head, or does it stay derived-only (doc 04 §4.3's default)? This is the one place Phase 5 can change Phase 4's *default* production architecture — do so only if the learned head shows a real, statistically supported improvement, not a directionally-positive-but-noisy one at this subset's likely small n.

---

## 8. Definition of done (Phase 5)

- [ ] `PAMPA_clusters.csv` is downloaded and cached; `verify_tokenizer_parity` has been run and passed (or the discrepancy it found has been resolved) before any PAMPA fine-tuning happens.
- [ ] At least one PAMPA fine-tune checkpoint has been reproduced using the authors' exact confirmed hyperparameters (§2.2), with `transfer_learning=False` (full unfrozen fine-tune, matching their real production runs), and its own held-out PAMPA metric has been sanity-checked against the authors' published numbers.
- [ ] The warm-start ablation has been run across >=5 folds x 3 seeds x 2 arms (or a documented, justified smaller grid), with paired head-init seeds and one shared fine-tuning protocol (freeze -> top-k-unfreeze, reusing Phase 4's P2 code unmodified) across all three fine-tunes per doc 03 §6 checklist.
- [ ] Stage-1 (frozen-probe) and stage-2 (unfrozen) results are both reported, separately, for HC50 and MIC, with step-count-at-stop logged for every run.
- [ ] The paired Wilcoxon test on per-fold deltas has been run as the pre-registered primary comparison, and the result (whichever direction it points) is reported per-task, not collapsed into one verdict.
- [ ] The dual-labeled subset size has been measured against the real dataset and reported before any learned-TI-head training was undertaken.
- [ ] If the subset supported it: the learned TI head (`residual` mode primary) has been trained and compared against derived TI via Spearman ρ + paired Wilcoxon on held-out folds, and a decision about the default architecture has been recorded either way.
- [ ] `pytest` passes for the full `tests/ablations/` suite in §6.
- [ ] Both workstreams' results are written to a machine-readable + human-readable report (§4) rather than left as scattered run directories.

---

## 9. Open risks and questions carried into implementation

Restated from doc 03 §5/§6 and doc 04 §4.3/§7, as concrete things that can break this specific scaffold:

1. **Full unfrozen PAMPA fine-tuning on ~6,700 examples risks catastrophic forgetting/narrow overfitting** (doc 03 Risk B) — mitigated by comparing stage-1 frozen-probe results between arms (§2.3's reason for choosing freeze/unfreeze over LoRA), but not eliminated; if arm (b) underperforms only at stage 2, say so explicitly rather than attributing it to "permeability features don't transfer."
2. **Single-replicate PAMPA reproduction confounds "PAMPA history in general" with "this one noisy draw"** (doc 03 Risk C) — budget 2-3x replicate reproduction if compute allows; if only one replicate is built, flag it as a named limitation in the report (§4), not silently.
3. **Step-count confound between arms under shared early stopping** (doc 03 Risk D/§2.4) — log `steps_at_stop` for every run regardless; only run the fixed-step-budget robustness variant if the logged numbers actually diverge substantially, to avoid double the compute by default.
4. **Tokenizer parity between the locked backbone and `aaronfeller/PeptideMTR`** (doc 03 §1 point 4, §2.4) is unverified until `verify_tokenizer_parity` actually runs against the real locked model — this is a hard blocker on step 3 of the Workstream A runbook (§7), not a nice-to-have check.
5. **Dual-labeled subset size is completely unknown** (doc 04 §7 item 6, §3.1, §7 Workstream B step 1) — this could make Workstream B's formal ablation infeasible; the runbook explicitly budgets for a "report the count and stop" outcome as a legitimate, non-failure result.
6. **Reproducing via `clamp.training` primitives instead of the authors' actual Lightning script** (§2.2's stated tradeoff) means this is a faithful-to-hyperparameters reproduction, not a byte-identical one — if the reproduced checkpoint's own PAMPA-holdout metric (step 4 of the Workstream A runbook) diverges substantially from the authors' published numbers, that's evidence the reimplementation has a real discrepancy worth chasing down before trusting it as arm (b), not something to wave past.
7. **LoRA-across-all-blocks was explicitly not built** (§0, §2.3) — if a future need arises to compare against the authors' newer classification-harness convention directly, that's a new scoped addition, not something this design silently covers.
8. **This doc's `AblationResultRow`/report format (§4) is new, project-specific tooling**, not shared with anything upstream — if Phase 4's own `metrics.json` format (doc 12 §10) evolves, reconcile the two rather than letting them drift into two incompatible reporting conventions for what is, at the data level, the same kind of per-fold/per-task metric.
</content>
