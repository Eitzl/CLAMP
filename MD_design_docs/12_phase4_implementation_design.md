# Phase 4 Implementation Design: Core Multitask Implementation (P2 & P3)

*Executable companion to `08_implementation_roadmap.md` Phase 4 ("Core Multitask Implementation — P2 & P3"), translating `04_task5_multitask_architecture_plan.md` into an actual module layout, class/function signatures, training-script structure, checkpoint format, and test plan — the way `09_phase1_implementation_design.md` did for Phase 1. This is a design document, not a finished implementation: code below is illustrative scaffold, and every section with a real design choice presents options with a recommendation rather than assuming one. As with doc 09, decisions are pinned where the upstream research docs already resolved them (pooling, head shape, loss form, batching strategy) and left open where they genuinely are (task weighting scheme, mean-pool-vs-BOS, per-fold vs combined reporting).*

---

## 0. Scope, dependencies, and non-goals

**In scope** (mirrors doc 04 and doc 08's Phase 4 bullets):
- The shared architecture: mask-aware mean pooling, the shared `Linear+SiLU` projection ("disruption propensity" layer), and per-task 2-layer MLP heads.
- **P2** — single-task HC50 training: two-stage protocol (frozen-encoder head warm-up, then partial unfreeze at low LR).
- **P3** — multitask addition: adding the MIC head to an already-trained P2 model, the copy-vs-Xavier `head_mic` init ablation, the masked multitask loss, and the stratified batch sampler.
- Checkpoint format, training-script/CLI structure, and a test plan sized like doc 09's (pure-logic unit tests, not full training-loop integration tests in CI).

**Explicitly out of scope for this doc** (other phases per `08_implementation_roadmap.md`):
- Generating the base embeddings, clustering, and homology-aware splitting themselves (Phase 2) — this doc **consumes** their output as a data contract (§2) but does not build the clustering/splitting pipeline.
- The frozen-probe vs. short-fine-tune model-selection pilot that locks the backbone variant (Phase 3 in the roadmap's numbering — confusingly also called "Phase 3," but it's the *model selection* phase, not this doc's P3) — this doc **consumes** that decision as a data contract (§2) but does not build the pilot grid.
- The warm-start-encoder and TI-head ablations (Phase 5, doc 13).
- Any of the data-pull/convert/dedup/normalize pipeline (Phase 1, already implemented — see `src/clamp/data/`).

**Naming collision, flagged up front:** the roadmap (doc 08) numbers its five phases 1–5, and its **Phase 3** is "The Model Selection Pilot." Separately, doc 02/doc 04's research narrative calls the *training curriculum stages* **P0–P4** (P0 = data, P1 = baseline, P2 = single-task HC50, P3 = multitask addition, P4 = full evaluation/ablations). This doc's title, "Phase 4 (P2 & P3)," is doc 08's roadmap Phase 4, which *implements* curriculum stages P2 and P3. Every "P2"/"P3" reference below is the **curriculum stage** sense, not the roadmap phase sense — same overloaded letter, different axis, exactly as doc 08 itself set it up. Keep this straight when this doc, doc 13, and doc 08 are read side by side.

**New runtime dependencies this phase introduces** (none of Phase 1's `clamp.data` code needs them; illustrative, not applied to `pyproject.toml` by this doc):

```toml
dependencies = [
    # ...Phase 1 deps unchanged...
    "torch>=2.3",
    "transformers>=4.44",
    "safetensors>=0.4",
    "scipy>=1.12",         # spearmanr/pearsonr for eval metrics (§9)
]

[dependency-groups]
dev = [
    # ...Phase 1 dev deps unchanged...
]
```

No `accelerate`/`lightning`/`peft` dependency is added in this phase — the two-stage freeze/unfreeze protocol (§6) is a few dozen lines of plain `torch` (parameter `requires_grad` toggling + per-group optimizer LRs), and single-GPU/CPU training at this dataset size (hundreds to low thousands of peptides, per doc 07 §2.1/§3.4) does not need a training-loop framework. Revisit only if Phase 5's LoRA path (doc 13) or a later distributed-training need arises.

---

## 1. Package layout

**Recommendation: two new subpackages under the existing `clamp` namespace**, siblings to `clamp.data`, not nested inside it — the model/training code has a different lifecycle (GPU-bound, iterative, checkpoint-producing) than the data pipeline's batch/idempotent stages, and doc 09's `src/`-layout + `pyproject.toml` scaffold already supports adding packages cleanly.

```
CLAMP/
├── src/
│   └── clamp/
│       ├── data/                     # Phase 1, unchanged
│       ├── models/
│       │   ├── __init__.py
│       │   ├── pooling.py            # masked_mean_pool (§3)
│       │   ├── heads.py              # make_head, TaskHead (§4)
│       │   └── multitask.py          # LysisMultitaskModel (§5)
│       └── training/
│           ├── __init__.py
│           ├── config.py             # TrainRunConfig, StageConfig (§7)
│           ├── splits.py             # load_split_table, SplitTable (§2.1)
│           ├── datasets.py           # PeptideRegressionDataset, collate_fn (§8)
│           ├── sampler.py            # StratifiedTaskBatchSampler (§8.3)
│           ├── losses.py             # masked_mse, multitask_loss, UncertaintyWeightedLoss (§9)
│           ├── freezing.py           # freeze_encoder, unfreeze_top_k_blocks, param_groups (§6.3)
│           ├── loop.py               # run_stage, EarlyStopper (§7.4)
│           ├── checkpoint.py         # save_checkpoint, load_checkpoint, CheckpointMeta (§10)
│           ├── metrics.py            # spearman, pearson, rmse, mae, per_cluster_metrics, paired_wilcoxon (§9.4)
│           └── cli.py                # `clamp-train` typer app (§11)
├── tests/
│   ├── data/                         # Phase 1, unchanged
│   └── training/
│       ├── test_pooling.py
│       ├── test_heads.py
│       ├── test_multitask_model.py
│       ├── test_losses.py
│       ├── test_sampler.py
│       ├── test_freezing.py
│       ├── test_checkpoint.py
│       ├── test_metrics.py
│       └── test_smoke_train_stage.py  # tiny-config end-to-end, no real HF download (§12)
└── data/
    └── runs/                          # gitignored checkpoint/metrics artifact root (§13)
        ├── p2_hc50/{run_id}/
        └── p3_multitask/{run_id}/
```

Why `models/` and `training/` as two packages rather than one `clamp.model` grab-bag: `models/` holds pure `nn.Module` definitions with **no training-loop concerns** (no optimizer, no data loading, no checkpointing) — this makes them independently importable and unit-testable with plain tensors (§12), matching how doc 09 kept `convert/` (pure transform logic) separate from `pipeline.py` (orchestration). `training/` is the analogue of Phase 1's `pipeline.py` + `cli.py`: it owns the loop, the data plumbing, and the CLI.

---

## 2. Data contracts consumed from Phase 2 and Phase 3 (roadmap sense)

This doc does not build Phase 2 (splitting) or Phase 3 (model selection) — docs 10/11 own those, written concurrently with this one. To make this design buildable without waiting on those docs' exact internal structure, the contracts below are stated as **interfaces this phase requires**, deliberately narrow and easy to satisfy from either doc's actual output format. If docs 10/11 land on a materially different schema, reconcile by adapting `training/splits.py`'s loader (§2.1) and `training/config.py`'s backbone field (§2.2) — nothing else in this phase should need to change, which is the point of isolating the contract to two small modules.

### 2.1 Split table (from Phase 2)

**Required interface:** a table, one row per unique `peptide_uid`, loadable as a pandas DataFrame with at minimum:

| column | type | meaning |
|---|---|---|
| `peptide_uid` | str | joins against `processed/dataset.parquet` (Phase 1 output, `src/clamp/data/schema.py`) |
| `cluster_id` | int | the joint embedding-cluster (or homology-graph-community) assignment from doc 07 §3.1/§3.3 — computed **once**, jointly over the HC50 ∪ MIC union, per doc 07's non-negotiable joint-split rule |
| `fold` | int | which LOCO rotation this row's cluster plays which role in (doc 07 §3.5's k outer folds) |
| `role` | str, one of `{"train", "val", "test"}` | this row's role **within** `fold` |

Expected physical location: `data/processed/splits/split_table.parquet`, following Phase 1's `data/{raw,interim,processed,datasheet}` convention (doc 09 §13) — recommend Phase 2 land it under `processed/` alongside `dataset.parquet`, not a new top-level directory, since it's a derived table keyed on the same `peptide_uid`.

```python
# training/splits.py
import pandas as pd
from pathlib import Path

class SplitTable:
    """Thin wrapper around the Phase-2 split parquet. Isolates this phase
    from Phase 2's exact internal column names beyond the four above."""

    def __init__(self, df: pd.DataFrame):
        required = {"peptide_uid", "cluster_id", "fold", "role"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"split table missing required columns: {missing}")
        self.df = df

    @classmethod
    def load(cls, path: Path) -> "SplitTable":
        return cls(pd.read_parquet(path))

    def role_uids(self, fold: int, role: str) -> set[str]:
        m = (self.df["fold"] == fold) & (self.df["role"] == role)
        return set(self.df.loc[m, "peptide_uid"])

    def n_folds(self) -> int:
        return int(self.df["fold"].nunique())
```

**Open coordination item:** if Phase 2 instead delivers *both* an embedding-cluster split and a homology-graph split (doc 07 §3.2 runs both, recommends the homology-graph one as final), confirm with doc 10 which one is the canonical `split_table.parquet` this phase should train against — do not silently pick one inside this phase's code. Record whichever was chosen in the run's `training_meta.json` (§10) so every checkpoint is traceable to the split version it was trained on.

### 2.2 Locked backbone choice (from Phase 3 / roadmap, model-selection pilot)

**Required interface:** a single HF repo id string plus the depth metadata needed for the unfreeze-fraction calculation (§6.2). Doc 05 already gives concrete, verified values for every candidate (`aaronfeller/peptideclm-2-{hybrid,mlm,mtr}-{small,base,large}`; `embed_dim`/`num_blocks`/`num_heads` per size tag) — this phase does not re-derive them, it consumes whichever one the pilot locks.

Recommend Phase 3's decision be handed off as a small committed config file (not a hardcoded string in this phase's code, since re-running the pilot or revisiting the choice shouldn't require an `src/` edit):

```json
// configs/locked_backbone.json  (produced by Phase 3, consumed here)
{
  "hf_repo_id": "aaronfeller/peptideclm-2-hybrid-small",
  "num_blocks": 14,
  "embed_dim": 512,
  "max_seq_len": 2048,
  "selection_rationale": "doc 05/doc 11 pilot result, see MD_design_docs/11_..."
}
```

`training/config.py`'s `TrainRunConfig.backbone` field (§7.1) reads this file by default, with a CLI override for re-running against a different variant without editing the locked file. Until doc 11 exists, treat `aaronfeller/peptideclm-2-hybrid-small` (doc 05's stated default, pending its own pilot) as the placeholder value for every code example and test fixture in this doc — this is exactly analogous to how doc 09 treated `helm.py` as a stub pending an unresolved upstream choice (doc 09 §6, §16 item 3), not a silent hardcoded assumption.

---

## 3. Module: `models/pooling.py`

Direct implementation of doc 04 §2's resolved decision — mask-aware mean pooling, not the naive `mean(dim=1)` or BOS-token pooling doc 02 sketched:

```python
import torch

def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """(B, T, D), (B, T) -> (B, D). attention_mask: 1=real token, 0=pad.
    doc 04 §2 — matches the PeptideCLM-2 authors' own finetune_ensemble.py
    RegressionModel.forward, verified against their released code, not guessed."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts
```

`→ OPEN` (doc 04 §2, §7 item 1): a cheap one-off mean-pool-vs-BOS-token ablation on real HC50 data is worth running once P2 is up — implement `bos_pool(last_hidden_state) -> last_hidden_state[:, 0, :]` alongside this function so the ablation is a one-line swap in `LysisMultitaskModel.forward` (§5), not a rewrite, but do **not** wire a config flag for it in the default P2/P3 path until that ablation has actually run — an unused config knob for an untested option is the same premature-work smell doc 09 flagged for the HELM stub.

---

## 4. Module: `models/heads.py`

Direct implementation of doc 04 §4.1's resolved decision — a 2-layer MLP per task, matching the PeptideCLM-2 authors' own regression fine-tuning head and MTR pretraining head shapes, not a bare linear probe:

```python
import torch.nn as nn

def make_head(d: int, dropout: float = 0.2) -> nn.Sequential:
    """doc 04 §4.1. One task head: Linear(d,d) -> SiLU -> Dropout -> Linear(d,1)."""
    return nn.Sequential(
        nn.Linear(d, d),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(d, 1),
    )
```

No separate `TaskHead` class is needed beyond this factory — `nn.ModuleDict({name: make_head(d) for name in tasks})` (§5) is sufficient, and keeping heads as plain `nn.Sequential` (rather than a custom subclass) is what makes the copy-init ablation (§7.3) a one-line `load_state_dict` call rather than custom clone logic.

---

## 5. Module: `models/multitask.py`

The refined end-to-end architecture from doc 04 §6, with the P2→P3 head-addition mechanics from doc 04 §5.1 built in from day one — **P2 is not a separately-architected single-head model that gets retrofitted later; it is this same class with `tasks=("hc50",)`.**

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from clamp.models.pooling import masked_mean_pool
from clamp.models.heads import make_head


class LysisMultitaskModel(nn.Module):
    """Shared PeptideCLM-2 encoder -> masked mean pool -> shared
    'disruption propensity' projection -> per-task MLP heads.
    TI is derived post-hoc (doc 04 §4.3), not a head, in this phase's
    default architecture — see doc 13 for the learned-TI-head ablation.
    """

    def __init__(
        self,
        backbone: str,
        tasks: tuple[str, ...] = ("hc50",),
        dropout: float = 0.2,
        max_length: int = 2048,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(backbone, trust_remote_code=True, use_safetensors=True)
        self.max_length = max_length
        d = self.encoder.config.hidden_size
        self.shared_projection = nn.Sequential(nn.Linear(d, d), nn.SiLU())
        self.heads = nn.ModuleDict({t: make_head(d, dropout) for t in tasks})

    def add_task(self, name: str, warm_start_from: str | None = None, dropout: float = 0.2) -> None:
        """P3: add a new task head to an already-constructed (and possibly
        already-trained) model. warm_start_from: existing task name whose
        weights to copy (ablation arm 'a', doc 04 §5.2); None for fresh
        Xavier/Kaiming init (arm 'b', nn.Linear's default init)."""
        if name in self.heads:
            raise ValueError(f"task {name!r} already present")
        d = self.encoder.config.hidden_size
        new_head = make_head(d, dropout)
        if warm_start_from is not None:
            new_head.load_state_dict(self.heads[warm_start_from].state_dict())
        self.heads[name] = new_head

    def encode(self, smiles_batch: list[str]) -> torch.Tensor:
        """Tokenize + encode + pool + project. Split out from forward()
        so §7's frozen-probe evaluation and doc 13's ablations can reuse
        the shared embedding without re-deriving per-task predictions."""
        tok = self.tokenizer(
            smiles_batch, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_length,
        )
        tok = {k: v.to(next(self.parameters()).device) for k, v in tok.items()}
        out = self.encoder(**tok)
        pooled = masked_mean_pool(out.last_hidden_state, tok["attention_mask"])
        return self.shared_projection(pooled)

    def forward(self, smiles_batch: list[str]) -> dict[str, torch.Tensor]:
        shared = self.encode(smiles_batch)
        return {t: head(shared).squeeze(-1) for t, head in self.heads.items()}
```

Note the `encode`/`forward` split versus doc 04 §6's single-function sketch: this is a small, deliberate addition beyond the research doc, because (a) §7.2's frozen-probe diagnostic needs the pooled/projected embedding *without* running it through any head, and (b) doc 13's warm-start ablation and TI-head ablation both need direct access to `shared` — pulling it out now avoids two near-duplicate encoder forward passes later.

**Tokenization caveat carried from doc 04 §6:** verify the exact `AutoModel` output keys (`last_hidden_state` presence, `trust_remote_code=True` requirement) against the actual installed `peptideclm-2` revision before treating this as drop-in — this is custom-code-on-the-Hub, not a stock `transformers` architecture, and the model card's exact loading incantation should be re-confirmed once Phase 3's pilot has actually loaded it.

---

## 6. Freezing / unfreezing (`training/freezing.py`)

Direct implementation of doc 04 §5.3's resolved two-stage protocol, generalized so P2 and P3 call the same helpers.

### 6.1 Stage semantics

- **Stage "frozen"**: `encoder` and `shared_projection` both frozen (`requires_grad_(False)`); only `heads` (all currently-present tasks) train.
- **Stage "partial_unfreeze"**: unfreeze the **top-k** transformer blocks (nearest the output, per doc 03 §4's clarification of "top") + final layer norm + `shared_projection`, at a lower LR than the heads; `heads` continue training at the higher LR.

```python
# training/freezing.py
import torch.nn as nn

def freeze_all(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)

def unfreeze_top_k_blocks(encoder: nn.Module, k: int, block_attr: str = "encoder.layer") -> None:
    """Unfreeze the last k transformer blocks (nearest the pooled output)
    plus the final layer norm, by parameter name pattern. block_attr is
    the dotted path to the block list on the loaded AutoModel — confirm
    against the actual peptideclm-2 module tree (doc 04 §6 caveat) before
    relying on the exact string; this is a placeholder pending that check."""
    ...

def unfreeze_fraction(num_blocks: int, fraction: float = 0.175) -> int:
    """doc 03 §4's 'fixed fraction of depth, ~15-20%, not a fixed block
    count' rule, so unfreeze depth stays comparable across small/base/large.
    small (14 blocks) -> 2, base (24) -> 4, large (32) -> 5-6."""
    return max(1, round(num_blocks * fraction))

def param_groups(model, lr_heads: float, lr_shared: float, lr_encoder: float) -> list[dict]:
    """Differential LR groups for stage 'partial_unfreeze', per doc 04 §5.3
    step 2: lowest for encoder blocks, higher for shared_projection,
    highest for task heads."""
    return [
        {"params": model.heads.parameters(), "lr": lr_heads},
        {"params": model.shared_projection.parameters(), "lr": lr_shared},
        {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": lr_encoder},
    ]
```

**Open item, flagged rather than guessed:** the exact dotted path to PeptideCLM-2's transformer block list (`encoder.layer[i]` is the stock HF BERT/RoBERTa-style convention, but this is a custom `trust_remote_code=True` model) needs to be confirmed once against the actual loaded module (`print(model.encoder)` or `named_modules()`) before `unfreeze_top_k_blocks` can be implemented for real — this is the same category of "verify before treating as drop-in" caveat doc 04 §6 already flags for the forward pass itself.

---

## 7. P2: single-task HC50 training curriculum

### 7.1 Config

```python
# training/config.py
from pydantic import BaseModel

class StageConfig(BaseModel):
    name: str                      # "frozen" | "partial_unfreeze"
    max_epochs: int
    lr_heads: float
    lr_shared: float = 0.0          # ignored in "frozen" stage
    lr_encoder: float = 0.0         # ignored in "frozen" stage
    unfreeze_fraction: float | None = None   # only for "partial_unfreeze"
    early_stopping_patience: int = 4         # matches PeptideCLM-2 authors' own default (doc 03 §4)
    batch_size: int = 32
    seed: int = 0

class TrainRunConfig(BaseModel):
    backbone: str                            # from configs/locked_backbone.json, §2.2
    tasks: tuple[str, ...]                   # ("hc50",) for P2; ("hc50","mic") for P3
    split_path: str = "data/processed/splits/split_table.parquet"
    dataset_path: str = "data/processed/dataset.parquet"
    fold: int                                # which LOCO outer rotation (doc 07 §3.5)
    stages: list[StageConfig]
    output_dir: str = "data/runs"
    dropout: float = 0.2
    max_length: int = 2048
```

### 7.2 Stage 1 — frozen-encoder head training

Freeze `encoder` + `shared_projection` via `freeze_all`; train `heads["hc50"]` to convergence (val-loss early stopping, patience from `StageConfig`) using plain per-task MSE (masked loss, §9, degenerates to ordinary MSE when only one task is present and every row has a label). Report this stage's held-out metric **on its own**, not just as an intermediate step — doc 04 §5.3/doc 03 §4 both flag the frozen-probe result as a clean diagnostic worth keeping (in Phase 5's warm-start ablation this specific number is the primary comparison; recording it here for free means Phase 5 doesn't need to re-derive a stage-1-only run from scratch).

### 7.3 Stage 2 — partial unfreeze at low LR

Call `unfreeze_top_k_blocks(model.encoder, k=unfreeze_fraction(num_blocks, cfg.unfreeze_fraction))`, unfreeze `shared_projection`, build `param_groups` with differential LRs, continue training with the same early-stopping criterion. Save best-checkpoint-by-val-loss; evaluate the stage-2 best checkpoint as P2's reportable result (do not cherry-pick between stage-1 and stage-2 checkpoints post hoc — same discipline doc 03 §4 calls out for the warm-start ablation, applies equally here).

### 7.4 Training loop shape

```python
# training/loop.py
from dataclasses import dataclass

@dataclass
class EarlyStopper:
    patience: int
    best: float = float("inf")
    bad_epochs: int = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best:
            self.best, self.bad_epochs = val_loss, 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def run_stage(model, stage_cfg, train_loader, val_loader, task_masks_fn, loss_fn) -> "StageResult":
    """One frozen/partial_unfreeze stage: standard epoch loop, masked
    multitask loss (§9) each step, early stopping on val loss, checkpoint
    saved on every val-loss improvement. Returns best val metrics + the
    path of the best checkpoint for this stage."""
    ...
```

Kept deliberately plain (no Trainer-framework abstraction) — at this data scale (single fold ≤ a couple thousand examples, per doc 07 §3.5's compute estimate of "minutes to low tens of minutes per run") a bespoke ~100-line loop is easier to reason about and debug than adopting a training-framework's callback system for one project's needs.

---

## 8. Data loading, collation, and the stratified batch sampler

### 8.1 Dataset

```python
# training/datasets.py
import torch
from torch.utils.data import Dataset

class PeptideRegressionDataset(Dataset):
    """One row per peptide_uid x task-presence combination is NOT how this
    is shaped -- unlike Phase 1's PeptideRecord (one row per assay-record,
    doc 09 §3), P2/P3 training needs one row per *peptide_uid*, with
    per-task columns and per-task masks, since a batch element feeds one
    shared encoder forward pass regardless of how many tasks it has labels
    for. Building this table (replicate-row aggregation to one row per
    peptide_uid, e.g. mean over replicate hc50_log_uM/mic_log_uM values,
    or leaving replicates as separate rows and accepting a peptide can
    appear more than once per epoch) is a genuine open design choice --
    see doc09-style flag in §14 item 1."""

    def __init__(self, df, tasks: tuple[str, ...]):
        self.smiles = df["smiles"].tolist()
        self.targets = {t: torch.tensor(df[f"{t}_log_uM"].fillna(0.0).to_numpy(), dtype=torch.float32) for t in tasks}
        self.masks = {t: torch.tensor(df[f"{t}_log_uM"].notna().to_numpy(), dtype=torch.float32) for t in tasks}
        self.tasks = tasks

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict:
        return {
            "smiles": self.smiles[idx],
            "targets": {t: self.targets[t][idx] for t in self.tasks},
            "masks": {t: self.masks[t][idx] for t in self.tasks},
        }


def collate_fn(batch: list[dict]) -> dict:
    """Tokenization happens inside LysisMultitaskModel.encode (per-batch,
    since padding must be computed within a batch), so collate here just
    stacks the raw smiles strings and target/mask tensors."""
    return {
        "smiles": [b["smiles"] for b in batch],
        "targets": {t: torch.stack([b["targets"][t] for b in batch]) for t in batch[0]["targets"]},
        "masks": {t: torch.stack([b["masks"][t] for b in batch]) for t in batch[0]["masks"]},
    }
```

### 8.2 Masked losses

Direct implementation of doc 04 §3.2/§9's spec:

```python
# training/losses.py
import torch

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    n = mask.sum()
    if n == 0:
        return None
    return (((pred - target) ** 2) * mask).sum() / n

def multitask_loss(preds: dict, targets: dict, masks: dict, weights: dict[str, float] | None = None):
    weights = weights or {t: 1.0 for t in preds}
    total, n_active = 0.0, 0
    for t in preds:
        l = masked_mse(preds[t], targets[t], masks[t])
        if l is not None:
            total = total + weights[t] * l
            n_active += 1
    return total if n_active else None


class UncertaintyWeightedLoss(torch.nn.Module):
    """doc 04 §3.5's cheap ablation, deferred to a P3 run flag, not the
    P3 default (naive equal-weighted sum is default). Learns log-sigma^2
    per task; clamps sigma^2 to a safety range given few-steps-per-epoch
    convergence risk flagged in doc 04 §3.5."""

    def __init__(self, tasks: tuple[str, ...], log_sigma2_clamp: tuple[float, float] = (-4.0, 4.0)):
        super().__init__()
        self.log_sigma2 = torch.nn.ParameterDict({t: torch.nn.Parameter(torch.zeros(())) for t in tasks})
        self.clamp = log_sigma2_clamp

    def forward(self, preds, targets, masks):
        total, n_active = 0.0, 0
        for t in preds:
            l = masked_mse(preds[t], targets[t], masks[t])
            if l is None:
                continue
            log_s2 = self.log_sigma2[t].clamp(*self.clamp)
            total = total + l / (2 * log_s2.exp()) + 0.5 * log_s2
            n_active += 1
        return total if n_active else None
```

**Default: `multitask_loss` with `weights={"hc50": 1.0, "mic": 1.0}`.** `UncertaintyWeightedLoss` is implemented as a drop-in alternative behind a config flag (`TrainRunConfig.loss_kind: Literal["naive_sum", "uncertainty"] = "naive_sum"`), matching doc 04 §3.5's "run as a P3 ablation, don't reach for GradNorm" recommendation.

### 8.3 Stratified batch sampler

Direct implementation of doc 04 §3.3's resolved option (b):

```python
# training/sampler.py
import random
from torch.utils.data import Sampler

class StratifiedTaskBatchSampler(Sampler):
    """Guarantees every batch contains examples from every task pool,
    rather than relying on random shuffling of a concatenated pool (doc 04
    §3.3). task_indices: {"hc50": [...], "mic": [...]} -- an index can
    appear in more than one task's pool (both-labeled peptides), which is
    deliberate, matching doc 04 §3.3's 'both-labeled subset should be
    drawable into either pool' requirement.
    """

    def __init__(self, task_indices: dict[str, list[int]], batch_size: int, seed: int = 0):
        self.task_indices = task_indices
        self.batch_size = batch_size
        self.per_task = max(1, batch_size // len(task_indices))
        self.rng = random.Random(seed)

    def __iter__(self):
        pools = {t: self._shuffled(idxs) for t, idxs in self.task_indices.items()}
        n_batches = min(len(idxs) // self.per_task for idxs in self.task_indices.values())
        for b in range(n_batches):  # drop_last=True equivalent, doc 04 §3.4
            batch = []
            for t, idxs in pools.items():
                batch.extend(idxs[b * self.per_task:(b + 1) * self.per_task])
            self.rng.shuffle(batch)
            yield batch

    def _shuffled(self, idxs: list[int]) -> list[int]:
        idxs = list(idxs)
        self.rng.shuffle(idxs)
        return idxs

    def __len__(self) -> int:
        return min(len(idxs) // self.per_task for idxs in self.task_indices.values())
```

For P2 (`tasks=("hc50",)`), `StratifiedTaskBatchSampler` degenerates to a single-pool shuffled sampler — same class handles both curriculum stages, no separate P2-only sampler needed. `drop_last` behavior is baked in (`n_batches` computed via floor division) per doc 04 §3.4's guard against degenerate end-of-epoch batches.

---

## 9. Evaluation metrics

```python
# training/metrics.py
import numpy as np
from scipy.stats import spearmanr, pearsonr, wilcoxon

def spearman(pred: np.ndarray, target: np.ndarray) -> float:
    return float(spearmanr(pred, target).statistic)

def pearson(pred: np.ndarray, target: np.ndarray) -> float:
    return float(pearsonr(pred, target)[0])

def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))

def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))

def per_cluster_metrics(pred, target, cluster_ids) -> "pd.DataFrame":
    """doc 07 §3.5's per-fold reporting requirement -- report each LOCO
    fold's metrics separately, not just a single pooled number, since
    fold-to-fold variance is itself a reportable quantity at this n."""
    ...

def paired_wilcoxon(deltas: np.ndarray):
    """Reused by doc 13's warm-start and TI ablations (both need a paired,
    nonparametric significance test across ~5-15 folds/seeds, doc 03 §2.5).
    Defined here, not duplicated in the ablations package."""
    return wilcoxon(deltas)
```

Report **Spearman ρ, Pearson r, RMSE, MAE** per task, per fold — matching QMAP's own reporting convention (doc 04 §1) so numbers are comparable to a published external baseline, and matching doc 07 §3.5's per-fold (not just pooled) reporting requirement. `paired_wilcoxon` is included in this phase's `metrics.py` specifically so Phase 5 (doc 13) can import it rather than re-deriving the same test in a separate module.

---

## 10. Checkpoint format

**Recommendation: a directory per run, not a single opaque `.pt` file** — this phase's checkpoints need to carry enough metadata (which tasks are present, which fold/split version, which init strategy) that a bare `state_dict` blob would force that information into an out-of-band naming convention instead of a structured, inspectable format.

```
data/runs/{p2_hc50,p3_multitask}/{run_id}/
├── model.safetensors        # full state_dict: encoder + shared_projection + heads
├── model_config.json        # backbone id, tasks present, dropout, max_length
├── training_meta.json       # stage-by-stage history, see below
└── metrics.json             # per-fold, per-task held-out metrics (§9)
```

```python
# training/checkpoint.py
from pydantic import BaseModel
from safetensors.torch import save_file, load_file

class CheckpointMeta(BaseModel):
    run_id: str
    backbone: str
    tasks: tuple[str, ...]
    fold: int
    split_table_hash: str          # content hash of the split_table.parquet used (§2.1) --
                                    # ties every checkpoint to an exact split version, same
                                    # spirit as doc 09 §13's dataset-versioning discipline
    stage_history: list[dict]      # [{"stage": "frozen", "epochs_run": N, "best_val_loss": x}, ...]
    mic_head_init: str | None = None   # "copy_from_hc50" | "xavier" | None (P2 has no mic head yet)
    seed: int
    git_commit: str | None = None

def save_checkpoint(model, meta: CheckpointMeta, out_dir) -> None: ...
def load_checkpoint(out_dir, backbone_override: str | None = None) -> tuple["LysisMultitaskModel", CheckpointMeta]: ...
```

**P2 -> P3 loading, concretely** (doc 04 §5.1's "no checkpoint surgery" requirement): `load_checkpoint` on a P2 run returns a `LysisMultitaskModel` with `tasks=("hc50",)`; P3's training script then calls `model.add_task("mic", warm_start_from="hc50" if arm == "copy" else None)` and continues training — the `nn.ModuleDict` mechanism (§5) means this is a state-dict-compatible load with one new key added, not a rename/remap operation.

`split_table_hash` matters specifically because doc 07's split artifact may be regenerated (different k, different random seed) over the project's lifetime — a checkpoint whose provenance doesn't pin the exact split version it was evaluated against cannot be trusted for a later comparison, the same failure mode doc 09 §13 flags for the dataset itself.

---

## 11. CLI

**Recommendation: `typer`, matching Phase 1's `clamp-data` convention** (doc 09 §11) — a new entry point `clamp-train`, not folded into `clamp-data`, since training is a distinct lifecycle (GPU jobs, longer-running, produces `data/runs/` not `data/processed/`).

```python
# training/cli.py
import typer
from clamp.training.config import TrainRunConfig

app = typer.Typer()

@app.command()
def p2(config_path: str, fold: int = 0) -> None:
    """Run the P2 (HC50 single-task) two-stage curriculum for one LOCO fold."""
    ...

@app.command()
def p3(
    config_path: str,
    fold: int = 0,
    p2_checkpoint: str = typer.Option(..., help="path to a completed P2 run directory"),
    mic_init: str = typer.Option("copy", help="'copy' or 'xavier' -- doc 04 §5.2 ablation arm"),
) -> None:
    """Run the P3 (multitask) curriculum for one LOCO fold, starting from a P2 checkpoint."""
    ...

@app.command()
def evaluate(checkpoint_dir: str, split_path: str = "data/processed/splits/split_table.parquet") -> None:
    """Re-run held-out evaluation for an existing checkpoint (e.g. after a metrics.py change)."""
    ...

@app.command(name="run-all-folds")
def run_all_folds(config_path: str, stage: str = "p2") -> None:
    """Loop p2/p3 over every LOCO fold in the locked split table (doc 07 §3.5)."""
    ...
```

```toml
[project.scripts]
clamp-data = "clamp.cli:app"
clamp-train = "clamp.training.cli:app"
```

---

## 12. Testing plan

Per doc 09's model: focus on **pure, high-risk logic**, not full GPU training runs in CI. One exception (last row) covers the actual training loop end-to-end, but against a tiny synthetic encoder, not a real HF download.

| Target | Approach | Why this one |
|---|---|---|
| `models/pooling.py` (`masked_mean_pool`) | Hand-constructed 2-3 example tensors with known padding, assert output equals manual mean over non-pad positions | This is precisely the "averages in padding-token embeddings" bug doc 04 §2 flags as a real risk once HC50-only and MIC-only sequences of different lengths share batches — worth a direct regression test, not just trusting the formula. |
| `models/heads.py` (`make_head`) | Assert output shape `(B, 1)` -> squeeze `(B,)`; assert dropout is present in `model.training` mode (activations differ across two forward calls) and absent in `eval()` mode | Cheap, catches an accidental `dropout=0` or missing-`squeeze` regression early. |
| `models/multitask.py` (`LysisMultitaskModel.add_task`) | Construct with `tasks=("hc50",)`, call `add_task("mic", warm_start_from="hc50")`, assert `heads["mic"].state_dict()` tensors are `torch.equal` to `heads["hc50"]`'s **at the moment of the copy** (before any further training step); separately assert `add_task("mic", warm_start_from=None)` produces different weights than `heads["hc50"]` | This is the exact mechanism the doc 04 §5.2 ablation depends on -- a bug here would silently invalidate the copy-vs-Xavier comparison without failing loudly elsewhere. |
| `training/losses.py` (`masked_mse`, `multitask_loss`) | Parametrized cases: all-masked-out task returns `None` and is excluded from the sum; single-task batch reduces to plain MSE; a batch with `n=0` for one task and `n>0` for the other doesn't raise or NaN | This is the degenerate-batch guard doc 04 §3.4 calls out explicitly -- test the exact scenario it warns about. |
| `training/losses.py` (`UncertaintyWeightedLoss`) | Assert `log_sigma2` values are learnable `nn.Parameter`s that receive gradients after one `.backward()`; assert clamping actually bounds an artificially large gradient step from pushing `log_sigma2` outside `log_sigma2_clamp` | Directly tests the "safety rail" doc 04 §3.5 recommends against degenerate variance collapse. |
| `training/sampler.py` (`StratifiedTaskBatchSampler`) | Construct with synthetic `task_indices` of deliberately imbalanced size (e.g. 100 MIC-only, 10 HC50-only, 5 both-labeled); iterate all batches; assert **every** batch contains at least one index from every task's pool | This is the exact failure mode (batches missing a task entirely) doc 04 §3.3 identifies as the reason option (b) was chosen over naive shuffling -- test the property, not just that the class runs. |
| `training/freezing.py` (`freeze_all`, `unfreeze_fraction`, `param_groups`) | Assert `freeze_all` sets every parameter's `requires_grad` to `False`; parametrized `unfreeze_fraction(num_blocks, 0.175)` cases for 14/24/32 -> 2/4/6 matching doc 03 §4's worked examples; assert `param_groups` produces exactly 3 groups with the expected `lr` values and that the encoder group only contains currently-`requires_grad=True` parameters | `unfreeze_fraction`'s exact rounding behavior is easy to get subtly wrong (off-by-one at the boundary) and doc 03 §4 gives concrete expected outputs to test against directly. |
| `training/checkpoint.py` (`save_checkpoint`/`load_checkpoint`) | Round-trip a small `LysisMultitaskModel` (real backbone swapped for a tiny synthetic encoder, see below) through save/load, assert every parameter tensor is `torch.equal` after reload, and `CheckpointMeta` fields survive JSON round-trip | Silent checkpoint corruption (e.g. a `state_dict` key mismatch after `add_task`) is exactly the kind of bug that would only surface much later, expensively, mid-P3-training. |
| `training/metrics.py` (`spearman`, `pearson`, `rmse`, `mae`, `paired_wilcoxon`) | Known-input/known-output cases (perfectly correlated, anti-correlated, and a hand-computed RMSE/MAE example); `paired_wilcoxon` against a `scipy` reference computation with a fixed synthetic delta array | Small pure functions wrapping well-tested `scipy` internals -- the risk is a wiring bug (wrong axis, wrong `.statistic` attribute), not the underlying math. |
| `training/loop.py` + `training/datasets.py` (end-to-end smoke test) | Build a **tiny fake encoder** (a 2-layer `nn.TransformerEncoder` over a 20-token toy vocabulary, hidden size 16, wrapped to expose `.config.hidden_size` and a `last_hidden_state`-shaped output, standing in for `AutoModel`) plus ~20 synthetic `(smiles-like string, label)` pairs; run one full `run_stage` call for 2 epochs; assert loss decreases and no `NaN`/`inf` appears | This is the one integration-style test in the suite, and it exists specifically so a wiring bug across `datasets.py` -> `sampler.py` -> `losses.py` -> `loop.py` -> `checkpoint.py` is caught in CI seconds, without ever downloading the real ~32M-parameter PeptideCLM-2 weights or requiring a GPU. |

**Explicitly not covered by this test plan:** real convergence behavior of P2/P3 against the actual locked backbone and real HC50/MIC data, the actual value of the copy-vs-Xavier ablation, and whether the stratified sampler's `per_task` quota choice is well-tuned for the real (likely far more imbalanced than any synthetic test fixture) HC50:MIC ratio. Those need one real run against real data per protocol change, exactly as doc 09 §12 argues for the Phase 1 pipeline's live pulls.

---

## 13. Storage layout

```
data/
└── runs/                              # gitignored, like raw/interim/processed (doc 09 §13)
    ├── p2_hc50/
    │   └── {run_id}/                  # run_id e.g. "fold0_seed0_20260723"
    │       ├── model.safetensors
    │       ├── model_config.json
    │       ├── training_meta.json
    │       └── metrics.json
    └── p3_multitask/
        └── {run_id}/                  # same shape, plus mic_head_init in training_meta.json
```

`run_id` should encode fold + seed + date at minimum, so re-running the same fold/seed combination after a code change produces a distinguishable new directory rather than a silent overwrite — same reasoning as doc 09 §13's `dataset_v{n}.parquet` convention, applied to training runs instead of datasets.

---

## 14. Runbook (execution order for this phase)

1. Confirm `configs/locked_backbone.json` exists (Phase 3/roadmap output, §2.2) and `data/processed/splits/split_table.parquet` exists (Phase 2 output, §2.1) — both are hard prerequisites, not soft ones; do not start P2 training against a placeholder split.
2. **Resolve the open item flagged in §8.1**: decide and implement the peptide_uid-level aggregation rule for `PeptideRegressionDataset` (mean over replicate `hc50_log_uM`/`mic_log_uM` values sharing a `peptide_uid`, vs. keeping replicate rows and accepting within-epoch repeats) — this is a real modeling decision Phase 1's schema deliberately left as separate rows (doc 09 §3's "one row per assay-record" convention), and Phase 4 is where it must finally be resolved, one way, before any training happens.
3. Implement `models/pooling.py`, `models/heads.py`, `models/multitask.py`; run their unit tests (§12) before touching any real backbone.
4. Implement `training/freezing.py`, `training/losses.py`, `training/sampler.py`; run their unit tests.
5. Implement `training/datasets.py`, `training/loop.py`, `training/checkpoint.py`, `training/metrics.py`; run the smoke test (§12 last row) against the tiny fake encoder.
6. `clamp-train p2 --config configs/p2_hc50.json --fold 0` against the real locked backbone and real split — the first real GPU run. Inspect `metrics.json`'s stage-1 (frozen-probe) numbers before letting stage 2 run, per §7.2's "report this on its own" recommendation.
7. Repeat step 6 across every LOCO fold (`run-all-folds`), report per-fold Spearman ρ/Pearson r/RMSE/MAE (§9) against the doc 05 physicochemical-descriptor baseline (roadmap Phase 3) and against QMAP's own published HC50 numbers (doc 04 §1).
8. `clamp-train p3 --config configs/p3_multitask.json --fold 0 --p2-checkpoint data/runs/p2_hc50/{best_run} --mic-init copy`, then repeat with `--mic-init xavier` — same fold, same seed, per doc 04 §5.2's ablation design.
9. Repeat step 8 across every LOCO fold for both `mic_init` arms; compare held-out MIC performance (and TI-ranking behavior, derived per §5/doc 04 §4.3) between arms using `paired_wilcoxon` (§9) on the per-fold deltas.
10. Decide the default `mic_init` policy for all future P3 runs based on step 9's result; record the decision (and the numbers behind it) in this doc's open-questions log or a short follow-up note — do not leave the ablation's outcome undocumented.

---

## 15. Definition of done (Phase 4)

- [ ] `LysisMultitaskModel` implements mask-aware mean pooling, the shared `Linear+SiLU` projection, and per-task 2-layer MLP heads exactly per doc 04 §2/§4.
- [ ] P2 two-stage curriculum (frozen head warm-up -> partial top-k unfreeze) runs to convergence on every locked LOCO fold for HC50, with stage-1 (frozen-probe) and stage-2 results both reported.
- [ ] P2's held-out Spearman ρ/Pearson r clears the roadmap Phase 3 physicochemical-descriptor baseline on at least a majority of folds; if it doesn't on some fold, that's a reported finding, not silently dropped.
- [ ] The masked multitask loss (`masked_mse`, `multitask_loss`) and the stratified batch sampler are implemented and pass the degenerate-batch and task-starvation tests in §12.
- [ ] P3 adds `head_mic` via `LysisMultitaskModel.add_task` with no checkpoint-key renaming/surgery, re-freezes `encoder` + `shared_projection` at P3 start, and runs the copy-vs-Xavier ablation across every LOCO fold with a paired seed.
- [ ] The copy-vs-Xavier ablation result is computed via `paired_wilcoxon` on per-fold deltas and a default policy is chosen and recorded (§14 step 10).
- [ ] Checkpoints follow the §10 directory format, include `CheckpointMeta` with `split_table_hash`, and a P2 checkpoint loads cleanly into a P3 training run.
- [ ] `pytest` passes for the full `tests/training/` suite in §12, including the tiny-fake-encoder smoke test, without downloading the real PeptideCLM-2 weights.
- [ ] Per-task, per-fold metrics (Spearman ρ, Pearson r, RMSE, MAE) are reported both combined and per-task, per doc 04 §5.3.3's recommendation, and compared against QMAP's published baselines on a matched footing.

---

## 16. Open risks and questions carried into implementation

1. **Peptide-uid-level aggregation for training rows is unresolved** (§8.1, §14 step 2) — Phase 1's schema deliberately keeps replicate assay-records as separate rows; Phase 4 must pick one aggregation rule (mean over replicates vs. accept within-epoch repeats) and this doc does not resolve it, only flags it as the first real blocking decision.
2. **Exact module path for `unfreeze_top_k_blocks`** (§6.1) is a placeholder pending inspection of the actual loaded `peptideclm-2` module tree — do not assume the stock HF BERT-style `encoder.layer[i]` naming without checking, since this is custom `trust_remote_code=True` code.
3. **Mean-pool vs. BOS-token pooling** (§3, doc 04 §7 item 1) — the authors' own preference is a strong prior, not a guarantee for this specific target; budget a cheap ablation once P2 is running, but do not block P2 on it.
4. **Naive-sum vs. uncertainty-weighted task loss** (§8.2, doc 04 §3.5, §7 item 2) — start naive per this doc's default; run `UncertaintyWeightedLoss` as a P3 ablation once the core copy-vs-Xavier ablation (§14 steps 8-10) is done, watching for the degenerate log-variance-collapse failure mode doc 04 flags given few optimizer steps per epoch at this data scale.
5. **Copy-init vs. fresh-init for `head_mic`** (§7.3, §14 steps 8-10) is the headline ablation this phase exists to run — settle by the paired Wilcoxon test, not by a priori argument.
6. **Combined vs. per-task cluster-holdout reporting** (§9, doc 04 §5.3.3) — this doc recommends reporting both; confirm this doesn't duplicate split-construction effort that should have been scoped into doc 10/11's deliverable instead.
7. **Split-table schema drift from doc 10/11's actual output** (§2.1) — the four-column contract here is a minimal interface, not a guess at doc 10/11's exact internal representation; reconcile `training/splits.py`'s loader against the real Phase 2 deliverable once it lands, and treat any mismatch as a `training/splits.py` fix, not a reason to change this phase's model/training code.
8. **Locked backbone choice from doc 11 may not be `hybrid-small`** (§2.2) — every code example/default in this doc uses doc 05's stated pending-pilot default as a placeholder; swap `configs/locked_backbone.json`'s contents once doc 11's pilot actually concludes, and re-verify `unfreeze_fraction`'s `num_blocks` input against whichever size tag is actually locked.
</content>
