# Task 5.1 Deep-Dive: HC50/MIC Multitask Architecture — Implementation Plan

*Deepens `02_peptideclm_transfer_learning_plan.md` §5.1 (multitask reframing) and §5.6 (architecture sketch) into an executable spec: pooling, loss, head design, and the P2→P3 curriculum. Everything here assumes the P0–P1 data/baseline work in doc 02 is already done and the encoder choice (PeptideCLM-2, hybrid-small) is fixed. Where doc 02 floated an option without resolving it, this doc picks one and says why — flagged `→ RESOLVED` — versus genuinely open items, flagged `→ OPEN`.*

---

## 1. QMAP's actual joint-regression setup, and what it does/doesn't validate

**Source:** Lavertu, Corbeil & Germain, *QMAP: a benchmark for standardized evaluation of antimicrobial peptide MIC and hemolytic activity regression*, bioRxiv 2026.02.03.703041 (also published *Scientific Reports*, 2026, `10.1038/s41598-026-56004-8`); code at `github.com/anthol42/QMAP`.

**Correction to how doc 02 §5.1 characterizes it:** QMAP is **a benchmark/evaluation harness, not a jointly-trained multitask model**. It does not itself propose a shared-encoder-plus-heads architecture. What it actually provides:

- **Two separate regression tasks** (MIC, HC50), each evaluated independently — the paper explicitly evaluates prior MIC models (Witten et al., BERT-AmPEP60, and others pulled into `eval_prev_works/`) and a **linear-probe baseline on frozen ESM2-650M embeddings** per task. No architecture in the paper shares parameters across MIC and HC50.
- What *is* shared is the **evaluation protocol**: identical homology-aware, 5-fold predefined splits (60% sequence-identity threshold, similarity-constrained train/test separation) applied to both tasks — "For consistency with MIC regression experiments, we adopt the same predefined test splits and independence constraints, enabling direct comparison across tasks." That's the "joint" part: a shared *yardstick*, not a shared *model*.
- **Dataset sizes are asymmetric and small**: HC50 (~800 peptides) vs. MIC-*E. coli* (~4,000 peptides), i.e. roughly a 1:5 ratio — directly relevant to the loss-weighting and batching discussion in §3 below.
- **Loss/training details for the baselines are thin.** The paper reports evaluation metrics (R², Kendall's τ, Spearman ρ, Pearson r, RMSE, MAE) but does not disclose the training loss used by the deep-learning baselines it re-evaluates; the one baseline whose training is fully specified (linear regression on ESM2-650M embeddings, no regularization, log₁₀-transformed labels) uses plain least squares.
- **TI is named as the design objective but never predicted directly**: "AMP development aims to maximize potency while minimizing toxicity, which can be expressed as minimizing the ratio MIC/HC50" — stated as motivation for evaluating both tasks, not implemented as a third target or model output anywhere in the benchmark.
- **Headline finding, worth carrying into your risk register**: MIC progress has been essentially flat for six years, degrading toward random in the *high-potency* regime (MIC < 10 µM — precisely the regime you care about for lead candidates), and **HC50 is predicted markedly worse than MIC across every metric**, with even the linear ESM2 baseline barely correlating — "protein language models capture only weak linearly accessible information relevant to hemolytic activity." This is a strong prior that HC50 is the harder head and probably needs more regularization / lower capacity than MIC, not less.

**What to actually take from QMAP:**
1. **Reuse its splits and metric suite as your eval protocol** (Spearman ρ, RMSE, MAE, plus the same high-potency-subset stratification) so your numbers are comparable to a published external baseline — this is a bigger win than anything in its (nonexistent) joint architecture.
2. **Do not treat "QMAP does joint regression" as precedent for a shared-encoder design** — that precedent doesn't exist in the literature yet; §5.1's multitask architecture is a genuine (reasonable) proposal on your part, not an implementation of something QMAP already validated. Say so explicitly if this shows up in a writeup — claiming otherwise is a citation risk.
3. TI-as-derived-ratio is exactly what QMAP's own framing suggests (§4 below agrees, independently).

---

## 2. Pooling strategy — resolved by the authors' own downstream code, not just the model card

The HF model card / repo README example (`huggingface.co/aaronfeller/peptideclm-2-hybrid-small`) only shows `outputs.last_hidden_state` and stops — no pooling shown, no CLS/BOS guidance. **But the PeptideCLM-2 repo's own regression fine-tuning code answers this directly**, and it does **not** match doc 02 §5.6's naive `mean(dim=1)` sketch — it does something one step more careful.

**Source:** `github.com/AaronFeller/PeptideCLM-2`, `training/01_regression_benchmarks_training_code/finetune_ensemble.py` (this is the exact script the authors used for their own permeability/stability/fibrillation regression benchmarks — `perm_external.csv` is literally the CycPeptMPDB-derived permeability task, i.e. the closest existing analog to what you're building).

The `RegressionModel.forward` in that file:

```python
# use the bos token's representation as the input to the regression head
# intermediate_output = self.intermediate_layer(outputs[0][:, 0, :])  # Use the first token's representation
...
if attention_mask is not None:
    attention_mask = attention_mask.unsqueeze(-1)
    masked_x = x * attention_mask
    sum_x = masked_x.sum(dim=1)
    attention_mask_sum = attention_mask.sum(dim=1).clamp(min=1e-6)
    mean_pool = sum_x / attention_mask_sum
else:
    mean_pool = x.mean(dim=1)
# use mean pooled output instead of the first token's representation
```

Two findings, both load-bearing:

1. **They tried BOS/CLS-token pooling and commented it out in favor of attention-mask-aware mean pooling.** That's the strongest signal available short of an ablation table in the paper itself — the authors' working code prefers mean pooling.
2. **Their mean pool is mask-aware, doc 02's sketch is not.** `last_hidden_state.mean(dim=1)` on a padded batch averages in the padding-token embeddings, which is a real bug at anything but batch size 1 or with an unpadded collator. This matters more here than usual because you'll have HC50-only and MIC-only sequences of different lengths sharing batches (§3), guaranteeing padding.
3. Also confirmed from `MTR_model.py`: the base transformer has **no built-in pooler/CLS output** — `forward` returns only the token-level sequence representation (`self.sequence_head` is the MLM logit head, not a pooler). So there is no "free" pooled representation to fall back on; pooling is entirely the downstream fine-tuner's responsibility, exactly as doc 02 §5.6 assumed in spirit, just not in the exact code.

**→ RESOLVED:** use attention-mask-aware mean pooling, matching the authors' own validated approach. Corrected pooling function:

```python
def masked_mean_pool(last_hidden_state, attention_mask):
    # last_hidden_state: (B, T, D); attention_mask: (B, T), 1=real token, 0=pad
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)   # (B, T, 1)
    summed = (last_hidden_state * mask).sum(dim=1)                    # (B, D)
    counts = mask.sum(dim=1).clamp(min=1e-6)                          # (B, 1)
    return summed / counts
```

`→ OPEN`: it's worth a cheap one-off ablation (mean-pool vs. BOS-token) on your own HC50 data once P2 is running, since "the authors preferred it for stability/fibril/perm benchmarks" isn't a guarantee it's best for a lysis-propensity target — but given it's the authors' own default across all three of their regression benchmarks (including the most closely related, permeability), treat mean-pool as the strong prior, not a coin flip.

---

## 3. Masked multitask loss — concrete spec

### 3.1 The masking problem, precisely

A training example is a `(smiles, y_hc50, mask_hc50, y_mic, mask_mic)` tuple. `mask_task ∈ {0,1}` indicates whether that peptide has a label for that task — **these are not mutually exclusive**: DBAASP peptides with both MIC and hemolysis reported have both masks set, which is the (presumably small) subset that actually determines TI quality (see §4). Given the QMAP dataset sizes above, expect roughly: MIC-labeled ≫ HC50-labeled ≫ both-labeled.

### 3.2 Per-batch masked loss

For a batch of size B, with predictions `pred_hc50, pred_mic ∈ R^B`:

```python
def masked_mse(pred, target, mask):
    # mask: (B,) float/bool; returns scalar, or None if no labeled examples in batch
    n = mask.sum()
    if n == 0:
        return None
    sq_err = (pred - target) ** 2 * mask
    return sq_err.sum() / n

def multitask_loss(preds, targets, masks, weights):
    total, active = 0.0, 0
    for t in preds:                       # t in {"hc50", "mic"}
        l = masked_mse(preds[t], targets[t], masks[t])
        if l is not None:
            total = total + weights[t] * l
            active += 1
    return total  # if active == 0 the batch is degenerate — see §3.4
```

Because `masked_mse` divides by the **count of labeled examples in that batch**, not batch size, each task's loss is already a proper per-example mean regardless of how sparsely that task is represented in the batch — this is the mechanism that keeps a batch with (say) 28 MIC-labeled and 4 HC50-labeled examples from letting MIC dominate the loss *magnitude*. What it does **not** fix is *gradient variance*: a mean over 4 examples is a noisier gradient estimate for `head_hc50` (and for the shared trunk, via backprop) than a mean over 28 is for `head_mic`. That's a batch-composition problem, not a loss-formula problem — see §3.3.

### 3.3 Batching: don't let random shuffling starve the minority task

Given the ~1:5 (or worse) HC50:MIC label ratio, naively concatenating both label sets and using a single `DataLoader(shuffle=True)` means a meaningful fraction of batches will contain **zero** HC50-labeled examples (that batch silently trains only `head_mic` + trunk-via-MIC-gradient-only) and most others will contain very few. Two fixes, pick one:

- **(a) Oversample the minority task** — same trick the PeptideCLM-2 authors already use for skewed regression targets in `finetune_ensemble.py` (`train_df.groupby(pd.cut(train_df['value'], bins=5)).apply(lambda x: x.sample(n=max_bin_size, replace=True))`), applied across tasks instead of within one task's value distribution: repeat HC50 rows (with replacement) until the merged pool is roughly balanced per epoch.
- **(b) Stratified/interleaved batch sampler** — maintain two index pools (HC50-labeled, MIC-labeled; overlap is fine, an example can appear in both pools) and construct each batch by drawing a fixed sub-quota from each, e.g. `batch_size=32` → 12 guaranteed-HC50 + 20 guaranteed-MIC-or-either. Cleaner than (a) because it doesn't distort the *label value* distribution within a task, only the *task presence* distribution across batches.

**→ RESOLVED: prefer (b).** It directly targets the actual failure mode (batches missing a task entirely) without touching label-value sampling, which you're already handling separately per doc 02 §5.4 (cluster splits) and don't want to compound. Concretely: `torch.utils.data.BatchSampler` fed by two `SubsetRandomSampler`s over the HC50-index-set and MIC-index-set respectively, zipped per step; the "both-labeled" subset should be allowed to be drawn into either pool so it isn't systematically underused.

### 3.4 Degenerate batches
Guard for the case where a stratification bug or edge-case dataset produces a batch with `active == 0` for one task all-around (both masks all-zero, which shouldn't happen under (b) but can happen at end-of-epoch boundary effects with drop_last=False) — just skip that task's loss term for that step (already handled by `masked_mse` returning `None`), and set `drop_last=True` on both underlying samplers so partial trailing batches don't produce a single-task-only step at every epoch boundary.

### 3.5 Task weighting — pick simple, justify from the data regime

Three options, per your prompt:

- **Naive sum** (`weights = {"hc50": 1.0, "mic": 1.0}`): given the per-task mean normalization in §3.2 and stratified batching in §3.3, the two loss terms already arrive at roughly comparable *scale* and *gradient-noise* per step. The main remaining asymmetry is real-world label noise: HC50 (broth microdilution / RBC lysis assays across heterogeneous species/protocols in DBAASP+HemoPI) is typically noisier than MIC. Equal weighting doesn't correct for that, but it also doesn't require estimating it.
- **Uncertainty weighting** (Kendall, Gal & Cipolla, *Multi-task Learning Using Uncertainty to Weigh Losses*, CVPR 2018): learn two scalars `log σ_hc50², log σ_mic²`, loss `= Σ_t [ L_t / (2σ_t²) + log σ_t ]`. Cheap (2 extra parameters, no extra forward pass), principled, and it's specifically designed for exactly "combine two noisy regression heads of different intrinsic difficulty" — which QMAP's own finding (HC50 much harder to predict than MIC) says is your situation. Known failure mode: with few optimizer steps per epoch (small-data regime → few batches), the log-variance scalars can be noisy/slow to converge and occasionally drift toward degenerate solutions (one task's weight collapsing) — worth clamping `σ_t²` to a reasonable range as a safety rail.
- **GradNorm** (Chen et al. 2018): balances *gradient norms at the shared layer* across tasks, adjusted every step. **→ RESOLVED against, for now.** It needs a per-step estimate of each task's gradient norm at the shared trunk, which is itself a noisy quantity when a given step's batch has only ~12 labeled examples for one task; you'd be meta-optimizing a weighting signal that's noisier than the signal it's trying to balance. This is a "needs thousands of steps and reasonably large batches to be worth its complexity" method, not obviously true here yet.

**Default recommendation: equal-weighted naive sum after per-task masked-mean normalization (§3.2), with stratified batching (§3.3) doing the real work of keeping both tasks well-represented.** Run **uncertainty weighting as a P3 ablation** (cheap to add, direct upgrade path if naive-sum training shows one head stalling or dominating) — do not reach for GradNorm unless/until the dataset is meaningfully larger (QMAP's own MIC set, ~4k, is likely still on the small side for it).

---

## 4. Head design: MLP over bare linear, TI as derived (not learned) by default

### 4.1 Linear vs. MLP head

Doc 02 §5.6's sketch uses `nn.Linear(d, 1)` per task. The PeptideCLM-2 authors' own regression fine-tuning code does **not** do this — both their downstream regression head (`finetune_ensemble.py`: `intermediate_layer = Linear(d,d)` → `Dropout(0.2)` → `regression_head = Linear(d,1)`) and their native MTR pretraining heads (`MTR_model.py`: `nn.Sequential(Linear(d,d), SiLU(), Linear(d, size))`) are small 2-layer MLPs, not bare linear probes.

**→ RESOLVED: use a 2-layer MLP head per task**, matching the authors' pattern:

```python
def make_head(d, dropout=0.2):
    return nn.Sequential(
        nn.Linear(d, d),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(d, 1),
    )
```

Reasoning beyond "match the source repo": a bare linear head on a frozen or lightly-adapted embedding is a strictly weaker probe than an MLP, and given QMAP's finding that HC50 signal is only "weakly linearly accessible" in a comparable PLM embedding space (ESM2-650M linear probe barely correlates), a linear `head_hc50` risks reproducing that exact failure. One extra nonlinear layer is cheap in parameters (this is the dominant place you *can* afford capacity, since the shared trunk is what you're trying to keep from overfitting on small n) and costs little in overfitting risk if dropout is present.

### 4.2 Should there be an explicit shared "disruption propensity" layer?

Doc 02 §5.1's narrative — shared mechanism (amphipathicity/charge/hydrophobicity) vs. divergent selectivity — is cleaner to implement if there's a literal shared bottleneck between the pooled embedding and the two task-specific MLPs, rather than two independent MLPs reading the same pooled vector with no shared adaptation at all:

```
pooled_embedding (d)
      │
      ▼
shared_projection = Linear(d, d) + SiLU     # the "membrane-disruption propensity" embedding
      │                     │
      ▼                     ▼
  head_hc50 (MLP)      head_mic (MLP)
```

This isn't in doc 02's sketch but is a small, well-justified addition: it gives you a **named, inspectable object** (`shared_projection`'s output) that operationalizes "general disruption propensity" as an actual tensor you can probe (e.g., correlate its principal components against charge/hydrophobic-moment baselines from P1), rather than that concept living only informally in the frozen/unfrozen encoder layers. It also gives a clean answer for §5's warm-start question (below): the shared_projection is exactly the thing that should *not* be reinitialized when MIC is added in P3, while the task MLPs are where task-specific selectivity is expected to live.

### 4.3 TI: derived quantity by default, learned head deferred to an ablation

Doc 02 floats both. Recommendation: **derive `TI_pred = pred_hc50 - pred_mic` (in log space) post-hoc; do not add a third trained `head_TI` in the default architecture.**

Reasoning:
- **Label-noise compounding.** A directly-labeled TI target would have to come from peptides with *both* HC50 and MIC measured (often in different papers, different labs, different bacterial strains/RBC sources per doc 02's own §6.4 open question #1) — i.e., the TI label itself is already a ratio of two independently noisy, heterogeneously-sourced measurements. Training a third head *against* that ratio doesn't remove the noise, it just gives the model a third noisy target to fit, on the smallest available subset (both-labeled peptides, likely the scarcest slice of the whole dataset). Deriving TI from the two marginal heads' predictions, which are each trained on their *full* respective label sets (not just the both-labeled intersection), uses strictly more data per component.
- **Ratio-of-predictions vs. ratio-of-labels is not obviously worse.** Since `log(HC50) − log(MIC) = log(HC50/MIC)`, deriving TI from two well-calibrated marginal log-predictions recovers the same quantity a learned head would target, without the extra noisy label.
- **The counter-argument (a learned head could capture nonlinear selectivity signal the marginals miss) is real but currently untestable-cheaply.** A learned `head_TI` only has something extra to learn if selectivity is *not* well-approximated by the difference of two independently-optimized heads reading the same shared embedding — e.g. if there's an interaction term the shared trunk represents but that gets lost when you optimize each marginal head only against its own task loss. This is plausible but speculative without data in hand.

**→ RESOLVED (default): derived TI.** **→ OPEN (P4 ablation, not default):** once both marginal heads are trained, fit a third small head on the both-labeled subset that takes `[pooled_embedding, pred_hc50, pred_mic]` as input and predicts either the residual `TI_actual − TI_derived` or `TI_actual` directly; compare its held-out ranking performance (Spearman ρ on cluster-holdout TI) against the plain derived quantity. This isolates exactly the question doc 02 leaves open — "does a learned head capture selectivity signal not in the marginals" — as a controlled, cheap follow-up rather than a day-one architecture commitment, and it's the right kind of experiment to only run once you already have real held-out numbers for the marginal heads (P4 per doc 02 §5.5), not before.

---

## 5. Training curriculum: P2 → P3 head-addition mechanics

Doc 02 §5.5 specifies P2 (HC50 single-task, freeze→unfreeze) then P3 (add MIC, masked multitask) but not the mechanics of the addition itself. Concretely:

### 5.1 Architecture continuity across P2 and P3
Build the P2 model with the full shared architecture from day one — `encoder → masked_mean_pool → shared_projection → {head_hc50}` — with `head_mic` simply absent from the `ModuleDict` until P3, rather than architecting P2 as a plain single-head model and retrofitting later. This avoids any checkpoint-surgery/key-renaming step when P3 starts; you're just adding a new entry to an existing `nn.ModuleDict` and loading everything else from the P2 checkpoint by name.

### 5.2 Initializing `head_mic` at P3: warm-start from `head_hc50`, not fresh init — with a caveat, run as an ablation

Argument for warm-starting (copying `head_hc50`'s weights as the initial state of `head_mic`, then letting both diverge under the multitask loss): in **log-concentration space**, both labels are "concentration threshold at which a membrane-disruption effect is observed" — HC50 (concentration causing 50% hemolysis) and MIC (minimum concentration inhibiting bacterial growth) are both, directionally, *lower when the peptide disrupts membranes more readily*. That's precisely doc 02 §5.1's "shared mechanism" claim operationalized: a peptide that's a strong general membrane-disruptor should show up as *low* on both scales before assay-specific selectivity shifts them apart. If that framing is right, `head_hc50`'s already-trained linear mapping from `shared_projection` to a log-concentration output is a much better prior for `head_mic` than Xavier/Kaiming random init — it starts the new head somewhere chemically sensible instead of at a random point the small MIC-labeled set (relative to HC50's already-small set) then has to move a long way from scratch.

Caveat, and why this needs to be an ablation rather than an assumption: the two labels are not on the same scale by construction (different unit conventions, different concentration ranges typical of each assay) even after your log-transform + normalization step (doc 02 §5.5/P0) — if that normalization isn't perfectly matched, a copied head starts with a systematic offset that random init wouldn't have, and the model has to unlearn it. This is exactly the kind of small, cheap, easy-to-get-wrong-a-priori design choice that should be settled empirically rather than argued from first principles.

**→ OPEN, run both as a controlled P3 ablation:**
- (a) `head_mic` initialized as a copy of trained `head_hc50` weights.
- (b) `head_mic` initialized fresh (Xavier/Kaiming, matching whatever init P2 used for `head_hc50` before its own training).

Identical everything else (data, splits, schedule); compare held-out-cluster MIC performance (and, secondarily, whether TI ranking improves) between (a) and (b). Given this is a single extra training run, it's cheap enough that there's no reason to guess instead of testing.

### 5.3 Re-freezing the encoder when `head_mic` is added

**→ RESOLVED: yes, re-freeze `encoder` (and, in this refined architecture, `shared_projection` too) at the start of P3**, mirroring the same two-stage protocol P2 already uses, for a straightforward reason: `head_mic` starts either randomly initialized or copied-but-uncalibrated (§5.2), and its first few gradient steps will be large and somewhat arbitrary relative to what the trunk has already learned from HC50. Backpropagating those early, noisy `head_mic` gradients into an already-reasonably-fit `shared_projection`/encoder risks partially undoing P2's progress before `head_mic` has learned anything useful to teach the trunk. Concrete P3 sub-schedule:

1. **P3a — head warm-up (frozen trunk).** Freeze `encoder` + `shared_projection`; train only `head_mic` (and, since it's now getting gradient again as part of the multitask loss, allow `head_hc50` to keep training too, still frozen trunk) for a short warm-up (a handful of epochs, early-stopped on val loss same as P2's protocol) using the masked multitask loss and stratified batching from §3.
2. **P3b — joint fine-tune.** Unfreeze the top N encoder blocks + `shared_projection` (same "top-layers-only, low LR" pattern P2 used going from its own frozen→unfrozen stage), continue with the masked multitask loss, now with differential learning rates: lowest for encoder blocks, higher for `shared_projection`, highest for the two task heads — since the heads have the most catching-up to do and the encoder the least (it's already been adapted once).
3. Use the **same cluster-holdout split methodology** (doc 02 §5.4) for the P3 validation set as P2, but note MIC's larger label count (§1) means its clusters are a different partition than HC50's — decide up front whether P3 evaluates against a combined split covering both label sets or reports per-task cluster-holdout numbers separately (recommend: report both; a combined figure is needed for the TI ablation in §4.3, per-task numbers are needed to compare against QMAP's baselines on a matched footing).

---

## 6. Refined end-to-end sketch

Supersedes doc 02 §5.6's sketch with the pooling, head, and loss decisions above. Still illustrative — verify exact `AutoModel` output keys against the installed `peptideclm-2` revision before treating this as drop-in.

```python
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

BACKBONE = "aaronfeller/peptideclm-2-hybrid-small"  # verify variant on HF

def masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts

def make_head(d, dropout=0.2):
    return nn.Sequential(
        nn.Linear(d, d), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d, 1),
    )

class LysisMultitaskModel(nn.Module):
    """Shared PeptideCLM-2 encoder -> shared 'disruption propensity' projection
    -> per-task MLP heads (HC50, MIC). TI is derived post-hoc, not a head."""
    def __init__(self, backbone=BACKBONE, tasks=("hc50", "mic"), dropout=0.2):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
        self.enc = AutoModel.from_pretrained(backbone, trust_remote_code=True, use_safetensors=True)
        d = self.enc.config.hidden_size
        self.shared_projection = nn.Sequential(nn.Linear(d, d), nn.SiLU())
        self.heads = nn.ModuleDict({t: make_head(d, dropout) for t in tasks})

    def add_task(self, name, warm_start_from=None, dropout=0.2):
        """P3: add a new task head. warm_start_from: existing task name to copy
        weights from (ablation arm 'a'), or None for fresh init (arm 'b')."""
        d = self.enc.config.hidden_size
        new_head = make_head(d, dropout)
        if warm_start_from is not None:
            new_head.load_state_dict(self.heads[warm_start_from].state_dict())
        self.heads[name] = new_head

    def forward(self, smiles_batch):
        x = self.tok(smiles_batch, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        x = {k: v.to(self.enc.device) for k, v in x.items()}
        out = self.enc(**x)
        pooled = masked_mean_pool(out.last_hidden_state, x["attention_mask"])
        shared = self.shared_projection(pooled)
        return {t: head(shared).squeeze(-1) for t, head in self.heads.items()}

def masked_mse(pred, target, mask):
    n = mask.sum()
    if n == 0:
        return None
    return (((pred - target) ** 2) * mask).sum() / n

def multitask_loss(preds, targets, masks, weights=None):
    weights = weights or {t: 1.0 for t in preds}
    total, n_active = 0.0, 0
    for t in preds:
        l = masked_mse(preds[t], targets[t], masks[t])
        if l is not None:
            total = total + weights[t] * l
            n_active += 1
    return total if n_active else None

def derive_ti(pred_log_hc50, pred_log_mic):
    return pred_log_hc50 - pred_log_mic  # log(HC50/MIC); larger = more selective
```

Freeze/unfreeze helpers, the stratified batch sampler (§3.3), and the P3a/P3b schedule (§5.3) are training-loop concerns, omitted here for brevity but should live alongside this module, not inside it.

---

## 7. Open questions carried forward

1. **Mean-pool vs. BOS-token, empirically, on your own data** (§2) — the authors' preference is a strong prior, not a guarantee for a lysis-propensity target specifically.
2. **Naive-sum vs. uncertainty-weighted task loss** (§3.5) — start naive, ablate uncertainty weighting in P3, watch for degenerate log-variance collapse given few steps/epoch.
3. **Copy-init vs. fresh-init for `head_mic`** (§5.2) — argued both ways above; settle by running both, not by picking one.
4. **Derived vs. learned TI** (§4.3) — derived by default; the learned-head ablation is only worth running once P4 marginal-head numbers exist.
5. **Combined vs. per-task cluster-holdout reporting for P3/P4** (§5.3.3) — recommend both, but this duplicates split-construction effort worth scoping into P0's deliverable.
6. **Both-labeled subset size** — nothing found in QMAP or DBAASP's public docs gives an exact count of peptides with *both* MIC and HC50 measured under directly comparable conditions; this number gates how meaningful the TI ablation (§4.3) and the "shared mechanism" hypothesis itself (§5.2) actually are, and should be measured as part of doc 02's P0 data build before committing engineering time to the TI ablation.

---

## Source pointers
- QMAP: Lavertu, Corbeil & Germain, bioRxiv 2026.02.03.703041, `biorxiv.org/content/10.64898/2026.02.03.703041v1`; also *Scientific Reports* 2026, `10.1038/s41598-026-56004-8`; code `github.com/anthol42/QMAP`.
- PeptideCLM-2 repo (pooling + head architecture ground truth): `github.com/AaronFeller/PeptideCLM-2`, specifically `training/01_regression_benchmarks_training_code/finetune_ensemble.py` and `.../model/MTR_model.py`.
- PeptideCLM-2 model card / usage example: `huggingface.co/aaronfeller/peptideclm-2-hybrid-small`; collection `huggingface.co/collections/aaronfeller/peptideclm-2`.
- Uncertainty task-weighting: Kendall, Gal & Cipolla, *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics*, CVPR 2018.
- GradNorm: Chen et al., *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*, ICML 2018.
