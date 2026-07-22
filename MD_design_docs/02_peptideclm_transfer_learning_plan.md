# PeptideCLM → Membrane-Permeabilization Predictor: Design & Transfer-Learning Plan

*Covers Tasks 3 (what membrane PeptideCLM scores against), 4 (can it be changed), 5 (transfer-learning plan for permeabilization/lysis), and 6 (further developments, feasibility, open questions). Companion to `01_permeation_vs_penetration_review.md`, which establishes the penetration-vs-permeabilization distinction this plan depends on.*

---

## Task 3 — What membrane does PeptideCLM's score correspond to?

**An artificial phospholipid membrane (PAMPA) — not a living cell.**

PeptideCLM (Feller & Wilke, *J. Chem. Inf. Model.* 2025) is a BERT-style transformer pretrained by masked language modeling on **SMILES strings** of ~23M molecules (~10.8M peptides + ~12.6M small molecules). The **permeability head was fine-tuned on the PAMPA-only subset of CycPeptMPDB**:

- The authors explicitly subset CycPeptMPDB to peptides with **PAMPA** results and **excluded the cell-based assays** (Caco-2, MDCK, RRCK).
- They removed points flagged "undetectable" (coded −10) to avoid aggregation/measurement artifacts.
- The label is **log P_exp (log cm/s)**, the effective PAMPA permeability coefficient. (The paper reports ROC/PR curves, i.e. a permeable-vs-not framing on top of the continuous PAMPA value — worth confirming against the repo when you wire up the head.)

**What PAMPA physically is:** a passive-diffusion assay across a filter impregnated with phospholipid dissolved in an inert organic solvent (typically egg lecithin ± cholesterol in dodecane on a PVDF filter). It has **no transporters, no efflux pumps, no cytoskeleton, no membrane potential** — pure transcellular passive permeation of the intact molecule. So the score answers: *"how well does this peptide passively diffuse across a defined artificial lipid film?"* — the **drug/penetration** sense from the review doc, and the opposite of the permeabilization/lysis you care about.

**PeptideCLM-2** (bioRxiv 2026.01.06.697994; HF collection `aaronfeller/peptideclm-2`) is the successor suite: 9 models (32M–337M params), trained on >100M molecules with three objectives — MLM, multi-task regression to 99 RDKit descriptors (MTR), and an MLM-MTR hybrid — and benchmarked on membrane diffusion, aggregation, and **cell targeting**. Its permeability results still trace to the same CycPeptMPDB/PAMPA lineage; the added value is a stronger, more general **encoder**, which is what matters for transfer.

---

## Task 4 — Can the membrane be changed?

**Not at inference. The membrane is not a model input — it is baked into the training labels.**

PeptideCLM/-2 take **only a SMILES string** as input. There is no argument for membrane type, lipid composition, or assay. The model never "knows" it is predicting PAMPA; it only learned a mapping from chemistry → the numbers it was trained on, and those numbers happened to come from PAMPA. Consequently:

- You **cannot** flip a switch to score against a bacterial or RBC membrane.
- To target a different membrane you must **re-fine-tune on data labeled against that membrane/assay.** The pretrained encoder is membrane-agnostic; the *head* is where "PAMPA-ness" lives.

**Precedent that this works:** the CPMP model (molecular-attention transformer, PMC11933047) was fine-tuned **separately** on PAMPA, Caco-2, RRCK, and MDCK subsets of CycPeptMPDB to produce assay-specific predictors — i.e., swapping the target membrane is done by swapping the fine-tuning dataset. This is precisely the mechanism your Task 5 relies on, generalized from "a different permeability assay" to "a permeabilization/lysis assay."

This is also the deep parallel to your own data: **whatever hemolysis/MIC assay your labels come from silently defines the membrane your new model predicts** (human RBC vs. specific bacterial species vs. LUV composition). You are not escaping the "membrane is in the label" property — you are choosing which label to inherit it from.

---

## Task 5 — Transfer-learning plan (base → permeabilization/lysis head)

### 5.0 One correction to the framing, then everything else follows
Your description — "replace the layer that predicts penetration with one that predicts permeation" — is right in spirit, with one fix: **do not warm-start from the PAMPA-fine-tuned checkpoint. Start from the base pretrained encoder** (`PeptideCLM-23M-all` for v1, or a `peptideclm-2` MLM/hybrid variant) and attach a **fresh** head trained on lysis labels.

Why: penetration (passive crossing) and permeabilization (disruption) are mechanistically different and partly **anti-correlated** (a good quiet permeant is not a good membrane-wrecker). The PAMPA head encodes the wrong task and risks **negative transfer**. The thing worth reusing is the **encoder's general representation of peptide chemistry**, learned from 100M+ molecules — that is task-agnostic and useful for any peptide property. So this is standard foundation-model transfer: *reuse the backbone, train a new head*, not "the penetration head is close to the lysis head."

*(Keep the PAMPA-checkpoint start as a small ablation — see 5.6 — to empirically test whether any penetration features transfer.)*

### 5.1 Reframe the target as multitask (this is stronger than "unfortunately unrelated")
You noted HC50 and MIC seem unrelated. They are actually an **ideal multitask pair**:

- They **share** the upstream mechanism (amphipathicity, charge, hydrophobicity all drive membrane disruption) → a **shared encoder** captures "membrane-disruption propensity" once.
- They **diverge** on **membrane selectivity** (anionic bacterial vs. zwitterionic/cholesterol mammalian) → **separate heads** capture the assay-specific part.
- The quantity you actually want to design for — the **therapeutic index TI = HC50/MIC** — is exactly the *difference* between the two heads. Predict both, and selectivity falls out.

The QMAP benchmark (bioRxiv 2026) already treats MIC and HC50 as a joint regression problem with shared evaluation, so there is precedent and a ready-made yardstick.

Proposed architecture: shared PeptideCLM-2 encoder → mean-pooled embedding → two (or three) regression heads: `head_HC50`, `head_MIC`, and optionally `head_TI` or a derived `log(HC50) − log(MIC)`. Use masking so peptides with only one label still train the shared trunk.

### 5.2 Model choice
- **Primary: PeptideCLM-2**, hybrid (MLM-MTR) small/medium variant. Released with weights + code (`github.com/AaronFeller/PeptideCLM-2`, `huggingface.co/collections/aaronfeller/peptideclm-2`), loads cleanly via `AutoModel(..., trust_remote_code=True, use_safetensors=True)`, all variants take SMILES. The paper's own finding — descriptor-guided (MTR) pretraining helps most at *smaller* model scale — suggests a hybrid small/medium model is a sensible, cheap starting point.
- **Fallback: PeptideCLM v1** (`aaronfeller/PeptideCLM-23M-all`). Note the friction: its custom SMILES-Pair-Encoding tokenizer **cannot be loaded through vanilla `transformers`** — you must pull the `tokenizer/` dir from the repo and use `SMILES_SPE_Tokenizer`. Also: the v1 **pretraining dataset had a cyclization ring-numbering bug fixed in v1.1** — use v1.1 if you ever pretrain, and be cautious reusing v1 cyclic artifacts.

### 5.3 Data (you chose to reconstruct — good; here's the source stack)
- **Primary: DBAASP.** Best single source for AMPs with **MIC and hemolytic/HC50** plus the modification metadata you need (N-/C-terminal modifications, D-amino acids, cyclization), and it often provides monomer/SMILES-level structure. This is what makes SMILES conversion tractable.
- **Supplement:** DRAMP, APD3, dbAMP for coverage; HemoPI/HemoPI2 for hemolysis; **QMAP's curated, homology-split sets** so you can benchmark against published baselines rather than a private split.
- **For the cyclic long game:** CycPeptMPDB (permeability, not lysis) and any cyclic-AMP activity sets you can find — cyclic *lysis* data is scarce, which is a real constraint (see 6.2).

**SMILES conversion (your flagged risk).** Linear canonical peptides → SMILES is mechanical (`RDKit Chem.MolFromSequence`, or HELM notation). Modifications are the hard part; concrete tooling:
- **`CycloPs_v2`** — Feller's own peptide-SMILES generation library (linked from the PeptideCLM repo), purpose-built to emit peptide SMILES including cyclization. Most aligned with the model's expected input distribution.
- **HELM → SMILES** toolchains for terminal/backbone modifications.
- **DBAASP-provided structures** where available (skip conversion entirely).
Budget real time here; if modification metadata is missing or inconsistent, you fall back to plain-backbone SMILES and lose the exact advantage (amidation, D-residues) that motivated using a chemical LM. Accepting that for a first pass, as you said, is reasonable — just measure how much data you keep at full chemical fidelity.

### 5.4 Splits (non-negotiable)
Use **homology-aware or embedding-cluster splits**, not random. Both Feller & Wilke (leave-one-cluster-out k-means on embeddings) and QMAP (sequence-homology constraints) show random splits massively inflate performance via train/test leakage — the exact overfitting trap on small peptide datasets. Cluster with the PeptideCLM-2 embeddings themselves for consistency.

### 5.5 Phased plan
- **P0 — Data build.** Pull DBAASP (+supplements), normalize units (everything to µM; log-transform HC50/MIC), resolve assay/species heterogeneity (ideally restrict to human RBC HC50 + a defined bacterial panel for MIC to keep the "membrane in the label" clean), convert to SMILES at max fidelity, dedupe, cluster, split. **Deliverable: a versioned dataset + datasheet noting assay/species per label.**
- **P1 — Baselines.** Physicochemical-descriptor regressor (charge, μH, H, length) and, if you have it, your existing ESM-2 sequence model. These are the bars PeptideCLM-2 must clear to justify the SMILES complexity.
- **P2 — Single-task fine-tune (HC50 first).** Base PeptideCLM-2 encoder + one regression head. Freeze encoder → train head → unfreeze top layers with low LR (standard two-stage fine-tune). Establishes the transfer path end-to-end.
- **P3 — Multitask.** Add MIC (and TI) heads on the shared encoder with label masking. Compare single- vs. multi-task on held-out clusters.
- **P4 — Eval + ablation.** Report on cluster-holdout with regression metrics (MAE, Spearman ρ) and, for TI, ranking metrics; run the 5.6 ablation.

### 5.6 Minimal loading sketch (PeptideCLM-2 + regression head)
Illustrative, not drop-in — confirm exact repo names/pooling against the model card.

```python
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

BACKBONE = "aaronfeller/peptideclm-2-hybrid-small"  # verify variant on HF

class PermeabilizationHead(nn.Module):
    """Shared PeptideCLM-2 encoder + multitask regression heads (HC50, MIC)."""
    def __init__(self, backbone=BACKBONE, tasks=("hc50", "mic")):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
        self.enc = AutoModel.from_pretrained(
            backbone, trust_remote_code=True, use_safetensors=True
        )
        d = self.enc.config.hidden_size
        self.heads = nn.ModuleDict({t: nn.Linear(d, 1) for t in tasks})

    def forward(self, smiles):
        x = self.tok(smiles, return_tensors="pt", padding=True, truncation=True)
        h = self.enc(**x).last_hidden_state.mean(dim=1)   # mean-pool
        return {t: head(h).squeeze(-1) for t, head in self.heads.items()}

# Two-stage fine-tune: freeze encoder for warm-up, then unfreeze top blocks.
# Masked multitask loss: only backprop the head whose label exists per sample.
```

**Ablation to actually test your premise:** train (a) from the base MLM/hybrid checkpoint vs. (b) from the PAMPA-fine-tuned checkpoint, identical everything else. If (b) ≤ (a), you've empirically confirmed penetration features don't transfer to permeabilization — a publishable negative result and a clean justification for the design.

---

## Task 6 — Further developments, feasibility, open questions

### 6.1 Feasibility — honest read
- **Engineering: easy.** Fine-tuning a released transformer with regression heads on 10³–10⁵ labels runs comfortably on a single Colab/consumer GPU. No blockers.
- **Science: moderate-to-hard, and worth stating up front.** QMAP found **HC50 is intrinsically hard to predict and MIC progress has stalled for six years.** Expect modest R²/ρ, not near-perfect regression. The value is a *ranking/prioritization* tool and a selectivity signal, not an oracle.
- **Biggest risk = domain shift in the direction you care about.** PeptideCLM was validated on **cyclic, drug-like, short (2–15-mer)** peptides. Most *lysis* training data is **linear, longer, cationic** AMPs. So you'll train mostly on linear data — and your long-term deployment interest is cyclic (see 6.3), which is a shift the other way. Quantify this: hold out a cyclic cluster and watch performance.

### 6.2 The cyclic-peptide rationale actually strengthens the model choice
Your long-term lean toward cyclic AMP production is well-supported and is the strongest single argument for using a chemical LM over ESM-2. Cyclization confers, across multiple studies: **proteolytic/protease stability, longer half-life, and often improved potency and bacterial-vs-mammalian selectivity** (ACS Omega 2024, CE-03/CE-05, acsomega.4c11466; Dathe-group hexapeptides; C-LR18, PMC11939470 — MIC held at 4 µM vs. 128 µM for the linear form after protease treatment; ultrashort amide-cyclized WKR-cyl, 2026). A **sequence model literally cannot see cyclization; a SMILES model can.** The catch (6.1): cyclic *lysis* datasets are thin, so near-term you train on linear and rely on the encoder's cyclic pretraining to generalize — testable, not guaranteed.

### 6.3 Ideas worth queuing
- **Predict TI directly**, or add a monotonic selectivity head, so the model optimizes the design objective rather than two proxies.
- **Uncertainty estimates** (ensemble or MC-dropout) — essential given noisy HC50 labels, so downstream selection can prefer confident high-TI candidates.
- **Active learning loop** with your wet-lab: model proposes, you assay the highest-information peptides, retrain. Cheapest path to beating QMAP baselines on *your* chemical space.
- **MD-derived features** (insertion depth, tilt, order-parameter perturbation from short CG-MARTINI runs) as auxiliary regression targets — a physics-grounded analog of PeptideCLM-2's RDKit-descriptor MTR objective.
- **Couple to a generator** (e.g., a peptide generative model) with this model as the scoring/filter function, for closed-loop design toward high TI.

### 6.4 Open questions for you
1. **Label scope:** lock HC50 to human RBC and MIC to a fixed bacterial panel/species? Mixing species/assays reintroduces the "which membrane" ambiguity into your own labels.
2. **Chemical fidelity floor:** what fraction of your DBAASP pull actually has clean modification metadata (amidation, D-aa, cyclization)? That number decides whether the SMILES route beats a plain ESM-2 baseline.
3. **Head vs. full fine-tune** given dataset size — do you want to start frozen-encoder (safer on small n) or full fine-tune (needs more data, more regularization)?
4. **PeptideCLM-2 variant:** hybrid-small to start, or benchmark a couple of sizes? The paper suggests small-scale benefits most from the descriptor-guided objective.
5. **Deployment target:** are we prioritizing near-term linear-AMP screening, or building the pipeline cyclic-first from the start (which changes data-sourcing priorities)?

---

## Source pointers (see companion review doc for full list)
- PeptideCLM: Feller & Wilke, *JCIM* 2025, 10.1021/acs.jcim.4c01441; repo `AaronFeller/PeptideCLM`; weights `huggingface.co/aaronfeller`.
- PeptideCLM-2: bioRxiv 2026.01.06.697994; `github.com/AaronFeller/PeptideCLM-2`; `huggingface.co/collections/aaronfeller/peptideclm-2`.
- CycPeptMPDB: Li et al., 2023, PMC10091415. Per-assay fine-tuning precedent: CPMP, PMC11933047.
- QMAP (MIC+HC50 regression benchmark): bioRxiv 2026.02.03.703041.
- Cyclic-AMP advantages: ACS Omega 2024 (acsomega.4c11466); C-LR18 PMC11939470; WKR-cyl 2026.
- DBAASP (data source): dbaasp.org.