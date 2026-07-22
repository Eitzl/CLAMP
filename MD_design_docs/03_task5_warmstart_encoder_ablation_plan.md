# Warm-Start Ablation — Execution Plan for §5.0 / §5.6 (Base Encoder vs. PAMPA-Fine-Tuned Checkpoint)

*Companion to `02_peptideclm_transfer_learning_plan.md`. That doc states the position — don't warm-start from the PAMPA-fine-tuned checkpoint, and keep it only as an ablation (§5.0, §5.6) — but doesn't specify how to build arm (b), how to keep "everything else identical," how many reruns make the result trustworthy, or what could quietly invalidate the comparison. This doc is the execution plan for exactly that slice. All GitHub/HF claims below were checked directly against the live repos on 2026-07-22; see Source pointers.*

---

## 1. Does a PAMPA-fine-tuned checkpoint actually exist to download? No.

**Checked:** the HF collection `huggingface.co/collections/aaronfeller/peptideclm-2`, the full HF user page `huggingface.co/aaronfeller`, and the file trees of `github.com/AaronFeller/PeptideCLM` and `github.com/AaronFeller/PeptideCLM-2` (both repos' default branch is `master`, not `main`).

**What's published:**
- HF collection: 9 PeptideCLM-2 checkpoints — `peptideclm-2-{mlm,mtr,hybrid}-{small,base,large}` (31.7M/0.1B/0.3B params) — all **base pretrained** (MLM / multi-task-RDKit-regression / MLM-MTR-hybrid objectives), plus the pretraining dataset. None are described as fine-tuned on PAMPA, permeability, or any downstream task.
- HF user page: same list, plus v1's `PeptideCLM-23M-all`, `PeptideCLM-12M-smol`, `PeptideCLM-11M-pep` — again, base pretrained only.
- `github.com/AaronFeller/PeptideCLM` (v1): ships `example_training_script.py` and a note that "Finetuned models are trained to predict membrane penetration for cyclic peptides from CycPeptMPDB," but **no saved fine-tuned weights** — only the training script and `clustered_data/all_clusters.csv` (6,702 rows: `SMILES, PAMPA, cluster`, 6 k-means clusters, sizes 494–1546).
- `github.com/AaronFeller/PeptideCLM-2`: goes further — it contains the *entire* fine-tuning harness and the **results** of a 3×-replicated PAMPA fine-tuning study per architecture/objective (`figure_generation/results/PAMPA_results/{MLM,MTR,Hybrid}/*_study_{1,2,3}.csv`, plus tokenizer/masking-ablation variants), but these are `metrics.csv` / `predictions.csv` outputs, not `.ckpt` model weights. The identical-lineage cluster file also lives here as `data/PAMPA_clusters.csv` (same 6,702-row structure).

**Conclusion:** arm (b) of the §5.6 ablation ("train from the PAMPA-fine-tuned checkpoint") is **not runnable off the shelf**. You must reproduce the PAMPA fine-tune yourself before you can run the ablation. Fortunately, the repo gives you everything needed to do that faithfully:

**Reproduction recipe (arm-(b) checkpoint construction):**
1. Data: `PeptideCLM-2/data/PAMPA_clusters.csv` (or v1's `clustered_data/all_clusters.csv` if you use v1) — already k-means clustered into 6 leave-one-cluster-out folds, no extra split work needed.
2. Code: `PeptideCLM-2/training/01_regression_benchmarks_training_code/finetune_ensemble.py`, model class `models/MTR_model.py`. Confirmed hyperparameters read directly from the script:
   - Tokenizer: `AutoTokenizer.from_pretrained("aaronfeller/PeptideMTR")`, `model_max_length=2048`.
   - Per-size LR: small (32M, 14 blocks) → `3e-4`; base/"medium" (114M, 24 blocks) → `1e-4`; large (337M, 32 blocks) → `5e-5`. Optimizer: `AdamW`. `gradient_clip_val=0.1`, `precision='bf16-mixed'`.
   - `max_epochs=10`, `val_check_interval=0.2` (validate 5×/epoch), `EarlyStopping(monitor='val_loss', patience=4, mode='min')`, checkpoint-by-best-val-loss.
   - Batch size default 16 (train), 128 (val/test, no shuffle).
   - Training-set target distribution is rebalanced by resampling 5 value-bins to equal size before each fold's training (`pd.cut(train_df['value'], bins=5)` + upsample-with-replacement) — a detail worth carrying into your HC50/MIC training too if you want comparable protocol.
   - Splitting is **nested**: outer loop = leave-one-cluster-out test fold; inner loop = ensemble over the remaining clusters as train/val folds. The multi-run driver script (`finetune_ensemble_multi-run.sh`) additionally runs each configuration in **3 replicate "rounds"** (`_round1/2/3`), matching the `_study_1/2/3` result files you can see in the repo.
3. **Critical confirmed detail:** the actual production PAMPA-study `sbatch`/`srun` calls in `finetune_ensemble_multi-run.sh` **omit** the script's `--transfer_learning` flag. Reading the script, that flag is what freezes the encoder and trains only the head; omitting it means **all parameters are trainable** — i.e., the published PAMPA fine-tuning results were produced by **full, unfrozen fine-tuning**, not a frozen-encoder linear probe. This matters for what arm (b) actually represents (see Risk A/B in §5).
4. Practical note: the script loads a **local pretraining `.ckpt`** (Lightning checkpoint with `state_dict`), not the HF `AutoModel` you'd load for the ablation itself. You will need to either (a) run this exact script against a locally-saved copy of the HF-hosted base checkpoint's weights (convert HF `safetensors` → the `MTR_model` state-dict format used here), or (b) reimplement the same hyperparameters against the `AutoModel(..., trust_remote_code=True)` loading path from the design doc's §5.6 sketch. Either is fine as long as you fix the discrepancy explicitly and document which one you used — don't let the ablation's arm (b) accidentally use a different tokenizer or param initialization convention than the base checkpoint used for arm (a) (the tokenizer here is `aaronfeller/PeptideMTR`, not necessarily the tokenizer shipped with the specific `peptideclm-2-hybrid-*` repo you pick — verify parity).
5. If you fall back to v1 (`PeptideCLM-23M-all`) per the design doc's §5.2 fallback clause, use v1's `clustered_data/all_clusters.csv` + `example_training_script.py` (not directly inspected for hyperparameters here — check before relying on it) and remember v1's SMILES-Pair-Encoding tokenizer requires the custom `SMILES_SPE_Tokenizer`, not vanilla `AutoTokenizer`.

**Net effect on the plan:** budget real engineering time for step 3-4 above (constructing arm (b) at all) before you can even start the ablation proper — it is not a "load a checkpoint and go" task as the phrase "the PAMPA-fine-tuned checkpoint" in §5.6 might suggest.

---

## 2. Concretizing "identical everything else"

### 2.1 Data & splits
Lock one HC50/MIC dataset + split, generated once, reused unchanged by both arms and every seed/fold combination.

Recommend adopting **QMAP's** released pipeline (`pip install qmap-benchmark`, `github.com/anthol42/QMAP`) rather than re-deriving DBAASP→SMILES→cluster tooling from scratch, as a large fraction of the design doc's flagged-as-hard §5.3/§5.4 work is already solved there:
- `DBAASPDataset` gives filterable, chainable access to DBAASP with **SMILES already attached**, N-/C-terminal modification flags (`ACT`/`AMD`), common noncanonical residues (Ornithine `O`, DAB `B`), D-amino-acid detection, disulfide/amide bond typing, per-bacterial-species MIC and consensus HC50 — i.e., most of the "chemical fidelity floor" concern in the source doc's §6.4 Q2 is pre-solved for whatever subset you select.
- `QMAPBenchmark(split=0..4)` gives a **predefined 5-way homology-aware split**: `get_train_mask()` uses a Rust-accelerated (`pwiden_engine`) BLOSUM-alignment identity computation to exclude any training sequence too similar (default threshold 0.6) to a benchmark test sequence — directly satisfying the source doc's §5.4 "homology-aware, not random" requirement, and letting you cite a standardized, third-party benchmark rather than a private split.
- Caveat: QMAP bakes in its own label/scope conventions (which species, how multi-study values are consensused, log-transform by default in `compute_metrics(log=True)`). Decide explicitly whether to adopt QMAP's scope as-is (fast, comparable to their published leaderboard) or use only the *splitting algorithm* while keeping your own label curation (needed if you want to lock HC50 to human RBC / a specific bacterial panel per the source doc's §6.4 Q1 — check whether QMAP's species selection already matches that choice before assuming it does).

### 2.2 Head architecture
Use one shared `PermeabilizationHead` module definition (as sketched in the source doc's §5.6) for both arms, freshly Xavier-initialized each run — the **only** thing that differs between arm (a) and arm (b) is the encoder's `state_dict` at the start of training. Concretely:
- Fix a **head-init seed** separately from the data-shuffle/dropout seed.
- For replicate `i`, use the *same* head-init seed `i` in both arm (a) run `i` and arm (b) run `i`. This makes the comparison **paired** (same fold, same head initialization, only encoder history differs) — which is what licenses the paired significance test in §2.5, and is standard practice for isolating one factor in an otherwise-matched ablation.

### 2.3 Optimizer / schedule
PeptideCLM-2's own regression fine-tuning script (`finetune_ensemble.py`) uses a flat AdamW LR with the scheduler code commented out. Their newer classification script (`classification_finetuning_v2.py`) instead computes a `total_steps`-aware scheduler. For the ablation, prefer an **explicit, reproducible schedule** (e.g., linear warmup + cosine or linear decay tied to a fixed total step count) over "flat LR + early stopping," since early stopping alone can silently let the two arms train for different effective step counts (see §2.4) — an explicit schedule at least bounds how much that can vary.

### 2.4 Compute budget / step count (the subtlety the source doc doesn't address)
Because arm (a) and arm (b) start from **different initial losses** on HC50/MIC (plausibly lower for arm (b) if any PAMPA-adjacent features transfer positively, plausibly *higher* if PAMPA fine-tuning over-specialized/forgot general chemistry — see Risk B/D), "same number of epochs, same early-stopping criterion" is not automatically "same optimization budget" in any meaningful sense.
- Fix the **number of optimizer steps** from your locked dataset size and batch size, identical for both arms, as the primary comparison.
- Also run with the shared early-stopping criterion (val loss on the shared held-out fold) as a secondary comparison, but **log and report the step-count-at-stop for both arms**. If they diverge substantially, that asymmetry is itself a reportable finding, not something to normalize away silently.

### 2.5 What "(b) ≤ (a)" should operationally mean
The source doc's §5.6 says: *"If (b) ≤ (a), you've empirically confirmed penetration features don't transfer... a publishable negative result."* Concretize before running anything:

- **Metric:** report HC50 and MIC **separately** (Spearman ρ or PCC on the locked held-out folds — PCC matches QMAP's own reporting convention, so numbers are directly comparable to their published baselines). Do not blend into one score first — the source doc's own §5.1 multitask framing (shared mechanism, divergent selectivity) makes it plausible that PAMPA features help one property and hurt the other; a single combined "(b) ≤ (a)" verdict would mask that.
- **Significance test given small held-out folds:** QMAP itself reports **min/mean/max across 5 splits**, not a p-value — precedent that ~5 is the realistic practical n here, and that point estimates alone are not trusted. Recommended test: for each of the (≥5) held-out folds, arm (a) and arm (b) share that exact fold (paired by §2.2's design) → compute the paired difference in ρ per fold → **Wilcoxon signed-rank test** on those paired differences (nonparametric, appropriate at n≈5–15, doesn't assume normality of a metric QMAP shows can range from −0.18 to 0.29 for HC50, i.e. close to the noise floor). Do **not** run a t-test (or any test) on pooled per-peptide residuals across folds — that overstates effective n once cluster/fold structure is accounted for, which is exactly the leakage trap the source doc's §5.4 warns against, now applied to the ablation's own statistics rather than just the train/test split.
- **Pre-register the expected effect size honestly:** QMAP's published baselines get mean PCC ≈ 0.07 (range −0.18 to 0.29) for HC50 "full," and only ≈0.36–0.56 for E. coli MIC "full" (dropping to 0.16–0.33 on the harder "high efficiency" subset). Given that noise floor, a real possibility is that arms (a) and (b) are statistically indistinguishable from **each other and from a trivial baseline** — treat that outcome as informative (the ablation was underpowered / both encoders extract similarly little signal), not as silent evidence against the source doc's anti-correlation hypothesis.

---

## 3. How many seeds/reruns to trust the result

Precedent found directly in the repos:
- PeptideCLM-2's own newer LoRA-based benchmarking (`figure_generation/results/runs_LoRA_highrank/{amp_hgt,cellppd,thpep}/...`) uses exactly **3 seeds** (`seed_101`, `seed_202`, `seed_303`) per (dataset, model) cell.
- Their PAMPA regression fine-tuning uses **3 replicate rounds** (`_round1/2/3` / `_study_1/2/3`) layered on top of a 6-cluster leave-one-cluster-out nested CV.
- QMAP's benchmark structure is **5 predefined homology-safe splits**.

Recommendation: combine both axes rather than picking one. For the single-task (HC50-first, per the source doc's §5.5 P2) ablation:
- **≥5 held-out folds** (from the locked QMAP-style split, §2.1) **× 3 seeds** (head-init/data-shuffle/dropout, paired per §2.2) = **15 runs per arm, 30 total**. This is a modest multiple of PeptideCLM-2's own published compute budget (3 seeds × a fixed fold set) and is exactly what's needed to run the paired Wilcoxon test in §2.5 without further design work.
- Do not go below **3 seeds**. A single seed cannot distinguish "arm (b) is genuinely worse" from "arm (b) got an unlucky head initialization or dropout draw," which is precisely the confound the ablation exists to rule out.
- If budget allows, also replicate the **construction of arm (b) itself** (i.e., redo the PAMPA fine-tune 2–3× with different seeds before ever touching HC50/MIC data) — see Risk C in §5 for why this matters.

---

## 4. Two-stage fine-tune, concretized (applies to P2, both arms)

The source doc's §5.5/§5.6 says: *"Freeze encoder → train head → unfreeze top layers with low LR (standard two-stage fine-tune)."* Concretely, for both arm (a) and arm (b):

**Stage 1 — frozen-encoder head training.** Freeze `self.enc` entirely (`requires_grad=False` on every encoder parameter). This is literally the pattern PeptideCLM-2 already ships behind its `--transfer_learning` flag in `finetune_ensemble.py` (lines ~355–369: sets all params to `requires_grad=False`, then re-enables only the regression-head and pooling/"intermediate" layer) — reuse that toggle logic, pointed at your HC50/MIC head instead of theirs. Train to convergence (val-loss early stopping) on the locked train/val split.

*Why this stage is worth reporting on its own, beyond being step 1 of the recipe:* with the encoder frozen, the pooled embedding fed to the head is **exactly** the base or PAMPA-checkpoint encoder's representation, untouched by any HC50/MIC-specific adaptation. A difference between arm (a) and arm (b) at this stage is the cleanest possible read on "do penetration features transfer" — there is no stage-2 confound yet (§5, Risk B) to explain it away. Report this frozen-probe result for both arms as a primary diagnostic, not just an intermediate step.

**Stage 2 — partial unfreeze at low LR.** Unfreeze the **top-k transformer blocks**, warm-started from the stage-1 checkpoint, at a reduced LR relative to stage 1. Concretize "top layers" using the actual architecture depths found in PeptideCLM-2's own config (`MTR_model.num_blocks` in `finetune_ensemble.py::model_configs`): **small = 14 blocks, base = 24 blocks, large = 32 blocks**. Recommend unfreezing a fixed **fraction** of depth (≈15–20%) rather than a fixed block count, so model sizes stay comparable — e.g., last 2 blocks for small, last 3–4 for base, last 5–6 for large — plus the final layer norm and pooling. "Top" here means the blocks nearest the output/pooling head (the ones whose activations feed directly into the mean-pooled embedding the head consumes), not top in a token-position sense.

**Stopping criteria (both stages, both arms):** monitor val loss (or val Spearman ρ on the locked val fold), early-stopping patience 3–4 epochs — matching PeptideCLM-2's own `patience=4` — or the fixed-step-budget variant from §2.4 if you use that instead. Save best-checkpoint-by-criterion; evaluate the stage-2 best checkpoint only. Do not cherry-pick between stage-1 and stage-2 checkpoints post hoc for either arm — that would reintroduce exactly the "which result did you happen to report" bias flagged in Risk F.

**Flag — the two-stage recipe is not what the authors currently use.** The *currently released* PeptideCLM-2 classification harness (`classification_finetuning_v2.py`) does not do freeze→unfreeze staging at all. It wraps the whole encoder in **LoRA** — `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["qkv_proj"])` applied to every attention block simultaneously via `peft.get_peft_model` — and trains the LoRA adapters (plus a small FC head) end to end in a single stage, no staged unfreezing. This is the method behind their newest "LoRA_highrank" benchmark runs referenced in §3. For this ablation, **pick one protocol explicitly and use it for both arms** (freeze→top-k-unfreeze *or* LoRA-across-all-blocks) — mixing them between arms would itself be a confound, and picking whichever looks better after the fact would reintroduce the Risk F bias. If engineering time is tight, LoRA-all-blocks is the better-tested path (it's the authors' own current harness for small-peptide-dataset fine-tuning); if the cleaner mechanistic story ("penetration head learned the wrong thing") matters more than convenience, freeze→top-k-unfreeze is more interpretable, precisely because its stage-1 frozen-probe result isolates representation-only differences before any adaptation happens.

---

## 5. Risks — does the ablation actually test what it claims to?

**Risk A — is the encoder even different between arms?** Checked directly: **no**, this is not automatically a problem. The actual PAMPA-study production runs (`finetune_ensemble_multi-run.sh`) omit `--transfer_learning`, meaning the published PAMPA fine-tuning used **full, unfrozen** fine-tuning — the encoder genuinely changes weight values during PAMPA fine-tuning. So arm (b)'s encoder is not literally identical to arm (a)'s. Good — the ablation is not vacuous by that specific route. But this cuts the other way too:

**Risk B — full fine-tune on ~6,700 PAMPA examples is a small-data full-fine-tune of a 32M–337M-parameter transformer.** Real risk of catastrophic forgetting or narrow overfitting to PAMPA's specific chemical distribution (short, cyclic, drug-like peptides — see source doc §6.1) rather than cleanly encoding "generalizable but wrong-task" penetration features. If arm (b) underperforms arm (a) on HC50/MIC, you cannot cleanly attribute that to "penetration features are anti-correlated with permeabilization" versus "PAMPA-only full fine-tuning narrowed/degraded the general representation, independent of task relatedness." **Mitigation:** compare the §4 stage-1 (frozen-encoder probe) results for both arms side by side. If arm (b)'s *frozen* encoder is already worse than arm (a)'s frozen encoder, that's closer to genuine evidence of a learned-but-task-wrong representation (no forgetting from your own HC50/MIC fine-tuning is possible yet — the encoder is frozen). If arm (b) only falls behind *after* stage 2 (its own encoder unfrozen again on your small HC50/MIC set), forgetting/overfitting during that step is at least as plausible an explanation as inherited PAMPA-task specificity.

**Risk C — checkpoint-specific noise predates either ablation arm.** The repo's own `_study_1/2/3` / `_round1/2/3` replicates exist precisely because pretraining-plus-PAMPA-fine-tuning is noisy run to run. Your reproduced arm-(b) checkpoint (§1) is one draw from that same noisy process. Fine-tuning it only once for the ablation conflates "PAMPA history in general" with "which particular PAMPA-fine-tune replicate you happened to reproduce." **Mitigation:** reproduce arm (b) 2–3× independently (different seeds, same recipe) before running the HC50/MIC ablation on top, and treat "which PAMPA-checkpoint replicate" as a nested random factor — or, at minimum, explicitly flag single-replicate arm (b) as a named limitation rather than silent.

**Risk D — effective step-count confound (restated from §2.4).** If arm (b) starts at lower initial loss on HC50/MIC (plausible if some feature transfer is real and positive), it may converge in fewer steps under a shared early-stopping patience, quietly turning "identical everything else" into "different realized training length." Report step-count-at-stop for both arms; if they differ substantially, add the fixed-step-budget comparison from §2.4 as a robustness check.

**Risk E — single-metric framing can hide the more interesting result.** HC50 and MIC could show opposite-signed ablation outcomes — the source doc's own §5.1 (shared mechanism, divergent membrane selectivity) makes this plausible, not exotic. A single averaged "(b) ≤ (a)" verdict would erase a finding that's scientifically more informative (e.g., PAMPA-derived features hurt MIC transfer more than HC50, consistent with PAMPA being measured on a synthetic lipid film closer in some respects to a bacterial-membrane mimic than to cholesterol-rich mammalian membranes — worth checking against the review doc's membrane-composition tables rather than assumed). Report HC50 and MIC ablation results **separately** as the primary output.

**Risk F — framing risk (methodological, not statistical).** The source doc explicitly frames "(b) ≤ (a)" as *"a publishable negative result."* That framing creates pressure to select, after the fact, whichever combination of {HC50 result, MIC result, frozen-probe result, full-fine-tune result} best supports the pre-stated conclusion. Pre-register the primary comparison (which metric, which stage, which test from §2.5) before arm (b) is ever trained, or the ablation risks manufacturing the same false confidence the source doc is trying to guard against by running it in the first place.

---

## 6. Summary checklist handed back

- [ ] **Pick the shared base architecture** (`peptideclm-2-hybrid-{small,base,large}` vs. v1 `PeptideCLM-23M-all`) once, for both the main HC50/MIC track (§5.2 of the source doc) and this ablation — arm (a) and arm (b) must be the same architecture/size, or the comparison is meaningless.
- [ ] **Reproduce arm (b)** using `PeptideCLM-2/training/01_regression_benchmarks_training_code/finetune_ensemble.py` against `data/PAMPA_clusters.csv`, confirming full (unfrozen) fine-tuning to match the authors' own production recipe (§1) — this is real engineering work, not a checkpoint download.
- [ ] **Pick one fine-tuning protocol** (freeze→top-k-unfreeze vs. LoRA-across-all-blocks, §4) and use it identically for (i) the arm-(b) PAMPA reproduction, (ii) arm (a)'s HC50/MIC fine-tune, and (iii) arm (b)'s HC50/MIC fine-tune — three fine-tunes, one protocol.
- [ ] **Adopt or fork `qmap-benchmark`** (`pip install qmap-benchmark`) for the locked DBAASP data pull + 5-way homology-aware split (§2.1), unless its baked-in label/species conventions conflict with the source doc's §6.4 Q1 decision (human-RBC-only HC50 / fixed bacterial panel).
- [ ] **Budget ≥15 runs/arm** (≥5 folds × 3 seeds, paired by head-init seed) for the single-task HC50-first ablation; add 2–3× replication of arm (b)'s *construction* if Risk C is a concern given available compute.
- [ ] **Report per-metric, per-stage results** (HC50 vs. MIC separately; stage-1 frozen-probe vs. stage-2 unfrozen; step-count-at-stop) rather than one collapsed "(b) ≤ (a)" verdict, and run the paired Wilcoxon test (§2.5) as the pre-registered primary comparison.

---

## Source pointers
- HF collection (base checkpoints only, confirmed no PAMPA-tuned checkpoint): `huggingface.co/collections/aaronfeller/peptideclm-2`; user page `huggingface.co/aaronfeller`.
- PeptideCLM v1 repo: `github.com/AaronFeller/PeptideCLM` (default branch `master`) — `clustered_data/all_clusters.csv`, `example_training_script.py`, `functions/`, `tokenizer/`.
- PeptideCLM-2 repo: `github.com/AaronFeller/PeptideCLM-2` (default branch `master`) — `data/PAMPA_clusters.csv`, `data/amp_{train,val,test}.csv`, `data/CellPPD_*`, `data/THPep_main90_smiles_classes.csv`; `training/00_pretraining/`; `training/01_regression_benchmarks_training_code/finetune_ensemble.py` and `finetune_ensemble_multi-run.sh`; `training/02_classification_benchmarks_training_code/scripts/classification_finetuning_v2.py` and `experiment/benchmark_manifest.json`; published-but-weightless PAMPA study results at `figure_generation/results/PAMPA_results/{MLM,MTR,Hybrid}/`; LoRA benchmark results at `figure_generation/results/runs_LoRA_highrank/{amp_hgt,cellppd,thpep}/`.
- QMAP: `github.com/anthol42/QMAP` (package `qmap-benchmark`); paper "QMAP: A Benchmark for Standardized Evaluation of Antimicrobial Peptide MIC and Hemolytic Activity Regression," Lavertu, Corbeil & Germain, bioRxiv Feb 2026 (`biorxiv.org/content/10.64898/2026.02.03.703041v1`), also in *Scientific Reports* (`nature.com/articles/s41598-026-56004-8`); docs at `QMAP/docs/references/{DBAASPDataset,QMAPBenchmark,train_test_split}.md`.
- DBAASP (underlying data source, size/composition context): `dbaasp.org`; DBAASP v3, *Nucleic Acids Research* 49(D1):D288 (2021).
