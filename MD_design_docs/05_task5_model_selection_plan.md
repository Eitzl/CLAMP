# Task 5.2 Deep-Dive: Model Choice — Verified Inventory, Benchmarking Protocol, and Decision Rule

*Expands §5.2 of `02_peptideclm_transfer_learning_plan.md` into something executable. Everything below was checked directly against the live HuggingFace API, the live GitHub repos, and the paper (via PMC), on 2026-07-22 — not re-derived from the source doc's prose. Where I could not verify a claim first-hand, it's called out explicitly in §7.*

---

## 0. Bottom line

**Default: `aaronfeller/peptideclm-2-hybrid-small`, but only after the §4 pilot — do not skip straight to full P2–P4 on a guessed variant.** The paper never tested anything resembling HC50/MIC regression, and the one benchmark it ran that's *closest* to our task (antimicrobial-activity classification) was actually won by **MLM**, not hybrid, at large scale — which cuts against blindly trusting the "hybrid helps most at small scale" heuristic for our specific labels. Run the small pilot in §4 first; it costs a few GPU-hours at most.

**v1 fallback (`aaronfeller/PeptideCLM-23M-all`) should be a last resort**, not a coin-flip alternative: it costs real, avoidable engineering friction (custom tokenizer, hand-built classification head, an unfixed cyclization bug baked into the released weights) that v2 does not have. See §6 for the exact trigger conditions.

---

## 1. What's actually there (verified inventory)

### 1.1 PeptideCLM-2 — primary candidate

Confirmed live via the HF collection page, `huggingface.co/aaronfeller`, and `curl`'d `config.json` for each repo (not guessed):

| HF repo ID (exact) | Objective | Size tag | Params | `embed_dim` | `num_blocks` | `num_heads` | `vocab_size` | `max_seq_len` |
|---|---|---|---|---|---|---|---|---|
| `aaronfeller/peptideclm-2-mlm-small` | MLM | small | 31.7M | 512 | 14 | 8 | 405 | 2048 |
| `aaronfeller/peptideclm-2-mtr-small` | MTR | small | 31.7M | 512 | 14 | 8 | 405 | 2048 |
| `aaronfeller/peptideclm-2-hybrid-small` | MLM+MTR | small | 31.7M | 512 | 14 | 8 | 405 | 2048 |
| `aaronfeller/peptideclm-2-mlm-base` | MLM | base | ~114M | 768 | 24 | 12 | 405 | 2048 |
| `aaronfeller/peptideclm-2-mtr-base` | MTR | base | ~114M | 768 | 24 | 12 | 405 | 2048 |
| `aaronfeller/peptideclm-2-hybrid-base` | MLM+MTR | base | ~114M | 768 | 24 | 12 | 405 | 2048 |
| `aaronfeller/peptideclm-2-mlm-large` | MLM | large | ~337M | 1024 | 32 | 16 | 405 | 2048 |
| `aaronfeller/peptideclm-2-mtr-large` | MTR | large | ~337M | 1024 | 32 | 16 | 405 | 2048 |
| `aaronfeller/peptideclm-2-hybrid-large` | MLM+MTR | large | ~337M | 1024 | 32 | 16 | 405 | 2048 |

Plus the pretraining/descriptor dataset: `huggingface.co/datasets/aaronfeller/peptideclm-2-pretraining-data`.

**Correction to the source doc:** there is **no "medium" tier**. The three sizes are `small` / `base` / `large` (32M / 114M / 337M). §5.2's phrase "hybrid small/medium" should read "hybrid small (or base)."

**Naming quirk worth knowing:** the HF model card and the auto-loaded custom code both refer to this internally as **"PeptideMTR"** / `ChemPepMTR` (the model class is literally `ChemPepMTR.MLM_model`, and the model card says "part of the PeptideMTR suite"). The paper's own bioRxiv v1 title was *"PeptideMTR: Scaling SMILES-Based Language Models for Therapeutic Peptide Engineering"*; later versions retitled to *"Scaling SMILES-based chemical language models for therapeutic peptide engineering."* Same project, same weights, just be aware if you're searching for it independently — "PeptideMTR" and "PeptideCLM-2" are the same thing.

**GitHub repo** (`github.com/AaronFeller/PeptideCLM-2`, confirmed via GitHub API) contains:
- `README.md`, `LICENSE` (MIT)
- `tokenizer/` — `build_tokenizer.ipynb`, `my_tokenizer.py`, `unique_chars.txt` (tokenizer *construction* code; the *usable* artifact is already baked into each HF repo as a standard `tokenizer.json`, see §2)
- `data_processing/` — `canonicalize.py`, `normalize_descriptors.py`, `analyze_pretraining_overlap.py`, `data_processing.ipynb`
- `training/00_pretraining/`, `training/01_regression_benchmarks_training_code/`, `training/02_classification_benchmarks_training_code/`, `training/03_PepMSND_training_code/` — **these are directly reusable as a starting template for our own HC50/MIC regression fine-tuning script**, since `01_regression_benchmarks_training_code` is exactly the "attach a regression head, fine-tune" pattern we need.

**Paper:** Feller, Secor, Swanson, Wilke, Deibler. *"Scaling SMILES-based chemical language models for therapeutic peptide engineering."* bioRxiv, DOI `10.64898/2026.01.06.697994` (latest seen: v5, June 23 2026); also indexed as PMC12803269. Authors span Novo Nordisk Molecular AI (Feller, Secor, Swanson, Deibler) and UT Austin Integrative Biology / Wilke lab (Feller, Wilke). Note the DOI prefix is `10.64898`, not bioRxiv's usual `10.1101` — unusual but consistently reported across independent search hits, not a transcription error on my part as far as I can tell.

### 1.2 PeptideCLM v1 — fallback

Confirmed present on `huggingface.co/aaronfeller`: `PeptideCLM-23M-all`, `PeptideCLM-11M-pep`, `PeptideCLM-12M-smol` (all tagged `fill-mask`, MIT license).

**GitHub repo** (`github.com/AaronFeller/PeptideCLM`) contains `tokenizer/`, `example_training_script.py`, `clustered_data/` (the PAMPA/CycPeptMPDB clustered fine-tuning data), and two analysis notebooks.

**Tokenizer friction confirmed directly from the model card**, verbatim loading pattern:
```python
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer

def get_tokenizer():
    vocab_file = 'tokenizer/new_vocab.txt'
    splits_file = 'tokenizer/new_splits.txt'
    return SMILES_SPE_Tokenizer(vocab_file, splits_file)
```
This requires the `tokenizer/` directory from the GitHub repo on your Python path — it is **not** shipped inside the HF model repo and **cannot** be loaded via `AutoTokenizer.from_pretrained(...)`, confirmed by the README's own admission: *"I attempted to port my custom tokenizer to HuggingFace, but was unable to."*

**Loading-code inconsistency I found (not in the source doc):** the GitHub README's usage example loads the model as
```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('aaronfeller/model_name')
```
while the HF model card for `PeptideCLM-23M-all` itself uses
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained("aaronfeller/PeptideCLM-23M-all", device_map="auto")
```
These are two different `AutoModelFor*` heads on the *same* base checkpoint, and neither is a plain regression head — meaning **v1 doesn't hand you a fine-tuning head "for free" either.** You'd write the same custom `nn.Linear` regression-head wrapper for v1 as for v2 (per the source doc's §5.6 sketch); the *only* extra v1 tax is the tokenizer plumbing. Worth knowing so nobody picks v1 under the mistaken impression it's "more finished."

**v1.1 status — checked directly against the Zenodo record (`10.5281/zenodo.14194469`, fetched via the Zenodo API):**
> *"This version update includes changes to `Generated_peptides.csv` to fix cyclization. The prior upload did not have ring closures generated correctly as SMILES strings. **The model in the publication was trained on the dataset containing errors**, however to support the community we decided it would be best to release a 10M peptide SMILES dataset for use in future pretraining applications."*

This confirms the source doc's implicit read, precisely: **v1.1 is a data-only correction with no corresponding retrained checkpoint.** The `PeptideCLM-23M-all` / `-11M-pep` / `-12M-smol` weights on HF today are still the ones trained on the buggy (v1.0) cyclization data — there is no "v1.1 weights" to download, and the fix is only relevant to you if *you* pretrain from scratch on the corrected corpus. If we use v1 as a fallback and fine-tune its existing checkpoint, we inherit whatever cyclic-SMILES representation quality it learned from the ring-closure-buggy examples in its pretraining set — a real, if probably small, defect to log rather than a hypothetical.

---

## 2. Confirmed loading mechanics

### 2.1 PeptideCLM-2 — verified working pattern (from the live README, cross-checked against HF API metadata)

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "aaronfeller/peptideclm-2-hybrid-small"  # swap per §4/§6

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)

smiles_string = "NCC(=O)NCC(=O)O"  # glycyl-glycine
inputs = tokenizer(smiles_string, return_tensors="pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state   # (1, seq_len, embed_dim) — mean-pool per source doc §5.6
```
This matches the source doc's §5.6 sketch essentially exactly — good sign it wasn't guessed.

**Why `trust_remote_code=True` is actually required (not cargo-culted):** the model repo's `config.json` declares
```json
"auto_map": {
    "AutoConfig": "config.model_config",
    "AutoModel": "ChemPepMTR.MLM_model"
},
"architectures": ["MLM_model"]
```
`MLM_model` is not a class registered in vanilla `transformers` — it only resolves by pulling `ChemPepMTR.py`/`config.py` from the model repo at load time. So this is genuine remote-code execution, not boilerplate; treat it with the same supply-chain caution as any `trust_remote_code=True` load (pin a `revision=<commit sha>` in `from_pretrained` for reproducibility, and don't do this on a machine you don't want executing arbitrary author-supplied Python).

**Tokenizer is actually standard, unlike v1** — `tokenizer_config.json` reports `"tokenizer_class": "PreTrainedTokenizer"` and the repo ships a normal serialized `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`. There is **no `auto_map` entry for `AutoTokenizer`**, meaning the tokenizer is a plain fast tokenizer and — unverified but very likely, see §7 — probably loads fine with `AutoTokenizer.from_pretrained(model_name)` even *without* `trust_remote_code=True`; the flag in the README's example is presumably there for uniformity with the model load, not because the tokenizer needs it. **First cheap thing to check when you actually start implementing:** confirm this, since dropping `trust_remote_code` for the tokenizer call reduces the remote-code attack surface by half.

### 2.2 PeptideCLM v1 — exact steps to de-risk the fallback

```bash
git clone https://github.com/AaronFeller/PeptideCLM.git
cd PeptideCLM
pip install torch transformers datasets SmilesPE pandas
```
```python
import sys
sys.path.append("PeptideCLM")   # path to the cloned repo root

from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
tokenizer = SMILES_SPE_Tokenizer(
    vocab_file="PeptideCLM/tokenizer/new_vocab.txt",
    splits_file="PeptideCLM/tokenizer/new_splits.txt",
)

from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained("aaronfeller/PeptideCLM-23M-all")
# No trust_remote_code needed here — the v1 architecture is a standard
# BERT-family class already registered in transformers; only the
# tokenizer is nonstandard.
```
For our use case you then discard the MLM head, take `model.base_model` (or equivalent, confirm attribute name against the actual `config.json` once loaded) as the encoder, mean-pool, and attach the same `PermeabilizationHead`-style regression wrapper as in the source doc's §5.6 — same as you'd do for v2.

---

## 3. What the paper's benchmarks actually say (grounding, not just the abstract's general claim)

Pulled from the paper via PMC12803269 (treat exact numeric values here as best-effort — see caveat in §7):

- **Confirmed scaling-transition finding** (matches source doc's citation): at 32M params, descriptor-guided pretraining (MTR/hybrid) gives a real advantage on CycPeptMPDB permeability fine-tuning (R²≈0.38 MTR vs. ≈0.24 MLM). At 114M and 337M, "this dependency on the pretraining method became negligible" — MLM catches up.
- **Aggregation (fibrillation, AUROC):** scales cleanly with size regardless of objective — 32M ≈0.69 → 114M ≈0.75 → 337M ≈0.82, vs. a Morgan-fingerprint baseline of 0.58. Bigger unambiguously helps here.
- **Antimicrobial-activity classification (AmpHGT benchmark, MCC) — the closest published proxy to our MIC task, but only reported at the 337M/large tier:**
  - MLM: **0.884 ± 0.013** (best)
  - MTR: 0.853 ± 0.039
  - Hybrid: 0.850 ± 0.014
  - (vs. AmpHGT's own graph-transformer baseline: 0.797)
  - **This is the important counter-signal:** on the one task in the paper that structurally resembles ours, plain **MLM beat hybrid**, inverting the "hybrid wins" heuristic the source doc leaned on. AmpHGT is a binary active/inactive classification task on natural + noncanonical AMPs, explicitly designed to test OOD generalization to noncanonical residues — not MIC regression, but close in spirit.
- **Blood-stability / half-life (PepMSND, MCC, 337M only):** MTR best (0.749) > Hybrid (0.741) > MLM (0.729). Mixed signal again — objective ranking is task-dependent, not universal.
- **No hemolysis, HC50, toxicity, or lysis benchmark exists anywhere in the paper.** This directly reinforces source doc §6.1's domain-shift risk: nothing published tells us how any PeptideCLM-2 variant behaves on membrane-lysis-style labels. We are extrapolating from adjacent tasks, not reading off a result.
- **Fine-tuning protocol used in the paper** (useful as a starting hyperparameter set for our own pilot in §4): full fine-tune (not just frozen-embedding probing, which the paper itself found weak — "R²<0.30 regardless of model scale" for frozen ElasticNet/ExtraTrees probes), LR **1e-5**, batch size **16**, dropout **0.1**, early stopping, max **10 epochs**, 3 random seeds per config. Pretraining itself used 8×H100, global batch 512, 3 pseudo-epochs — irrelevant to us (we're not pretraining), included only for completeness.

**Takeaway:** the paper gives real evidence that (a) objective choice is task-dependent and non-obvious, and (b) our specific tasks are untested. That's the argument for §4's pilot rather than committing to hybrid-small on priors alone.

---

## 4. Concrete pilot benchmarking protocol

**Goal:** pick a variant with 1-2 days of small-scale experimentation instead of a full P2–P4 commitment on a guess.

### 4.1 Candidates to run (5, not all 9 — but see note)
| Candidate | Why included |
|---|---|
| `peptideclm-2-hybrid-small` | Source doc's prior / paper's general small-scale finding |
| `peptideclm-2-mtr-small` | Isolates whether MTR alone (no MLM half) beats hybrid at small scale for our labels |
| `peptideclm-2-mlm-small` | Isolates whether the "descriptor guidance helps at small scale" effect even replicates on our task, vs. plain MLM |
| `peptideclm-2-mlm-large` | The paper's actual best performer on the closest analog task (AMP classification) — the strongest concrete counter-hypothesis to "start small+hybrid" |
| `PeptideCLM-23M-all` (v1) | Fallback baseline, different lineage/tokenizer entirely — costs the §2.2 setup tax but is worth pricing in now rather than after committing to v2 |

If pilot compute is genuinely cheap (see §4.3 — it should be), running the full 3×3 v2 grid instead of this subset of 4 costs little extra and gives a cleaner picture; treat the 4 above as the floor, not the ceiling.

### 4.2 Procedure
1. **Pilot data:** the first fully-cleaned slice of the P0 dataset build (source doc §5.5) — a few hundred to ~2,000 labeled HC50/MIC peptides is enough to rank variants; you don't need the full dataset to see which encoder is in the right ballpark. Reuse the *same* homology/cluster split machinery planned for the real run (source doc §5.4) so the pilot isn't invalidated by leakage the real run won't have.
2. **Baseline:** run the P1 physicochemical-descriptor regressor (source doc §5.5) on the same pilot split first — this is the bar every candidate must clear.
3. **Two passes per candidate**, mirroring the paper's own finding that frozen probing is weak:
   - **Frozen-encoder probe** (cheap, minutes): mean-pooled embeddings → ElasticNet or small MLP head. Useful as a fast filter, but per §3 expect low R² regardless of encoder — don't eliminate a candidate on this alone.
   - **Short full fine-tune** (the real signal): start from the paper's own hyperparameters as defaults — LR 1e-5, batch 16, dropout 0.1, early stopping, cap at 10 epochs — but drop to 1-2 seeds instead of 3 for the pilot to save compute; re-run with 3 seeds only for the eventual winner.
4. **Metrics:** MAE and Spearman ρ per task (HC50, MIC) on the held-out cluster split, identical protocol across every candidate and the P1 baseline.
5. **Decision gate:** pick whichever candidate clears the P1 baseline by the largest margin relative to its compute cost (§4.3), at an absolute performance level you're willing to build P2–P4 on top of. If **nothing** clears the P1 baseline meaningfully, that's a real finding — surface it before sinking more time into the SMILES-LM route (it would mean the P1 baseline in source doc §5.5, or ESM-2, might be the better foundation for this project, at least for a first pass).

### 4.3 Rough compute/VRAM estimates (derived from verified param counts, standard rules of thumb — not measured; see §7)

Using the common estimate of ~16 bytes/param for full-fine-tune mixed-precision AdamW (fp16 weights + fp16 grads + fp32 master weights + fp32 Adam m/v):

| Model | Params | Optimizer+weights footprint | Practical note |
|---|---|---|---|
| `*-small` (any objective) | 31.7M | ~0.5 GB | Trivial — runs on essentially any GPU, even a laptop GPU, batch 16 with room to spare |
| `*-base` | ~114M | ~1.8 GB | Comfortable on any 8–12 GB consumer GPU |
| `*-large` | ~337M | ~5.4 GB | Comfortable on a single 16–24 GB GPU (e.g., RTX 3090/4090, A10, L4); fine even with modest activation overhead at `max_seq_len=2048`, batch 16 |
| `PeptideCLM-23M-all` (v1) | 23M | ~0.4 GB | Same ballpark as v2-small |

Frozen-encoder probing needs only forward-pass (inference) memory — negligible for all variants, effectively free even on CPU for a pilot-sized dataset.

**Conclusion: compute is not the constraint for this pilot or for the real P2–P4 run.** Even the 337M model is a same-day, single-GPU job at our label-count scale (10³–10⁵, per source doc §6.1). This reinforces running the full candidate set in §4.1 rather than trimming it to save GPU time — the actual cost driver is data prep and split construction, not model size.

---

## 5. De-risking the v1 fallback path — summary

Already detailed in full in §1.2 and §2.2; the checklist form:

1. `git clone https://github.com/AaronFeller/PeptideCLM.git` and add it to `sys.path` — the tokenizer is **not** pip-installable or HF-hosted.
2. Load the tokenizer via `SMILES_SPE_Tokenizer(vocab_file, splits_file)` pointing at `tokenizer/new_vocab.txt` and `tokenizer/new_splits.txt` inside the cloned repo.
3. Load the encoder via plain `AutoModelForMaskedLM.from_pretrained("aaronfeller/PeptideCLM-23M-all")` — no `trust_remote_code` needed for the model itself, only the tokenizer requires manual wiring.
4. Build your own regression head exactly as you would for v2 — v1 does not ship one "for free" despite the GitHub README implying a classification head exists (`AutoModelForSequenceClassification`) inconsistent with the model card's `AutoModelForMaskedLM` example.
5. **Accept, don't try to fix:** the released checkpoint was pretrained on the v1.0 (ring-closure-buggy) cyclization data. v1.1 is Zenodo-only, data-only, pretraining-only — there is no corrected checkpoint to substitute in. If cyclic-peptide fidelity from the encoder matters to you, this is a standing, unfixable-without-repretraining caveat on the v1 fallback specifically (v2 was trained on a >100M-molecule corpus assembled later; whether it inherited the same bug class was **not verified** — flagged in §7).

---

## 6. Decision rule

1. **Run the §4 pilot before committing to any variant for P2–P4.** This is the whole point of this document — don't let §0's stated default substitute for the empirical check when the compute cost of checking is this low.
2. **If the pilot shows no separation beyond seed noise** (plausible on a few-hundred-to-2,000-row pilot): default to **`peptideclm-2-hybrid-small`**. It's the cheapest candidate, matches the source doc's original reasoning, and matches the paper's general (if task-mismatched) small-scale finding — when the data can't tell you otherwise, prefer the cheapest option with a plausible prior.
3. **If `base` or `large` clearly beats `small` by more than pilot noise, and stays within your compute comfort** (per §4.3, it almost certainly will): move up a size tier. Compute is not the limiting factor here, so don't let inertia keep you on `small` if a bigger model is measurably better.
4. **If MLM beats hybrid/MTR at whatever size you land on** (a real, paper-precedented outcome for AMP-adjacent tasks — see §3): **go with MLM.** Do not force the hybrid choice on the strength of the paper's general abstract-level claim once your own pilot data disagrees; the paper itself shows the effect is task-dependent, and HC50/MIC were never tested by the original authors.
5. **Escalate to the PeptideCLM v1 fallback only if:**
   - (a) `trust_remote_code=True` loading breaks for **all** v2 variants and can't be fixed quickly (e.g., `ChemPepMTR.py` incompatible with your installed `transformers` version — check this compatibility explicitly, since `custom_code` repos are notoriously version-sensitive and the author has not pinned a tested `transformers` version in the README), **or**
   - (b) the v2 encoder empirically underperforms the P1 physicochemical baseline on the pilot while v1 does not (unlikely, but the v1 pilot run is cheap enough per §4.3 to just check rather than assume).
   - **Do not** fall back to v1 merely because it's "the known quantity" or "already used once in the source doc's framing" — the tokenizer/head-building friction in §5 is real, avoidable engineering cost that v2 doesn't carry, so v1 should only be paid for if v2 is actually broken or actually worse.
6. **Record the decision.** Log the pilot's numbers (MAE/ρ per candidate per task, compute time) in the P0 dataset datasheet or a short standalone model-selection note before starting P2, so whoever picks this back up doesn't have to re-run the pilot to know why a particular variant was chosen.

---

## 7. Open questions / what I could not verify

- **Paper Table 1 numbers are second-hand.** I read them via a fetch tool's parsed summary of the PMC HTML page, not the raw table/PDF. Treat the AUROC/MCC figures in §3 as best-effort transcription — re-pull the actual table from PMC12803269 or the bioRxiv PDF before citing these numbers in anything external-facing. (Architecture dimensions in §1.1, by contrast, I pulled directly from each model's live `config.json` via `curl` — those are first-hand and authoritative; note they conflict with one figure the same PMC summary reported — "6 layers, 384 hidden dim" for `small` — which is simply wrong per the config files. Trust the `config.json` values.)
- **Did not execute a live Python load.** No internet-connected `transformers` install was available in this session to actually run `AutoModel.from_pretrained(..., trust_remote_code=True)` end-to-end. The loading mechanics in §2 are inferred from the README example, the HF API's `auto_map`/`config.json` metadata, and the tokenizer-config `tokenizer_class` field — strong evidence, but not a confirmed successful run. **First concrete task for whoever implements this: literally run the §2.1 snippet once and log the `transformers` version it worked against**, before writing project code that depends on it.
- **No independent/third-party benchmarking of PeptideCLM-2 exists yet** — it's a very recent preprint (Feb–June 2026 revisions), single-lab, no replication found via search.
- **Whether v2's pretraining corpus inherited the same cyclization ring-closure bug as v1's is unverified.** V2's corpus is described as ">100M molecules" assembled later and separately from v1's Zenodo dataset, but I found no explicit statement either confirming it used corrected cyclization generation or ruling out the same class of bug. Worth a direct question to the authors or a spot-check (attempt `RDKit Chem.MolFromSmiles` on a sample of cyclic peptides recoverable from `aaronfeller/peptideclm-2-pretraining-data`) before leaning heavily on v2's cyclic-peptide representations.
- **`aaronfeller/peptideclm-2-pretraining-data` dataset repo** was confirmed to exist (via the collection listing, ~118M rows) but not inspected in depth — size on disk, exact license, and column schema weren't checked.
- **Tokenizer vocab coverage for modified/D-amino-acid SMILES** (relevant to source doc §5.3's "chemical fidelity" concern) was not tested for either v1's SPE vocab or v2's 405-token vocab. Once the DBAASP SMILES conversion (source doc §5.3) produces real examples, run a quick tokenization pass to check for excessive `[UNK]`/fallback-splitting on modified residues before assuming either tokenizer handles them gracefully.
- **Whether `AutoTokenizer.from_pretrained(model_name)` works for v2 *without* `trust_remote_code=True`** (flagged as likely in §2.1 given `tokenizer_class: PreTrainedTokenizer` and no `AutoTokenizer` entry in `auto_map`) was not actually tested — cheap to confirm, worth doing before writing it into project code either way.

---

## Source pointers

- PeptideCLM-2 HF collection: `huggingface.co/collections/aaronfeller/peptideclm-2`
- All PeptideCLM-2 model repos + configs: `huggingface.co/aaronfeller/peptideclm-2-{mlm,mtr,hybrid}-{small,base,large}` (verified via HF API `config.json` fetch, 2026-07-22)
- PeptideCLM-2 pretraining/descriptor data: `huggingface.co/datasets/aaronfeller/peptideclm-2-pretraining-data`
- PeptideCLM-2 GitHub: `github.com/AaronFeller/PeptideCLM-2` (README, `tokenizer/`, `data_processing/`, `training/{00_pretraining,01_regression_benchmarks_training_code,02_classification_benchmarks_training_code,03_PepMSND_training_code}`)
- Paper: Feller, Secor, Swanson, Wilke, Deibler, *"Scaling SMILES-based chemical language models for therapeutic peptide engineering,"* bioRxiv `10.64898/2026.01.06.697994` (v5, June 23 2026); PMC12803269
- PeptideCLM v1 HF repos: `huggingface.co/aaronfeller/PeptideCLM-23M-all`, `-11M-pep`, `-12M-smol`
- PeptideCLM v1 GitHub: `github.com/AaronFeller/PeptideCLM` (README, `tokenizer/`, `example_training_script.py`, `clustered_data/`)
- v1 pretraining dataset + v1.1 correction note: Zenodo `10.5281/zenodo.14194469` (fetched via Zenodo API, 2026-07-22) — confirms v1.1 is data-only, no retrained weights released
- AmpHGT benchmark (antimicrobial-activity classification, cited as PeptideCLM-2's closest published proxy task to our MIC target)
- PepMSND benchmark (blood-stability/half-life, cited for objective-ranking comparison)
