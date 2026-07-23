# Phase 2 Implementation Design: Embedding, Clustering, & Splitting

*Executable companion to `08_implementation_roadmap.md` Phase 2 ("Embedding, Clustering, & Splitting — The Yardstick"), translating `07_task5_splitting_strategy_plan.md` into an actual package addition on top of the merged Phase 1 pipeline: module boundaries, data contracts, CLI, storage layout, and a testing plan. Like `09_phase1_implementation_design.md`, this is a design document, not a finished implementation — every section with a real design choice presents options with a recommendation. This doc is grounded in the actual merged `src/clamp/` layout (not a fictional one): it extends `clamp.data`'s `PeptideRecord`/`pipeline.py`/manifest conventions rather than inventing new ones.*

---

## 0. Scope and non-goals

**In scope** (mirrors doc 08's Phase 2 bullets and doc 07's recommendations):
- Deduplicate the Phase 1 `processed/dataset.parquet` down to one row per unique `peptide_uid` and generate mean-pooled final-layer embeddings from the **base pretrained** PeptideCLM-2 encoder.
- Run joint k-means clustering (k swept 4–6) over the union of every peptide carrying an HC50 label, an MIC label, or both — one clustering pass, not two independent ones.
- Run an independent sequence-identity graph partition (homology cross-check) over the same peptide set.
- Build the actual train/val/test fold assignments (a leave-one-cluster-out, LOCO, rotation) that Phase 4's training loops will consume.
- Validate the resulting folds against QMAP's published leakage reference numbers and log the diagnostics doc 07 §5 calls for.
- Produce a versioned "split" artifact + report, the Phase 2 analogue of Phase 1's datasheet.

**Explicitly out of scope for this doc:**
- Any model training, heads, or losses (Phase 4, doc 04).
- The model-selection pilot itself (Phase 3, doc 05) — Phase 2 produces the folds Phase 3 evaluates candidates *on*, but doc 11 owns the pilot grid, frozen-probe/fine-tune protocol, and architecture decision.
- Re-pulling or re-normalizing source data — this doc consumes `data/processed/dataset.parquet` as a read-only input and never modifies Phase 1's pipeline.
- Building the cyclic-holdout ablation as a first-class, permanently-wired pipeline stage (doc 07 §4) — the clustering/partition machinery below makes it *possible* (§9), but running it as a formal ablation is deferred to whenever Phase 5's ablation work actually needs it, since doc 07 §4.2 flags the cyclic-fraction number as unmeasured until real data exists.

---

## 1. Relationship to Phase 1's output — the one load-bearing contract

Everything in this doc reads `data/processed/dataset.parquet` (Phase 1's Definition-of-Done deliverable) and nothing else. Concretely, this doc's code depends on exactly these columns existing and meaning what `src/clamp/data/schema.py`'s `PeptideRecord` says they mean: `peptide_uid`, `smiles`, `chemical_fidelity_tier`, `sequence_canonical`, `is_cyclic`, `cyclization_type`, `hc50_value_uM`/`hc50_log_uM`, `mic_value_uM`/`mic_log_uM`.

One property of Phase 1's `dedup.py` that this doc relies on and is worth stating explicitly since it isn't obvious from the schema alone (verified by reading `dedup.py` directly, not assumed): **every row sharing a `peptide_uid` already carries an identical `smiles` value** — `dedup_records` propagates the metadata-richest record's `smiles`/`smiles_source`/`chemical_fidelity_tier` onto every other row in the group rather than leaving them as originally converted. This is exactly what makes "group by `peptide_uid`, take one row" a safe way to get a canonical SMILES per peptide for embedding — there is no need to re-resolve conflicting SMILES here, Phase 1 already did it.

**Rows excluded before anything else in this doc runs:** any row with `chemical_fidelity_tier == FAILED` (no usable SMILES was ever generated — nothing to embed) and any row where both `hc50_value_uM` and `mic_value_uM` are null (not part of the "union of peptides with either label" this phase's clustering is scoped to, per doc 07 §3.3). Log both exclusion counts in the split report (§8) — they're exactly the kind of number that should be visible, not silently dropped.

---

## 2. Package layout

**Recommendation: a new `src/clamp/splitting/` subpackage**, sibling to `src/clamp/data/`, plus a small shared-infrastructure extraction from `data/pipeline.py`.

```
src/clamp/
├── config.py                       # extended, not replaced — see §5
├── cli.py                          # unchanged (clamp-data)
├── pipeline_utils.py                # NEW — Stage/manifest runner, factored out of data/pipeline.py
├── data/                            # unchanged (Phase 1)
│   └── pipeline.py                  # updated to import Stage/run_pipeline from pipeline_utils
└── splitting/                       # NEW (this doc)
    ├── __init__.py
    ├── cli.py                       # typer app, entry point `clamp-split`
    ├── schema.py                    # PeptideEmbedding, ClusterAssignment, FoldAssignment, etc. (§4)
    ├── embed.py                     # encoder loading + mask-aware mean pooling (§6)
    ├── cluster.py                   # PCA + k-means sweep + selection (§7)
    ├── homology.py                  # sequence-identity graph partition (§8)
    ├── folds.py                     # LOCO fold construction, per-task filtering (§9)
    ├── validate.py                  # leakage diagnostics (§10)
    ├── report.py                    # split report, mirrors data/datasheet.py's split (§11)
    └── pipeline.py                  # orchestrator (§12)
```

```
tests/
└── splitting/
    ├── __init__.py
    ├── fixtures/                    # tiny synthetic embeddings + sequences, not live model output
    ├── test_embed.py
    ├── test_cluster.py
    ├── test_homology.py
    ├── test_folds.py
    ├── test_validate.py
    └── test_report.py
```

**Why a new subpackage rather than growing `clamp.data`:** Phase 1's `data/` package is about *acquiring and cleaning records*; this phase is about *deriving a fixed evaluation geometry* from an already-finished dataset — a genuinely different concern with different dependencies (torch/transformers/scikit-learn vs. requests/rdkit/pydantic) and a different rerun cadence (Phase 1 reruns when a source changes; this phase reruns only when the dataset or the split protocol changes). Keeping them separate also means Phase 3/4 can depend on `clamp.splitting` without pulling in the source-puller machinery, and vice versa.

**Why extract `pipeline_utils.py` now:** `data/pipeline.py`'s `Stage`/manifest-cache pattern (doc 09 §10) is exactly the right shape for this phase's own linear DAG (embed → cluster → homology-partition → agreement-check → build-folds → validate). Rather than copy-pasting `Stage`, `_content_hash`, `_load_manifest`/`_save_manifest`, and `run_pipeline` into a second module, factor the generic (non-Phase-1-specific) parts out of `clamp/data/pipeline.py` into `clamp/pipeline_utils.py` once, and have both `clamp.data.pipeline` and `clamp.splitting.pipeline` import from there. This is a small, mechanical refactor of already-merged Phase 1 code — call it out explicitly as the one piece of this doc that touches existing files, not just adds new ones.

---

## 3. New dependencies

```toml
dependencies = [
    # ... existing Phase 1 deps unchanged ...
    "torch>=2.2",
    "transformers>=4.40",
    "scikit-learn>=1.4",       # PCA, KMeans, silhouette/DB/CH scores, ElasticNet (validate.py's threshold probe)
    "biopython>=1.83",         # Bio.Align.PairwiseAligner (Needleman-Wunsch + BLOSUM45), for pairwise identity
    "graph-part>=1.0",         # primary homology-partition tool (doc 07 §3.2)
    "python-igraph>=0.11",     # Leiden fallback, if graph-part degenerates (§8.3)
    "leidenalg>=0.10",         # ditto
]
```

`transformers`/`torch` are new, real, load-bearing dependencies from this phase onward — Phase 1 had none. `graph-part` is pip-installable per doc 07's own verification (`pip install graph-part`). `biopython`'s `Bio.Align.PairwiseAligner` supports affine-gap global (Needleman-Wunsch-mode) alignment with an arbitrary substitution matrix including BLOSUM45 — a lighter dependency than reimplementing QMAP's Rust `pwiden` engine, and adequate at this project's scale (hundreds–low-thousands of peptides means low-thousands-squared pairwise comparisons for the leakage check, not QMAP's DBAASP-scale problem).

---

## 4. Data model (`splitting/schema.py`)

```python
from enum import StrEnum
from pydantic import BaseModel

class EmbeddingModel(StrEnum):
    HYBRID_SMALL = "aaronfeller/peptideclm-2-hybrid-small"

class PeptideEmbedding(BaseModel):
    peptide_uid: str
    embedding: list[float]           # fixed length = encoder embed_dim (512 for *-small)
    model_name: str
    model_revision: str               # pinned commit sha — never "main" (see §6.4)
    pooling: str = "mean"


class ClusterAssignment(BaseModel):
    peptide_uid: str
    k: int                            # the k actually selected (§7.3), same for every row
    cluster_id: int
    pca_components: int


class PartitionMethod(StrEnum):
    GRAPH_PART = "graph_part"
    LEIDEN_FALLBACK = "leiden_fallback"

class HomologyCommunity(BaseModel):
    peptide_uid: str
    community_id: int
    method: PartitionMethod
    identity_threshold: float


class Role(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class FoldAssignment(BaseModel):
    fold_id: int                      # which LOCO rotation (0..k-1)
    peptide_uid: str
    role: Role
    cluster_id: int                   # the cluster this peptide_uid belongs to (constant across folds)
```

Same design rationale as doc 09 §3: pydantic models validated at construction, `model_dump()` for the parquet round-trip. One deliberate asymmetry from `PeptideRecord`: `FoldAssignment` is **peptide-level**, not assay-record-level. A peptide with three MIC replicate rows in `dataset.parquet` gets one `cluster_id` and one `role` per fold — `folds.py` (§9) is responsible for the join back onto the per-assay-record rows, not for tracking fold membership per row.

---

## 5. Config additions

Extend the existing `clamp.config.Settings` (do not create a parallel settings object — one validated settings surface per doc 09 §4's original reasoning still applies):

```python
class Settings(BaseSettings):
    # ... existing Phase 1 fields unchanged ...

    embedding_model: str = "aaronfeller/peptideclm-2-hybrid-small"
    embedding_model_revision: str = ""     # MUST be set to a real pinned sha before first real run (§6.4)
    embedding_batch_size: int = 32
    embedding_device: str = "cuda"          # falls back to cpu at runtime if unavailable

    kmeans_k_min: int = 4
    kmeans_k_max: int = 6
    kmeans_pca_variance: float = 0.99
    kmeans_min_cluster_size: int = 40       # doc 07 §3.4's floor, applied against the HC50-labeled subset
    kmeans_random_state_search: int = 20    # number of seeds tried per k, per Feller & Wilke's own notebook

    homology_identity_threshold: float = 0.60   # doc 07 §2.4 default — re-derive before trusting (§8.2)
    homology_gap_open: float = -5.0
    homology_gap_extend: float = -1.0

    @property
    def embeddings_dir(self) -> Path:
        return self.data_root / "embeddings"

    @property
    def splits_dir(self) -> Path:
        return self.data_root / "splits"
```

Leaving `embedding_model_revision` defaulted to `""` rather than a guessed sha is deliberate: `embed.py` (§6.4) should refuse to run against an unpinned revision, forcing whoever runs this for the first time to resolve and record a real commit sha, exactly as doc 05 §7's "first concrete task" flags for Phase 3. Don't paper over that with a plausible-looking fake default.

---

## 6. Module: `splitting/embed.py`

### 6.1 Loading the encoder

```python
from transformers import AutoTokenizer, AutoModel
import torch

def load_encoder(model_name: str, revision: str, device: str) -> tuple:
    if not revision:
        raise ValueError(
            "embedding_model_revision is unset — resolve and pin a real commit "
            "sha before running embed (see §6.4); refusing to load `main`."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(
        model_name, revision=revision, trust_remote_code=True, use_safetensors=True
    )
    model.to(device).eval()
    return tokenizer, model
```

`trust_remote_code=True` is required for the model (PeptideCLM-2's `config.json` declares a custom `auto_map` resolving to `ChemPepMTR.MLM_model`, confirmed directly against the live config per doc 05 §2.1) but **not** attempted for the tokenizer — doc 05 §2.1 found the tokenizer repo metadata reports a plain `PreTrainedTokenizer` with no `AutoTokenizer` `auto_map` entry, so the tokenizer call omits the flag to minimize remote-code execution surface. This is one of doc 05 §7's "cheap to confirm, worth doing before writing it into project code" items — confirm it loads without the flag the first time this is actually run, and note the result in the split report (§11) rather than assuming it silently.

### 6.2 Mask-aware mean pooling

```python
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean over real (non-pad) tokens only — matches PeptideCLM-2's own
    documented convention (doc 07 §3.1: 'final-layer mean-pooled
    embeddings', 'mean-pooled across the sequence dimension'), not the
    CLS-token convention v1/Feller & Wilke's clustering notebook used."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts
```

**Forward-compat note, not a requirement of this doc:** this is the same mask-aware mean-pooling operation `04_task5_multitask_architecture_plan.md` plans to build as a shared model component for Phase 4's heads. Keep `mean_pool`'s signature (`last_hidden_state`, `attention_mask` → pooled tensor) generic enough that Phase 4 can import it directly from `clamp.splitting.embed` rather than reimplementing it, but don't block this doc on that — if Phase 4's needs turn out to differ (e.g. needing gradients flowing through, which this function already supports since it's plain tensor ops with no `torch.no_grad()` baked in), it can be lifted into a shared `clamp.nn` module then.

### 6.3 Batch embedding

```python
def embed_smiles(
    smiles: list[str], tokenizer, model, device: str, batch_size: int
) -> np.ndarray:
    """Returns an (n, embed_dim) array, one row per input SMILES, in input order."""
    ...  # tokenize in batches, forward pass under torch.no_grad(), mean_pool, .cpu().numpy(), concatenate

def embed_dataset(dataset_path: Path, settings: Settings) -> "EmbeddingReport":
    """Reads processed/dataset.parquet, dedupes to unique (peptide_uid, smiles)
    per §1's excluded-rows rule, embeds, writes embeddings/peptide_embeddings.parquet."""
    ...
```

### 6.4 Pinning the model revision — a blocking runbook step, not a footnote

Per doc 05 §7's most concrete open item: nobody has actually executed `AutoModel.from_pretrained(..., trust_remote_code=True)` against this checkpoint end-to-end yet. **Runbook step 1 (§13) is: load the model once, confirm it works, resolve `main` to a real commit sha via the HF API (`huggingface_hub.HfApi().model_info(model_name).sha` or the web UI's commit history), and set `CLAMP_EMBEDDING_MODEL_REVISION` (or `embedding_model_revision` in `.env`) to that sha before running `embed` for real.** This is exactly the same supply-chain caution doc 05 §2.1 already calls for (`trust_remote_code=True` is genuine remote code execution) applied concretely: pin, don't float.

---

## 7. Module: `splitting/cluster.py`

### 7.1 Dimensionality reduction

```python
from sklearn.decomposition import PCA

def reduce_dimensionality(embeddings: np.ndarray, variance: float) -> np.ndarray:
    """PCA to the minimum number of components retaining `variance` fraction
    of variance — Feller & Wilke's exact threshold (doc 07 §1.2), reused
    verbatim since it's model-agnostic. Expect far fewer than their 153
    components at this project's smaller, less chemically diverse scale —
    log the actual number selected, don't hardcode an expectation."""
    pca = PCA(n_components=variance, svd_solver="full")
    return pca.fit_transform(embeddings)
```

### 7.2 K-means sweep and selection

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class KSweepResult(BaseModel):
    k: int
    random_state: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    cluster_sizes: list[int]
    min_hc50_labeled_cluster_size: int   # the binding floor, see §7.3

def sweep_k(
    reduced: np.ndarray,
    peptide_uids: list[str],
    hc50_labeled_uids: set[str],
    k_min: int,
    k_max: int,
    n_random_states: int,
) -> list[KSweepResult]:
    """For each k in [k_min, k_max] and each of n_random_states seeds, fit
    KMeans, score it, and record the smallest resulting cluster's count of
    HC50-labeled peptides specifically (not just smallest cluster overall —
    see §7.3 for why that's the number that matters)."""
    ...

def select_k(results: list[KSweepResult], min_cluster_size: int) -> KSweepResult:
    """Joint silhouette/DB/CH selection (doc 07 §1.2's rule, reused
    verbatim), restricted to candidates whose min_hc50_labeled_cluster_size
    clears `min_cluster_size` — reject a k/seed combination on cluster-size
    grounds before ranking survivors by cluster-quality metrics, exactly as
    doc 07 §3.4 specifies."""
    ...
```

### 7.3 Resolving doc 07's k-range ambiguity — one joint k, not per-task k ranges

Doc 07 contains two statements about `k` that are in tension if read independently: §3.3 requires **one shared clustering pass over the union** of HC50/MIC peptides (a hard requirement — a peptide's cluster membership must be identical regardless of which head uses it, per the shared-encoder leakage argument), while §3.4 separately suggests "k = 4–5 for HC50, k = 5–6 for MIC" as if they could be tuned independently. **This doc resolves the tension in favor of §3.3, which is the harder, non-negotiable requirement:** there is exactly **one** k-means run, over the joint peptide set, producing exactly one `cluster_id` per `peptide_uid`. The §3.4 "smaller k for HC50" concern is real but gets satisfied a different way: `select_k` above rejects any candidate k/seed whose smallest cluster's **HC50-labeled peptide count** (not its raw peptide count) falls below `kmeans_min_cluster_size` — since HC50 is the smaller, more fragile label set (doc 07 §2.1: as few as ~800 peptides plausible), this is the binding constraint that naturally biases the selected k downward without needing two separate clustering runs. Report the actual selected k and every candidate's scores in the split report (§11) so this isn't an invisible decision.

---

## 8. Module: `splitting/homology.py`

### 8.1 Pairwise identity

```python
from Bio.Align import PairwiseAligner, substitution_matrices

def build_aligner(gap_open: float, gap_extend: float) -> PairwiseAligner:
    aligner = PairwiseAligner(mode="global")   # global = Needleman-Wunsch
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM45")
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend
    return aligner

def percent_identity(seq_a: str, seq_b: str, aligner: PairwiseAligner) -> float:
    """Matched positions / alignment length of the best global alignment —
    same definition QMAP used (doc 07 §2.2), for direct comparability of
    the reference numbers in §10."""
    ...
```

### 8.2 Threshold — don't blindly reuse QMAP's 60%

Doc 07 §3.2 is explicit that QMAP's 60% threshold is dataset-specific (derived from *their* PeptideAtlas-vs-DBAASP 99th-percentile reference) and should be re-derived, not assumed. **Concrete plan:** compute the same PeptideAtlas-vs-this-project's-dataset max-identity 99th percentile once real data exists (a small standalone script/notebook, not part of the automated pipeline, since PeptideAtlas is a one-time external reference pull, not a rerun-per-change dependency), and set `homology_identity_threshold` from that measurement. **Until that measurement exists, `homology_identity_threshold` defaults to QMAP's 60% as a documented placeholder** — flag this explicitly in the split report (§11) as "using QMAP's threshold, not a re-derived one" so nobody mistakes the placeholder for a validated number.

### 8.3 Partitioning — GraphPart first, Leiden fallback if it degenerates

```python
class CommunityResult(BaseModel):
    communities: dict[str, int]        # peptide_uid -> community_id
    method: PartitionMethod
    test_fraction_achieved: float

def partition_graph_part(
    sequences: dict[str, str], threshold: float
) -> CommunityResult:
    """Wraps the `graph_part` CLI/API. QMAP's own experience (doc 07 §2.3)
    is a direct, concrete warning: vanilla GraphPart applied to a
    DBAASP-derived set produced an EMPTY test set because the similarity
    structure was too dense for strict partitioning. Detect that failure
    mode explicitly (test_fraction_achieved far below the ~20% target, or
    zero) rather than silently accepting a degenerate partition."""
    ...

def partition_leiden_fallback(
    sequences: dict[str, str], threshold: float, test_fraction: float = 0.20, seed: int = 0
) -> CommunityResult:
    """Reimplements QMAP's own fix (doc 07 §2.3): build the similarity
    graph directly, run Leiden community detection (`leidenalg`,
    modularity-based), then randomly assign whole communities to the test
    set until test_fraction is reached, then post-filter any remaining
    training sequence whose identity to any test sequence exceeds
    `threshold`. Only invoked if partition_graph_part degenerates."""
    ...

def run_homology_partition(sequences: dict[str, str], settings: Settings) -> CommunityResult:
    result = partition_graph_part(sequences, settings.homology_identity_threshold)
    if result.test_fraction_achieved < 0.10:   # degenerate — GraphPart's own failure mode per §2.3
        result = partition_leiden_fallback(sequences, settings.homology_identity_threshold)
    return result
```

### 8.4 What the homology partition is used for — resolving doc 08 vs. doc 07's framing

There is a second real tension worth resolving explicitly rather than picking one source silently. Doc 08's roadmap phrasing frames the homology partition as strictly a **cross-check** against the embedding clusters ("Execute a sequence-identity graph partition... as an independent check against your embedding clusters"). Doc 07 §3.2, read on its own, argues the homology partition should actually be the **final, authoritative** split ("Recommend the final split be the sequence-identity-graph partition... with the embedding-cluster split reported as a secondary/reported check"), because its leakage guarantee is directly measurable by construction while an embedding cluster's is not.

**This doc adopts doc 08's simpler framing as the default, with doc 07's stronger recommendation wired in as an explicit, triggered fallback — not silently dropped:**
- **Default:** the embedding-based k-means clusters (§7) are what `folds.py` (§9) turns into the actual LOCO train/val/test rotation Phase 3/4 train and evaluate on. This is cheaper (one clustering pass vs. building a second full partition-based split), matches doc 08's literal roadmap deliverable list, and is what the rest of this project's design docs (04's shared-encoder architecture, 05's pilot) are written assuming exists.
- **Homology partition's role:** an independently-derived artifact used exclusively for §10's leakage validation (the max-identity(test→train) diagnostic and the cluster-vs-community agreement check) — it does not itself feed `folds.py` under normal operation.
- **The escalation trigger:** if §10's validation shows the embedding-cluster LOCO folds fail the leakage bar (max-identity(test→train) distribution is not comparably low to QMAP's reference range, §10.1) — the split-selection gate in the Definition of Done (§14) is **not met**, and the documented next step is exactly doc 07 §3.2's original recommendation: re-derive folds from the homology partition instead of the k-means clusters, or re-roll the k-means seed with the leakage diagnostic added as a secondary selection criterion (§7.3 already adds a size-floor criterion; this would add a leakage-floor criterion the same way). `folds.py` (§9) is written to accept **either** a `ClusterAssignment` table or a `HomologyCommunity` table as its partition input for exactly this reason — the fallback is a config change, not a rewrite.

---

## 9. Module: `splitting/folds.py`

```python
def build_loco_folds(
    partition: pd.DataFrame,       # peptide_uid -> cluster_id (or community_id), one row per peptide
    validation_strategy: str = "rotate",   # "rotate" or "fixed_smallest", doc 07 §3.4/§3.5
) -> list[FoldAssignment]:
    """One rotation per distinct cluster_id: that cluster is TEST, one other
    cluster is VAL (per validation_strategy), the rest are TRAIN. This is
    the 'cheaper hybrid' from doc 07 §3.5 — outer LOCO rotation kept (the
    leakage-critical part), inner 5-fold CV replaced by a single fixed
    validation cluster, cutting compute from up to 30 runs to k runs."""
    ...

def filter_for_task(
    fold_assignments: list[FoldAssignment], dataset: pd.DataFrame, task: Literal["hc50", "mic"]
) -> pd.DataFrame:
    """Joins fold_assignments onto dataset.parquet by peptide_uid, filtered
    to rows where this task's *_value_uM is non-null. A peptide's role
    (train/val/test) is identical across tasks by construction — this
    function only ever subsets rows, never reassigns a role."""
    ...
```

`validation_strategy` defaults to `"rotate"` (each non-test cluster gets a turn as validation across the outer rotations, matching Feller & Wilke's own notebook pattern per doc 07 §3.5) with `"fixed_smallest"` (always use the smallest remaining cluster as validation) available as the doc 07 §3.4 fallback for when HC50's small n makes sacrificing a whole cluster to validation too costly — implement both from day one since doc 07 explicitly flags this as "a real trade-off to make explicitly, not silently default on," not a hypothetical to add later.

---

## 10. Module: `splitting/validate.py`

Direct implementation of doc 07 §5's five checks, each a standalone function so the split report (§11) can run and log all five independently rather than one monolithic "validate" black box:

```python
def max_identity_test_to_train(
    fold_assignments: list[FoldAssignment], sequences: dict[str, str], aligner: PairwiseAligner
) -> pd.DataFrame:
    """Per fold, per test peptide: max percent_identity against every train
    peptide in that fold. Returns long-form (fold_id, peptide_uid, max_identity).
    Report median/90th-pctile per fold and compare against QMAP's own
    published reference (homology split: median ~50%, 90th pctile ~57%;
    random split: median ~90%, 90th pctile ~100%) — doc 07 §5 point 1."""
    ...

def embedding_nn_distance(
    fold_assignments: list[FoldAssignment], embeddings: pd.DataFrame
) -> pd.DataFrame:
    """Per test peptide, nearest-neighbor distance to any train peptide in
    embedding space; compare within-cluster vs. cross-cluster NN distance
    distributions. Doc 07 §5 point 2 — cheap since embeddings already exist."""
    ...

def threshold_sensitivity_probe(
    dataset: pd.DataFrame, embeddings: pd.DataFrame, sequences: dict[str, str],
    task: Literal["hc50", "mic"], thresholds: list[float] = [1.0, 0.8, 0.6, 0.4],
) -> pd.DataFrame:
    """ElasticNet on frozen embeddings, re-split at each identity threshold
    (peptides within `threshold` identity of any test peptide excluded from
    train), confirm apparent PCC/Spearman rho rises as threshold rises —
    doc 07 §5 point 3, QMAP's Fig. 3A reproduced on this project's own data.
    NOTE: this frozen-probe machinery is the same shape as doc 11 (Phase 3)
    §4's frozen-encoder probe — reuse doc 11's probe function here rather
    than writing a second one, once Phase 3 exists; for Phase 2 alone, an
    ElasticNet-only inline implementation is enough."""
    ...

def label_balance_check(fold_assignments: list[FoldAssignment], dataset: pd.DataFrame) -> pd.DataFrame:
    """Per fold, per task: mean/variance/min/max of the log-transformed
    label across train/val/test. Doc 07 §5 point 4 — flags a cluster that
    accidentally concentrates extreme values."""
    ...

def composition_check(fold_assignments: list[FoldAssignment], dataset: pd.DataFrame) -> pd.DataFrame:
    """Per fold: fraction cyclic, fraction dual-labeled (has both HC50 and
    MIC). Doc 07 §5 point 5."""
    ...
```

**Agreement diagnostic** (doc 07 §3.2, feeding the §8.4 escalation decision): compute the Adjusted Rand Index between the k-means `cluster_id` labels and the homology `community_id` labels over the shared peptide set. High agreement is reassuring; low agreement is reported, not treated as a bug per se (embedding clusters can legitimately separate on chemistry axes sequence identity doesn't capture) — but it is the first thing to check if §10's leakage numbers come back bad.

---

## 11. Module: `splitting/report.py`

Same shape as Phase 1's `datasheet.py` (doc 09 §9): a reporting module, not a transform one. Emits both a machine-readable `splits/split_report.json` and a human-readable `splits/split_report.md` from the same underlying numbers — don't hand-author one separately from the other. Contents, directly enumerable from the sections above:
- Peptide counts: total union, excluded (`FAILED` fidelity / no label), embedded.
- The full k-sweep table (§7.2) and which k/seed was selected and why (§7.3).
- Per-fold sizes and train/val/test proportions (doc 07 §3.4's "report actual sizes, don't assume balance").
- The homology partition's method used (GraphPart vs. Leiden fallback, §8.3) and the identity threshold used, flagged as placeholder-vs-re-derived (§8.2).
- The ARI agreement number between embedding clusters and homology communities.
- All five §10 validation tables, with an explicit **pass/fail** read against the DoD gate in §14.
- Whether the escalation trigger in §8.4 fired, and if so, which fallback path was taken.

---

## 12. Orchestration (`splitting/pipeline.py`)

Same manifest-file pattern as Phase 1 (doc 09 §10), via the shared `pipeline_utils.py` (§2):

```python
STAGES: list[Stage] = [
    Stage("embed", _embed, inputs=["processed/dataset.parquet"], output="embeddings/peptide_embeddings.parquet"),
    Stage("cluster", _cluster, inputs=["embeddings/peptide_embeddings.parquet"], output="splits/cluster_assignments.parquet"),
    Stage("homology_partition", _homology, inputs=["processed/dataset.parquet"], output="splits/homology_communities.parquet"),
    Stage("build_folds", _build_folds, inputs=["splits/cluster_assignments.parquet"], output="splits/folds/"),
    Stage("validate", _validate, inputs=["splits/folds/", "splits/homology_communities.parquet"], output="splits/split_report.json"),
]
```

`homology_partition` has no dependency on `cluster` (they're independent, run-in-either-order artifacts per §8.4) but `validate` depends on both — the manifest's content-hash mechanism (unchanged from doc 09 §10) handles this correctly since it hashes whatever's actually on disk, not a hardcoded assumption about stage order.

---

## 13. CLI (`splitting/cli.py`, entry point `clamp-split`)

```toml
[project.scripts]
clamp-data = "clamp.cli:app"
clamp-split = "clamp.splitting.cli:app"
```

```python
app = typer.Typer()

@app.command()
def embed(force: bool = False): ...

@app.command()
def cluster(force: bool = False): ...

@app.command(name="homology-partition")
def homology_partition(force: bool = False): ...

@app.command(name="build-folds")
def build_folds(force: bool = False): ...

@app.command()
def validate() -> None:
    """Prints the pass/fail read against the §14 DoD gate to stdout in
    addition to writing split_report.{json,md}."""
    ...

@app.command(name="run-all")
def run_all(force: bool = False): ...
```

A separate entry point from `clamp-data` (rather than adding subcommands to the existing app) mirrors §2's package-boundary reasoning: distinct dependency footprint, distinct rerun cadence, distinct "am I done" question (Phase 1 asks "is the dataset built," this asks "is the split safe").

---

## 14. Testing plan

| Target | Approach | Why this one |
|---|---|---|
| `embed.mean_pool` | Pure-tensor unit tests: known `last_hidden_state`/`attention_mask` pairs with hand-computed expected output, including a padded-batch case | This is the one place a padding bug silently produces wrong embeddings for every downstream stage — worth locking down independent of any real model. |
| `embed.embed_smiles`/`embed_dataset` | **Not** unit-tested against the live HF model in CI — mock `load_encoder` to return a tiny deterministic fake `nn.Module` (fixed-size random projection) so the batching/dedup/output-shape logic is tested without a network call or GPU | Same philosophy as doc 09 §12's DBAASP-pull mocking: never hit the live network in CI. |
| `cluster.reduce_dimensionality`, `sweep_k`, `select_k` | Pure `pytest` unit tests on small synthetic embedding arrays with known cluster structure (e.g. 3 well-separated Gaussian blobs) — confirm `select_k` correctly rejects a candidate whose smallest HC50-labeled cluster falls below the floor | Pure logic, cheap, and this is exactly the §7.3 tie-break decision worth a regression test — construct a synthetic case where the naively-best-scoring k has an undersized HC50 cluster and confirm it's rejected in favor of the next-best. |
| `homology.percent_identity` | Parametrized cases against known alignments (identical sequences → 100%, a hand-computed short-sequence pair) | Small pure function, correctness matters for every downstream leakage number. |
| `homology.partition_graph_part`/`partition_leiden_fallback`/`run_homology_partition` | Unit tests on tiny synthetic sequence sets, including a deliberately-dense synthetic set constructed to reproduce GraphPart's empty-test-set failure mode, confirming the fallback actually triggers | Directly tests the §8.3 escalation logic doc 07 warns is a real risk, not just the happy path. |
| `folds.build_loco_folds` | Unit tests on synthetic cluster assignments: confirm every cluster gets exactly one turn as test, `"rotate"` vs `"fixed_smallest"` validation strategies behave as documented, and every `peptide_uid` gets exactly one role per fold | This is the leakage-critical bookkeeping — a peptide silently appearing in both train and test within the same fold is the single worst bug this module could ship. |
| `folds.filter_for_task` | Unit test confirming a peptide's role is identical across the HC50-filtered and MIC-filtered views | Directly tests doc 07 §3.3's "identical role no matter which head uses it" requirement. |
| `validate.*` | Unit tests on small synthetic fold assignments + sequences/embeddings with known, hand-computable expected diagnostics (e.g. a synthetic case with a known max-identity value) | Confirms the diagnostic math itself is right before trusting it on real data. |
| `report.py` | Snapshot-style test: given fixed synthetic inputs, confirm `split_report.json`'s and `split_report.md`'s numbers agree with each other | Same drift-prevention rationale as doc 09 §9's datasheet. |

**Fixtures:** small synthetic embedding arrays and synthetic peptide sequences (10–30 peptides, hand-constructed cluster/community structure), **not** captured real-model output — unlike Phase 1's fixtures (real API responses), there's no equivalent "capture once, reuse" story for live model embeddings, since the whole point of these tests is to validate the logic independent of which encoder produced the numbers.

**Explicitly not covered by this test plan:** the real embedding run against the actual pulled dataset through the actual PeptideCLM-2 checkpoint, and the real GraphPart/Leiden partition against actual peptide sequences at real scale. Those need `clamp-split run-all` actually executed once against live data (§15) — no unit test substitutes for seeing the real k-sweep table or the real max-identity distribution.

---

## 15. Storage layout

```
data/
├── embeddings/
│   └── peptide_embeddings.parquet     # peptide_uid, embedding (list<float>), model_name, model_revision, pooling
├── splits/
│   ├── cluster_assignments.parquet    # peptide_uid, k, cluster_id, pca_components
│   ├── cluster_sweep.json             # every (k, random_state) candidate's scores — the full §7.2 table
│   ├── homology_communities.parquet   # peptide_uid, community_id, method, identity_threshold
│   ├── folds/
│   │   └── fold_{i}.parquet           # fold_id, peptide_uid, role, cluster_id — one file per LOCO rotation
│   ├── agreement_report.json          # ARI between cluster_id and community_id
│   └── split_report.{json,md}         # §11
```

All new, gitignored, same convention as Phase 1's `data/` tree (doc 09 §13). Parquet for the same dtype-preservation reasons doc 09 §13 gives, with one addition worth calling out: `embedding` as a `list<float>` parquet column is a fixed-length (512-element, for `*-small`) list per row — both pandas and pyarrow handle this natively, no need to explode into 512 separate columns.

---

## 16. Runbook (execution order for Phase 2)

1. **Resolve and pin `embedding_model_revision`** (§6.4) — load the model once by hand, confirm `trust_remote_code=True` works against your installed `transformers` version, log the version that worked, resolve `main` to a commit sha, set it in `.env`. Blocking, do this before step 3.
2. `uv sync` after adding the §3 dependencies.
3. `clamp-split embed` — embeds every unique, non-`FAILED`, at-least-one-label peptide. Expect low thousands of peptides at most (doc 07 §2.1) — this is a same-session job, not a multi-hour pull like Phase 1's DBAASP puller.
4. `clamp-split cluster` — runs the k-sweep, selects k, writes `cluster_assignments.parquet` + `cluster_sweep.json`. **Inspect `cluster_sweep.json` by hand** before proceeding — confirm the selected k's smallest HC50-labeled cluster genuinely clears the floor, don't just trust the automation silently.
5. `clamp-split homology-partition` — independent of step 4, can run before or after.
6. `clamp-split build-folds` — turns the (default: embedding-cluster) partition into the actual LOCO fold files.
7. `clamp-split validate` — runs all five §10 diagnostics plus the ARI agreement check, writes `split_report.{json,md}`, prints the pass/fail DoD read.
8. **Read `split_report.md` by hand.** If the leakage-bar check fails, follow the §8.4 escalation path (re-derive folds from the homology partition, or re-roll the k-means seed with a leakage-floor criterion added) and re-run from step 6 or step 4 respectively — do not proceed to Phase 3 on a failed validation gate.

Equivalently, `clamp-split run-all` executes 3–7 in order; running individually the first time is preferable so a failure in `validate` doesn't obscure whether `embed` actually produced sane vectors.

---

## 17. Definition of done (Phase 2)

- [ ] `embeddings/peptide_embeddings.parquet` contains one row per unique, non-`FAILED`, at-least-one-label `peptide_uid`, embedded via a **pinned** model revision (not `main`).
- [ ] `splits/cluster_assignments.parquet` reflects exactly one joint k-means run (not per-task runs) over the union of HC50/MIC peptides, with the selected k's rationale (scores, size-floor check) recorded in `cluster_sweep.json`.
- [ ] `splits/homology_communities.parquet` exists, generated via GraphPart or the documented Leiden fallback, with the threshold used explicitly flagged as placeholder-vs-re-derived.
- [ ] `splits/folds/fold_{i}.parquet` exist for every k rotation, and `folds.filter_for_task` produces per-task train/val/test sets where every peptide's role is identical regardless of task.
- [ ] `split_report.{json,md}` contains all five §10 diagnostics plus the ARI agreement number, and has been **read** — not just generated — by whoever runs this, per the same "a report nobody reads doesn't close anything" standard doc 09 §15 held Phase 1's datasheet to.
- [ ] The max-identity(test→train) distribution has been explicitly compared against QMAP's reference numbers (median ≈50%, 90th pctile ≈57% for a homology-safe split; median ≈90%, 90th pctile ≈100% for a leaky one) and the result — pass or fail — is recorded, not implied.
- [ ] If the leakage bar failed, the §8.4 escalation path was actually taken (not just noted as an option) and the resulting folds re-validated.
- [ ] `pytest` passes for the full `tests/splitting/` suite in §14.

---

## 18. Open risks carried into implementation

Restated from doc 07 as concrete things that can break this specific scaffold:

1. **GraphPart may degenerate on this project's own data, not just QMAP's DBAASP-scale data** — §8.3's fallback exists for exactly this reason; don't be surprised if it triggers on the very first real run, and don't treat that as a bug in the fallback rather than an expected possibility.
2. **The 60% homology threshold is a QMAP-derived placeholder, not validated for this project's dataset** (§8.2) — re-derive the PeptideAtlas-vs-this-dataset reference distribution before treating any pass/fail read against it as final.
3. **`trust_remote_code=True` for the model load is genuine remote-code execution** (§6.1, doc 05 §2.1) — pin the revision (§6.4), and treat any `transformers` version bump as a reason to re-verify `ChemPepMTR.py` compatibility before assuming the pinned sha still loads cleanly.
4. **Feller & Wilke's exact PAMPA-dataset size is unverified** (doc 07 §1.3/§6.1) — this project's smaller-k recommendation (§7.3) is reasoned from first principles about this project's own likely n, not a direct apples-to-apples comparison; re-check if that assumption turns out wrong once real peptide counts are in.
5. **CD-HIT/MMseqs2 vs. exact Needleman-Wunsch for the homology check is an unresolved empirical question** (doc 07 §2.2/§6) — this doc picks Biopython's exact global aligner for correctness and direct QMAP comparability, but if the joint peptide count grows enough that O(n²) pairwise alignment in `validate.max_identity_test_to_train` becomes a real bottleneck, revisit with an empirical CD-HIT-2D/MMseqs2 comparison rather than assuming either is unusable at AMP lengths.
6. **The cyclic-holdout ablation (doc 07 §4) is explicitly not built as a standing pipeline stage here** (§0) — the cluster/community assignments this doc produces make it straightforward to check post hoc whether one cluster is cyclic-enriched, but doing that check and turning it into a formal LOCO rotation is left to whenever Phase 5's ablation work needs it.
