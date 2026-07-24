# Phase 2 (steps 1 & 2) — Colab-generated embeddings + joint clusters

Local tracker only (not committed — `data/embeddings/` and `data/splits/` are
gitignored, same convention as Phase 1's `data/` tree).

Produced by two Colab notebooks, output files downloaded and dropped in
locally.

## Inputs
- `data/processed/dataset.parquet` (Phase 1 output, 251,346 rows / 39,029
  unique `peptide_uid`s) as of the 2026-07-23 full production pull.
- Deduped to unique peptides with a usable SMILES (non-`failed` fidelity
  tier) and at least one label, canonicalized, conflict-flagged per
  `has_smiles_conflict` → 34,971 rows going into embedding/clustering.

## Step 1 & 2 — embeddings + joint k-means clustering

`notebooks/phase2_embed_and_cluster.ipynb`, run 2026-07-23.

- `data/embeddings/peptide_embeddings_colab.npy` — (34971, 1024) float32,
  row order matches `data/splits/joint_clusters_k4_colab.csv`.
- `data/splits/joint_clusters_k4_colab.csv` — `peptide_uid`,
  `smiles_canonical`, `has_smiles_conflict`, `cluster_label`.

Encoder: `aaronfeller/peptideclm-2-mlm-large` (337M params), mean-pooled
final-layer hidden states, mask-aware. **Revision not pinned** (loaded
from `main`) — re-resolve to a commit sha before this is treated as a
reproducible artifact rather than a one-off. Tokenizer: `max_length=2048`,
`trust_remote_code=True`.

Clustering: joint k-means over the full 34,971-peptide union (HC50 ∪ MIC),
swept k=4–6, k selected by max silhouette score. **Sweep metrics
(silhouette/DB/CH per k) were printed in the notebook but not saved to
disk** — only the winning k=4 assignment was exported. Cluster sizes (k=4):
0=8122, 1=9084, 2=7867, 3=9898.

## Step 3 — LOCo split + chemical homology audit

`notebooks/colab_loco_split_and_audit.ipynb`, run 2026-07-24. Not the
doc 10-specced approach (GraphPart/Leiden on sequence identity) — this
dataset only has SMILES, no amino-acid sequence column, so MMSeqs2/GraphPart
can't run on it. Uses connected-components clustering on a k-NN graph over
the embedding cosine distance (threshold 0.01, swept first — see notebook
Cell 2), bin-packed into 80/10/10 train/val/test by largest-deficit-first
cluster assignment, then an independent **Tanimoto similarity audit**
(Morgan/ECFP4 fingerprints, radius 2, 2048 bits) checking every val/test
peptide against the full set.

- `data/splits/loco_split_colab.parquet` — `joint_clusters_k4_colab.csv`'s
  columns plus `embedding_cluster` (15,996 clusters) and `split`
  (train/val/test, 27977/3497/3497 — clean 80/10/10 by construction).
- `data/splits/tanimoto_leak_audit_colab.csv` — `query_uid`, `target_uid`,
  `tanimoto`, `split_query`, `split_target` for every cross-split pair with
  Tanimoto ≥ 0.85 (83,575 pairs, 4,156 of 6,994 val/test peptides / 59.4%
  affected).

**Known caveat, deliberately not blocking on it:** the 59.4% cross-split
overlap rate looks alarming at face value, but a quick check (`same_smiles`
on the Tanimoto==1.0 subset) found **0 of 1,975 perfect-similarity pairs are
actually identical SMILES** — i.e. distinct molecules are scoring as
maximally similar, which points at a Morgan-fingerprint saturation/collision
artifact (e.g. short peptides or common substructures saturating a
fixed-length 2048-bit vector) rather than genuine near-duplicate leakage.
Decision: proceed with this split for the first working system, revisit the
audit methodology as a Phase 5 ablation (see `14_ablations_list.md`) rather
than re-deriving the split now.

## Not yet done
No sanity-checking against `dataset.parquet` peptide_uid coverage, no
`split_report`-style validation against QMAP's leakage baseline (doc 10
§10/§11), no per-task fold filtering. The Tanimoto-vs-real-duplicate
question above is unresolved, just deprioritized.
