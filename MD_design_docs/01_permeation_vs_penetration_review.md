# Peptide Membrane "Penetration" vs. "Permeation": A Sourced Review

*Reference document for the PeptideCLM transfer-learning project. Covers Tasks 1 (permeation vs. penetration) and 2 (how to evaluate/predict permeabilization). All claims are drawn from peer-reviewed papers or curated databases; sources are listed inline and collected at the end.*

---

## 0. The one-sentence problem

The words **permeation, permeability, penetration, and diffusion** are used in **two different research communities to mean two nearly opposite things**, and the confusion is baked into the literature you will be reading — including PeptideCLM's own abstract, which uses "membrane penetration," "membrane diffusion," and "membrane permeation" interchangeably for the *same* phenomenon.

- **Drug / pharmacokinetics sense** (PeptideCLM, PAMPA, Caco-2, cyclic-peptide oral bioavailability): the **intact peptide passively crosses an undisturbed membrane** and comes out the other side. The membrane stays intact. This is what the field usually calls *permeability*.
- **Biophysics / antimicrobial sense** (your project): the peptide **permeabilizes — disrupts — the membrane**, forming pores or dissolving the bilayer, so that contents leak out and the cell lyses. The membrane is destroyed.

Your Tasks 1 and 5 use "permeation" in the **second** sense. PeptideCLM was built for the **first**. Getting this straight is the whole conceptual crux of the project, so this document keeps the two meanings rigorously separate and uses:

- **Penetration / translocation** = intact peptide crosses intact membrane (drug sense).
- **Permeabilization / lysis** = peptide disrupts membrane → leakage → death (your sense).

---

## 1. Penetration / translocation (crossing an intact membrane)

Two sub-phenomena share this label.

### 1a. Passive permeability (the PAMPA / Caco-2 world)
A molecule partitions into the bilayer, diffuses across, and partitions out, without disrupting it. Governed by lipophilicity (logP/logD), molecular size, hydrogen-bond donor count, polar surface area (TPSA), net charge, and — critically for peptides — **conformational flexibility ("chameleonicity")**: cyclic peptides shield polar backbone groups via intramolecular H-bonds in the membrane and expose them in water. This is why cyclic-peptide permeability does **not** follow Lipinski's rule of five (Li et al., *CycPeptMPDB*, 2023; Feller & Wilke, 2025). Assays: PAMPA (pure artificial lipid film, passive only), Caco-2 / MDCK / RRCK (cell monolayers, which add transporters and efflux).

### 1b. Cell-penetrating peptides (CPPs)
Short, usually cationic/arginine-rich peptides (TAT, penetratin/Antp, oligoarginine) that carry cargo into cells. After ~20 years of debate it is accepted that CPPs use **both** energy-dependent **endocytosis** and energy-independent **direct translocation**, and the balance depends on peptide sequence, extracellular concentration, and membrane composition — not simply on the number of positive charges (Jiao et al. / Duchardt studies; Ruzza, *Biochemistry* 2015; Kauffman et al. review, PMC6964662; Cell-Penetrating Peptide review, PMC3103903). Translocation dominates at low concentration; endocytosis switches on above a threshold.

### The boundary is genuinely blurry — this matters for your model
Direct CPP translocation is thought to proceed through **transient pore formation, carpet-like perturbation, inverted micelles, or membrane thinning** (PMC11279660; PMC6599048) — i.e., the *same* physical events used to describe AMP permeabilization. Many amphipathic CPPs are cytotoxic/lytic at higher concentrations, and some AMPs are also cell-penetrating and act on intracellular targets (Greco et al., *Sci. Rep.* 2020). So penetration and permeabilization are not two disjoint physics; they are **points on a continuum distinguished by degree and outcome**: does the peptide cross and leave the bilayer intact (penetration), or does it accumulate past a critical peptide:lipid ratio and rupture it (permeabilization)? This is the single most important nuance for the transfer-learning premise, and it is developed in the plan document.

---

## 2. Permeabilization / lysis (your target)

### 2a. Mechanistic models
For a membrane-active AMP to work, peptide must first **accumulate on the membrane surface up to a critical local concentration** (Melo/Brogden; Frontiers in Neuroscience 2017, fnins.2017.00073). Beyond that threshold, the accepted models split into two families (Brogden 2005; Wimley 2010; Matsuzaki, *Membrane Permeabilization Mechanisms*, Springer 2019):

- **Pore-forming (transmembrane):**
  - *Barrel-stave* — peptides insert perpendicular and bundle into a rigid channel, hydrophobic faces out, hydrophilic faces lining an aqueous pore (classic example: alamethicin).
  - *Toroidal pore* — peptides remain associated with lipid head groups and bend the bilayer so lipids line the pore with them (melittin, magainin). Often shows a threshold peptide:lipid ratio.
- **Surface-acting:**
  - *Carpet* — peptides blanket the bilayer surface and, above a threshold, disrupt it.
  - *Detergent* — at high concentration, carpet peptides solubilize the membrane like a surfactant (Bechinger & Lohner 2006).

The exact mechanism for any given AMP is usually unresolved and debated; recent work argues some "pores" are better described as **transient, non-structural water channels** rather than fixed geometries (PNAS 2025, pnas.2517944122; "Latest developments on membrane-disrupting peptides," PMC10244799). Practically: you are predicting an **emergent permeabilization propensity**, not a single clean mechanism.

An important quantitative point (Wimley): **true pore-formers permeabilize vesicles at very low peptide:lipid ratios**, whereas most AMPs only leak vesicles at high ratios, and *almost any* membrane-binding peptide leaks membranes at sufficiently high concentration ("Interfacial Activity Model," PMC2955829). Concentration is therefore never separable from the readout.

### 2b. Selectivity: which membrane is being permeabilized
Permeabilization is **membrane-specific**, unlike a single PAMPA number:

- **Bacterial** membranes are anionic (phosphatidylglycerol, cardiolipin, LPS/teichoic acids) → strong electrostatic attraction to cationic AMPs.
- **Mammalian** membranes are largely zwitterionic (phosphatidylcholine) and contain **cholesterol**, which rigidifies the bilayer and resists insertion.

This asymmetry is the entire basis of AMP therapeutic selectivity — peptides can be tuned to prefer bacterial over eukaryotic membranes (patent US9234004 shows (LLKK)₃-type peptides lysing bacteria at concentrations giving <5% hemolysis; Nature *Sci. Rep.* 2023, s41598-023-43274-9, rational selectivity design). **Whatever assay/membrane your training labels come from is implicitly encoded in the label** — exactly the situation PeptideCLM has with PAMPA.

### 2c. Experimental endpoints (your ground truth)
- **MIC** (minimum inhibitory concentration): antimicrobial potency against a bacterial membrane. **Lower = more potent.**
- **HC50 / HD50 / MHC** (concentration causing 50% hemolysis / minimum hemolytic concentration): toxicity against mammalian RBC membranes. **Higher = safer** (see correction below).
- **Dye/calcein leakage from LUVs/GUVs**: the most *direct* biophysical permeabilization readout, but far scarcer data.
- **Biophysical structure**: CD, SAXS/SANS, neutron reflectometry, impedance, and MD reveal *how* (pore vs. surface), used to validate mechanism rather than screen.

> **Correction to a common inversion.** For hemolysis, **you want a HIGH HC50, not a low one.** HC50 is the concentration needed to lyse 50% of red blood cells, so a *high* HC50 means it takes a lot of peptide to cause damage → **low toxicity → good**. A *low* HC50 means a small amount lyses cells → toxic → bad. The design goal is **low MIC and high HC50 simultaneously**, summarized by the **therapeutic/selectivity index TI = HC50 / MIC** (higher is better). Reported AMP therapeutic indices of ~50–500 indicate good selectivity (Greco et al., *Sci. Rep.* 2020; US12033725; US8252737: "a high therapeutic index indicates high safety and minimal toxicity"). This matters directly for the multitask objective in the plan doc: the two heads pull in opposite directions on membrane disruption, and TI is the quantity that reconciles them.

---

## 3. Task 2 — How to evaluate and predict permeabilization ability

Two layers: measure it, or predict it.

### 3a. Experimental evaluation (ground truth generation)
Screen with MIC (bacterial) and hemolysis/HC50 (mammalian); confirm mechanism and membrane target with LUV dye-leakage and biophysics (CD for secondary structure on membrane binding; SAXS/NR for peptide location and pore geometry); MD simulation (GROMACS with CHARMM36 or coarse-grained MARTINI, bacterial-mimetic vs. mammalian-mimetic bilayers) as the expensive mechanistic gold standard.

### 3b. Computational prediction — three tiers of increasing cost
1. **Physicochemical / QSAR descriptors.** The classic, interpretable levers, all cheap to compute from sequence: **net positive charge, mean hydrophobicity (H), amphipathicity, hydrophobic moment (μH), helical propensity, length, and hydrophobic-face angle.** Hydrophobicity in particular correlates with *both* antimicrobial potency and hemolysis — which is exactly why selectivity, not raw potency, is the hard part.
2. **Sequence-based ML (protein language models).** ESM-2 embeddings + a regression head predict MIC/HC50 from sequence. Fast and strong for canonical linear peptides, but **cannot represent C-terminal amidation, D-amino acids, cyclization, or non-canonical residues** — it only sees the 20-letter alphabet.
3. **Chemical language models (SMILES).** PeptideCLM / PeptideCLM-2 operate on the full chemical structure, so they *can* represent modifications, cyclization, and non-canonical chemistry. This is the tier your project targets, and its trade-offs (data conversion, domain shift) are covered in the plan doc.
4. **Physics-based (MD / free-energy).** Most accurate for mechanism, far too slow for library-scale screening; useful for validating a handful of top candidates or generating features.

### 3c. A realistic note on predictability
The **QMAP benchmark** (bioRxiv 2026, homology-aware MIC + HC50 regression) reassessed AMP potency/toxicity predictors and found **limited progress over six years, poor performance on high-potency MIC, and low predictability of hemolytic activity.** Treat HC50 regression as an intrinsically hard, noisy-label problem — this shapes expectations for the transfer-learning model and argues strongly for homology-aware evaluation splits.

---

## Sources

**PeptideCLM / permeability**
- Feller & Wilke, "Peptide-aware chemical language model successfully predicts membrane diffusion of cyclic peptides," *J. Chem. Inf. Model.* 2025, 65(2):571–579. DOI 10.1021/acs.jcim.4c01441. Preprint: bioRxiv 2024.08.09.607221.
- Feller et al., "Scaling SMILES-Based Chemical Language Models for Therapeutic Peptide Engineering" (PeptideCLM-2 / PeptideMTR), bioRxiv 2026.01.06.697994.
- Li et al., "CycPeptMPDB: A Comprehensive Database of Membrane Permeability of Cyclic Peptides," 2023. PMC10091415.

**AMP permeabilization mechanism**
- Wimley, "Interfacial Activity Model," PMC2955829.
- Matsuzaki, "Membrane Permeabilization Mechanisms," Springer 2019.
- "Latest developments on the mechanism of action of membrane disrupting peptides," PMC10244799.
- "Membrane-Active AMPs: Translating Mechanistic Insights to Design," Front. Neurosci. 2017, fnins.2017.00073.
- "Structural pores not required… transient water channels," *PNAS* 2025, pnas.2517944122.

**Cell-penetrating peptides / translocation**
- "Mechanisms of Cellular Uptake of Cell-Penetrating Peptides," PMC3103903.
- "Internalization mechanisms of cell-penetrating peptides," PMC6964662.
- Ruzza et al., "Translocation Mechanism(s) of CPPs… Artificial Membrane Bilayers," *Biochemistry* 2015, 10.1021/bi501392n.
- "CPP-Mediated Biomolecule Transportation in Artificial Lipid Vesicles and Living Cells," PMC11279660.

**Selectivity / therapeutic index**
- Greco et al., "Correlation between hemolytic activity, cytotoxicity and systemic in vivo toxicity of synthetic AMPs," *Sci. Rep.* 2020, s41598-020-69995-9.
- "Rational design of cell-selective AMPs (QL17)," *Sci. Rep.* 2023, s41598-023-43274-9.

**Benchmark**
- "QMAP: A Benchmark for Standardized Evaluation of AMP MIC and Hemolytic Activity Regression," bioRxiv 2026.02.03.703041.
