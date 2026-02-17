# NICG Chromatin Model

## Overview

This repository builds and exports **nucleosome-resolution coarse-grained chromatin fibers** in the spirit of the NICG (Nucleosome Interaction Coarse-Grained) model described in the accompanying manuscript [[Gu, C., S. Takada, and G.B. Brandani. bioRxiv (2025)](https://www.biorxiv.org/content/10.1101/2025.08.04.668597v1.abstract)]. 

The builder constructs chains of:

* **Nucleosomes** (histone core + nucleosomal DNA beads ± H1 beads),
* **Linker DNA beads** between nucleosomes,
* Optional **leading/trailing linker overhangs**, and
* Optional **epigenetic states** (acetylation and BRD4-associated acetylation) and **linker histone H1**.

It then writes:

1. a **PDB** for visualization,
2. a **LAMMPS data file** (Atoms/Bonds/Angles), and
3. a **bond-coefficient include file** (`in.bond_settings`) with equilibrium distances extracted from the generated geometry.

The model is intended to be simulated with **LAMMPS** using Langevin dynamics, with nonbonded Lennard-Jones + Debye–Hückel electrostatics (for charged bead types). 

## NICG model summary

### Bead representation and geometry

Each nucleosome is represented by point particles:

* **8 histone beads** (H2A/H2B/H3/H4, one bead each),
* **14 nucleosomal DNA beads** (147 bp at 10.5 bp/bead),
* Optional **2 H1 beads** (globular + C-terminal tail). 

Geometry:

* Nucleosomal DNA beads are arranged on a superhelix (radius ~40 Å, pitch ~25 Å).
* Histone beads are arranged on a coaxial superhelix with the same pitch and smaller radius (~18 Å). 
* H1 beads (if present) are placed on the dyad axis; the first H1 bead is ~70 Å from the nucleosome center. 

All beads use a common size parameter **σ = 35 Å**, chosen to match roughly one DNA helical turn (10.5 bp). 

### Bead types / epigenetic states

The manuscript describes **20 bead types**, grouped into **5 interaction classes**:

1. linker DNA
2. canonical nucleosome
3. acetylated nucleosome
4. acetylated nucleosome associated with BRD4
5. linker histone H1 

This builder encodes those states via `ac` (acetylation) and `brd4` flags per nucleosome, and an `H1` flag for whether H1 is present on that nucleosome.

### Bonded interactions (connectivity)

The potential energy includes:

* **Harmonic bonds** between consecutive linker DNA beads, and between any bead pairs that are within a cutoff distance in the reference nucleosome geometry (implemented here by bonding pairs closer than ~38 Å). 
* **Morse bonds** on the nucleosome DNA entry/exit region, replacing harmonic bonds to permit “breathing” / unwrapping dynamics. 
* **Angle potentials** along the DNA chain to reproduce DNA bending elasticity/persistence length. Additional angles to stabilize H1 positioning when present. 

### Nonbonded interactions

Nonlocal interactions are:

* Lennard-Jones interactions with strengths depending on bead classes (weaker attractions for acetylated–acetylated, stronger when BRD4-associated nucleosomes are involved to capture bridging). 
* Debye–Hückel electrostatics for charged bead types; in particular, linker DNA and H1 carry charges (−7.35e and +5e per bead, respectively). 

## What `nicg_chromatin_builder.py` does

### 1) Reads a “multi-chain” top-level input

You run the script with `-f chains.txt`. That file is a *list of chains*:

```
chain chain1.txt
chain chain2.txt
...
```

Each `chain*.txt` describes one chromatin fiber.

### 2) Parses one chain definition

A chain file supports these keys:

* `nuc N`: Number of nucleosomes in the chain.

* `ll l0 l1 ... lN`: Linker lengths in number of coarse-grained linker-DNA beads** (not bp).
You must provide **N+1 integers**:
`l0` = leading overhang linker beads (before nucleosome 0),
`l1..l(N-1)` = internal linkers between nucleosomes,
`lN` = trailing overhang linker beads (after last nucleosome).

* `H1 0/1 0/1 ...`: Linker histone flags (length N; optional; defaults to all 0).

* `ac 0/1 0/1 ...`: Acetylated nucleosome flags (length N; optional; defaults to all 0).

* `brd4 0/1 0/1 ...`: BRD4-bound nucleosome flags (length N; optional; defaults to all 0).

* `cg coords.txt`: Path coordinates (Nx3). When provided, nucleosomes are placed along a spline fit to these points.

### 3) Builds the chain

There are two modes:

#### A) No `cg` provided (straight fiber)

The script builds a straight fiber with alternating linkers and nucleosomes, where each new linker is extended from the last two DNA beads of the previous nucleosome.

#### B) `cg` provided (reference backbone)

If `cg` is provided:

1. The coordinates are converted into a **cubic spline backbone**.
2. Nucleosome dyad positions are mapped along the spline in proportion to total DNA content (nucleosomal DNA + linkers).
3. Each nucleosome is placed using a local frame (tangent/normal) and a twist increment.
4. Linker beads between nucleosomes are initially placed along a straight segment and then “snapped” to uniform spacing between the *actual* exit DNA bead of nucleosome i and entry DNA bead of nucleosome i+1.

For the first nucleosome, the leading linker DNA overhang is repositioned so that it emanates from the first two DNA beads of the first nucleosomal DNA

### 4) Emits outputs

For each run (potentially multiple chains appended):

* `chromatin.pdb` (or `-o`): the chromatin structure for visualization.

* `data.chromatin` (or `-d`): the main LAMMPS data file containing the list of Atoms, Bonds, and Angles.

* `in.bond_settings` (or `-b`): a helper file that writes LAMMPS `bond_coeff` lines using the equilibrium distances measured from the generated geometry, to be included in the LAMMPS input file for the simulation.

At the end, the script recenters the whole structure by subtracting the overall center-of-mass.

---

## Minimal example

`chains.txt`:

```
chain input_chain1.txt
```

`input_chain1.txt`:

```
nuc 3
ll 2 5 5 2
H1 0 1 0
ac  0 1 1
brd4 0 0 1
# cg backbone.txt   # optional
```

Run:

```bash
python nicg_chromatin_builder.py -f chains.txt -o chromatin.pdb -d data.chromatin -b in.bond_settings
```

---

## Additional utility scripts

In addition to the core chromatin builder, this repository provides two companion scripts that automate **input generation** and **backmapping** for NICG chromatin simulations.

---

### `nicg_define_fiber.py` — stochastic chromatin fiber generator

`nicg_define_fiber.py` generates **1D chromatin fiber definitions** suitable as direct input for `nicg_chromatin_builder.py`.

It constructs a single chromatin chain with:

* stochastic **A/B compartment/domain organization**,
* per-nucleosome **epigenetic states** (acetylation, BRD4 binding, linker histone H1),
* realistic **linker length distributions**, and
* optional **leading/trailing overhangs**.

The script is intended to generate *physically motivated random chromatin fibers* with user-defined statistical properties.

#### Key features

* **Domains** alternating between A and B. Domain lengths are exponentially distributed. Independent mean domain sizes can be specified for A and B compartments.
* **Epigenetics**.
Acetylation is allowed only in A compartments (with compartment-specific probabilities).
BRD4 binding is conditional on acetylation.
Linker histone H1 is excluded from A compartments and enriched in B compartments.
Global target fractions are recovered statistically.
* **Linker lengths**.
Internal linker lengths are sampled one by one from a truncated normal distribution (in bp).
Negative linker lengths are rejected.
The input total fiber length is matched exactly by adjusting the trailing overhang.

**Builder-compatible output**

The script writes a builder-compatible output that defines the chromatin fiber:

```
nuc N
ll  l0 l1 ... lN
H1  0/1 ...
ac  0/1 ...
brd4 0/1 ...
cg <coords>   # optional
```

Produces auxiliary BED files (`nucs.bed`) and a JSON summary (`settings.used.json`) for validation and reproducibility.

#### Typical usage

```bash
python nicg_define_fiber.py --outdir fiber_inputs --config params.json
python nicg_chromatin_builder.py -f fiber_inputs/chains.txt
```

This script can be useful to generate **large, statistically controlled chromatin fibers** for NICG simulations.

---

### `nicg_backmapping.py` — coarse-grained to all-atom backmapping

`nicg_backmapping.py` reconstructs an **all-atom chromatin model** from a coarse-grained NICG configuration.

Starting from a coarse-grained chromatin PDB produced by `nicg_chromatin_builder.py`, the script backmaps:

* nucleosomal DNA,
* histone cores,
* linker DNA,
* and optional linker histone H1

using **reference all-atom structures** stored in the `pdb/` directory.

#### Backmapping strategy

* Each coarse-grained nucleosome is aligned to a **reference all-atom nucleosome template**.
* DNA base-pair frames are extracted from the reference structure.
* Linker DNA is rebuilt using **spline-based interpolation** between nucleosome entry/exit frames.
* Base-pair geometry is reconstructed using canonical nucleotide templates (`aa_*.pdb`).
* Kabsch alignment and frame interpolation ensure continuity of DNA geometry across nucleosomes.

The resulting structure is a **fully atomistic chromatin fiber** consistent with the coarse-grained geometry.

#### Inputs

* Coarse-grained chromatin PDB (`chromatin_cg.pdb`)
* All-atom nucleosome template (`nucleosome_template.pdb`)
* Nucleotide library (`lib/aa_A.pdb`, `aa_T.pdb`, etc.)

#### Output

* All-atom chromatin PDB suitable for visualization, further atomistic refinement, or conversion to other MD engines.

#### Example usage

```bash
python nicg_backmapping.py \
    -cg chromatin_cg.pdb \
    -nuc PDB/nucleosome_template.pdb \
    -libdir PDB \
    -o chromatin_allatom.pdb \
    -scale 10.0
```

---

## Example NICG workflow

A complete NICG simulation pipeline is therefore:

1. **Generate a stochastic fiber**  
   ```
   nicg_define_fiber.py
   ```
2. **Build the coarse-grained chromatin model**  
   ```
   nicg_chromatin_builder.py
   ```
3. **Run CG molecular dynamics in LAMMPS**  
4. **Backmap selected configurations to all-atom resolution**  
   ```
   nicg_backmapping.py
   ```

---

## LAMMPS examples

Complete examples to perform simulations in LAMMPS, prepared with the above scripts, are found in the folder `examples/`.
>>>>>>> 4a340b8 (Initial commit: scripts for structure setup and backmapping examples)
