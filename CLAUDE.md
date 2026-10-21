# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Python tools for the NICG (Nucleosome Interaction Coarse-Grained) chromatin model (Gu, Takada & Brandani, bioRxiv 2025). The repo has three standalone scripts in `NICG/` and runnable LAMMPS examples in `examples/`. There is no package, build system or test suite. Dependencies are only `numpy` and `scipy`. Simulations run with LAMMPS (`lmp`), which is outside this repo.

## Pipeline and commands

```bash
# 1. (optional) generate a stochastic fiber definition -> chains.txt, input_chain1.txt, nucs.bed, settings.used.json
python NICG/nicg_define_fiber.py --config settings.input.json [--outdir DIR] [--seed N] [--cg coords.xyz]

# 2. build the CG model -> PDB, LAMMPS data file, in.bond_settings
python NICG/nicg_chromatin_builder.py -f chains.txt -o chromatin.pdb -d data.chromatin -b in.bond_settings -hbox 400.0
#    other flags: -gdna (default 0.35), -gnuc (0.0), -qH1 (H1 charge, 10.), -scalepdb (PDB coordinate scale)

# 3. simulate
lmp < in.run            # fiber_100kb runs `lmp < in.relax` first and in.run reads its restart file

# 4. backmap a CG PDB to all-atom
python NICG/nicg_backmapping.py -cg chromatin.pdb -nuc NICG/pdb/<template>.pdb -libdir NICG/pdb -o allatom.pdb [-scale 10.0]
```

Each `examples/*/run.sh` calls the scripts as bare `nicg_*.py`, so `NICG/` must be on `PATH`. The fastest way to check builder changes is to rebuild an example (e.g. `examples/n12nrl189` or `examples/dna`) and diff the regenerated `data.*` and `in.bond_settings` against the committed versions. The builder checks its own atom, bond and angle counts and raises on a mismatch.

## Input formats

- `chains.txt`: one `chain <file>` line per chain. Several chains are appended into one system, and chains without a `cg` backbone are offset by 300 Å in x.
- Chain file keys: `nuc N`, `ll l0 … lN` (N+1 linker lengths in **beads**, not bp, including the leading and trailing overhangs), `H1`/`ac`/`brd4` 0/1 lists of length N, and `cg <coords file>` (an Nx3 backbone, spline-fitted). `nuc 0` with a single `ll` value builds DNA only. Lines starting with `#` are comments.
- `nicg_define_fiber.py` works in bp and converts to beads with 10.5 bp/bead. Its defaults live in the `Settings` dataclass and are overridden by the JSON config.

## Builder architecture (`nicg_chromatin_builder.py`)

- `generate_nucleosome` builds one reference nucleosome: 8 histone beads, 14 DNA beads (12 core + 2 entry/exit) and optionally 2 H1 beads. The bead **name** (e.g. `DLN`, `DNC`, `DEX`, `H2A`, `DAN`/`DAE`/`A2A…` for acetylated, `DBN`/`DBE`/`B2A…` for BRD4, `H1`) maps to a fixed type ID from 1 to 20.
- `build_chromatin` places nucleosomes either along a straight fiber or along the spline backbone given by `cg`, then generates linker DNA. Bonds are created between bead pairs closer than about 38 Å in the reference geometry. Entry/exit DNA bonds use Morse potentials (`morse` or, for acetylated nucleosomes, `morse2`) so the DNA can unwrap.
- **Type renumbering:** `fix_atom_types` and `fix_bonds` compact atom and bond types to a contiguous range 1..N based on the types actually present. Atom type IDs in the data file therefore match the 1–20 scheme only when every bead class is present (as in fiber_100kb). Systems without acetylation, BRD4 or H1 end up with fewer types. This is why `pair_coeff` lines for types 8–20 are commented out in `examples/n12nrl189/in.run`. Keep the LAMMPS `pair_coeff` ranges consistent with the types present.
- `generate_bond_settings` writes one `bond_coeff` per bond type, using the mean measured distance as the equilibrium length. Morse r0 is set to 0.98× the measured distance. Force constants are left as LAMMPS variables (`${kbond}`, `${kmorse}`, `${kmorse2}`, `${alphamorse}`), which are defined in `in.run`/`in.relax`. `in.bond_settings` is regenerated on every build and must match the data file it was built with. A few bond types (DEX–DNC Morse bonds and two histone–DNA harmonic keys) merge the two sides of the dyad, which have slightly different reference lengths. The builder prints `# WARNING ... std of distances` for these on every nucleosome build. This is expected and matches the published parameterisation.

## Backmapping (`nicg_backmapping.py`)

This script identifies beads by **PDB atom name**, not by type. DNA beads are those whose names start with `D`. `DLN`, `DEX`, `DAE` and `DBE` are treated as linker DNA, so the entry/exit beads belong to the linker segment. Each nucleosome must have exactly 12 remaining core DNA beads. Renaming beads in the builder will therefore break backmapping. The script aligns all-atom nucleosome templates from `NICG/pdb/` to the CG beads and rebuilds linker DNA with splines and the `aa_{A,C,G,T}.pdb` nucleotide templates.

## LAMMPS conventions (examples)

The examples use `units real` (Å, fs, kcal/mol) and `atom_style full`. Pair interactions use `lj/cut/coul/debye`, with the Debye length computed from the salt concentration in the input and σ = 35 Å. Bonds use `hybrid harmonic morse` and angles use `cosine`. Dynamics use Langevin with a 10 fs timestep. Model parameters (`kpair`, `kpair2`, `kpairb`, `kH1`, `kmorse`, …) are set as variables at the top of each `in.*` file. `.gitignore` excludes LAMMPS outputs (`log.*`, `restart.*`, `*.dcd`, `*.xtc`).
