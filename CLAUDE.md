# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Python tools for the NICG (Nucleosome Interaction Coarse-Grained) chromatin model (Gu, Takada & Brandani, Nucleic Acids Res. 54, gkag535 (2026), doi:10.1093/nar/gkag535). The repo has three standalone scripts in `NICG/` and runnable LAMMPS examples in `examples/`. There is no package, build system or test suite. Dependencies are only `numpy` and `scipy`. Simulations run with LAMMPS (`lmp`), which is outside this repo.

## Pipeline and commands

```bash
# 1. (optional) generate a stochastic fiber definition -> chains.txt, input_chain1.txt, nucs.bed, settings.used.json
python NICG/nicg_define_fiber.py --config settings.input.json [--outdir DIR] [--seed N] [--cg coords.xyz]

# 2. build the CG model -> PDB, LAMMPS data file, in.bond_settings
python NICG/nicg_chromatin_builder.py -f chains.txt -o chromatin.pdb -d data.chromatin -b in.bond_settings -hbox 400.0
#    other flags: -box X1 X2 Y1 Y2 Z1 Z2 (any box instead of the cube of -hbox), -gdna (default 0.35), -gnuc (0.0),
#    -qH1 (H1 charge, 10.), -scalepdb (PDB coordinate scale);
#    linker torsion: -tw or -twh/-twN/-twC/-twsL, writes -t in.dihedral_settings; -h prints all options and the input format

# 3. simulate
lmp < in.run            # fiber_100kb runs `lmp < in.relax` first and in.run reads its restart file

# 4. backmap a CG PDB to all-atom
python NICG/nicg_backmapping.py -cg chromatin.pdb -nuc NICG/pdb/<template>.pdb -libdir NICG/pdb -o allatom.pdb [-scale 10.0]
```

Each `examples/*/run.sh` calls the scripts as bare `nicg_*.py`, so `NICG/` must be on `PATH`. The fastest way to check builder changes is to rebuild an example (e.g. `examples/n12nrl189`, `examples/dna`, or `examples/n12nrl189_torsion` for the torsion) and diff the regenerated `data.*`, `in.bond_settings` and `in.dihedral_settings` against the committed versions. `examples/llps_ll30bp` (45 fibers in a slab box, `-box`) also exercises the `cg` placement: its `setup_fibers.py` writes `chains.txt` and `fibers/` (one chain file and one 2-point backbone per fiber) before the builder runs. The `cg` placement spaces the linker beads evenly between nucleosomes, so linker bonds start off their 35 Å rest length (32.3 Å with 3 beads per linker, 43 Å with 2). The builder checks its own atom, bond and angle counts and raises on a mismatch.

## Input formats

- `chains.txt`: one `chain <file>` line per chain. Several chains are appended into one system, and chains without a `cg` backbone are offset by 300 Å in x.
- Chain file keys: `nuc N`, the linker DNA, `H1`/`ac`/`brd4` 0/1 lists of length N, and `cg <coords file>` (an Nx3 backbone, spline-fitted). Lines starting with `#` are comments.
- Linker DNA, exactly one of: `ll l0 … lN` (N+1 lengths in **beads**, including the leading and trailing overhangs; not allowed with the torsion), `llbp L0 … LN` (the same in bp; between nucleosomes L = NRL − 147), or `nucpos s1 … sN` + `genlen G` (1-based first bp of each 147-bp nucleosome and total length; converted to llbp in `nucpos_to_llbp`). bp lengths give round(L/10.5) beads (`linker_beads`). `nuc 0` builds DNA only from one `ll`/`llbp` value or `genlen` alone.
- `nicg_define_fiber.py` works in bp and writes `nucpos`/`genlen` (BED start + 1; it requires `nucleosomal_dna_bp` = 147). Its defaults live in the `Settings` dataclass and are overridden by the JSON config.

## Builder architecture (`nicg_chromatin_builder.py`)

- `generate_nucleosome` builds one reference nucleosome: 8 histone beads, 14 DNA beads (12 core + 2 entry/exit) and optionally 2 H1 beads. The bead **name** (e.g. `DLN`, `DNC`, `DEX`, `H2A`, `DAN`/`DAE`/`A2A…` for acetylated, `DBN`/`DBE`/`B2A…` for BRD4, `H1`) maps to a fixed type ID from 1 to 20. Histone beads 0–7 are `H2A H2B H4 H3 H3 H4 H2B H2A` (named after the histone each lies on in 1KX5; beads 2–5 were `H3 H4 H4 H3` before 2026-11-10). The Morse rules in `generate_nucleosome_bonds` select the DEX–H3 and DEX–H2A pairs (and `A3`/`DAE`, `B3`/`DBE`) by name, so renaming histone beads means changing those rules too.
- `build_chromatin` places nucleosomes either along a straight fiber or along the spline backbone given by `cg`, then generates linker DNA. Bonds are created between bead pairs closer than about 38 Å in the reference geometry. Entry/exit DNA bonds use Morse potentials (`morse` or, for acetylated nucleosomes, `morse2`) so the DNA can unwrap.
- **Type renumbering:** `fix_atom_types` and `fix_bonds` compact atom and bond types to a contiguous range 1..N based on the types actually present. Atom type IDs in the data file therefore match the 1–20 scheme only when every bead class is present (as in fiber_100kb). Systems without acetylation, BRD4 or H1 end up with fewer types. This is why `pair_coeff` lines for types 8–20 are commented out in `examples/n12nrl189/in.run`. Keep the LAMMPS `pair_coeff` ranges consistent with the types present.
- **Linker torsion** (optional): one dihedral per internal linker over beads 7 (H2A) and 20 (last DNC) of nucleosome i and 9 (first DNC) and 0 (H2A) of nucleosome i+1, selected by position. `torsion_parameters` gives D0 = `TORSION_C0` + tw n (c0 = 42.20 deg, from a zero-linker dinucleosome of two 1KX5 copies joined by one ideal 10.5-bp step, NICG mapped as in the backmapping; see the comment above `TORSION_H`) and the sd from the linker length L in bp (the torsion's nucleosome length `-twN` only enters there, as n = L + 148 − N free steps; the bead geometry always has 147 bp), `von_mises_kappa` the concentration, and `generate_dihedral_settings` writes `dihedral_style spherical` coefficients (25 terms: von Mises × flat-top factor 1 − cos⁸θ, m = 4 fixed) using the LAMMPS variable `T`. Without torsion options no dihedrals are written, and the outputs are those of the builder before the torsion was added (byte-identical apart from the H3/H4 names, which changed later).
- `generate_bond_settings` writes one `bond_coeff` per bond type, using the mean measured distance as the equilibrium length. Morse r0 is set to 0.98× the measured distance. Force constants are left as LAMMPS variables (`${kbond}`, `${kmorse}`, `${kmorse2}`, `${alphamorse}`), which are defined in `in.run`/`in.relax`. `in.bond_settings` is regenerated on every build and must match the data file it was built with. A few bond types (DEX–DNC Morse bonds and two histone–DNA harmonic keys) merge the two sides of the dyad, which have slightly different reference lengths. The builder prints `# WARNING ... std of distances` for these on every nucleosome build. This is expected and matches the published parameterisation.

## Backmapping (`nicg_backmapping.py`)

This script identifies beads by **PDB atom name**, not by type. DNA beads are those whose names start with `D`. `DLN`, `DEX`, `DAE` and `DBE` are treated as linker DNA, so the entry/exit beads belong to the linker segment. Each nucleosome must have exactly 12 remaining core DNA beads. Renaming beads in the builder will therefore break backmapping. The script aligns all-atom nucleosome templates from `NICG/pdb/` to the CG beads and rebuilds linker DNA with splines and the `aa_{A,C,G,T}.pdb` nucleotide templates.

## LAMMPS conventions (examples)

The examples use `units real` (Å, fs, kcal/mol) and `atom_style full`. Pair interactions use `lj/cut/coul/debye`, with the Debye length computed from the salt concentration in the input and σ = 35 Å. Bonds use `hybrid harmonic morse` and angles use `cosine`. Dynamics use Langevin with a 10 fs timestep. Model parameters (`kpair`, `kpair2`, `kpairb`, `kH1`, `kmorse`, …) are set as variables at the top of each `in.*` file. `.gitignore` excludes LAMMPS outputs (`log.*`, `restart.*`, `*.dcd`, `*.xtc`).
