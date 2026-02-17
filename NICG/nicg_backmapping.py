#!/usr/bin/env python3
"""
chromatin_backmap.py

Backmap a coarse-grained chromatin model (1 bead / DNA turn, 1 bead / core
histone, 2 beads / linker histone) to an all-atom chromatin model using:

  - an all-atom nucleosome template (127 bp DNA + histones),
  - reference ideal nucleotides in lib/aa_*.pdb,
  - spline-based linker geometry with boundary conditions taken from
    the first and last base-pair frames of each nucleosome DNA.

Usage (example):

  python chromatin_backmap.py \
      -cg chromatin_cg.pdb \
      -nuc nucleosome_template.pdb \
      -libdir lib \
      -o chromatin_allatom.pdb \
      -scale 10.0

Coordinates in input PDBs are multiplied by `scale` to recover Å.
"""

import sys
import os
import math
import argparse
import numpy as np
from numpy.linalg import svd, norm
from scipy.interpolate import CubicSpline


# ======================================================================
# Basic PDB parsing/writing helpers
# ======================================================================

class PDBAtom:
    def __init__(self, index, name, resname, chain, resid, x, y, z,
                 occupancy=1.0, bfactor=1.0, element=""):
        self.index   = index
        self.name    = name
        self.resname = resname
        self.chain   = chain
        self.resid   = resid
        self.x       = x
        self.y       = y
        self.z       = z
        self.occ     = occupancy
        self.bfac    = bfactor
        self.elem    = element or (name.strip()[0] if name.strip() else "X")

    def coord(self):
        return np.array([self.x, self.y, self.z], dtype=float)

    def set_coord(self, r):
        self.x, self.y, self.z = float(r[0]), float(r[1]), float(r[2])


def read_pdb(fname, scale=1.0):
    atoms = []
    with open(fname) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            idx = int(line[6:11])
            name = line[12:16]
            resname = line[17:20]
            chain = line[21]
            resid = int(line[22:26])
            x = float(line[30:38]) * scale
            y = float(line[38:46]) * scale
            z = float(line[46:54]) * scale
            occ = float(line[54:60]) if len(line) >= 60 else 1.0
            bfac = float(line[60:66]) if len(line) >= 66 else 1.0
            elem = line[76:78].strip() if len(line) >= 78 else ""
            atoms.append(PDBAtom(idx, name, resname, chain, resid, x, y, z, occ, bfac, elem))
    return atoms


def check_pdb_coord_range(atoms, inv_scale=1.0):
    """
    Check whether coordinates divided by `inv_scale` will fit in PDB 8.3 format.
    """
    max_abs = 0.0
    for a in atoms:
        max_abs = max(max_abs, abs(a.x * inv_scale), abs(a.y * inv_scale), abs(a.z * inv_scale))
    # Rough safe bound for 8.3 fields: 9999.99 Å
    if max_abs > 9999.99:
        raise RuntimeError(
            f"Coordinates would exceed PDB column width after scaling by 1/{inv_scale}. "
            f"Max |coord| ~ {max_abs:.2f} Å. Consider using a smaller scale factor or another format."
        )


def write_pdb(atoms, fname, inv_scale=1.0):
    """
    Write PDB, dividing coordinates by inv_scale to go back to PDB units.
    """
    check_pdb_coord_range(atoms, inv_scale=inv_scale)
    with open(fname, "w") as out:
        for a in atoms:
            x = a.x * inv_scale
            y = a.y * inv_scale
            z = a.z * inv_scale
            out.write(
                "ATOM  %5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %-2s\n" %
                (a.index % 100000,
                 a.name[:4],
                 a.resname[:3],
                 a.chain,
                 a.resid % 10000,
                 x, y, z,
                 a.occ, a.bfac,
                 a.elem[-2:].rjust(2))
            )
        out.write("END\n")


# ======================================================================
# Linear algebra helpers (Kabsch, rotation, etc.)
# ======================================================================

def kabsch(P, Q):
    """
    Compute optimal rotation matrix R and translation t such that:
        Q ≈ R @ P + t
    P, Q: (N,3) arrays
    Returns R (3x3), t (3,)
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    assert P.shape == Q.shape
    # centers
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    P0 = P - cP
    Q0 = Q - cQ
    # covariance
    C = P0.T @ Q0
    # SVD
    U, S, Vt = svd(C)
    # determinant sign correction
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    # optimal rotation
    R = Vt.T @ D @ U.T
    # translation
    t = cQ - R @ cP
    return R, t


def rotation_about_axis(axis, angle_deg):
    """
    Rodrigues rotation around given axis (3,) by angle in degrees.
    """
    axis = np.asarray(axis, dtype=float)
    axis_norm = norm(axis)
    if axis_norm < 1e-12:
        return np.eye(3)
    axis /= axis_norm
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    x, y, z = axis
    R = np.array([
        [c + x*x*(1-c),     x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s,   c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s,   z*y*(1-c) + x*s, c + z*z*(1-c)]
    ], dtype=float)
    return R


def rotation_axis_angle(R):
    """
    Given a rotation matrix R, return (axis, angle) such that
        R ~ exp(angle * [axis]_x)
    angle in radians, axis normalized.
    """
    R = np.asarray(R, float)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if abs(theta) < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0
    rx = R[2,1] - R[1,2]
    ry = R[0,2] - R[2,0]
    rz = R[1,0] - R[0,1]
    axis = np.array([rx, ry, rz], float)
    axis /= (2.0 * math.sin(theta))
    return axis, theta


def sqrt_rotation(R):
    """
    Matrix square-root of a rotation: R_s such that R_s @ R_s ≈ R.
    Implemented via axis-angle: half the rotation angle.
    """
    axis, theta = rotation_axis_angle(R)
    half_deg = math.degrees(0.5 * theta)
    return rotation_about_axis(axis, half_deg)


def Rz_deg(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], float)

def Ry_deg(angle_deg):
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0.0,  s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0,  c]], float)

# ======================================================================
# Template DNA: select DNA, 12 COMs, base-pair frames from aa_*.pdb
# ======================================================================

def is_allatom_dna_resname(resname):
    rn = resname.strip()
    return rn.startswith("D") or rn in ("DA", "DT", "DG", "DC")


def select_dna_atoms(template_atoms):
    """
    Select DNA atoms from the all-atom nucleosome template.
    """
    dna = []
    for a in template_atoms:
        if is_allatom_dna_resname(a.resname):
            dna.append(a)
    return dna


def get_dna_resid_order(dna_atoms):
    """
    Return sorted unique residue IDs for DNA.
    """
    resids = sorted({a.resid for a in dna_atoms})
    return resids


# ======================================================================
# Nucleotide library (aa_A.pdb etc.) and base-pair frame analysis
# ======================================================================

def read_standard_base(fname):
    """
    Read a single-nucleotide PDB (aa_A.pdb etc) into:

      atom_names: list[str]
      coords    : (3,N) array

    in the canonical local basis.
    """
    atom_names = []
    coords = []
    with open(fname) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            name = line[12:16]
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            atom_names.append(name)
            coords.append([x, y, z])
    coords = np.array(coords, dtype=float).T  # 3 x N
    return atom_names, coords


def load_nucleotide_library(libdir):
    bases = {}
    for bt in ["A", "C", "G", "T"]:
        fname = os.path.join(libdir, f"aa_{bt}.pdb")
        if not os.path.isfile(fname):
            raise RuntimeError(f"Cannot find reference nucleotide file {fname}")
        anames, crd = read_standard_base(fname)
        bases[bt] = (anames, crd)
    return bases


class Nucleotide:
    """
    Minimal nucleotide container for analysis.
    """
    def __init__(self, basetype, atomnames, positions):
        self.basetype  = basetype          # 'A','T','C','G'
        self.atomnames = list(atomnames)   # list of atom names (stripped)
        self.positions = np.array(positions, dtype=float)  # (M,3)

    def get_natoms(self):
        return self.positions.shape[0]


def compute_bp_frames_from_template(template_atoms, bases_lib):
    """
    Re-implementation of the DNA base-pair frame analysis, but only returns:

      bp_centers: list of np.array(3,)
      bp_frames : list of 3x3 rotation matrices

    template_atoms: all atoms of nucleosome template (including histones).
    bases_lib: dict {'A':(names, coords), ...} from load_nucleotide_library()
    """

    # key atoms as in original script
    list_key_atoms = {
        'A': ['N9','C8','N7','C5','C6','N1','C2','N3','C4'],
        'G': ['N9','C8','N7','C5','C6','N1','C2','N3','C4'],
        'T': ['N1','C2','N3','C4','C5','C6'],
        'C': ['N1','C2','N3','C4','C5','C6'],
    }
    base_types = ['A','T','C','G']

    # Build reference bases from the base library, keeping only key atoms
    ref_bases = {}
    for bt in base_types:
        anames, coords = bases_lib[bt]  # coords: 3 x N
        key_names = []
        key_coords = []
        for name, col in zip(anames, coords.T):
            if name.strip() in list_key_atoms[bt]:
                key_names.append(name.strip())
                key_coords.append(col)
        if len(key_names) == 0:
            raise RuntimeError(f"No key atoms found in reference base {bt}")
        ref_bases[bt] = (key_names, np.array(key_coords, dtype=float))  # (M,3)

    # Extract DNA atoms from template and group into strands by chain+resid
    dna_atoms = [a for a in template_atoms if is_allatom_dna_resname(a.resname)]
    # Sort by chain, resid, atom index
    dna_atoms.sort(key=lambda a: (a.chain, a.resid, a.index))

    # Build nucleotides for each chain (strand)
    strands_dict = {}  # chain -> list of Nucleotide
    current_chain = None
    current_resid = None
    atomnames = []
    positions = []
    basetype = None

    for a in dna_atoms:
        bt = a.resname.strip()[-1]  # 'A','T','C','G'
        if bt not in base_types:
            continue
        name = a.name.strip()
        if name not in list_key_atoms[bt]:
            continue  # ignore non-key atoms
        if (a.chain != current_chain) or (a.resid != current_resid):
            # store previous nucleotide
            if current_chain is not None:
                strands_dict.setdefault(current_chain, []).append(
                    Nucleotide(basetype, atomnames, positions)
                )
            # start new nucleotide
            current_chain = a.chain
            current_resid = a.resid
            basetype = bt
            atomnames = []
            positions = []
        atomnames.append(name)
        positions.append(a.coord())

    # store last nucleotide
    if current_chain is not None and len(positions) > 0:
        strands_dict.setdefault(current_chain, []).append(
            Nucleotide(basetype, atomnames, positions)
        )

    # Take two DNA strands (by sorted chain IDs)
    chain_ids = sorted(strands_dict.keys())
    if len(chain_ids) != 2:
        raise RuntimeError(f"Expected exactly 2 DNA strands, found {len(chain_ids)}")
    strands = [strands_dict[chain_ids[0]], strands_dict[chain_ids[1]]]

    # invert second strand so that it runs 3'->5' relative to the first
    strands[1] = strands[1][::-1]

    # some checks
    Nbp = len(strands[0])
    if Nbp != len(strands[1]):
        raise RuntimeError("Number of bases in 2 strands differ")
    basepair_types = ['AT','TA','CG','GC']
    for i in range(Nbp):
        nt0 = strands[0][i].basetype
        nt1 = strands[1][i].basetype
        if (nt0 + nt1) not in basepair_types:
            raise RuntimeError(f"Non-Watson-Crick pair: {nt0}{nt1} at bp {i+1}")

    # Align each experimental nucleotide to its reference base to get base frames
    def kabsch_rotation(P_ref, P_exp):
        # P_ref, P_exp: (N,3)
        R, _ = kabsch(P_ref, P_exp)
        return R

    base_frames = [[], []]  # per strand

    for i_strand in [0, 1]:
        for i in range(Nbp):
            nucleotide_exp = strands[i_strand][i]
            bt_exp = nucleotide_exp.basetype
            pos_exp = np.array(nucleotide_exp.positions, dtype=float)  # (M,3)
            ref_names, ref_pos = ref_bases[bt_exp]                     # (M,3)
            if pos_exp.shape[0] != ref_pos.shape[0]:
                raise RuntimeError("Mismatch in number of key atoms between exp and ref")

            # enforce same order of atoms via name matching
            # build mapping from name -> coord for exp
            exp_map = {name: pos for name, pos in zip(nucleotide_exp.atomnames, pos_exp)}
            ref_coords_ordered = []
            exp_coords_ordered = []
            for name in ref_names:
                if name not in exp_map:
                    raise RuntimeError(f"Key atom {name} missing in experimental base")
                ref_coords_ordered.append(ref_pos[[ref_names.index(name)], :][0])
                exp_coords_ordered.append(exp_map[name])

            ref_coords_ordered = np.array(ref_coords_ordered, dtype=float)
            exp_coords_ordered = np.array(exp_coords_ordered, dtype=float)

            com_ref = ref_coords_ordered.mean(axis=0)
            com_exp = exp_coords_ordered.mean(axis=0)
            ref_centered = ref_coords_ordered - com_ref
            exp_centered = exp_coords_ordered - com_exp

            R = kabsch_rotation(ref_centered, exp_centered)
            origin = (R @ (-com_ref)) + com_exp
            base_frames[i_strand].append((origin, R))

    # Compute base-pair frames from paired bases
    Rflip = np.diag([1.0, -1.0, -1.0])  # 180° around x
    bp_centers = []
    bp_frames = []

    for i in range(Nbp):
        o1, R1 = base_frames[0][i]
        o2, R2 = base_frames[1][i]
        o1 = np.array(o1, dtype=float)
        o2 = np.array(o2, dtype=float)
        R1 = np.array(R1, dtype=float)
        R2 = np.array(R2, dtype=float)

        R2p = R2 @ Rflip
        R2p1 = R2p.T @ R1
        R2p1sqrt = sqrt_rotation(R2p1)
        Rb = R2p @ R2p1sqrt
        ob = 0.5 * (o1 + o2)  # pivot=0

        bp_centers.append(ob)
        bp_frames.append(Rb)

    return bp_centers, bp_frames  # lists of length Nbp


def compute_bp_section_anchors(bp_centers, n_sections=12):
    """
    Given a list/array of bp_centers (Nbp,3), divide them into
    n_sections contiguous blocks along the DNA and return
    one representative center per block (the middle bp).

    This provides 12 "turn" anchors lying on the true helical axis,
    suitable for aligning to the 12 CG nucleosomal beads.
    """
    centers = np.asarray(bp_centers, dtype=float)
    Nbp = centers.shape[0]
    if Nbp < n_sections:
        raise RuntimeError(f"Not enough base pairs ({Nbp}) to make {n_sections} sections")

    anchors = []
    for k in range(n_sections):
        i_start = int(round(k      * Nbp / n_sections))
        i_end   = int(round((k+1) * Nbp / n_sections))
        if i_end <= i_start:
            i_end = i_start + 1
            if i_end > Nbp:
                i_end = Nbp
        # choose middle index in this block
        i_mid = (i_start + i_end - 1) // 2
        anchors.append(centers[i_mid])

    return np.array(anchors, dtype=float)  # (n_sections, 3)


def build_linker_atomic_coordinates(
        bp_names, bp_parms,
        bp_frame_origin, bp_frame_basis,
        bases_lib,
        chain_id_A="X", chain_id_B="Y",
        start_resid_A=1, start_resid_B=1,
        start_atom_index=1):
    """
    Construct atomic coordinates for a dsDNA segment (two strands)
    using the same logic as dna_build_from_params.build_atomic_coordinates.

    bp_names: list like ["A-T", "G-C", ...]
    bp_parms: list[[shear,stretch,stagger,buckle,propeller,opening]]
    bp_frame_origin: (Nbp,3)
    bp_frame_basis:  (Nbp,3,3)
    bases_lib: dict['A'|'C'|'G'|'T'] -> (atom_names, coords(3,N))
               as returned by load_nucleotide_library.

    Returns:
        atoms_out : list[PDBAtom]
        next_atom_index, next_resid_A, next_resid_B
    """
    num_bp = len(bp_parms)
    if num_bp == 0:
        return [], start_atom_index, start_resid_A, start_resid_B

    # unpack base library
    map_base_atom_names = {bt: bases_lib[bt][0] for bt in bases_lib}
    map_base_atom_coors = {bt: bases_lib[bt][1] for bt in bases_lib}

    atoms_out = []
    atom_index = start_atom_index

    Rflip = np.diag([1.0, -1.0, -1.0])  # 180° around x

    for ibp in range(num_bp):
        shear, stretch, stagger, buckle, prop, opening = bp_parms[ibp]

        # same "BucTwist"/phi logic as dna_build_from_params
        BucTwist = np.hypot(buckle, prop)
        if BucTwist < 1e-10:
            phi = 0.0
        else:
            phi_sign = 1.0 if buckle > 0 else -1.0
            phi = np.rad2deg(np.arccos(prop / BucTwist)) * phi_sign

        # local base frames in bp frame
        local_base1_basis = (
            Rz_deg(-phi)                @
            Ry_deg(+BucTwist/2.0)       @
            Rz_deg(phi + opening/2.0)
        )
        local_base2_basis = (
            Rz_deg(-phi)                @
            Ry_deg(-BucTwist/2.0)       @
            Rz_deg(phi - opening/2.0)
        )
        local_base2_basis = local_base2_basis @ Rflip

        bp_o = bp_frame_origin[ibp]
        bp_R = bp_frame_basis[ibp]

        # local translation for base origins (half shear/stretch/stagger)
        local_base1_origin = bp_R @ (np.array([shear, stretch, stagger], float) / 2.0)
        global_base1_origin = bp_o + local_base1_origin
        global_base1_basis  = bp_R @ local_base1_basis
        global_base2_origin = bp_o - local_base1_origin
        global_base2_basis  = bp_R @ local_base2_basis

        # base types in this pair, e.g. "A-T"
        bpname = bp_names[ibp]
        b1 = bpname[0]
        b2 = bpname[2]

        coor_std_base1 = map_base_atom_coors[b1]
        coor_std_base2 = map_base_atom_coors[b2]

        # NOTE: here we use the standard base coordinates *as-is*,
        # exactly like dna_build_from_params, i.e. no COM centering.
        coor_new_base1 = global_base1_basis @ coor_std_base1 + global_base1_origin[:, None]
        coor_new_base2 = global_base2_basis @ coor_std_base2 + global_base2_origin[:, None]

        names1 = map_base_atom_names[b1]
        names2 = map_base_atom_names[b2]

        # residue IDs (simple incremental scheme)
        resid_A = start_resid_A + ibp
        resid_B = start_resid_B + ibp

        # Strand A atoms
        resname1 = f" D{b1}"
        for j, aname in enumerate(names1):
            xyz = coor_new_base1[:, j]
            atoms_out.append(
                PDBAtom(
                    index=atom_index,
                    name=aname,
                    resname=resname1,
                    chain=chain_id_A,
                    resid=resid_A,
                    x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                    occupancy=1.0,
                    bfactor=1.0,
                    element=aname.strip()[0] if aname.strip() else "C"
                )
            )
            atom_index += 1

        # Strand B atoms
        resname2 = f" D{b2}"
        for j, aname in enumerate(names2):
            xyz = coor_new_base2[:, j]
            atoms_out.append(
                PDBAtom(
                    index=atom_index,
                    name=aname,
                    resname=resname2,
                    chain=chain_id_B,
                    resid=resid_B,
                    x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                    occupancy=1.0,
                    bfactor=1.0,
                    element=aname.strip()[0] if aname.strip() else "C"
                )
            )
            atom_index += 1

    next_resid_A = start_resid_A + num_bp
    next_resid_B = start_resid_B + num_bp

    return atoms_out, atom_index, next_resid_A, next_resid_B


# ======================================================================
# CG DNA parsing and nucleosome/linker segmentation
# ======================================================================

def is_cg_dna_bead(atom):
    # CG beads: names like 'DLN','DNC','DEX','DAN', etc. (start with 'D')
    return atom.name.strip().startswith("D")


def is_linker_dna_bead(atom):
    # Note that here the linker DNA includes the nucleosomal DNA entry/exit beads
    return (atom.name.strip() == "DLN" or atom.name.strip() == "DEX" or atom.name.strip() == "DAE" or atom.name.strip() == "DBE")


def extract_cg_dna_beads(cg_atoms):
    """
    Extract DNA beads (both nucleosomal and linker) in atom index order.
    """
    dna_beads = [a for a in cg_atoms if is_cg_dna_bead(a)]
    dna_beads.sort(key=lambda a: a.index)
    return dna_beads


def segment_nucleosomes_and_linkers(dna_beads):
    """
    From ordered DNA beads, segment into:
       - nucleosomal blocks (12 consecutive non-linker beads each),
       - linker blocks of linker-type beads between/around them.

    Returns:
      nucleosomes: list of dicts with {'dna_beads': [atoms], 'start_index': int, 'end_index': int}
      linkers:     list of dicts with {'dna_beads': [atoms]}
                   with length Nnuc+1:
                     linkers[0]   : linker before first nucleosome
                     linkers[i]   : linker between nucleosome i-1 and i  (1 <= i <= Nnuc-1)
                     linkers[Nnuc]: linker after last nucleosome
    """
    n_total = len(dna_beads)
    # indices of all non-linker beads
    non_idx = [i for i, b in enumerate(dna_beads) if not is_linker_dna_bead(b)]
    if len(non_idx) % 12 != 0:
        raise RuntimeError(f"Number of nucleosomal DNA beads ({len(non_idx)}) is not a multiple of 12.")

    Nnuc = len(non_idx) // 12
    nucleosomes = []
    linkers = []

    # First, build nucleosomes as contiguous 12-bead blocks
    for k in range(Nnuc):
        start = non_idx[12*k]
        end   = non_idx[12*k + 11]
        if end - start + 1 != 12:
            raise RuntimeError(
                f"Nucleosome {k}: expected 12 contiguous non-linker beads, "
                f"but got indices {start}..{end}."
            )
        nuc_beads = dna_beads[start:end+1]
        nucleosomes.append({
            "dna_beads": nuc_beads,
            "start_index": start,
            "end_index": end
        })

    # Now, build linkers in the gaps (and before/after)
    # linker before first nucleosome
    first_start = nucleosomes[0]["start_index"]
    leading = [dna_beads[i] for i in range(0, first_start) if is_linker_dna_bead(dna_beads[i])]
    linkers.append({"dna_beads": leading})

    # linkers between nucleosomes
    for k in range(1, Nnuc):
        prev_end = nucleosomes[k-1]["end_index"]
        this_start = nucleosomes[k]["start_index"]
        mid = [dna_beads[i] for i in range(prev_end+1, this_start) if is_linker_dna_bead(dna_beads[i])]
        linkers.append({"dna_beads": mid})

    # linker after last nucleosome
    last_end = nucleosomes[-1]["end_index"]
    trailing = [dna_beads[i] for i in range(last_end+1, n_total) if is_linker_dna_bead(dna_beads[i])]
    linkers.append({"dna_beads": trailing})

    return nucleosomes, linkers


# ======================================================================
# Spline & local frames for linker DNA
# ======================================================================

def compute_spline_path(coords):
    """
    coords: (M,3) array
    Returns:
      spline_x, spline_y, spline_z, total_length, u (M,)
      where u is arc-length parametrization in [0,1].
    """
    coords = np.asarray(coords, dtype=float)
    M = coords.shape[0]
    if M < 2:
        raise RuntimeError("Need at least 2 points to fit a spline for linker DNA.")
    d = np.zeros(M)
    for i in range(1, M):
        d[i] = d[i-1] + norm(coords[i] - coords[i-1])
    L = d[-1]
    if L <= 0:
        raise RuntimeError("Zero total length for linker path.")
    u = d / L
    spline_x = CubicSpline(u, coords[:,0])
    spline_y = CubicSpline(u, coords[:,1])
    spline_z = CubicSpline(u, coords[:,2])
    return spline_x, spline_y, spline_z, L, u


def spline_point(spline_x, spline_y, spline_z, u):
    return np.array([spline_x(u), spline_y(u), spline_z(u)], dtype=float)


def spline_tangent(spline_x, spline_y, spline_z, u):
    dx = spline_x.derivative()(u)
    dy = spline_y.derivative()(u)
    dz = spline_z.derivative()(u)
    t = np.array([dx, dy, dz], dtype=float)
    t_norm = norm(t)
    if t_norm < 1e-12:
        # fallback
        return np.array([0.0, 0.0, 1.0])
    return t / t_norm


def spline_normal(spline_x, spline_y, spline_z, u, t):
    """
    As in your CG builder: try curvature; if small, choose arbitrary perpendicular.
    """
    d2x = spline_x.derivative(2)(u)
    d2y = spline_y.derivative(2)(u)
    d2z = spline_z.derivative(2)(u)
    k = np.array([d2x, d2y, d2z], dtype=float)
    if norm(k) < 1e-8:
        # choose arbitrary perpendicular
        if abs(t[0]) < 0.9:
            v = np.array([1.0, 0.0, 0.0])
        else:
            v = np.array([0.0, 0.0, 1.0])
        n = np.cross(t, v)
        n_norm = norm(n)
        if n_norm < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return n / n_norm
    # remove component along tangent
    k = k - np.dot(k, t) * t
    if norm(k) < 1e-8:
        if abs(t[0]) < 0.9:
            v = np.array([1.0, 0.0, 0.0])
        else:
            v = np.array([0.0, 0.0, 1.0])
        k = np.cross(t, v)
    n = k / norm(k)
    return n


def build_linker_bp_centers_and_frames(
        linker_dna_beads,
        exit_center, entry_center,
        exit_bp_frame, entry_bp_frame,
        Nbp,
        approx_twist_deg=34.0):
    """
    Build bp centers and bp frames for a linker between two nucleosomes.

    Parameters
    ----------
    linker_dna_beads : list[PDBAtom]
        DLN beads between nucleosome i and i+1 (can be empty).
    exit_center : (3,) array
        Center of the *last* nucleosomal bp of nucleosome i (all-atom).
    entry_center : (3,) array
        Center of the *first* nucleosomal bp of nucleosome i+1 (all-atom).
    exit_bp_frame : (3,3) array
        Rotation matrix of the last nucleosomal bp of nucleosome i
        (columns are x,y,z axes) from dna_structure_analysis logic.
    entry_bp_frame : (3,3) array
        Rotation matrix of the first nucleosomal bp of nucleosome i+1.
    Nbp : int
        Number of bp to place in the linker.
    approx_twist_deg : float
        Target twist per bp (B-DNA ~ 34°).

    Returns
    -------
    bp_centers : (Nbp,3) array
    bp_frames  : list of Nbp (3,3) rotation matrices
    L_link     : float, spline arc-length
    """
    if Nbp <= 0:
        return np.zeros((0, 3), float), [], 0.0

    # --- 1. Spline path through exit, linker beads, entry ---
    coords = [np.asarray(exit_center, float)]
    coords.extend([b.coord() for b in linker_dna_beads])
    coords.append(np.asarray(entry_center, float))
    coords = np.array(coords, float)

    spline_x, spline_y, spline_z, L, u_nodes = compute_spline_path(coords)

    # Parameter positions for each bp center (include endpoints)
    us = np.linspace(0.0, 1.0, Nbp)
    bp_centers = np.zeros((Nbp, 3), float)
    for i, u in enumerate(us):
        bp_centers[i] = spline_point(spline_x, spline_y, spline_z, u)

    # Force exact endpoint positions for safety
    bp_centers[0]  = np.asarray(exit_center, float)
    bp_centers[-1] = np.asarray(entry_center, float)

    # --- 2. Tangents from finite differences on bp_centers ---
    tangents = np.zeros((Nbp, 3), float)
    for i in range(Nbp):
        if i == 0:
            t = bp_centers[1] - bp_centers[0]
        elif i == Nbp - 1:
            t = bp_centers[-1] - bp_centers[-2]
        else:
            t = bp_centers[i+1] - bp_centers[i-1]
        norm_t = np.linalg.norm(t)
        if norm_t < 1e-8:
            # fallback: use spline derivative directly
            du = us[1] - us[0] if Nbp > 1 else 1e-3
            dx = spline_x.derivative()(us[i])
            dy = spline_y.derivative()(us[i])
            dz = spline_z.derivative()(us[i])
            t = np.array([dx, dy, dz], float)
            norm_t = np.linalg.norm(t)
            if norm_t < 1e-8:
                t = np.array([0.0, 0.0, 1.0])
                norm_t = 1.0
        tangents[i] = t / norm_t

    # --- 3. Parallel transport frames + twist along the curve ---
    bp_frames = [None] * Nbp

    # First frame: z = tangent at 0; x,y from exit_bp_frame projected
    z0 = tangents[0]
    # take x from exit_bp_frame projected onto plane ⟂ z0
    x_exit = exit_bp_frame[:, 0]
    x0 = x_exit - np.dot(x_exit, z0) * z0
    norm_x0 = np.linalg.norm(x0)
    if norm_x0 < 1e-8:
        # try y of exit frame
        y_exit = exit_bp_frame[:, 1]
        x0 = y_exit - np.dot(y_exit, z0) * z0
        norm_x0 = np.linalg.norm(x0)
        if norm_x0 < 1e-8:
            # arbitrary perpendicular
            if abs(z0[0]) < 0.9:
                v = np.array([1.0, 0.0, 0.0])
            else:
                v = np.array([0.0, 1.0, 0.0])
            x0 = np.cross(z0, v)
            norm_x0 = np.linalg.norm(x0)
    x0 /= norm_x0
    y0 = np.cross(z0, x0)
    y0 /= np.linalg.norm(y0)
    bp_frames[0] = np.column_stack([x0, y0, z0])

    # total desired twist between first and last bp (before end correction)
    nominal_twist_step = approx_twist_deg

    for i in range(1, Nbp):
        R_prev = bp_frames[i-1]
        z_prev = R_prev[:, 2]
        z_new  = tangents[i]

        # rotation that takes z_prev to z_new (parallel transport)
        dot_zz = np.clip(np.dot(z_prev, z_new), -1.0, 1.0)
        angle_zz = math.acos(dot_zz)
        if angle_zz < 1e-8:
            R_align = np.eye(3)
        else:
            axis = np.cross(z_prev, z_new)
            norm_a = np.linalg.norm(axis)
            if norm_a < 1e-8:
                R_align = np.eye(3)
            else:
                axis /= norm_a
                # Rodrigues
                ca = math.cos(angle_zz)
                sa = math.sin(angle_zz)
                ux, uy, uz = axis
                R_align = np.array([
                    [ca + ux*ux*(1-ca),     ux*uy*(1-ca) - uz*sa, ux*uz*(1-ca) + uy*sa],
                    [uy*ux*(1-ca) + uz*sa,  ca + uy*uy*(1-ca),    uy*uz*(1-ca) - ux*sa],
                    [uz*ux*(1-ca) - uy*sa,  uz*uy*(1-ca) + ux*sa, ca + uz*uz*(1-ca)]
                ], float)

        R_trans = R_align @ R_prev  # parallel-transported frame with z aligned

        # now apply nominal helical twist about new z axis
        R_twist = rotation_about_axis(R_trans[:, 2], nominal_twist_step)
        bp_frames[i] = R_twist @ R_trans

    # --- 4. Smoothly correct twist to match entry_bp_frame at last bp ---
    # We assume tangents(N-1) is reasonably aligned already; we just fix rotation around z.
    R_last = bp_frames[-1]
    z_last = R_last[:, 2]
    x_last = R_last[:, 0]

    # project entry x onto plane ⟂ z_last
    x_entry = entry_bp_frame[:, 0]
    x_entry_proj = x_entry - np.dot(x_entry, z_last) * z_last
    norm_xe = np.linalg.norm(x_entry_proj)
    if norm_xe < 1e-8:
        # fallback: use entry y
        y_entry = entry_bp_frame[:, 1]
        x_entry_proj = y_entry - np.dot(y_entry, z_last) * z_last
        norm_xe = np.linalg.norm(x_entry_proj)
    if norm_xe < 1e-8:
        # can't resolve, just return as is
        return bp_centers, bp_frames, L

    x_entry_proj /= norm_xe

    # angle from x_last to x_entry_proj around z_last
    # (signed)
    v1 = x_last - np.dot(x_last, z_last) * z_last
    v2 = x_entry_proj
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    dot_x = np.clip(np.dot(v1, v2), -1.0, 1.0)
    delta = math.acos(dot_x)
    sign = 1.0 if np.dot(np.cross(v1, v2), z_last) >= 0.0 else -1.0
    delta *= sign  # signed twist correction (radians)

    # distribute this twist correction linearly along linker
    if Nbp > 1:
        for i in range(Nbp):
            frac = float(i) / float(Nbp - 1)
            phi = math.degrees(delta * frac)
            R_i = bp_frames[i]
            R_corr = rotation_about_axis(R_i[:, 2], phi)
            bp_frames[i] = R_corr @ R_i

    return bp_centers, bp_frames, L


def build_terminal_linker_bp_centers_and_frames(
        linker_dna_beads,
        boundary_center,
        boundary_bp_frame,
        Nbp,
        approx_twist_deg=34.0,
        approx_rise=3.4):
    """
    Build bp centers and frames for a terminal linker attached to a single
    nucleosome boundary (entry or exit).

    Parameters
    ----------
    linker_dna_beads : list[PDBAtom]
        Linker beads on the 'free' side (can be 1 or more beads).
        The spline runs from boundary_center through these beads.
    boundary_center : (3,) array
        Center of the boundary base pair (entry or exit) of the nucleosome.
    boundary_bp_frame : (3,3) array
        Rotation matrix of that boundary base pair (columns are x,y,z axes).
    Nbp : int
        Number of bp to place in the linker.
    approx_twist_deg : float
        Preferred twist per bp (B-DNA ~ 34°).
    approx_rise : float
        Target average rise per bp in Å (default ~3.4).

    Returns
    -------
    bp_centers : (Nbp,3) array
    bp_frames  : list of Nbp (3,3) rotation matrices
    L_link     : float, spline arc-length (before scaling)
    """
    if Nbp <= 0 or len(linker_dna_beads) == 0:
        return np.zeros((0, 3), float), [], 0.0

    # 1. Spline path from boundary center outwards along linker beads
    coords = [np.asarray(boundary_center, float)]
    coords.extend([b.coord() for b in linker_dna_beads])
    coords = np.array(coords, float)  # shape (M,3), M >= 2

    spline_x, spline_y, spline_z, L, u_nodes = compute_spline_path(coords)

    # parameter positions for each bp center, initially in [0,1]
    us = np.linspace(0.0, 1.0, Nbp)
    bp_centers = np.zeros((Nbp, 3), float)
    for i, u in enumerate(us):
        bp_centers[i] = spline_point(spline_x, spline_y, spline_z, u)

    # enforce exact boundary center for i=0
    bp_centers[0] = np.asarray(boundary_center, float)

    # ------------------------------------------------------------------
    # NEW: stretch terminal linker so total span matches desired rise
    # ------------------------------------------------------------------
    # current straight-line distance between first and last bp centers
    vec = bp_centers[-1] - bp_centers[0]
    curr_len = np.linalg.norm(vec)

    # desired straight-line distance for Nbp base pairs
    # (Nbp-1) steps of approx_rise along the helical axis
    if Nbp > 1:
        desired_len = approx_rise * (Nbp - 1)
    else:
        desired_len = 0.0

    if curr_len > 1e-6 and desired_len > 0.0:
        scale = desired_len / curr_len
        # scale all centers radially outwards from the boundary center
        origin0 = bp_centers[0].copy()
        for i in range(1, Nbp):
            delta = bp_centers[i] - origin0
            bp_centers[i] = origin0 + scale * delta
    # ------------------------------------------------------------------

    # 2. Tangents from finite differences on *scaled* bp_centers
    tangents = np.zeros((Nbp, 3), float)
    for i in range(Nbp):
        if i == 0:
            t = bp_centers[1] - bp_centers[0]
        elif i == Nbp - 1:
            t = bp_centers[-1] - bp_centers[-2]
        else:
            t = bp_centers[i+1] - bp_centers[i-1]
        norm_t = np.linalg.norm(t)
        if norm_t < 1e-8:
            dx = spline_x.derivative()(us[i])
            dy = spline_y.derivative()(us[i])
            dz = spline_z.derivative()(us[i])
            t = np.array([dx, dy, dz], float)
            norm_t = np.linalg.norm(t)
            if norm_t < 1e-8:
                t = np.array([0.0, 0.0, 1.0])
                norm_t = 1.0
        tangents[i] = t / norm_t

    # 3. Parallel transport + twist, starting from the nucleosome boundary frame
    bp_frames = [None] * Nbp

    # First frame: z = tangent at 0; x,y from boundary frame projected
    z0 = tangents[0]
    x_boundary = boundary_bp_frame[:, 0]
    x0 = x_boundary - np.dot(x_boundary, z0) * z0
    norm_x0 = np.linalg.norm(x0)
    if norm_x0 < 1e-8:
        y_boundary = boundary_bp_frame[:, 1]
        x0 = y_boundary - np.dot(y_boundary, z0) * z0
        norm_x0 = np.linalg.norm(x0)
        if norm_x0 < 1e-8:
            if abs(z0[0]) < 0.9:
                v = np.array([1.0, 0.0, 0.0])
            else:
                v = np.array([0.0, 0.0, 1.0])
            x0 = np.cross(z0, v)
            norm_x0 = np.linalg.norm(x0)
    x0 /= norm_x0
    y0 = np.cross(z0, x0)
    y0 /= np.linalg.norm(y0)
    bp_frames[0] = np.column_stack([x0, y0, z0])

    nominal_twist_step = approx_twist_deg

    for i in range(1, Nbp):
        R_prev = bp_frames[i-1]
        z_prev = R_prev[:, 2]
        z_new  = tangents[i]

        dot_zz = np.clip(np.dot(z_prev, z_new), -1.0, 1.0)
        angle_zz = math.acos(dot_zz)
        if angle_zz < 1e-8:
            R_align = np.eye(3)
        else:
            axis = np.cross(z_prev, z_new)
            norm_a = np.linalg.norm(axis)
            if norm_a < 1e-8:
                R_align = np.eye(3)
            else:
                axis /= norm_a
                ca = math.cos(angle_zz)
                sa = math.sin(angle_zz)
                ux, uy, uz = axis
                R_align = np.array([
                    [ca + ux*ux*(1-ca),     ux*uy*(1-ca) - uz*sa, ux*uz*(1-ca) + uy*sa],
                    [uy*ux*(1-ca) + uz*sa,  ca + uy*uy*(1-ca),    uy*uz*(1-ca) - ux*sa],
                    [uz*ux*(1-ca) - uy*sa,  uz*uy*(1-ca) + ux*sa, ca + uz*uz*(1-ca)]
                ], float)

        R_trans = R_align @ R_prev
        R_twist = rotation_about_axis(R_trans[:, 2], nominal_twist_step)
        bp_frames[i] = R_twist @ R_trans

    # no end-frame correction here (only one boundary)
    return bp_centers, bp_frames, L


# ======================================================================
# Build poly-AG linker DNA at all-atom resolution
# ======================================================================

def build_polyAG_linker_allatom(
        bp_centers, bp_frames,
        bases_lib,
        start_atom_index, start_resid,
        chain_id_A="X", chain_id_B="Y"):
    """
    Build an ideal dsDNA linker as poly-AG:

        strand A: A G A G A G ...
        strand B: T C T C T C ...

    using dna_build_from_params-style base construction on given
    bp centers & frames.

    Parameters
    ----------
    bp_centers : (Nbp,3)
    bp_frames  : list of Nbp (3,3)
    bases_lib  : dict from load_nucleotide_library
    start_atom_index : int
    start_resid : int
        Starting residue index for strand A (strand B uses same range).

    Returns
    -------
    linker_atoms : list[PDBAtom]
    next_atom_index, next_resid
        next_resid is the next available residue index for strand A/B.
    """
    Nbp = len(bp_centers)
    if Nbp == 0:
        return [], start_atom_index, start_resid

    # 1) Build bp_names and bp_parms (ideal bp: all zeros)
    bp_names = []
    bp_parms = []
    for i in range(Nbp):
        if i % 2 == 0:
            bp_names.append("A-T")
        else:
            bp_names.append("G-C")
        bp_parms.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    bp_frame_origin = np.asarray(bp_centers, float)
    bp_frame_basis  = np.asarray(bp_frames, float)

    # 2) Call the 3DNA-style base constructor
    linker_atoms, next_atom_index, next_resid_A, next_resid_B = build_linker_atomic_coordinates(
        bp_names, bp_parms,
        bp_frame_origin, bp_frame_basis,
        bases_lib,
        chain_id_A=chain_id_A,
        chain_id_B=chain_id_B,
        start_resid_A=start_resid,
        start_resid_B=start_resid,
        start_atom_index=start_atom_index
    )

    # We keep strand A/B residue ranges identical; return next_resid_A as "next"
    return linker_atoms, next_atom_index, next_resid_A


def compute_and_report_linker_twist_rise(linker_id, bp_centers, bp_frames):
    """
    For a linker with given bp centers and frames, compute step rise and twist
    between neighboring base pairs and print them for reference.

    Twist is measured as the effective rotation about the local z-axis
    using the 2x2 xy block of R_step, which avoids the axis-angle
    sign ambiguity.
    """
    Nbp = len(bp_centers)
    if Nbp < 2:
        print(f"# Linker {linker_id}: fewer than 2 base pairs, no twist/rise to report.", file=sys.stderr)
        return

    rises = []
    twists = []

    #print(f"# Linker {linker_id}: step-wise rise and twist (bp i -> i+1):", file=sys.stderr)

    for i in range(Nbp - 1):
        c_i = np.asarray(bp_centers[i], dtype=float)
        c_j = np.asarray(bp_centers[i+1], dtype=float)
        R_i = np.asarray(bp_frames[i], dtype=float)
        R_j = np.asarray(bp_frames[i+1], dtype=float)

        # step vector
        dc = c_j - c_i

        # local helical axis ~ z-axis of bp i
        z_i = R_i[:, 2]

        # rise = projection of dc on local z-axis
        rise = float(np.dot(dc, z_i))

        # relative rotation R_step taking frame i to frame j
        R_step = R_i.T @ R_j

        # Effective twist about local z: use xy-block
        # For a pure z-rotation, R_step[0,0] = cosθ, R_step[1,0] = sinθ
        twist_rad = math.atan2(R_step[1,0], R_step[0,0])
        twist_deg = math.degrees(twist_rad)

        # If you want twist always in [0, 360) instead of (-180, 180]:
        # if twist_deg < 0.0:
        #     twist_deg += 360.0

        rises.append(rise)
        twists.append(twist_deg)

        #print(f"#   step {i+1:4d}: rise = {rise:8.3f} Å, twist = {twist_deg:8.3f} deg", file=sys.stderr)

    avg_rise = sum(rises) / len(rises)
    avg_twist = sum(twists) / len(twists)
    print(f"# Linker {linker_id}: average rise = {avg_rise:8.3f} Å, average twist = {avg_twist:8.3f} deg", file=sys.stderr)

# ======================================================================
# Main backmapping driver
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Backmap CG chromatin to all-atom resolution.")
    parser.add_argument("-cg",    required=True, help="Input CG chromatin PDB (from chromatin_builder).")
    parser.add_argument("-nuc",   required=True, help="All-atom nucleosome template PDB (127 bp).")
    parser.add_argument("-libdir", required=True, help="Directory with aa_A.pdb, aa_C.pdb, aa_G.pdb, aa_T.pdb.")
    parser.add_argument("-o",     required=True, help="Output all-atom chromatin PDB.")
    parser.add_argument("-scale", type=float, default=1.0,
                        help="Scale factor used when input PDB coordinates were shrunk. "
                             "Coordinates are multiplied by this to recover Å (default 1.0).")
    parser.add_argument("--approx_twist", type=float, default=34.0,
                        help="Preferred twist per bp for linker DNA (degrees, default 34).")
    args = parser.parse_args()

    scale = args.scale

    # --- Load nucleosome template
    template_atoms = read_pdb(args.nuc, scale=scale)
    template_dna   = select_dna_atoms(template_atoms)

    # --- Load nucleotide library (used for both analysis and linker building)
    bases_lib = load_nucleotide_library(args.libdir)

    # --- Compute base-pair frames for template DNA (all-atom)
    bp_centers_template, bp_frames_template = compute_bp_frames_from_template(
        template_atoms, bases_lib
    )
    Nbp_template = len(bp_centers_template)
    if Nbp_template < 2:
        raise RuntimeError("Template DNA has too few base pairs for boundary analysis.")

    # Use bp index 0 as "entry" and Nbp-1 as "exit" boundaries
    bp_entry_idx = 0
    bp_exit_idx  = Nbp_template - 1

    # --- Define 12 anchor points along the template DNA axis
    ref_turn_anchors = compute_bp_section_anchors(bp_centers_template, n_sections=12)

    # --- Load CG chromatin and extract DNA beads
    cg_atoms = read_pdb(args.cg, scale=scale)
    dna_beads = extract_cg_dna_beads(cg_atoms)
    nucleosomes, linkers = segment_nucleosomes_and_linkers(dna_beads)

    print(f"# Found {len(nucleosomes)} nucleosomes and {len(linkers)} linker segments.", file=sys.stderr)

    # --- Backmap nucleosomes (align template via 12 COMs, then transform atoms and bp frames)
    allatom_atoms = []
    atom_index = 1
    resid = 1

    nuc_info = []  # per nucleosome: dict with atoms, exit/entry centers & frames

    for k, nuc in enumerate(nucleosomes):
        nuc_dna_beads = nuc["dna_beads"]
        if len(nuc_dna_beads) != 12:
            raise RuntimeError(f"Nucleosome {k} does not have 12 DNA beads (has {len(nuc_dna_beads)}).")
        cg_turn_pos = np.array([b.coord() for b in nuc_dna_beads], dtype=float)  # (12,3)

        # Align template via 12 bp-axis anchors (not residue COMs)
        R_align, t_align = kabsch(ref_turn_anchors, cg_turn_pos)

        # Transform all template atoms
        nuc_atoms = []
        for a in template_atoms:
            r0 = a.coord()
            r1 = R_align @ r0 + t_align
            new_atom = PDBAtom(
                index=atom_index,
                name=a.name,
                resname=a.resname,
                chain=chr(ord('A') + (k % 2)),  # e.g. alternate A/B
                resid=resid + (a.resid - 1),
                x=r1[0], y=r1[1], z=r1[2],
                occupancy=a.occ,
                bfactor=a.bfac,
                element=a.elem
            )
            nuc_atoms.append(new_atom)
            atom_index += 1

        # Transform boundary basepair centers & frames
        entry_center_local = np.asarray(bp_centers_template[bp_entry_idx], dtype=float)
        exit_center_local  = np.asarray(bp_centers_template[bp_exit_idx], dtype=float)
        entry_frame_local  = np.asarray(bp_frames_template[bp_entry_idx], dtype=float)
        exit_frame_local   = np.asarray(bp_frames_template[bp_exit_idx], dtype=float)

        entry_center_global = R_align @ entry_center_local + t_align
        exit_center_global  = R_align @ exit_center_local  + t_align
        entry_frame_global  = R_align @ entry_frame_local
        exit_frame_global   = R_align @ exit_frame_local

        nuc_info.append({
            "atoms": nuc_atoms,
            "entry_center": entry_center_global,
            "exit_center": exit_center_global,
            "entry_frame": entry_frame_global,
            "exit_frame": exit_frame_global
        })
        allatom_atoms.extend(nuc_atoms)

        # Advance residue numbering so the next nucleosome does not collide
        max_res_templ = max(a.resid for a in template_atoms)
        resid += max_res_templ + 10  # gap of 10 for safety

    # --- Build linkers as poly-AG double-stranded DNA
    Nnuc = len(nucleosomes)
    # linkers has length Nnuc+1:
    #   linkers[0]   : before first nucleosome
    #   linkers[i]   : between nuc i-1 and i  (1..Nnuc-1)
    #   linkers[Nnuc]: after last nucleosome

    # 3a. Leading linker (if any) before first nucleosome
    leading_linker = linkers[0]["dna_beads"]
    if len(leading_linker) > 0:
        N_linker_beads = len(leading_linker)
        Nbp_link = int(round(10.5 * N_linker_beads))
        print(f"# Leading linker: {N_linker_beads} CG beads -> {Nbp_link} bp.", file=sys.stderr)

        entry_center_0 = nuc_info[0]["entry_center"]
        entry_frame_0  = nuc_info[0]["entry_frame"]

        bp_centers_link, bp_frames_link, L_link = build_terminal_linker_bp_centers_and_frames(
            leading_linker,
            boundary_center=entry_center_0,
            boundary_bp_frame=entry_frame_0,
            Nbp=Nbp_link,
            approx_twist_deg=args.approx_twist
        )

        compute_and_report_linker_twist_rise("start", bp_centers_link, bp_frames_link)

        chain_A = chr(ord('A'))       # same convention as nucleosome 0
        chain_B = chr(ord('A') + 1)   # or whatever you prefer
        linker_atoms, atom_index, resid = build_polyAG_linker_allatom(
            bp_centers_link, bp_frames_link, bases_lib,
            start_atom_index=atom_index,
            start_resid=resid,
            chain_id_A=chain_A,
            chain_id_B=chain_B
        )
        allatom_atoms.extend(linker_atoms)

    # 3b. Internal linkers between nucleosomes i-1 and i
    for i in range(1, Nnuc):
        lnk_beads = linkers[i]["dna_beads"]
        N_linker_beads = len(lnk_beads)
        if N_linker_beads == 0:
            continue

        Nbp_link = int(round(10.5 * N_linker_beads))
        print(f"# Linker {i}: {N_linker_beads} CG beads -> {Nbp_link} bp.", file=sys.stderr)

        exit_center_i_1  = nuc_info[i-1]["exit_center"]
        exit_frame_i_1   = nuc_info[i-1]["exit_frame"]
        entry_center_i   = nuc_info[i]["entry_center"]
        entry_frame_i    = nuc_info[i]["entry_frame"]

        bp_centers_link, bp_frames_link, L_link = build_linker_bp_centers_and_frames(
            lnk_beads,
            exit_center_i_1, entry_center_i,
            exit_frame_i_1, entry_frame_i,
            Nbp=Nbp_link,
            approx_twist_deg=args.approx_twist
        )

        compute_and_report_linker_twist_rise(i, bp_centers_link, bp_frames_link)

        chain_A = chr(ord('A') + ((i-1) % 2))
        chain_B = chr(ord('A') + (i % 2))
        linker_atoms, atom_index, resid = build_polyAG_linker_allatom(
            bp_centers_link, bp_frames_link, bases_lib,
            start_atom_index=atom_index,
            start_resid=resid,
            chain_id_A=chain_A,
            chain_id_B=chain_B
        )
        allatom_atoms.extend(linker_atoms)

    # 3c. Trailing linker (if any) after last nucleosome
    trailing_linker = linkers[Nnuc]["dna_beads"]
    if len(trailing_linker) > 0:
        N_linker_beads = len(trailing_linker)
        Nbp_link = int(round(10.5 * N_linker_beads))
        print(f"# Trailing linker: {N_linker_beads} CG beads -> {Nbp_link} bp.", file=sys.stderr)

        exit_center_last = nuc_info[-1]["exit_center"]
        exit_frame_last  = nuc_info[-1]["exit_frame"]

        bp_centers_link, bp_frames_link, L_link = build_terminal_linker_bp_centers_and_frames(
            trailing_linker,
            boundary_center=exit_center_last,
            boundary_bp_frame=exit_frame_last,
            Nbp=Nbp_link,
            approx_twist_deg=args.approx_twist
        )

        compute_and_report_linker_twist_rise("end", bp_centers_link, bp_frames_link)

        chain_A = chr(ord('A') + ((Nnuc-1) % 2))
        chain_B = chr(ord('A') + (Nnuc % 2))
        linker_atoms, atom_index, resid = build_polyAG_linker_allatom(
            bp_centers_link, bp_frames_link, bases_lib,
            start_atom_index=atom_index,
            start_resid=resid,
            chain_id_A=chain_A,
            chain_id_B=chain_B
        )
        allatom_atoms.extend(linker_atoms)

    # --- Write output PDB (coordinates divided by scale to match PDB convention)
    write_pdb(allatom_atoms, args.o, inv_scale=scale)
    print(f"# All-atom chromatin PDB written to {args.o}", file=sys.stderr)

if __name__ == "__main__":
    main()
