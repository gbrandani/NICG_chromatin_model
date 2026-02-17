#!/usr/bin/env python

import numpy as np
import sys
import string
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

# Define the Atom class
class Atom:
    def __init__(self, index, atom_type, position,mass, mol_index=1, res_index=1, name='NA', resname='CHR', charge=0.):
        self.index = index          # Atom index starting from 1
        self.atom_type = atom_type  # Atom type as specified
        self.position = position    # Position as a 3D numpy array
        self.mol_index = mol_index  # Molecule index starting from 1
        self.res_index = res_index  # residue index
        self.name = name
        self.resname = resname
        self.charge = charge
        self.mass= mass
    def print(self):
        print('index: {}'.format(self.index))
        print('type: {}'.format(self.atom_type))
        print('mol_index: {}'.format(self.mol_index))
        print('name: {}'.format(self.name))
        print('resname: {}'.format(self.resname))
        print('position: {} {} {}'.format(self.position[0],self.position[1],self.position[2]))
        print('charge: {}'.format(self.charge))
        print('mass: {}'.format(self.mass))
        print('')

class Pair:
    def __init__(self, type1, type2, distance=35., epsilon=1.):
        self.type1 = type1        # type of atom 1
        self.type2 = type2        # type of atom 2
        self.distance = distance  # equilibrium distance
        self.epsilon = epsilon    # epsilon for interaction strength
    def print(self):
        print('types: {} {}'.format(self.type1,self.type2))
        print('distance: {}'.format(self.distance))
        print('epsilon: {}'.format(self.epsilon))
        print('')

class Bond:
    def __init__(self, index, bond_type, index1, index2, distance=35., key=0, potential='harmonic'):
        self.index = index          # Bond index starting from 1
        self.bond_type = bond_type  # Bond type as specified
        self.index1 = index1        # index of atom 1
        self.index2 = index2        # index of atom 2
        self.distance = distance    # equilibrium distance
        self.key = key              # key to define bond type
        self.potential = potential  # bond potential, e.g., harmonic or morse
    def print(self):
        print('index: {}'.format(self.index))
        print('type: {}'.format(self.bond_type))
        print('indices: {} {}'.format(self.index1,self.index2))
        print('distance: {}'.format(self.distance))
        print('key: ',self.key)
        print('potential: ',self.potential)
        print('')

class Angle:
    def __init__(self, index, bond_type, index1, index2, index3):
        self.index = index          # Bond index starting from 1
        self.bond_type = bond_type  # Bond type as specified
        self.index1 = index1        # index of atom 1
        self.index2 = index2        # index of atom 2
        self.index3 = index3        # index of atom 3
    def print(self):
        print('index: {}'.format(self.index))
        print('type: {}'.format(self.bond_type))
        print('indices: {} {} {}'.format(self.index1,self.index2,self.index3))
        print('')

def compute_spline_path(cg_coords):
    """
    Compute a cubic spline from the reference CG coordinates
    """
    # Compute cumulative distance along the path
    distances = np.zeros(len(cg_coords))
    for i in range(1, len(cg_coords)):
        distances[i] = distances[i-1] + np.linalg.norm(cg_coords[i] - cg_coords[i-1])
    
    # Normalize to [0, 1] for spline parameter
    t = distances / distances[-1]
    
    # Create cubic splines for each coordinate
    spline_x = CubicSpline(t, cg_coords[:, 0])
    spline_y = CubicSpline(t, cg_coords[:, 1])
    spline_z = CubicSpline(t, cg_coords[:, 2])
    
    return spline_x, spline_y, spline_z, distances[-1]

def compute_spline_tangent(spline_x, spline_y, spline_z, t):
    """
    Compute derivatives (tangent vectors) at parameter t
    """
    dx = spline_x.derivative()(t)
    dy = spline_y.derivative()(t)
    dz = spline_z.derivative()(t)
    tangent = np.array([dx, dy, dz])
    tangent_norm = tangent / np.linalg.norm(tangent)
    return tangent_norm

def compute_spline_normal(spline_x, spline_y, spline_z, t):
    """
    Compute curvature center for normal vector calculation
    """
    # First derivative (tangent)
    tangent = compute_spline_tangent(spline_x, spline_y, spline_z, t)
    
    # Second derivative (curvature)
    d2x = spline_x.derivative(2)(t)
    d2y = spline_y.derivative(2)(t)
    d2z = spline_z.derivative(2)(t)
    curvature = np.array([d2x, d2y, d2z])
    
    # If curvature is too small, use a default normal
    curvature_norm = np.linalg.norm(curvature)
    if curvature_norm < 1e-10:
        # Use arbitrary perpendicular vector
        if abs(tangent[0])<1.:
            normal = np.cross(tangent, np.array([1.,0.,0.]))
        else:
            normal = np.cross(tangent, np.array([0.,0.,1.]))
        normal = normal / np.linalg.norm(normal)
    else:
        normal = curvature / curvature_norm
        # Ensure normal is perpendicular to tangent
        normal = normal - np.dot(normal, tangent) * tangent
        normal = normal / np.linalg.norm(normal)
    
    return normal

def compute_rotation_matrix(tangent, normal, twist_angle):
    """
    Compute rotation matrix from tangent, normal, and twist angle
    """
    # Binormal (completes the coordinate system)
    binormal = np.cross(tangent, normal)
    binormal = binormal / np.linalg.norm(binormal)
    
    # Apply twist rotation around tangent axis
    cos_twist = np.cos(twist_angle)
    sin_twist = np.sin(twist_angle)
    
    # Rotate normal and binormal around tangent
    normal_rotated = cos_twist * normal + sin_twist * binormal
    binormal_rotated = -sin_twist * normal + cos_twist * binormal
    
    # Create rotation matrix
    rotation_matrix = np.column_stack([normal_rotated, binormal_rotated, tangent])
    return rotation_matrix

# Compute positions and orientations for nucleosomes along the spline path
def compute_nucleosome_positions_along_path(Nnuc, Llinkers, spline_x, spline_y, spline_z, total_contour_length):
    # Calculate total DNA length in base pairs
    total_dna_bp = 147 * Nnuc + 10.5 * sum(Llinkers)
    
    # Calculate cumulative positions for each nucleosome
    nucleosome_positions = []
    cumulative_bp = 10.5 * Llinkers[0]
    
    for i in range(Nnuc):
        # Add half nucleosome for dyad position
        cumulative_bp += 73.5
        
        # Map genomic position to contour length along spline
        t_param = (cumulative_bp / total_dna_bp)
        
        # Get position on spline
        position = np.array([spline_x(t_param), spline_y(t_param), spline_z(t_param)])
        
        # Get tangent (z-axis)
        tangent = compute_spline_tangent(spline_x, spline_y, spline_z, t_param)
        
        # Get normal (x-axis, pointing towards center)
        normal = compute_spline_normal(spline_x, spline_y, spline_z, t_param)
        
        # Calculate twist angle (-225 degrees per nucleosome, converted to radians)
        twist_angle = i * np.deg2rad(-225)
        
        # Compute rotation matrix
        rotation_matrix = compute_rotation_matrix(tangent, normal, twist_angle)
        
        # Offset nucleosome from path axis
        offset_distance = 50.0
        #offset_vector = normal * offset_distance
        offset_vector = -rotation_matrix[:,0] * offset_distance
        final_position = position + offset_vector

        nucleosome_positions.append({
            'position': final_position,
            'rotation_matrix': rotation_matrix,
            't_param': t_param,
            'path_position': position,
            'tangent': tangent,
            'normal': normal
        })

        # move from dyad to end of this nucleosome: remaining 73.5 bp
        cumulative_bp += 73.5
        # then add the internal linker after nuc i (for next nuc), i=0..Nnuc-2
        cumulative_bp += 10.5 * Llinkers[i + 1]
    
    return nucleosome_positions

def adjust_linker_dna_positions(linker_atoms, start_pos, end_pos):
    """
    Adjust linker DNA bead positions to be uniformly spaced along straight line
    between start_pos and end_pos
    """
    num_beads = len(linker_atoms)
    if num_beads == 0:
        return linker_atoms
    
    # Calculate direction and total distance
    direction = end_pos - start_pos
    total_distance = np.linalg.norm(direction)
    spacing = total_distance / (num_beads+1)
    direction = direction / total_distance
    # Reposition beads
    for i, atom in enumerate(linker_atoms):
        new_position = start_pos + direction*(i+1)*spacing
        atom.position = new_position
    
    return linker_atoms

# Function to output PDB lines
# modified by GU: avoid formatting error when coordinates exceed 8 digits
def pdbOut(atom, scale=1.0):
    pdbline = 'ATOM  {index:5d}  {name:3s} {resname:3s} {chain:1s}{mol_index:4d}    {x:>8.2f}{y:>8.2f}{z:>8.2f}\n'
    alphabet=list(string.ascii_uppercase)
    return pdbline.format(
        index=atom.index % 99999,
        name=atom.name,
        resname=atom.resname,
        chain=alphabet[(atom.mol_index-1)%len(alphabet)],
        mol_index=atom.res_index % 9999,
        x=scale*atom.position[0],
        y=scale*atom.position[1],
        z=scale*atom.position[2]
    )


# Function to read input parameters from a file
def read_input(filename):
    Nnuc = -1
    Llinkers = []
    H1s = []
    acetylations = []
    brd4s = []
    cg_coords=[]
    with open(filename, 'r') as f:
        lines = f.readlines()
        for l in lines:
            tokens=l.strip().split()
            key=tokens[0]
            if key=='nuc': # number of nucleosomes
                Nnuc = int(tokens[1])
            elif key=='ll': # linker lengths
                Llinkers = list(map(int, tokens[1:]))
            elif key=='H1': # linker histones (present or not)
                H1s = [bool(int(val)) for val in tokens[1:]]
            elif key=='ac': # acetylation (present or not)
                acetylations = [bool(int(val)) for val in tokens[1:]]
            elif key=='brd4': # brd4 (present or not)
                brd4s = [bool(int(val)) for val in tokens[1:]]
            elif key=='cg': # cg coordinates 
                coords_file = str(tokens[1])
                cg_coords = np.loadtxt(coords_file)
            elif key[0]=='#':
                continue
            else:
                raise RuntimeError('key',key,'not recognized, keys should be nuc, ll, H1, ac, brd4, or cg')
    # check consistency
    if Nnuc<0:
        raise RuntimeError('Nnuc<0')
    for L in Llinkers:
        if L<0:
            raise RuntimeError('linker<=0')
    if Nnuc>0 and len(Llinkers)!=Nnuc+1:
        raise RuntimeError('Nnuc!=0 and len(Llinkers)!=Nnuc+1')
    if Nnuc==0 and len(Llinkers)!=1:
        raise RuntimeError('Nnuc==0 and len(Llinkers)!=1')
    if Nnuc>0:
        if len(H1s)==0:
            for i in range(Nnuc):
                H1s.append(False)
        if len(acetylations)==0:
            for i in range(Nnuc):
                acetylations.append(False)
        if len(brd4s)==0:
            for i in range(Nnuc):
                brd4s.append(False)
        if len(H1s)!=Nnuc:
            raise RuntimeError('len(H1s)!=Nnuc')
        if len(acetylations)!=Nnuc:
            raise RuntimeError('len(acetylations)!=Nnuc')
    
    return Nnuc, Llinkers, H1s, acetylations, brd4s, cg_coords

# Function to generate a reference nucleosome
# modified by GU: now takes rotation matrix instead of angle as input
def generate_nucleosome(gamma_condensation=0.2, charge_H1=0., acetylation=False, brd4=False, H1=False, translation=np.array([0.,0.,0.]), rotation_matrix=np.eye(3), index_start=1, mol_index=1, res_index=1):

    # Geometry parameters
    scale = 1.05
    R = 41.9 / scale  # Angstrom
    P = 25.9 / scale  # Angstrom
    Rhis = 18.0       # Angstrom
    Reff = np.sqrt(R**2 + (P / (2 * np.pi))**2)
    Reff_his = np.sqrt(Rhis**2 + (P / (2 * np.pi))**2)
    Lnuc = 147  # bp
    Lhis = 126        # bp directly bound to histones
    Lbp1 = 10.5  # Number of bp per DNA bead
    alpha = 1.84
    alpha_his = 1.84 / Lnuc * Lhis
    Lbp1his = Lhis / 8  # Number of bp per histone bead

    # set types based on name
    types={}
    types['DLN']=1 # linker DNA
    types['DNC']=2 # nucleosomal DNA core 126 bps
    types['DEX']=3 # nucleosomal DNA exit
    types['H2A']=4
    types['H2B']=5
    types['H3'] =6
    types['H4'] =7
    # acetylated types
    types['DAN']=8 # nucleosomal DNA core 126 bps
    types['DAE']=9 # nucleosomal DNA exit
    types['A2A']=10
    types['A2B']=11
    types['A3'] =12
    types['A4'] =13
    # acetylated and bound to BRD4 types
    types['DBN']=14 # nucleosomal DNA core 126 bps
    types['DBE']=15 # nucleosomal DNA exit
    types['B2A']=16
    types['B2B']=17
    types['B3'] =18
    types['B4'] =19
    # linker histone
    types['H1'] =20
    
    masses={}
    masses['DLN']=1.0 # linker DNA
    masses['DNC']=1.0 # nucleosomal DNA core 126 bps
    masses['DEX']=1.0 # nucleosomal DNA exit
    masses['H2A']=1.0
    masses['H2B']=1.0
    masses['H3'] =1.0
    masses['H4'] =1.0
    # acetylated types
    masses['DAN']=1.0 # nucleosomal DNA core 126 bps
    masses['DAE']=1.0 # nucleosomal DNA exit
    masses['A2A']=1.0
    masses['A2B']=1.0
    masses['A3'] =1.0
    masses['A4'] =1.0
    # acetylated types
    masses['DBN']=1.0 # nucleosomal DNA core 126 bps
    masses['DBE']=1.0 # nucleosomal DNA exit
    masses['B2A']=1.0
    masses['B2B']=1.0
    masses['B3'] =1.0
    masses['B4'] =1.0
    # linker histone
    masses['H1'] = 1.0

    # Generate histone positions
    his_positions = []
    ids_his = np.arange(-Lhis/2 + Lbp1his/2, Lhis/2, Lbp1his)
    if (acetylation and brd4):
        his_names=['B2A','B2B','B3','B4','B4','B3','B2B','B2A']
    elif brd4:
        raise RuntimeError('brd4 binding without acetylation is not allowed')
    elif acetylation:
        his_names=['A2A','A2B','A3','A4','A4','A3','A2B','A2A']
    else:
        his_names=['H2A','H2B','H3','H4','H4','H3','H2B','H2A']
    charge_his=[0. for i in his_names]
    for i in ids_his:
        s = i * (2 * np.pi * Reff_his * alpha_his) / (Lhis - 1)
        x = Rhis * np.cos(s / Reff_his)
        y = Rhis * np.sin(s / Reff_his)
        z = -(P / (2 * np.pi * Reff_his)) * s
        his_positions.append(np.array([x, y, z]) / scale)
    # Atom types for histones from 4
    his_atoms = [Atom(index=0, atom_type=types[name], position=pos, name=name, charge=q, mass=masses[name]) for name, pos, q in zip(his_names,his_positions,charge_his)]

    # Generate nucleosomal DNA positions
    charge_single=-gamma_condensation*2.*Lbp1
    ids_dna = np.arange(-Lnuc/2 + Lbp1/2, Lnuc/2, Lbp1)
    if (acetylation and brd4):
        dna_names=['DBN' for i in ids_dna]
        dna_names[0] = 'DBE'
        dna_names[-1] = 'DBE'
    elif acetylation:
        dna_names=['DAN' for i in ids_dna]
        dna_names[0] = 'DAE'
        dna_names[-1] = 'DAE'
    else:
        dna_names=['DNC' for i in ids_dna]
        dna_names[0] = 'DEX'
        dna_names[-1] = 'DEX'
    dna_positions = []
    for i in ids_dna:
        s = i * (2 * np.pi * Reff * alpha) / (Lnuc - 1)
        x = R * np.cos(s / Reff)
        y = R * np.sin(s / Reff)
        z = -(P / (2 * np.pi * Reff)) * s
        dna_positions.append(np.array([x, y, z]))
    # Atom types for nucleosomal DNA
    dna_atoms = [Atom(index=0, atom_type=types[name], position=pos, name=name, charge=charge_single,mass=1.0) for name, pos in zip(dna_names,dna_positions)]

    # add H1
    H1_atoms = []
    if H1:
        name='H1'
        pos=np.array([70.,0.,0.])
        H1_atoms.append( Atom(index=0, atom_type=types[name], position=pos, name=name, charge=0.5*charge_H1,mass=masses[name]) )
        pos=np.array([70.+37.205,0.,0.])
        H1_atoms.append( Atom(index=0, atom_type=types[name], position=pos, name=name, charge=0.5*charge_H1,mass=masses[name]) )

    # Combine histones and nucleosomal DNA
    nucleosome = his_atoms + dna_atoms + H1_atoms
    len_his = len(his_atoms)

    # Update atom indices
    atom_index = index_start
    for atom in nucleosome:
        atom.index = atom_index
        atom.mol_index = mol_index
        atom.res_index = res_index
        atom_index += 1
    # add bonds for nucleosome
    # bonds have to be created before rotation otherwise there are issues in calculation of atom types, code assumes nucleosome along z
    bonds_nuc=generate_nucleosome_bonds(nucleosome)
    angles_nuc=generate_nucleosome_angles(nucleosome)
    # apply a rotation around x by 180 degrees so that nucleosome points in the +z direction
    nucleosome = transform_nucleosome(nucleosome, rotation_matrix=np.array([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.,]]))
    # translate so that first DNA bead is at the origin in the reference nucleosome
    start_pos = nucleosome[len_his].position
    nucleosome = transform_nucleosome(nucleosome, translation=-start_pos)
    # translate and rotate (around com) nucleosome if these are provided
    nucleosome = transform_nucleosome(nucleosome, translation=translation, rotation_matrix=rotation_matrix)

    return nucleosome,bonds_nuc,angles_nuc,atom_index

# Function to add linker DNA
def generate_linker_dna(start_pos, direction, num_beads, gamma_dna_condensation=1.):
    bead_distance = 35.0  # Angstrom
    charge_dna = -gamma_dna_condensation*2.*10.5
    linker_atoms = []
    for i in range(num_beads):
        position = start_pos + direction * bead_distance * (i + 1)
        atom = Atom(index=0, atom_type=1, position=position, mol_index=1, res_index=0, name='DLN', charge=charge_dna,mass=1.0)
        linker_atoms.append(atom)
    return linker_atoms

# generate pairs stabilizing the DNA entry/exit
def generate_nucleosome_exit_pairs(nucleosome):
    bonds=[]
    pairs=[]
    index=0
    id0=nucleosome[0].index # index of first nucleosome atom
    for i in range(len(nucleosome)):
        for j in range(i+1,len(nucleosome)):
            a1=nucleosome[i]
            a2=nucleosome[j]
            t1=a1.atom_type
            t2=a2.atom_type
            n1=a1.name
            n2=a2.name
            v=a2.position-a1.position
            d=np.linalg.norm(v)
            if d>38.: continue # too far for bond
            if (n1=='DEX' or n2=='DEX'): # make pair and continue
                pair=Pair(type1=t1, type2=t2, distance=d, epsilon=1.0)
                pairs.append(pair)
    return pairs

# generate bonds stabilizing nucleosome
def generate_nucleosome_bonds(nucleosome):
    bonds=[]
    index=0
    id0=nucleosome[0].index # index of first nucleosome atom
    bond_types={} # dictionary to set bond types based on key, type 1 reserved for linker DNA
    bond_types[(3,1)]=2 # bond along DNA backbone
    bond_types[(4,0)]=3 # nucleosomal dna exit-H4 bond
    bond_types[(4,1)]=4 # nucleosomal dna exit-H2A bond, do not include, as it does not discriminate binding
    bond_types[(4,2)]=5 # nucleosomal dna exit-dna bond
    # same but for acetylated
    bond_types[(5,0)]=6 # nucleosomal dna exit-H4 bond
    bond_types[(5,1)]=7 # nucleosomal dna exit-H2A bond, do not include, as it does not discriminate binding
    bond_types[(5,2)]=8 # nucleosomal dna exit-dna bond
    # same but for acetylated+brd4
    bond_types[(6,0)]=9 # nucleosomal dna exit-H4 bond
    bond_types[(6,1)]=10 # nucleosomal dna exit-H2A bond, do not include, as it does not discriminate binding
    bond_types[(6,2)]=11 # nucleosomal dna exit-dna bond
    core_histone_names=['H2A','H2B','H3','H4','A2A','A2B','A3','A4','B2A','B2B','B3','B4']
    DNC_names=['DNC','DAN','DBN']
    for i in range(len(nucleosome)):
        for j in range(i+1,len(nucleosome)):
            a1=nucleosome[i]
            a2=nucleosome[j]
            id1=a1.index
            id2=a2.index
            n1=a1.name
            n2=a2.name
            v=a2.position-a1.position
            d=np.linalg.norm(v)
            potential='harmonic'
            if d>38.: continue # too far for bond
            # make a new bond
            # assume histone beads come first, then DNA beads
            if (n1=='H4' and n2=='DEX'): # morse bond for nucleosome unwrapping
                key1=4
                key2=0
                potential='morse'
            elif (n1=='H2A' and n2=='DEX'): # morse bond for nucleosome unwrapping
                key1=4
                key2=1
                potential='morse'
            elif ( ((n1=='DNC' and n2=='DEX') or (n1=='DEX' and n2=='DNC')) and id2-id1>1 ): # morse bond for nucleosome unwrapping
                key1=4
                key2=2
                potential='morse'
            elif (n1=='A4' and n2=='DAE'): # morse bond for nucleosome unwrapping
                key1=5
                key2=0
                potential='morse2'
            elif (n1=='A2A' and n2=='DAE'): # morse bond for nucleosome unwrapping
                key1=5
                key2=1
                potential='morse2'
            elif ( ((n1=='DAN' and n2=='DAE') or (n1=='DAE' and n2=='DAN')) and id2-id1>1 ): # morse bond for nucleosome unwrapping
                key1=5
                key2=2
                potential='morse2'
            elif (n1=='B4' and n2=='DBE'): # morse bond for nucleosome unwrapping
                key1=6
                key2=0
                potential='morse2'
            elif (n1=='B2A' and n2=='DBE'): # morse bond for nucleosome unwrapping
                key1=6
                key2=1
                potential='morse2'
            elif ( ((n1=='DBN' and n2=='DBE') or (n1=='DBE' and n2=='DBN')) and id2-id1>1 ): # morse bond for nucleosome unwrapping
                key1=6
                key2=2
                potential='morse2'
            elif (n1 in core_histone_names and n2 in core_histone_names ): # his-his
                key1=1
                key2=id2-id1 # for his-his bonds, what matters is their distance along the superhelix
            elif (n1 in core_histone_names and n2 in DNC_names ): # his-nuc, ignore linker
                key1=2
                # each histone dimer interacts with 3 DNA beads, so their relative positions are captured by taking %2,3
                # but consider that the nucleosome is symmetric relative to the dyad
                shl1=id1-id0-3.5
                k1=int(round((abs(shl1)-0.5)%2))
                shl2=id2-id0-8-6.5 # map to SHL
                k2=int(round((abs(shl2)-0.5)%3)) # take modulus to conside position relative to histones
                # check if histone and DNA are on the same gyre
                dz=np.fabs(v[2]) # distance along z determines whether histone and DNA are part of the same gyre
                gyre_diff=int(round(dz/20.))
                # check if histone and DNA are on the same half side of the nucleosome
                side1=int(round(np.sign(shl1)))
                side2=int(round(np.sign(shl2)))
                side=side1*side2
                # combine all4 properties
                key2=(k1,k2,gyre_diff,side)
            elif (n1 in DNC_names and n2 in DNC_names ):
                key1=3
                key2=id2-id1 # for dna-dna bonds, what matters is their distance along the superhelix
            elif ( (n1 in DNC_names and n2=='H1') or (n1=='H1' and n2=='H1') ): # linker histone bonds
                key1=7
                key2=0
            else:
                continue
            key=(key1,key2)
            if key not in bond_types: # new bond type
                bond_types[key]=max(bond_types.values())+1
            bond_type=bond_types[key]
            index+=1
            bond=Bond(index=index, bond_type=bond_type, index1=id1, index2=id2, distance=d, key=(key1,key2), potential=potential)
            bonds.append(bond)
    # add bonds with exit DNA
    for i in range(8,22-1):
        a1=nucleosome[i]
        a2=nucleosome[i+1]
        id1=a1.index
        id2=a2.index
        n1=a1.name
        n2=a2.name
        v=a2.position-a1.position
        d=np.linalg.norm(v)
        if (n1 in DNC_names and n2 in DNC_names): continue # not near exit
        index+=1
        key=(3,id2-id1)
        bond_type=bond_types[key]
        bond=Bond(index=index, bond_type=bond_type, index1=id1, index2=id2, distance=d, key=key)
        bonds.append(bond)
    return bonds

# angles with exit DNA
def generate_nucleosome_angles(nucleosome):
    angles=[]
    index=0
    for i in range(8,22-2):
        a1=nucleosome[i]
        a2=nucleosome[i+1]
        a3=nucleosome[i+2]
        id1=a1.index
        id2=a2.index
        id3=a3.index
        bond_type=1
        index+=1
        angle=Angle(index=index, bond_type=bond_type, index1=id1, index2=id2, index3=id3)
        angles.append(angle)
    # add angles to stabilize H1
    if len(nucleosome)==22+2:
        if nucleosome[-1].name=='H1' and nucleosome[-2].name=='H1':
            a1=nucleosome[13]
            a2=nucleosome[14]
            a3=nucleosome[-2]
            id1=a1.index
            id2=a2.index
            id3=a3.index
            bond_type=1
            index+=1
            angle=Angle(index=index, bond_type=bond_type, index1=id1, index2=id2, index3=id3)
            angles.append(angle)
            a1=nucleosome[16]
            a2=nucleosome[15]
            a3=nucleosome[-2]
            id1=a1.index
            id2=a2.index
            id3=a3.index
            bond_type=1
            index+=1
            angle=Angle(index=index, bond_type=bond_type, index1=id1, index2=id2, index3=id3)
            angles.append(angle)
        else:
            raise RuntimeError('H1 not found')
    elif len(nucleosome)!=22:
        raise RuntimeError('incorrect number of nucleosome beads')
    return angles

# generate bonds and angles along linker DNA
def generate_linker_dna_bonds_and_angles(linker_atoms):
    bonds=[]
    angles=[]
    index=0
    for i in range(len(linker_atoms)-1):
        a1=linker_atoms[i]
        a2=linker_atoms[i+1]
        id1=a1.index
        id2=a2.index
        index+=1
        bond_type=1
        bond =Bond( index=index, bond_type=bond_type, index1=id1, index2=id2)
        bonds.append(bond)
        if i<len(linker_atoms)-2:
            a3=linker_atoms[i+2]
            id3=a3.index
            angle=Angle(index=index, bond_type=bond_type, index1=id1, index2=id2, index3=id3)
            angles.append(angle)
    return bonds,angles

def generate_bond_settings(bonds, filename):
    # ensure bonds of the same type have similar distances
    bond_types=[b.bond_type for b in bonds]
    Nbond_types=max(bond_types)
    if min(bond_types)!=1 or Nbond_types!=len(set(bond_types)):
        raise RuntimeError('inconsistent bond types')
    eq_distances={}
    potentials={}
    for i in range(1,Nbond_types+1):
        eq_distances[i]=np.array([b.distance for b in bonds if b.bond_type==i])
        potentials[i]  =[b.potential for b in bonds if b.bond_type==i][0]
        std=eq_distances[i].std()
        mean=eq_distances[i].mean()
        key=[b.key for b in bonds if b.bond_type==i][0]
        if std>0.01:
            RuntimeError('std of bond distances is too high')
        eq_distances[i]=mean
    with open(filename, 'w') as fout:
        for i in eq_distances:
            if potentials[i]=='morse':
                fout.write('bond_coeff {} morse ${{kmorse}} ${{alphamorse}} {:.3f} # eq: {:.3f}\n'.format(i,0.98*eq_distances[i],eq_distances[i]))
            elif potentials[i]=='morse2':
                fout.write('bond_coeff {} morse ${{kmorse2}} ${{alphamorse}} {:.3f} # eq: {:.3f}\n'.format(i,0.98*eq_distances[i],eq_distances[i]))
            elif potentials[i]=='harmonic':
                fout.write('bond_coeff {} harmonic ${{kbond}} {:.3f}\n'.format(i,eq_distances[i]))
            else:
                raise RuntimeError('bond potential not recognized, should be harmonic or morse')
    print(f'# bond settings distances written to {filename}')

# ensure indices go from 1 to Nbond_types
# and types go from 1 to Ntypes
def fix_bonds(bonds):
    # indices
    index=0
    for b in bonds:
        index+=1
        b.index=index
    # types
    bond_types=[b.bond_type for b in bonds]
    set_bond_types=set(bond_types)
    Nbond_types=len(set_bond_types)
    if min(bond_types)==1 and max(bond_types)==Nbond_types:
        return # nothing to fix for the bonds
    # reorder the bond_types so that these go from 1 to Nbond_types
    type_mapping = {old_type: new_type for new_type, old_type in enumerate(sorted(set_bond_types), start=1)}
    for b in bonds:
        b.bond_type = type_mapping[b.bond_type]

# ensure atom types go from 1 to Ntypes
def fix_atom_types(atoms):
    atom_types=[a.atom_type for a in atoms]
    set_atom_types=set(atom_types)
    Natom_types=len(set_atom_types)
    if min(atom_types)==1 and max(atom_types)==Natom_types:
        return # nothing to fix for the atoms
    # reorder the atom_types so that these go from 1 to Natom_types
    type_mapping = {old_type: new_type for new_type, old_type in enumerate(sorted(set_atom_types), start=1)}
    for b in atoms:
        b.atom_type = type_mapping[b.atom_type]

# Function to rotate and translate nucleosome
# modified by GU
# use rotation matrix 3*3 ndarray as input 
def transform_nucleosome(nucleosome, translation=np.array([0.,0.,0.]), rotation_matrix=np.eye(3)):
    transformed_nucleosome = []
    # Transform each atom
    for atom in nucleosome:
        rotated_position = rotation_matrix @ atom.position
        translated_position = rotated_position + translation
        transformed_atom = Atom(
            index=atom.index,
            atom_type=atom.atom_type,
            position=translated_position,
            mol_index=atom.mol_index,
            res_index=atom.res_index,
            name=atom.name,
            resname=atom.resname,
            charge=atom.charge,
            mass=atom.mass,
        )
        transformed_nucleosome.append(transformed_atom)
    return transformed_nucleosome
# Reposition leading linker beads to lie on a straight line extending from
# the first nucleosome, using the first 2 nucleosomal DNA beads to define direction.
# linker_atoms: list[Atom] in chain order (start -> end), where the END bead bonds to nuc_dna_atoms[0].
# nuc_dna_atoms: list[Atom] DNA beads of nucleosome 0 in their internal order (0,1,...)
def reposition_leading_linker_from_first_nuc(linker_atoms, nuc_dna_atoms, bead_distance=35.0):
    L = len(linker_atoms)
    if L == 0:
        return

    p0 = nuc_dna_atoms[0].position
    p1 = nuc_dna_atoms[1].position
    v = p0 - p1
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        raise RuntimeError("First two nucleosomal DNA beads coincide; cannot define leading linker direction.")
    u = v / nv  # outward direction

    # Fill positions so that the last linker bead (closest) is at p0 + u*d
    for j in range(L):
        k = L - j  # k = L, L-1, ..., 1
        linker_atoms[j].position = p0 + u * bead_distance * k

# to make sure the fiber is still along z, 
def initial_linker_direction_for_z_rise(bead_distance=35.0, dz_per_bead=3.53):
    dz = dz_per_bead / bead_distance
    dz = np.clip(dz, -1.0, 1.0)
    dx = np.sqrt(max(0.0, 1.0 - dz*dz))
    return np.array([dx, 0.0, dz], float)

# Main function to build the chromatin fiber
def build_chromatin(Nnuc,Llinkers,cg_coords=[],gamma_dna_condensation=1.,gamma_nuc_condensation=1.,acetylations=[],H1s=[],brd4s=[],charge_H1=0.):
    chromatin = []
    bonds=[]  # type,id1,id2
    angles=[] # type,id1,id2,id3
    atom_index = 1
    mol_index = 1
    res_index = 1
    
    if len(H1s)==0:
        H1s = [False] * Nnuc
    if len(acetylations)==0:
        acetylations = [False] * Nnuc
    if len(brd4s) == 0:
        brd4s = [False] * Nnuc
    if (len(acetylations)!=Nnuc or len(H1s)!=Nnuc or len(brd4s)!=Nnuc or len(Llinkers)!=Nnuc+1):
        raise RuntimeError('len(acetylations)!=Nnuc or len(H1s)!=Nnuc or len(brd4s)!=Nnuc or len(Llinkers)!=Nnuc+1')

    ref_nucleosome,tmp_bonds,tmp_angle,tmp_i = generate_nucleosome()
    # Generate reference nucleosome
    nucleosomal_dna_atoms_ref = [atom for atom in ref_nucleosome if atom.name[0]=='D']
    # v1: vector between first two nucleosomal DNA beads in reference nucleosome
    # will be used to decide rotation angle to align nucleosome's start with previous linker DNA
    start_pos_nuc = nucleosomal_dna_atoms_ref[0].position
    next_pos_nuc = nucleosomal_dna_atoms_ref[1].position
    v1 = next_pos_nuc - start_pos_nuc
    v1_xy = v1.copy()
    v1_xy[2] = 0.0  # Project onto xy-plane
    v1_norm = v1_xy / np.linalg.norm(v1_xy)

    # If using reference path
    if len(cg_coords) > 0:
        if len(cg_coords) < 2:
            raise RuntimeError('Need at least 2 points in cg_coords for spline fitting')
        # Compute spline path
        spline_x, spline_y, spline_z, total_contour_length = compute_spline_path(cg_coords)
        # Compute nucleosome positions along path
        nucleosome_data = compute_nucleosome_positions_along_path(Nnuc, Llinkers, spline_x, spline_y, spline_z, total_contour_length)

    # add leading linker DNA
    linker_dna = []
    num_linker_beads = Llinkers[0]
    if num_linker_beads > 0:
        start_pos = np.array([0.0, 0.0, 0.0])
        direction = initial_linker_direction_for_z_rise()
        linker_dna = generate_linker_dna(start_pos, direction, num_linker_beads, gamma_dna_condensation)

        for atom in linker_dna:
            atom.index = atom_index
            atom.mol_index = mol_index
            atom_index += 1
        chromatin.extend(linker_dna)

        # bonds/angles within linker
        bonds_linker, angles_linker = generate_linker_dna_bonds_and_angles(linker_dna)
        bonds.extend(bonds_linker)
        angles.extend(angles_linker)

        # set prev_prev_end_pos and prev_end_pos so nucleosome 0 placement works
        if num_linker_beads >= 2:
            prev_prev_end_pos = linker_dna[-2].position
            prev_end_pos      = linker_dna[-1].position
        else:  # num_linker_beads == 1
            prev_prev_end_pos = start_pos
            prev_end_pos      = linker_dna[-1].position
    else:
        prev_prev_end_pos = None
        prev_end_pos      = None
   
    # main loop over nucleosomes 
    for i in range(Nnuc):

        # build a straight fiber without any reference
        if len(cg_coords)==0:
            if i == 0 and (prev_end_pos is None):
                translation = np.array([0.0, 0.0, 0.0])
                rotation_matrix = np.eye(3)
            else:
                # Determine rotation angle to align nucleosome's start with previous linker DNA
                # v2: direction from previous linker DNA
                v2 = prev_end_pos - prev_prev_end_pos
                v2_xy = v2.copy()
                v2_xy[2] = 0.0  # Project onto xy-plane
                v2_norm = v2_xy / np.linalg.norm(v2_xy)
                # Translation to connect to previous linker DNA
                translation = prev_end_pos + v2
                # Compute angle between v1 and v2
                dot_prod = np.dot(v1_norm, v2_norm)
                # Ensure dot product is within valid range for arccos due to numerical errors
                dot_prod = np.clip(dot_prod, -1.0, 1.0)
                angle = np.arccos(dot_prod)
                # Determine sign of rotation using cross product
                cross_prod_z = np.cross(v1_norm, v2_norm)[2]
                if cross_prod_z < 0:
                    rotation_angle = -angle
                else:
                    rotation_angle = angle
                cos_theta = np.cos(rotation_angle)
                sin_theta = np.sin(rotation_angle)
                rotation_matrix = np.array([
                    [cos_theta, -sin_theta, 0],
                    [sin_theta,  cos_theta, 0],
                    [0,          0,         1]
                ])
        else:
            # Use reference path
            data = nucleosome_data[i]
            translation = data['position']
            rotation_matrix = data['rotation_matrix']
        
        # generate and transform nucleosome
        nucleosome,bonds_nuc,angles_nuc,atom_index = generate_nucleosome(
            gamma_condensation=gamma_nuc_condensation,
            charge_H1=charge_H1,
            acetylation=acetylations[i],
            brd4=brd4s[i],
            H1=H1s[i],
            index_start=atom_index,
            mol_index=mol_index,
            res_index=res_index,
            translation=translation,
            rotation_matrix=rotation_matrix
        )

        # add nucleosome to fiber
        bonds.extend(bonds_nuc)
        angles.extend(angles_nuc)
        chromatin.extend(nucleosome)

        last_dna_atoms = [atom for atom in nucleosome if atom.name[0]=='D']
        # Store current nucleosome's first DNA bead position for linker adjustment
        current_nuc_first_dna_pos = last_dna_atoms[0].position if last_dna_atoms else None

        # add bonds and angles to previous linker
        # DNA beads except those of last nucleosome
        last_dna_atoms_prev_nuc = [atom for atom in chromatin if atom.name[0]=='D'][:-len(last_dna_atoms)-num_linker_beads]
        if num_linker_beads >= 2:
            bonds.append( Bond( index=0,bond_type=1,index1=linker_dna[-1].index,index2=last_dna_atoms[0].index))
            angles.append(Angle(index=0,bond_type=1,index1=linker_dna[-1].index,index2=last_dna_atoms[0].index,index3=last_dna_atoms[1].index))
            angles.append(Angle(index=0,bond_type=1,index1=linker_dna[-2].index,index2=linker_dna[-1].index,   index3=last_dna_atoms[0].index))
        elif num_linker_beads == 1:
            bonds.append( Bond( index=0,bond_type=1,index1=linker_dna[-1].index,index2=last_dna_atoms[0].index))
            angles.append(Angle(index=0,bond_type=1,index1=linker_dna[-1].index,index2=last_dna_atoms[0].index,index3=last_dna_atoms[1].index))
            if i>0: # not first nucleosome, make sure there is another nucleosome before nucleosome
                angles.append(Angle(index=0,bond_type=1,index1=last_dna_atoms_prev_nuc[-1].index,index2=linker_dna[-1].index,index3=last_dna_atoms[0].index))
        elif i>0: # no linker, but there is a previous nucleosome to make a bond with
            bonds.append( Bond( index=0,bond_type=1,index1=last_dna_atoms_prev_nuc[-1].index,index2=last_dna_atoms[0].index))
            angles.append(Angle(index=0,bond_type=1,index1=last_dna_atoms_prev_nuc[-1].index,index2=last_dna_atoms[0].index,index3=last_dna_atoms[1].index))
            angles.append(Angle(index=0,bond_type=1,index1=last_dna_atoms_prev_nuc[-2].index,index2=last_dna_atoms_prev_nuc[-1].index,index3=last_dna_atoms[0].index))

        # linker adjustment if using path
        if len(cg_coords) > 0:
            if i==0 and len(linker_dna)>0:
                reposition_leading_linker_from_first_nuc(linker_dna, last_dna_atoms)
            if i>0:
                adjust_linker_dna_positions(linker_dna, prev_nuc_last_dna_pos, current_nuc_first_dna_pos)

        # Store current nucleosome's last DNA position for next iteration
        prev_nuc_last_dna_pos = last_dna_atoms[-1].position if last_dna_atoms else None

        # Add linker DNA if not the last nucleosome
        if i < Nnuc - 1:
            num_linker_beads = Llinkers[i+1]

            if len(cg_coords) == 0:
                # Straight fiber: use direction from last two DNA beads
                start_pos = last_dna_atoms[-1].position
                prev_pos = last_dna_atoms[-2].position
                direction = start_pos - prev_pos
                direction /= np.linalg.norm(direction)
                
                prev_prev_end_pos = prev_pos
                prev_end_pos = start_pos
            else:
                # Follow reference path: interpolate between nucleosome positions
                current_data = nucleosome_data[i]
                next_data = nucleosome_data[i + 1]
                
                start_pos = last_dna_atoms[-1].position
                end_pos = next_data['path_position'] + next_data['normal'] * 110.0  # Approximate next nucleosome start
                
                # Direction vector for linker
                direction = end_pos - start_pos
                direction_length = np.linalg.norm(direction)
                if direction_length > 0:
                    direction /= direction_length
                else:
                    direction = np.array([0.0, 0.0, 1.0])  # Fallback
                
                prev_prev_end_pos = start_pos
                prev_end_pos = end_pos

            linker_dna = generate_linker_dna(start_pos, direction, num_linker_beads, gamma_dna_condensation)

            # Update atom indices and molecule index
            for atom in linker_dna:
                atom.index = atom_index
                atom.mol_index = mol_index
                atom_index += 1

            # add linker to fiber
            chromatin.extend(linker_dna)

            # Update prev_end_pos and prev_prev_end_pos for next nucleosome
            # also, add bonds and angles between previous nucleosome and new linker
            if num_linker_beads >= 2:
                prev_prev_end_pos = linker_dna[-2].position
                prev_end_pos = linker_dna[-1].position
                bonds.append( Bond( index=0,bond_type=1,index1=last_dna_atoms[-1].index,index2=linker_dna[0].index))
                angles.append(Angle(index=0,bond_type=1,index1=last_dna_atoms[-2].index,index2=last_dna_atoms[-1].index,index3=linker_dna[0].index))
                angles.append(Angle(index=0,bond_type=1,index1=last_dna_atoms[-1].index,index2=linker_dna[0].index,     index3=linker_dna[1].index))
            elif num_linker_beads == 1:
                prev_prev_end_pos = start_pos
                prev_end_pos = linker_dna[-1].position
                bonds.append( Bond( index=0,bond_type=1,index1=last_dna_atoms[-1].index,index2=linker_dna[0].index))
                angles.append(Angle(index=0,bond_type=1,index1=last_dna_atoms[-2].index,index2=last_dna_atoms[-1].index,index3=linker_dna[0].index))
            else:
                # No linker DNA, use last two nucleosomal DNA beads
                prev_prev_end_pos = last_dna_atoms[-2].position
                prev_end_pos = last_dna_atoms[-1].position

            # add bonds for linker
            bonds_linker,angles_linker=generate_linker_dna_bonds_and_angles(linker_dna)
            bonds.extend(bonds_linker)
            angles.extend(angles_linker)

        else:
            # For the last nucleosome, update prev_end_pos for completeness
            last_dna_atoms = [atom for atom in nucleosome if atom.name[0]=='D']
            prev_prev_end_pos = last_dna_atoms[-2].position
            prev_end_pos = last_dna_atoms[-1].position

        res_index +=1

    # add trailing linker DNA
    trailing = Llinkers[Nnuc]
    if trailing > 0:
        last_dna_atoms = [atom for atom in chromatin if atom.name[0]=='D']  # or keep last_dna_atoms from final nuc
        start_pos = last_dna_atoms[-1].position
        prev_pos  = last_dna_atoms[-2].position
        direction = start_pos - prev_pos
        direction /= np.linalg.norm(direction)

        linker_dna = generate_linker_dna(start_pos, direction, trailing, gamma_dna_condensation)

        for atom in linker_dna:
            atom.index = atom_index
            atom.mol_index = mol_index
            atom_index += 1
        chromatin.extend(linker_dna)

        # bond/angle between nuc and first linker bead (same pattern you already use)
        if trailing >= 2:
            bonds.append(Bond(index=0, bond_type=1, index1=last_dna_atoms[-1].index, index2=linker_dna[0].index))
            angles.append(Angle(index=0, bond_type=1, index1=last_dna_atoms[-2].index, index2=last_dna_atoms[-1].index, index3=linker_dna[0].index))
            angles.append(Angle(index=0, bond_type=1, index1=last_dna_atoms[-1].index, index2=linker_dna[0].index, index3=linker_dna[1].index))
        else:  # trailing == 1
            bonds.append(Bond(index=0, bond_type=1, index1=last_dna_atoms[-1].index, index2=linker_dna[0].index))
            angles.append(Angle(index=0, bond_type=1, index1=last_dna_atoms[-2].index, index2=last_dna_atoms[-1].index, index3=linker_dna[0].index))

        # internal linker bonds/angles
        bonds_linker, angles_linker = generate_linker_dna_bonds_and_angles(linker_dna)
        bonds.extend(bonds_linker)
        angles.extend(angles_linker)

    # fix bonds and angles indices and types
    fix_bonds(bonds)
    fix_bonds(angles)

    return chromatin,bonds,angles

# Main function to build the DNA alone
def build_only_linker_dna(Nlinker, gamma_dna_condensation=1.):
    atom_index = 1
    mol_index = 1
    start_pos = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])  # Initially along z-axis

    # generate linker
    linker_dna = generate_linker_dna(start_pos, direction, Nlinker, gamma_dna_condensation)

    # Update atom indices and molecule index
    for atom in linker_dna:
        atom.index = atom_index
        atom.mol_index = mol_index
        atom_index += 1

    # add bonds for linker
    bonds,angles=generate_linker_dna_bonds_and_angles(linker_dna)

    # fix bonds and angles indices and types
    fix_bonds(bonds)
    fix_bonds(angles)
    # fix atom types
    fix_atom_types(linker_dna)

    return linker_dna,bonds,angles

# Function to write PDB file
def write_pdb(chromatin, filename, scale=1.0):
    with open(filename, 'w') as fout:
        for atom in chromatin:
            line = pdbOut(atom, scale=scale)
            fout.write(line)
    print(f'# PDB file written to {filename}')

# Function to LAMMPS data file
def write_lammps_data(chromatin, bonds, angles, filename, halfbox=-1):
    atom_indices= [a.index for a in chromatin]
    bond_indices= [b.index for b in bonds]
    angle_indices=[b.index for b in angles]
    atom_types= [a.atom_type for a in chromatin]
    bond_types= [b.bond_type for b in bonds]
    angle_types=[b.bond_type for b in angles]
    Natom_types =len(set(atom_types))
    Nbond_types =len(set(bond_types))
    Nangle_types=len(set(angle_types))
    if halfbox<=0.:
        halfbox=[0.,0.,0.]
        halfbox[0] = 0.5*( max([a.position[0] for a in chromatin]) - min([a.position[0] for a in chromatin]) )+35.
        halfbox[1] = 0.5*( max([a.position[1] for a in chromatin]) - min([a.position[1] for a in chromatin]) )+35.
        halfbox[2] = 0.5*( max([a.position[2] for a in chromatin]) - min([a.position[2] for a in chromatin]) )+35.
    else:
        halfbox=[halfbox,halfbox,halfbox]
    if (min(atom_indices)!=1 or max(atom_indices)!=len(chromatin)):
        raise RuntimeError('inconsistent atom indices')
    if (min(bond_indices)!=1 or max(bond_indices)!=len(bonds)):
        raise RuntimeError('inconsistent bond indices')
    if (min(angle_indices)!=1 or max(angle_indices)!=len(angles)):
        raise RuntimeError('inconsistent angle indices')
    if (min(atom_types)!=1 or max(atom_types)!=Natom_types):
        raise RuntimeError('inconsistent atom types')
    if (min(bond_types)!=1 or max(bond_types)!=Nbond_types):
        raise RuntimeError('inconsistent bond types')
    if (min(angle_types)!=1 or max(angle_types)!=Nangle_types):
        raise RuntimeError('inconsistent angle types')
    with open(filename, 'w') as fout:
        fout.write('LAMMPS data file\n\n')
        fout.write('{} atoms\n'.format(len(chromatin)))
        fout.write('{} bonds\n'.format(len(bonds)))
        fout.write('{} angles\n\n'.format(len(angles)))
        fout.write('{} atom types\n'.format(Natom_types))
        fout.write('{} bond types\n'.format(Nbond_types))
        fout.write('{} angle types\n\n'.format(Nangle_types))
        fout.write('-{:.1f} {:.1f} xlo xhi\n'.format(halfbox[0],halfbox[0]))
        fout.write('-{:.1f} {:.1f} ylo yhi\n'.format(halfbox[1],halfbox[1]))
        fout.write('-{:.1f} {:.1f} zlo zhi\n\n'.format(halfbox[2],halfbox[2]))
        fout.write('Masses\n\n')
        for i in range(1,Natom_types+1):
            mass=[a.mass for a in chromatin if a.atom_type==i][0]
            fout.write('{} {:.1f}\n'.format(i,mass))
        fout.write('\n')
        fout.write('Atoms # index molecule type q x y z\n\n')
        for a in chromatin:
            fout.write('{} {} {} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(a.index,a.mol_index,a.atom_type,a.charge,a.position[0],a.position[1],a.position[2]))
        fout.write('\n')
        fout.write('Bonds\n\n')
        for b in bonds:
            fout.write('{} {} {} {}\n'.format(b.index,b.bond_type,b.index1,b.index2))
        fout.write('\n')
        fout.write('Angles\n\n')
        for b in angles:
            fout.write('{} {} {} {} {}\n'.format(b.index,b.bond_type,b.index1,b.index2,b.index3))
        fout.write('\n')
    print(f'# LAMMPS data file written to {filename}')

# function to copy the fiber along x, y, z on a lattice
def copy_fiber(chromatin, bonds, angles, copies):

    # Unpack the number of copies along each axis
    nx_copies, ny_copies, nz_copies = copies

    # Compute the size of the input fiber along x, y, z
    x_positions = [atom.position[0] for atom in chromatin]
    y_positions = [atom.position[1] for atom in chromatin]
    z_positions = [atom.position[2] for atom in chromatin]
    xmin, xmax = min(x_positions), max(x_positions)
    ymin, ymax = min(y_positions), max(y_positions)
    zmin, zmax = min(z_positions), max(z_positions)
    dx = xmax - xmin if xmax - xmin > 0 else 1.0  # Avoid zero displacement
    dy = ymax - ymin if ymax - ymin > 0 else 1.0
    dz = zmax - zmin if zmax - zmin > 0 else 1.0
    # add something for safety
    dx += 35.
    dy += 35.
    dz += 35.

    # Initialize new lists for atoms, bonds, and angles
    new_chromatin = []
    new_bonds = []
    new_angles = []

    total_atoms = 0
    total_bonds = 0
    total_angles = 0

    N_atom_fiber = len(chromatin)
    N_bond_fiber = len(bonds)
    N_angle_fiber = len(angles)
    mol_index_offset = 0

    # Loop over copies along x, y, z
    for ix in range(nx_copies):
        for iy in range(ny_copies):
            for iz in range(nz_copies):
                # Compute shift vector for the current copy
                shift = np.array([ix * dx, iy * dy, iz * dz])

                # Atom index offset for the current copy
                atom_index_offset = total_atoms

                # Copy and shift atoms
                for atom in chromatin:
                    new_atom = Atom(
                        index=atom.index + atom_index_offset,
                        atom_type=atom.atom_type,
                        position=atom.position + shift,
                        mol_index=atom.mol_index + mol_index_offset,
                        res_index=atom.res_index,
                        name=atom.name,
                        resname=atom.resname,
                        charge=atom.charge,
                        mass=atom.mass
                    )
                    new_chromatin.append(new_atom)

                # Bond index offset for the current copy
                bond_index_offset = total_bonds

                # Copy and update bonds
                for bond in bonds:
                    new_bond = Bond(
                        index=bond.index + bond_index_offset,
                        bond_type=bond.bond_type,
                        index1=bond.index1 + atom_index_offset,
                        index2=bond.index2 + atom_index_offset,
                        distance=bond.distance,
                        key=bond.key,
                        potential=bond.potential
                    )
                    new_bonds.append(new_bond)

                # Angle index offset for the current copy
                angle_index_offset = total_angles

                # Copy and update angles
                for angle in angles:
                    new_angle = Angle(
                        index=angle.index + angle_index_offset,
                        bond_type=angle.bond_type,
                        index1=angle.index1 + atom_index_offset,
                        index2=angle.index2 + atom_index_offset,
                        index3=angle.index3 + atom_index_offset
                    )
                    new_angles.append(new_angle)

                # Update total counts
                total_atoms += N_atom_fiber
                total_bonds += N_bond_fiber
                total_angles += N_angle_fiber
                mol_index_offset += 1

    # Update box dimensions
    x_box_min = xmin
    x_box_max = xmin + dx * nx_copies
    y_box_min = ymin
    y_box_max = ymin + dy * ny_copies
    z_box_min = zmin
    z_box_max = zmin + dz * nz_copies
    x_length = x_box_max - x_box_min
    y_length = y_box_max - y_box_min
    z_length = z_box_max - z_box_min
    # Center the structure in the box
    x_center = x_box_min + x_length / 2
    y_center = y_box_min + y_length / 2
    z_center = z_box_min + z_length / 2
    center_shift = np.array([-x_center, -y_center, -z_center])
    for atom in new_chromatin:
        atom.position += center_shift

    return new_chromatin, new_bonds, new_angles

# function to append new chain to existing list of chromatin, bonds and angles
# written by GU
def append_chromatin(chromatin_old,bonds_old,angles_old,chromatin,bonds,angles):
    if len(chromatin_old)==0: 
        return chromatin,bonds,angles
    else:
        N_atom_old = len(chromatin_old)
        N_bond_old = len(bonds_old)
        N_angle_old = len(angles_old)
        
        #chain id and atom id need to be modified:
        mol_id = [atom.mol_index for atom in chromatin_old]
        mol_index_offset = max(mol_id)
        atom_id = [atom.index for atom in chromatin_old]
        atom_index_offset = max(atom_id)
        res_id = [atom.res_index for atom in chromatin_old]
        res_index_offset = max(res_id)
        for atom in chromatin:
            atom.index     = atom.index + atom_index_offset
            atom.mol_index = atom.mol_index + mol_index_offset
            atom.res_index = atom.res_index + res_index_offset
        chromatin_new=chromatin_old+chromatin
      
        # bond id and atom id in each bond need to be modified       
        bond_id = [bond.index for bond in bonds_old]
        bond_index_offset = max(bond_id)
        for bond in bonds:
            bond.index     = bond.index + bond_index_offset
            bond.index1    = bond.index1 + atom_index_offset
            bond.index2    = bond.index2 + atom_index_offset
        bonds_new = bonds_old + bonds
        
        # angle id and atom id in each angle
        angle_id = [angle.index for angle in angles_old]
        angle_index_offset = max(angle_id)
        for angle in angles:
            angle.index    = angle.index + angle_index_offset
            angle.index1   = angle.index1 + atom_index_offset
            angle.index2   = angle.index2 + atom_index_offset
            angle.index3   = angle.index3 + atom_index_offset
        angles_new = angles_old + angles
        
        return chromatin_new,bonds_new,angles_new
        
# function to Read CG coodinates (backbone of the chromatin structure)
# written by GU
def read_chains(filename):
    # initialize
    list_file=[]
    with open(filename, 'r') as f:
        lines = f.readlines()
        for l in lines:
            tokens=l.strip().split()
            key=tokens[0]
            if key=='chain': # one line is infomation of one polymer chain
                chain_file =  str(tokens[1])
                list_file.append(chain_file)
            elif key[0]=='#':
                continue
            else:
                raise RuntimeError('key',key,'not recognized, keys should be chain')
                
    return list_file

########################################################

# main function
if __name__ == '__main__':

    # input
    # modified by GU
    print('# syntax: chromatin_builder.py -f chains.txt -o chromatin.pdb -d data.chromatin -b in.bond_settings -hbox halfbox -gdna 0 -gnuc 0.35 -qH1 10. -scalepdb 0.1')
    input_file='chains.txt'
    pdb_file  ='chromatin.pdb'
    data_file ='data.chromatin'
    bond_file ='in.bond_settings'
    halfbox   =-1.
    gamma_nuc_condensation=0.0
    gamma_dna_condensation=0.35
    charge_H1=10.
    ref='' # reference structure to set atom positions
    scale_pdb=1.0 # scale for the pdb file
    for i in range(len(sys.argv)):
        if(sys.argv[i]=='-f'):   input_file=sys.argv[i+1]
        elif(sys.argv[i]=='-cg'):   cg_file=sys.argv[i+1]
        elif(sys.argv[i]=='-o'): pdb_file  =sys.argv[i+1]
        elif(sys.argv[i]=='-d'): data_file =sys.argv[i+1]
        elif(sys.argv[i]=='-b'): bond_file =sys.argv[i+1]
        elif(sys.argv[i]=='-hbox'): halfbox=float(sys.argv[i+1])
        elif(sys.argv[i]=='-gdna'): gamma_dna_condensation=float(sys.argv[i+1])
        elif(sys.argv[i]=='-gnuc'): gamma_nuc_condensation=float(sys.argv[i+1])
        elif(sys.argv[i]=='-qH1'):  charge_H1=float(sys.argv[i+1])
        elif(sys.argv[i]=='-ref'):  ref=sys.argv[i+1]
        elif(sys.argv[i]=='-scalepdb'): scale_pdb=float(sys.argv[i+1])

    list_chains = read_chains(input_file)
    i=1
    shift = np.array([300.,0.,0.]) # shift of chromatin chains
    chromatin_all=[]
    bonds_all=[]
    angles_all=[]
    for chain_file in list_chains:
        Nnuc, Llinkers, H1s, acetylations, brd4s, cg_coords  = read_input(chain_file)
        # Read input parameters
        print('# chain', i,":",Nnuc,'nucleosomes')

        # Build chromatin
        if Nnuc==0: # just DNA
            chromatin,bonds,angles = build_only_linker_dna(Llinkers[0], gamma_dna_condensation)
        else: # chromatin fiber with nucleosome+linker+nucleosome+...+linker+nucleosome
            chromatin,bonds,angles = build_chromatin(Nnuc, Llinkers, cg_coords, gamma_dna_condensation, gamma_nuc_condensation, acetylations=acetylations, H1s=H1s,brd4s=brd4s, charge_H1=charge_H1)
            expected_atoms =22*Nnuc+sum(Llinkers)+2*sum(H1s)
            expected_bonds =78*Nnuc+14*Nnuc+sum(Llinkers)-1+3*sum(H1s)
            expected_angles=        14*Nnuc+sum(Llinkers)-2+2*sum(H1s)
            # check expected number of atoms and bonds
            if len(chromatin)!=expected_atoms:
                raise RuntimeError('unexpected number of atoms',len(chromatin),expected_atoms)
            if len(bonds)!=expected_bonds:
                raise RuntimeError('unexpected number of bonds',len(bonds),expected_bonds)
            if len(angles)!=expected_angles:
                raise RuntimeError('unexpected number of angles',len(angles),expected_angles)

        # recenter and shift individual chromatin chains
        if len(cg_coords)==0:
            com=np.array([a.position for a in chromatin]).mean(axis=0)
            for a in chromatin:
                a.position -= com
                a.position += (i-1)*shift

        #append newly generated chain to all
        chromatin_all, bonds_all, angles_all = append_chromatin(chromatin_all,bonds_all,angles_all,chromatin,bonds,angles)
        i+=1

    # recenter again overall system
    com=np.array([a.position for a in chromatin_all]).mean(axis=0)
    for a in chromatin_all:
        a.position -= com
    
    # fix atom types
    fix_atom_types(chromatin_all)
    # fix bonds and angles indices and types
    fix_bonds(bonds_all)
    fix_bonds(angles_all)

    # check and get equilibrium distances
    generate_bond_settings(bonds_all, bond_file)

        # Write PDB file
    write_pdb(chromatin_all, pdb_file, scale=scale_pdb)

        # Write LAMMPS data file
    write_lammps_data(chromatin_all, bonds_all, angles_all, data_file, halfbox)

