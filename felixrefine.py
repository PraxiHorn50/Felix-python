# -*- coding: utf-8 -*-
"""
Started August 2024

@author: R.Beanland@warwick.ac.uk

!  Felix is free software: you can redistribute it and/or modify
!  it under the terms of the GNU General Public License as published by
!  the Free Software Foundation, either version 3 of the License, or
!  (at your option) any later version.
!
!  Felix is distributed in the hope that it will be useful,
!  but WITHOUT ANY WARRANTY; without even the implied warranty of
!  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!  GNU General Public License for more details.
!
!  You should have received a copy of the GNU General Public License
!  along with Felix.  If not, see <http://www.gnu.org/licenses/>

"""
# %% modules and subroutines

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import time
import imageio

# felix module
from pylix_modules import pylix as px
from pylix_modules import simulate as sim
from pylix_modules import pylix_dicts as fu
from pylix_modules import pylix_class as pc

path = os.getcwd()
start = time.time()
latest_commit_id = px.get_git()
# outputs
print("-----------------------------------------------------------------")
print(f"felixrefine:  version {latest_commit_id[:8]}")
print("felixrefine:  https://github.com/WarwickMicroscopy/Felix-python")
print("-----------------------------------------------------------------")

# initialise class objects
v = pc.Var()  # working variables used in the simulation
# initialise iteration count
v.iter_count = 0


# %% read felix.cif

# cif_dict is a dictionary of value-key pairs.  values are given as tuples
# with the second number the uncertainty in the first.  Nothing is currently
# done with these uncertainties...
cif_dict = px.read_cif('AnIso.cif')
v.update_from_dict(cif_dict)
# ====== extract cif data into working variables v
v.space_group = v.symmetry_space_group_name_h_m
if v.chemical_formula_structural is not None:
    v.chemical_formula = v.chemical_formula_structural
elif v.chemical_formula_sum is not None:
    v.chemical_formula = v.chemical_formula_sum
elif v.chemical_formula_iupac is not None:
    v.chemical_formula = v.chemical_formula_iupac
print("Material: " + v.chemical_formula)

# space group number and lattice type
if "space_group_symbol" in cif_dict:
    v.space_group = v.space_group_symbol.replace(' ', '')
elif "space_group_name_h_m_alt" in cif_dict:
    v.space_group = v.space_group_name_h_m_alt.replace(' ', '')
elif "symmetry_space_group_name_h_m" in cif_dict:
    v.space_group = v.symmetry_space_group_name_h_m.replace(' ', '')
elif "space_group_it_number" in cif_dict:
    v.space_group_number = int(v.space_group_it_number[0])
    reverse_space_groups = {v: k for k, v in fu.space_groups.items()}
    v.space_group = reverse_space_groups.get(v.space_group_number, "Unknown")
else:
    error_flag = True
    raise ValueError("No space group found in .cif")
v.lattice_type = v.space_group[0]
v.space_group_number = fu.space_groups[v.space_group]

# cell
v.cell_a = v.cell_length_a[0]
v.cell_b = v.cell_length_b[0]
v.cell_c = v.cell_length_c[0]
v.cell_alpha = v.cell_angle_alpha[0]*np.pi/180.0  # angles in radians
v.cell_beta = v.cell_angle_beta[0]*np.pi/180.0
v.cell_gamma = v.cell_angle_gamma[0]*np.pi/180.0
n_basis = len(v.atom_site_label)

#oxidation state and type label 





# symmetry operations
if "space_group_symop_operation_xyz" in cif_dict:
    v.symmetry_matrix, v.symmetry_vector = px.symop_convert(
        v.space_group_symop_operation_xyz)
elif "symmetry_equiv_pos_as_xyz" in cif_dict:
    v.symmetry_matrix, v.symmetry_vector = px.symop_convert(
        v.symmetry_equiv_pos_as_xyz)
else:
    error_flag = True
    raise ValueError("Symmetry operations not found in .cif")

# extract the basis from the raw cif values
# take basis atom labels as given, removing any trailing blanks
v.basis_atom_label = [s.rstrip() for s in v.atom_site_label]
# atom symbols, stripping any charge etc.
v.basis_atom_name = [''.join(filter(str.isalpha, name))
                     for name in v.atom_site_type_symbol]


# take care of any odd symbols, get the case right
for i in range(n_basis):
    name = v.basis_atom_name[i]
    if len(name) == 1:
        name = name.upper()
    elif len(name) > 1:
        name = name[0].upper() + name[1:].lower()
    v.basis_atom_name[i] = name
# take basis Wyckoff letters as given (maybe check they are only letters?)
v.basis_wyckoff = v.atom_site_wyckoff_symbol

# basis_atom_position = np.zeros([basis_count, 3])
v.basis_atom_position = \
    np.column_stack((np.array([tup[0] for tup in v.atom_site_fract_x]),
                     np.array([tup[0] for tup in v.atom_site_fract_y]),
                     np.array([tup[0] for tup in v.atom_site_fract_z])))


'''
#kappa refined
v.basis_atom_position[1][2]  = 0.998    #updating Nb z with refined
v.basis_atom_position[2] = [0.3837, 0.3811, 0.2287]#
v.basis_atom_position[0][2] = 0.2833

'''


#kirkland refined




v.basis_atom_position[1][2]  = 0.9994    #updating Nb z with refined
v.basis_atom_position[2] = [0.3972, 0.3783, 0.2289]#O
v.basis_atom_position[0][2] = 0.2917   #li




# Debye-Waller factor
# -----------------------------
# Debye–Waller: isotropic value
# -----------------------------



# occupancy, assume it's unity if not specified
if v.atom_site_occupancy is not None:
    v.basis_occupancy = np.array([tup[0] for tup in v.atom_site_occupancy])
else:
    v.basis_occupancy = np.ones([n_basis])

v.basis_atom_delta = np.zeros([n_basis, 3])  # ***********what's this


# %% read felix.inp
inp_dict = px.read_inp_file('felix.inp')
v.update_from_dict(inp_dict)

# thickness array
if (v.final_thickness > v.initial_thickness + v.delta_thickness):
    v.thickness = np.arange(v.initial_thickness, v.final_thickness,
                            v.delta_thickness)
    v.n_thickness = len(v.thickness)
else:
    # need np.array rather than float so wave_functions works for 1 or many t's
    v.thickness = np.array(v.initial_thickness)
    v.n_thickness = 1

# convert arrays to numpy
v.incident_beam_direction = np.array(v.incident_beam_direction, dtype='float')
v.normal_direction = np.array(v.normal_direction, dtype='float')
v.x_direction = np.array(v.x_direction, dtype='float')
v.atomic_sites = np.array(v.atomic_sites, dtype='int')

# crystallography exp(2*pi*i*g.r) to physics convention exp(i*g.r)
v.g_limit = v.g_limit * 2 * np.pi





 

#for i, atom in enumerate(atom_name):
    #if 'O' in atom:   # matches 'O1', 'O2-', etc.
       # pv_initial[i] = 0.8

#some initial reasonable pvs for testing
#print(v.basis_atom_name)


v.Basis_Pv = np.zeros_like(v.atom_site_label,dtype=float)
v.Basis_Kappa = np.zeros_like(v.atom_site_label,dtype=float)
atomic_number = np.array([fu.atomic_number_map[na] for na in v.basis_atom_name])
print(type(v.Basis_Kappa))
for i in range(len(atomic_number)):
    v.Basis_Pv[i]= fu.elements_info[atomic_number[i]]["pv"]
    v.Basis_Kappa[i] = 1.0  #set all kappa values to 1 initially 

#setting up initial pv values 



print(v.Basis_Kappa)

'''
#setting some initial values to match kirkland these act as our baseline
v.Basis_Kappa[0] = 1.0
v.Basis_Kappa[1] = 1.2
v.Basis_Kappa[2] = 0.98
'''



 #refined kappas
# kappas (default 1.0)

v.Basis_Kappa[0] = 1.0
v.Basis_Kappa[1] = 1.25
v.Basis_Kappa[2] = 0.98

#refined kappa values


#refined Pv values 

v.Basis_Pv[0] = 1
v.Basis_Pv[1] = 4.5
v.Basis_Pv[2] = 6.5



# expand per atom in full unit cell
 


 #print(unique_aniso_matrixes)
 #print(unique_aniso_matrixes.shape)
 
 # Step 1: define a dictionary of initial P_v guesses per element
 # For LiNbO3 using formal charges as we discussed
 # we just need a dictionary of the valence states of the atoms 


 

# output
print(f"Zone axis: {v.incident_beam_direction.astype(int)}")
if v.n_thickness == 1:
    print(f"Specimen thickness {v.initial_thickness/10} nm")
else:
    print(f"{v.n_thickness} thicknesses: {', '.join(map(str, v.thickness/10))} nm")
    
    

if v.scatter_factor_method == 0:
    print("Using Kirkland scattering factors")
elif v.scatter_factor_method == 1:
    print("Using Lobato scattering factors")
elif v.scatter_factor_method == 2:
    print("Using Peng scattering factors")
elif v.scatter_factor_method == 3:
    print("Using Doyle & Turner scattering factors")
elif v.scatter_factor_method == 4:
    print("using orbital HF scattering factors with Kappa formalism")
else:
    raise ValueError("No scattering factors chosen in felix.inp")

if 'S' in v.refine_mode:
    print("Simulation only, S")
elif 'A' in v.refine_mode:
    print("Refining Structure Factors, A")
    # needs error check for any other refinement
    # raise ValueError("Structure factor refinement
    # incompatible with anything else")
else:
    if 'B' in v.refine_mode:
        print("Refining Atomic Coordinates, B")
        # redefine the basis if necessary to allow coordinate refinement
        v.basis_atom_position = px.preferred_basis(v.space_group_number,
                                                   v.basis_atom_position,
                                                   v.basis_wyckoff)
    if 'C' in v.refine_mode:
        print("Refining Occupancies, C")
    if 'D' in v.refine_mode:
        print("Refining Isotropic Debye Waller Factors, D")
    if 'E' in v.refine_mode:
        print("Refining Anisotropic Uperp and Uparallel, E")
        #raise ValueError("Refinement mode E not implemented")
    if (len(v.atomic_sites) > n_basis):
        raise ValueError("Number of atomic sites to refine is larger than the \
                         number of atoms")
if 'F' in v.refine_mode:
    print("Refining Lattice Parameters, F")
if 'G' in v.refine_mode:
    print("Refining Lattice Angles, G")
if 'H' in v.refine_mode:
    print("Refining Convergence Angle, H")
if 'I' in v.refine_mode:
    print("Refining Accelerating Voltage, I")
if 'J' in v.refine_mode:
    print("Refining Kappa values, J")
if 'K' in v.refine_mode:
    print("Refining Pv vales, K")
    



if "atom_site_b_iso_or_equiv" in cif_dict:
    v.basis_U_iso = np.array([tup[0] for tup in v.atom_site_b_iso_or_equiv]) / (8 * np.pi**2)

elif "atom_site_u_iso_or_equiv" in cif_dict:
    v.basis_U_iso = np.array([tup[0] for tup in v.atom_site_u_iso_or_equiv])

else:
    raise ValueError("No isotropic displacement parameters in CIF")

v.basis_U_iso = np.asarray(v.basis_U_iso)

# --------------------------------
# 2. Allocate U_ij
# --------------------------------
v.U_ij = np.zeros((n_basis, 3, 3))

# --------------------------------
# 3. Read anisotropic arrays (if present)
# --------------------------------
v.has_aniso = np.zeros(n_basis, dtype=bool)  # default: no anisotropic component

if (
    v.atom_site_aniso_u_11 is not None and
    v.atom_site_aniso_u_22 is not None and
    v.atom_site_aniso_u_33 is not None
):
    # Convert to set for fast lookup
    aniso_labels = set(v.atom_site_aniso_label)
    # Boolean mask for atoms that have anisotropic U
    v.has_aniso = np.array([label in aniso_labels for label in v.atom_site_label])
#print(v.has_aniso)

if np.any(v.has_aniso):
    U11 = np.array([tup[0] for tup in v.atom_site_aniso_u_11])
    U22 = np.array([tup[0] for tup in v.atom_site_aniso_u_22])
    U33 = np.array([tup[0] for tup in v.atom_site_aniso_u_33])
    U12 = np.array([tup[0] for tup in v.atom_site_aniso_u_12])
    U13 = np.array([tup[0] for tup in v.atom_site_aniso_u_13])
    U23 = np.array([tup[0] for tup in v.atom_site_aniso_u_23])

#print(v.basis_U_iso)
# --------------------------------
# 4. Per-atom assignment (CONTROLLED)
# --------------------------------
#print (v.Debye_model)
aniso_idx = 0

for i in range(n_basis):
    # Case 1: force isotropic
    if v.Debye_model == 0:
        u = v.basis_U_iso[i]
        v.U_ij[i] = np.diag([u, u, u])
    
    # Case 2: allow anisotropic if available
    elif v.Debye_model == 1 and v.has_aniso[i]:
        # Fill from U11, U22, ... using a separate index
        v.U_ij[i] = np.array([
            [U11[aniso_idx], U12[aniso_idx], U13[aniso_idx]],
            [U12[aniso_idx], U22[aniso_idx], U23[aniso_idx]],
            [U13[aniso_idx], U23[aniso_idx], U33[aniso_idx]],
        ])
        aniso_idx += 1  # move to next anisotropic atom
    
    # Case 3: fallback to isotropic
    else:
        u = v.basis_U_iso[i]
        v.U_ij[i] = np.diag([u, u, u])



#set initial Uperp and Uparallel initially take iostropic terms and have U_parallel = U_perpendicular U22=U11=U33

# --- Option A: Parallel vs Perpendicular refinement (post-processing / refinement only) ---
# Can be commented out if you want to keep original CIF tensor

v.U_parallel_param = np.zeros(n_basis)
v.U_perp_param = np.zeros(n_basis)

#setting isotropic Li , Nb , O

Iso_list =np.array([0.219,0.459,0.541])

Refined_anisos= np.array([[0.00277,0.00277],[0.00332784,0.0056109],[0.00797715,0.0068116]]) #shape(atom,perp/parallel)


# convert to U_iso
Iso_list /= (8*np.pi**2)

for i in range(n_basis):
    # Extract current tensor
    U = v.U_ij[i]
    U[0,1] =0 
    U[0,2] =0
    U[1,2] =0
    # Define U_perp (fixed) and U_parallel (to refine)
    #U_perp = Iso_list[i]       # average in-plane, could also use isotropic baseline
    #U_parallel = Iso_list[i]                 # c-axis, refine this only
    U_perp = Refined_anisos[i][0]
    U_parallel = Refined_anisos[i][1]
    # Save for later scaling / plotting
    v.U_parallel_param[i] = U_parallel
    v.U_perp_param[i] = U_perp
    #U_perp = 0
    #U_parallel = 0

    # Build refined tensor for Option A
    v.U_ij[i] = np.array([
        [U_perp, U[0,1], U[0,2]],
        [U[0,1], U_perp,  U[1,2]],
        [U[0,2], U[1,2], U_parallel]   # this is the only component we will refine
    ])



print("Final U_ij:")
print(v.U_ij)

























# %% read felix.hkl

v.input_hkls, v.i_obs, v.sigma_obs = px.read_hkl_file("felix.hkl")
v.n_out = len(v.input_hkls)+1  # we expect 000 NOT to be in the hkl list


# %% set up refinement
# --------------------------------------------------------------------
# n_variables calculated depending upon Ug and non-Ug refinement
# --------------------------------------------------------------------
# Ug refinement is a special case, cannot do any other refinement alongside
# We count the independent variables:
# v.refined_variable = variable to be refined
# v.refined_variable_type = what kind of variable, as follows
# 0 = Ug amplitude
# 1 = Ug phase
# 2 = atom coordinate *** PARTIALLY IMPLEMENTED *** not all space groups
# 3 = occupancy
# 4 = B_iso
# 5 = B_aniso *** NOT YET IMPLEMENTED ***
# 61,62,63 = lattice parameters *** PARTIALLY IMPLEMENTED *** not rhombohedral
# 7 = unit cell angles *** NOT YET IMPLEMENTED ***
# 8 = convergence angle
# 9 = accelerating_voltage_kv *** NOT YET IMPLEMENTED ***
v.refined_variable = ([])  # array of floats, values to be refined
v.refined_variable_type = ([])  # array of integers corresponding to above
v.atom_refine_flag = ([])  # the index of the atom in the .cif, -1 if none
v.atom_refine_vec = ([])  # the direction of atom movement, [0,0,0] if none
nullvec = np.array([0, 0, 0])  # null vector for above
if 'S' not in v.refine_mode:
    v.n_variables = 0
    # count refinement variables
    '''
    if 'B' in v.refine_mode:  # Atom coordinate refinement
        # the input v.atomic_sites gives the index of the atom in the cif
        for i in range(len(v.atomic_sites)):
            # the [3, 3] matrix 'moves' returned by atom_move gives the
            # allowed movements for an atom (depending on its Wyckoff
            # symbol and space group) as row vectors with magnitude 1.
            # ***NB NOT ALL SPACE GROUPS IMPLEMENTED ***
            moves = px.atom_move(v.space_group_number,
                                 v.basis_wyckoff[v.atomic_sites[i]])
            degrees_of_freedom = np.sum(np.any(moves, axis=1))
            if degrees_of_freedom == 0:
                raise ValueError(f"Coordinate refinement of atom \
                                 {v.atomic_sites[i]} not possible")
            for j in range(degrees_of_freedom):
                v.atom_coord_vec = moves[j, :]  # the vector of movement
                # we refine the coordinate along the appropriate vector
                r_dot_v = np.dot(v.basis_atom_position[v.atomic_sites[i]],
                                 moves[j, :])
                v.refined_variable.append(r_dot_v)
                v.refined_variable_type.append(2)  # flag to say it's a coord
                v.atom_refine_flag.append(v.atomic_sites[i])  # atom index
                v.atom_refine_vec.append(moves[j, :])  # atom movement
    '''
    
    if 'B' in v.refine_mode:  # Atom coordinate refinement
        target_atom_idx =2  # refine only this atom
        i = target_atom_idx
        # Get allowed movement vectors for this atom
        moves = px.atom_move(v.space_group_number,
                             v.basis_wyckoff[v.atomic_sites[i]])
        degrees_of_freedom = np.sum(np.any(moves, axis=1))
        if degrees_of_freedom == 0:
            raise ValueError(f"Coordinate refinement of atom "
                             f"{v.atomic_sites[i]} not possible")
        
        # Add refinement variables only for allowed degrees of freedom
        for j in range(degrees_of_freedom):
            v.atom_coord_vec = moves[j, :]  # vector of allowed movement
            r_dot_v = np.dot(v.basis_atom_position[v.atomic_sites[i]], moves[j, :])
            v.refined_variable.append(r_dot_v)
            v.refined_variable_type.append(2)  # coordinate flag
            v.atom_refine_flag.append(v.atomic_sites[i])  # store atom index
            v.atom_refine_vec.append(moves[j, :])  # movement vector

    if 'C' in v.refine_mode:  # Occupancy
        for i in range(len(v.atomic_sites)):
            v.refined_variable.append(v.basis_occupancy[v.atomic_sites[i]])
            v.refined_variable_type.append(3)
            v.atom_refine_flag.append(v.atomic_sites[i])
            v.atom_refine_vec.append(nullvec)  # no atom movement

    if 'D' in v.refine_mode:  # Isotropic DW
        for i in range(len(v.atomic_sites)):
            v.refined_variable.append(v.basis_B_iso[v.atomic_sites[i]])
            v.refined_variable_type.append(4)
            v.atom_refine_flag.append(v.atomic_sites[i])
            v.atom_refine_vec.append(nullvec)  # no atom movement

    if 'E' in v.refine_mode:  # Anisotropic DW
        for i in range(len(v.atomic_sites)):
            atom = v.atomic_sites[i]
            if (i==2):
                # Append U_parallel first
                v.refined_variable.append(v.U_parallel_param[i])
                v.refined_variable_type.append(5)      # anisotropic DW type
                v.atom_refine_flag.append(i)
                v.atom_refine_vec.append(nullvec)
        
                # Append U_perp second
                v.refined_variable.append(v.U_perp_param[i])
                v.refined_variable_type.append(5)
                v.atom_refine_flag.append(i)
                v.atom_refine_vec.append(nullvec) 
    
          
        

    if 'F' in v.refine_mode:  # Lattice parameters
        # variable_type first digit=6 indicates lattice parameter
        # second digit=1,2,3 indicates a,b,c
        # This section needs work to include rhombohedral cells and
        # non-standard settings!!!
        v.refined_variable.append(v.cell_a)  # is in all lattice types
        v.refined_variable_type.append(61)
        v.atom_refine_flag.append(-1)  # -1 indicates not an atom
        v.atom_refine_vec.append(nullvec)  # no atom movement
        if v.space_group_number < 75:  # Triclinic, monoclinic, orthorhombic
            v.refined_variable.append(v.cell_b)
            v.refined_variable_type.append(62)
            v.atom_refine_flag.append(-1)
            v.refined_variable.append(v.cell_c)
            v.refined_variable_type.append(63)
            v.atom_refine_flag.append(-1)
            v.atom_refine_vec.append(nullvec)  # no atom movement
        elif 142 < v.space_group_number < 168:  # Rhombohedral
            # Need to work out R- vs H- settings!!!
            raise ValueError("Rhombohedral R- vs H- not yet implemented")
        elif (167 < v.space_group_number < 195) or \
             (74 < v.space_group_number < 143):  # Hexagonal or Tetragonal
            v.refined_variable.append(v.cell_c)
            v.refined_variable_type.append(63)
            v.atom_refine_flag.append(-1)
            v.atom_refine_vec.append(nullvec)  # no atom movement

    if 'G' in v.refine_mode:  # Unit cell angles
        # Not yet implemented!!! variable_type 7
        raise ValueError("Unit cell angle refinement not yet implemented")

    if 'H' in v.refine_mode:  # Convergence angle
        v.refined_variable.append(v.convergence_angle)
        v.refined_variable_type.append(8)
        v.atom_refine_flag.append(-1)
        v.atom_refine_vec.append(nullvec)  # no atom movement
        print(f"Starting convergence angle {v.convergence_angle} Å^-1")

    if 'I' in v.refine_mode:  # accelerating_voltage_kv
        v.refined_variable.append(v.accelerating_voltage_kv)
        v.refined_variable_type.append(9)
        v.atom_refine_flag.append(-1)
        v.atom_refine_vec.append(nullvec)  # no atom movement
        
        
    
    if  'J' in v.refine_mode:
        
        
        for i in range(len(v.atomic_sites)):
            
            v.refined_variable.append(v.Basis_Kappa[v.atomic_sites[i]])
            v.refined_variable_type.append(10)
            v.atom_refine_flag.append(v.atomic_sites[i])
            v.atom_refine_vec.append(nullvec)  # no atom movement

       
    if 'K' in v.refine_mode:  
        for i in range(len(v.atomic_sites)):
            
            v.refined_variable.append(v.Basis_Pv[v.atomic_sites[i]])
            v.refined_variable_type.append(11)
            v.atom_refine_flag.append(v.atomic_sites[i])
            v.atom_refine_vec.append(nullvec)  # no atom movement
       
        
       

    # Total number of independent variables
    v.n_variables = len(v.refined_variable)
    if v.n_variables == 0:
        raise ValueError("No refinement variables! \
        Check refine_mode flag in felix.v. \
            Valid refine modes are A,B,C,D,F,H,S")
    if v.n_variables == 1:
        print("Only one independent variable")
    else:
        print(f"Number of independent variables = {v.n_variables}")

    v.refined_variable = np.array(v.refined_variable)
    independent_delta = np.zeros(v.n_variables)
    v.refined_variable_type = np.array(v.refined_variable_type)
    v.refined_variable_atom = np.array(v.atom_refine_flag[:v.n_variables])


# # %% set up Ug refinement
# if 'A' in refine_mode:  # Ug refinement
#     print("Refining Structure Factors, A")
#     # needs error check for any other refinement
#     # raise ValueError("Structure factor refinement incompatible
#     # with anything else")
#     # we refine magnitude and phase for each Ug.  However for space groups
#     # with a centre of symmetry phases are fixed at 0 or pi, so only
#     # amplitude is refined (1 independent variable per Ug)
#     # Identify the 92 centrosymmetric space groups
#     centrosymmetric = [2, 10, 11, 12, 13, 14, 15, 47, 48, 49, 50, 51, 52,
#                        53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
#                        66, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85,
#                        86, 87, 88, 123, 124, 125, 126, 127, 128, 129, 130,
#                        131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
#                        141, 142, 147, 148, 162, 163, 164, 165, 166, 167,
#                        175, 176, 191, 192, 193, 194, 200, 201, 202,
#                        203, 204, 205, 206, 221, 222, 223, 224, 225, 226,
#                        227, 228, 229, 230]
#     if space_group_number in centrosymmetric:
#         vars_per_ug = 1
#     else:
#         vars_per_ug = 2

#     # set up Ug refinement
#     # equivalent g's identified by abs(h)+abs(k)+abs(l)+a*h^2+b*k^2+c*l^2
#     g_eqv = (10000*(np.sum(np.abs(g_matrix), axis=2) +
#                     g_magnitude**2)).astype(int)
#     # we keep track of individual Ug's in a matrix ug_eqv
#     ug_eqv = np.zeros([n_hkl, n_hkl], dtype=int)
#     # bit of a hack here - can skip Ug's in refinement using ug_offset, but
#     # should really be an input in felix.inp if it's going to be used
#     ug_offset = 0
#     # Ug number (position in ug_matrix)
#     i_ug = 1 + ug_offset
#     # the first column of the Ug matrix has g-vectors in ascending order
#     # we work through this list until we have identified the required no.
#     # of Ugs. The matrix ug_eqv identifies equivalent Ug's with an integer
#     # whose sign is that of the imaginary part of Ug. (may need to take
#     # care of floating point residuals where im(Ug) is nominally zero???)
#     j = 1  # number of Ug's processed
#     while j < no_of_ugs+1:
#         if ug_eqv[i_ug, 0] != 0:  # already in the list, skip it
#             i_ug += 1
#             continue
#         g_id = abs(g_eqv[i_ug, 0])
#         # update relevant locations in ug_eqv
#         ug_eqv[np.abs(g_eqv) == g_id] = j*np.sign(
#             np.imag(ug_matrix[i_ug, 0]))
#         # amplitude is type 1, always a variable
#         variable.append(ug_matrix[i_ug, 0])
#         variable_type.append(0)
#         if vars_per_ug == 2:  # we also adjust phase
#             variable.append(ug_matrix[i_ug, 0])
#             variable_type.append(1)
#         j += 1


# %% baseline simulation
print("-------------------------------")
print("Baseline simulation:")
# uses the whole v=Var class
sim.simulate(v)

# %% read in experimental images
if 'S' not in v.refine_mode:
    v.lacbed_expt = np.zeros([2*v.image_radius, 2*v.image_radius, v.n_out])
    # get the list of available images
    x_str = str(2*v.image_radius)
    dm3_folder = None
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            # Check if 'dm3' and the number x are in the folder name
            if 'dm3' in dirname.lower() and x_str in dirname:
                # Return the full path of the matching folder
                dm3_folder = os.path.join(dirpath, dirname)
                
    #dm3_folder = 'DM3_100x100' #diamond
    dm3_folder = 'DM3_84x84'  #Lithium Niobate
    
    if dm3_folder is not None:
        dm3_files = [file for file in os.listdir(dm3_folder)
                     if file.lower().endswith('.dm3')]
        # just match the indices in the filename to felix.hkl, expect the user
        # to ensure the data is of the right material!
        n_expt = v.n_out
        for i in range(v.n_out):
            g_string = px.hkl_string(v.hkl[v.g_output[i]])
            found = False
            for file_name in dm3_files:
                if g_string in file_name:
                    file_path = os.path.join(dm3_folder, file_name)
                    v.lacbed_expt[:, :, i] = px.read_dm3(file_path,
                                                       2*v.image_radius,
                                                       v.debug)
                    found = True
            if not found:
                n_expt -= 1
                print(f"{g_string} not found")

        # print experimental LACBED patterns
        w = int(np.ceil(np.sqrt(v.n_out)))
        h = int(np.ceil(v.n_out/w))
        fig, axes = plt.subplots(w, h, figsize=(w*5, h*5))
        text_effect = withStroke(linewidth=3, foreground='black')
        axes = axes.flatten()
        for i in range(v.n_out):
            axes[i].imshow(v.lacbed_expt[:, :, i], cmap='gist_earth')
            axes[i].axis('off')
            annotation = f"{v.hkl[v.g_output[i], 0]}{v.hkl[v.g_output[i], 1]}{v.hkl[v.g_output[i], 2]}"
            axes[i].annotate(annotation, xy=(5, 5), xycoords='axes pixels',
                             size=30, color='w', path_effects=[text_effect])
        for i in range(v.n_out, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
        # initialise correlation
        best_corr = np.ones(v.n_out)
        
        
#convert to tif for analysis in fiji
        '''
        dm3_folder = "DM3_100x100"
        tif_folder = "dm3_tif_diamond100"
        
        os.makedirs(tif_folder, exist_ok=True)
        
        for file_name in os.listdir(dm3_folder):
            if file_name.endswith(".dm3"):
                file_path = os.path.join(dm3_folder, file_name)
                img = px.read_dm3(file_path, 2*v.image_radius, debug=False)  # 2D array
                out_name = os.path.join(tif_folder, file_name.replace(".dm3", ".tif"))
                imageio.imwrite(out_name, img.astype('float64'))  # save as float T       
        
'''

# %% output - *** needs work, apply blur/find best blur 



sim.save_LACBED(v)


if v.image_processing == 1:
    print(f"  Blur radius {v.blur_radius} pixels")
if 'S' in v.refine_mode:
    #*** apply blur !!!
    # output simulated LACBED patterns
    sim.print_LACBED(v)
else:
    # figure of merit
    fom = sim.figure_of_merit(v)
    print(f"  Figure of merit {100*fom:.2f}%")
    print("-------------------------------")
    sim.print_LACBED(v)

# %% start refinement loop *** needs work
if 'S' not in v.refine_mode:
    # Initialise variables for refinement
    fit0 = fom*1.0
    v.best_fit = fom*1.0
    last_fit = fom*1.0
    # p is a vector along the gradient in n-dimensional space
    # we initially set these as
    p = np.ones(v.n_variables)
    # last_p = np.ones(v.n_variables)
    r3_var = np.zeros(3)  # for parabolic minimum
    r3_fom = np.zeros(3)
    # dunno what this is
    independent_delta = 0.0

    # for a plot
    v.fit_log = ([last_fit])

    # Refinement loop
    df = 1.0
    while df >= v.exit_criteria:
        # v.refined_variable is the working set of variables
        # best_var is the running best fit during this refinement cycle
        v.best_var = np.copy(v.refined_variable)
        # next_var is the predicted next (best) point
        v.next_var = np.copy(v.refined_variable)
        # if all variables have been refined and we're still in the loop, reset
        if np.sum(np.abs(p)) < 1e-10:
            p = np.ones(v.n_variables)

        # ===========individual variable minimisation
        # Go through the variables looking at three points in the hope
        # of capturing a minimum - if there is one, we take it and remove
        # that variable from multidimensional refinement, p[i] = 0.
        # Otherwise p[i] is the gradient for that variable.
        # We also get a predicted best starting point
        # for gradient descent, v.next_var
        for i in range(v.n_variables):
            # Skip variables already optimized
            if abs(p[i]) < 1e-10:
                p[i] = 0.0
                continue
            p[i] = sim.refine_single_variable(v, i)

        # if all variables have predicted minima, do a final simulation
        # if it's better, it will update v.best_fit and v.best_var accordingly
        if np.count_nonzero(p) == 0:
            print("Closing simulation for this cycle")
            v.refined_variable = np.copy(v.best_var)
            fom = sim.sim_fom(v, 0)
        else:
            # ===========vector descent
            # Downhill minimisation until we eliminate all variables
            while np.sum(np.abs(p)) > 1e-10:
                # the returned p will have an extra zero!
                p = sim.refine_multi_variable(v, p)
        # Update for next iteration
        df = last_fit - v.best_fit
        last_fit = np.copy(v.best_fit)
        v.refined_variable = np.copy(v.best_var)
        v.refinement_scale *= (1 - 1 / (2 * v.n_variables))
        print(f"Improvement in fit {100*df:.2f}%, will stop at {100*v.exit_criteria:.2f}%")
        print("-------------------------------")
        plt.plot(v.fit_log)
        # plt.scatter(var_pl, fit_pl)
        plt.show()
    
    print(f"Refinement complete after {v.iter_count} simulations.  Refined values: {v.best_var}")

# %% final print
sim.print_LACBED(v)
total_time = time.time() - start
print("-----------------------------------------------------------------")
# print(f"Beam pool calculation took {setup:.3f} seconds")
# print(f"Bloch wave calculation in {bwc:.1f} s ({1000*(bwc)/(4*v.image_radius**2):.2f} ms/pixel)")
print(f"Total time {total_time:.1f} s")
print("-----------------------------------------------------------------")
print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
