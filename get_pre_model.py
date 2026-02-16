from ase.build import surface, add_vacuum, make_supercell
from ase.visualize import view
from ase.io import read, write
import numpy as np
from ase.geometry import get_layers
from ase import Atoms
import os
import re
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import random
from tqdm import tqdm

def substitute_atoms_by_exact_count(structure, elements_to_replace, max_combinations=100):
    unique_structures = set()
    all_structures = []
    duplicate_warning = []

    element_counts = {atom.symbol: 0 for atom in structure}
    for atom in structure:
        element_counts[atom.symbol] += 1

    for original_element, new_elements, num_to_change_list in elements_to_replace:
        if original_element not in element_counts:
            raise ValueError(f"No element {original_element} in the structure, substitution cannot be performed")

        num_original_atoms = element_counts[original_element]
        total_to_change = sum(num_to_change_list)

        if total_to_change > num_original_atoms:
            raise ValueError(f"Total substitution count ({total_to_change}) exceeds the number of {original_element} atoms ({num_original_atoms})")

    while len(all_structures) < max_combinations:
        modified_structure = structure.copy()

        for original_element, new_elements, num_to_change_list in elements_to_replace:
            original_indices = [atom.index for atom in modified_structure if atom.symbol == original_element]
            num_original_atoms = len(original_indices)
            if num_original_atoms == 0:
                raise ValueError(f"No {original_element} atoms available for substitution in the structure")

            modified_indices = set()

            for new_element, num_to_change in zip(new_elements, num_to_change_list):
                if num_to_change > len(original_indices):
                    raise ValueError(f"Insufficient {original_element} atoms for substitution, cannot complete {new_element} substitution")

                random_indices = random.sample(original_indices, num_to_change)
                for idx in random_indices:
                    if idx not in modified_indices:
                        modified_structure[idx].symbol = new_element
                        modified_indices.add(idx)

                original_indices = [idx for idx in original_indices if idx not in modified_indices]

        unique_id = tuple((atom.symbol, tuple(atom.position)) for atom in modified_structure)
        if unique_id not in unique_structures:
            unique_structures.add(unique_id)
            all_structures.append(modified_structure)
        else:
            duplicate_warning.append(unique_id)
            print(f"Warning: Duplicate structure found! This structure already exists, skipping generation.")

    return all_structures

def reorder_atoms_by_elements(structure, element_order):
    ordered_atoms = []
    for element in element_order:
        ordered_atoms.extend([atom for atom in structure if atom.symbol == element])
    return Atoms(ordered_atoms, cell=structure.cell, pbc=structure.pbc)

def filter_sites_based_on_neighbors(fname, noc, excluded_site_types):
    nb_sym, _ = get_coord_atoms(fname, noc)
    site_type = "".join(sorted(nb_sym[:2]))
    return site_type not in excluded_site_types

def get_coord_atoms(fname, noc):
    model = read(fname)
    syminfo = model.get_chemical_symbols()

    for i in range(len(model)):
        if syminfo[i] == 'O':
            I_O = i
        elif syminfo[i] == 'H':
            I_H = i
        else:
            pass

    HEA_index = list(range(len(model)))
    HEA_index.remove(I_H)
    HEA_index.remove(I_O)

    dmat = model.get_distances(I_O, HEA_index, mic=True)
    np_dmat = np.array(dmat)
    srtindex = np_dmat.argsort()

    nb_sym = []
    nb_dis = []
    for i in range(noc):
        j = srtindex[i]
        nb_sym.append(model[j].symbol)
        nb_dis.append(dmat[j])

    return nb_sym, nb_dis

def create_slab_3d(orig_cell, miller, layers,
                   vacuum_z=(10, 5), xy_shift_fraction=0.1,
                   layer_offset=0):

    tags, dz = get_layers(orig_cell, miller)
    if isinstance(dz, np.ndarray) and len(dz) > 0:
        dz = np.mean(dz)
    else:
        dz = float(dz) if not isinstance(dz, float) else dz

    if layer_offset > 0:
        orig_cell = orig_cell.copy()
        var1 = dz * layer_offset
        var2 = orig_cell.positions[:, 1]
        var3 = orig_cell.positions[:, 2]
        orig_cell.positions[:, 2] -= dz * layer_offset

    slab = surface(orig_cell, miller, layers=layers, vacuum=0.0)

    cell = slab.cell.copy()
    current_z = cell[2, 2]
    target_z = current_z + vacuum_z[0] + vacuum_z[1]
    slab.positions[:, 2] += vacuum_z[1]
    cell[2, 2] = target_z

    shift_x = xy_shift_fraction * cell[0, 0]
    shift_y = xy_shift_fraction * cell[1, 1]
    slab.positions[:, 0] += shift_x
    slab.positions[:, 1] += shift_y
    slab.set_cell(cell, scale_atoms=False)
    return slab

def get_vasp_files():
    vasp_files = []
    for file in os.listdir('.'):
        if file.endswith('.vasp') and file.split('.')[0].isdigit():
            vasp_files.append(file)
    vasp_files.sort(key=lambda x: int(x.split('.')[0]))
    return vasp_files

def process_all_vasp_files():
    vasp_files = get_vasp_files()
    if not vasp_files:
        print("No vasp files named with numbers found!")
        return

    miller_indices = [
        ((0, 0, 1), 1, 0, 1),
        ((0, 0, 1), 1, 0.5, 2),
        ((1, 1, 1), 2, 1, 1)
    ]

    for vasp_file in vasp_files:
        file_num = vasp_file.split('.')[0]
        print(f"\nProcessing file: {vasp_file}")

        try:
            orig_cell = read(vasp_file)

            for miller, layers, offset, out_idx in miller_indices:
                miller_str = ''.join(str(i) for i in miller)
                slab = create_slab_3d(
                    orig_cell,
                    miller=miller,
                    layers=layers,
                    layer_offset=offset
                )

                output_file = f"{file_num}_{miller_str}_{out_idx}.vasp"
                write(output_file, slab, vasp5=True)

        except Exception as e:
            print(f"  Error processing file {vasp_file}: {str(e)}")

def add_adsorbates():
    adsorption_sites = {
        "001_1": [
            [5.34048, 3.49894, 12.02464],
            [5.34048, 1.65739, 12.02464]
        ],
        "001_2": [
            [3.49894, 5.34048, 12.02464],
            [5.34048, 5.34048, 12.02464]
        ],
        "111_1": [
            [9.90873,   3.01430,  12.87930],
            [8.60656,   3.01430,  12.87930]
        ]
    }

    file_names = [f for f in os.listdir('.') if re.match(r'^\d+_(001|111)_(\d+)\.vasp$', f)]

    for file_name in file_names:
        match = re.match(r'^(\d+)_(001|111)_(\d+)\.vasp$', file_name)
        if not match:
            continue

        struct_id, surface_type, struct_index = match.groups()
        surface_key = f"{surface_type}_{struct_index}"

        if surface_key not in adsorption_sites:
            print(f"  No adsorption sites defined for {surface_key}, skipping.")
            continue

        try:
            surf = read(file_name)
            base_name = os.path.splitext(file_name)[0]

            sites = adsorption_sites[surface_key]

            for i, ads_position in enumerate(sites, 1):
                ads_position = np.array(ads_position)

                oh_positions = np.array([
                    [0, 0, 0],
                    [0, 0, 1.01]
                ]) + ads_position

                surf_with_ads = surf.copy()
                surf_with_ads.extend(Atoms('OH', oh_positions))
                surf_with_ads.wrap()

                output_file = f"{base_name}_bridge_OH_{i}.vasp"
                write(output_file, surf_with_ads, vasp5=True)

        except Exception as e:
            print(f"  Error processing file {file_name}: {str(e)}")

def main():
    input_file = "Cu3Pt.xsd"
    output_file_template = "{}.vasp"

    elements_to_replace = [
        ("Cu", ["Fe", "Co", "Ni", "Zn"], [4, 7, 7, 1])
    ]

    element_order = ["Pt", "Fe", "Co", "Ni", "Cu", "Zn"]

    structure = read(input_file)

    max_combinations = 1000

    output_dir = "structure"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combined_structures = substitute_atoms_by_exact_count(structure, elements_to_replace, max_combinations)

    for i, combined_structure in tqdm(enumerate(combined_structures), total=len(combined_structures), desc="Generating structures"):
        ordered_structure = reorder_atoms_by_elements(combined_structure, element_order)
        fname = os.path.join(output_dir, output_file_template.format(i + 1))
        write(fname, ordered_structure, format="vasp", vasp5=True, direct=True)

    os.chdir(output_dir)
    process_all_vasp_files()

    add_adsorbates()

    print("\nComplete workflow finished!")

if __name__ == "__main__":
    main()

import os
import shutil

target_dirs = ["001", "111"]

for folder in target_dirs:
    os.makedirs(folder, exist_ok=True)

for filename in os.listdir("."):
    if not os.path.isfile(filename):
        continue

    if "_001_" in filename:
        shutil.move(filename, os.path.join("001", filename))
    elif "_111_" in filename:
        shutil.move(filename, os.path.join("111", filename))

print("✅ Files organized successfully.")