import os
import re
import pandas as pd
from ase.io import read
import numpy as np
from tqdm import tqdm


def get_surfcate(miller_index):
    if miller_index == '001':
        CN = 8
        second_region_count = 6
        third_region_count = 6
    elif miller_index == '111':
        CN = 9
        second_region_count = 8
        third_region_count = 5
    else:
        CN = None
        second_region_count = None
        third_region_count = None
    return CN, second_region_count, third_region_count


def get_coord_atoms(fname, noc):
    model = read(fname)
    model.set_constraint(None)
    model = model.repeat((3, 3, 1))
    model.set_pbc(True)

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

    return nb_sym, nb_dis, srtindex, model, I_O


def classify_neighbors(srtindex, model, I_O, second_region_count, third_region_count):
    region_1 = []
    region_2 = []
    region_3 = []

    first_neighbor_1 = model[srtindex[0]]
    first_neighbor_2 = model[srtindex[1]]
    region_1.append(first_neighbor_1.symbol)
    region_1.append(first_neighbor_2.symbol)

    visited = set()
    visited.add(srtindex[0])
    visited.add(srtindex[1])

    region_2_candidates = []
    for i in range(2, len(srtindex)):
        neighbor = model[srtindex[i]]
        z_neighbor = neighbor.position[2]

        dist_1 = model.get_distance(srtindex[i], srtindex[0], mic=True)
        dist_2 = model.get_distance(srtindex[i], srtindex[1], mic=True)

        if (abs(z_neighbor - first_neighbor_1.position[2]) < 1 or abs(z_neighbor - first_neighbor_2.position[2]) < 1):
            if srtindex[i] not in visited:
                region_2_candidates.append(neighbor.symbol)
                visited.add(srtindex[i])

        if len(region_2_candidates) >= second_region_count:
            break

    region_3_candidates = []
    for i in range(len(srtindex)):
        if i in visited or i == I_O:
            continue
        neighbor = model[srtindex[i]]
        z_neighbor = neighbor.position[2]
        if z_neighbor < 10:
            dist_1 = model.get_distance(srtindex[i], srtindex[0], mic=True)
            dist_2 = model.get_distance(srtindex[i], srtindex[1], mic=True)
            dist_sum = dist_1 + dist_2
            region_3_candidates.append((neighbor.symbol, dist_sum))

    region_3_candidates_sorted = sorted(region_3_candidates, key=lambda x: x[1])[:third_region_count]
    region_3 = [atom[0] for atom in region_3_candidates_sorted]

    return region_1, region_2_candidates, region_3


def get_element_ratios(fname):
    target_elements = ['Pt', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
    with open(fname, 'r') as f:
        lines = f.readlines()
        element_names = lines[5].split()
        counts = [int(x) for x in lines[6].split()]

    element_count_dict = {}
    for name, count in zip(element_names, counts):
        element_count_dict[name] = element_count_dict.get(name, 0) + count

    total_target_atoms = sum(
        element_count_dict.get(e, 0) for e in target_elements
    )

    ratios = {}
    for elem in target_elements:
        ratios[elem] = element_count_dict.get(elem, 0) / total_target_atoms if total_target_atoms > 0 else 0.0

    return ratios


vasp_files = [f for f in os.listdir('.') if re.match(r'^\d+_(111|001)_(\d+)_out_bridge_OH_(\d+)\.vasp$', f)]

file_info = []
for f in vasp_files:
    match = re.search(r'^(\d+)_(111|001)_(\d+)_out_bridge_OH_(\d+)\.vasp$', f)
    if match:
        idx = int(match.group(1))
        miller = match.group(2)
        site = int(match.group(3))
        oh_idx = int(match.group(4))
        file_info.append((idx, oh_idx, miller, site, f))

file_info_sorted = sorted(file_info, key=lambda x: (x[0], x[3], x[1]))

elements = ['Pt', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']

region_data = {
    'Vasp_File': [],
    'site_type': [],
    'Coordination_Number': [],
    'Pt1': [], 'Fe1': [], 'Co1': [], 'Ni1': [], 'Cu1': [], 'Zn1': [],
    'Pt2': [], 'Fe2': [], 'Co2': [], 'Ni2': [], 'Cu2': [], 'Zn2': [],
    'Pt3': [], 'Fe3': [], 'Co3': [], 'Ni3': [], 'Cu3': [], 'Zn3': [],
    'Pt_Ratio': [], 'Fe_Ratio': [], 'Co_Ratio': [], 'Ni_Ratio': [], 'Cu_Ratio': [], 'Zn_Ratio': []
}

with tqdm(total=len(file_info_sorted), desc="Processing files", unit="file") as pbar:
    for index, (hea_idx, oh_idx, miller_index, site_type, vasp_file) in tqdm(enumerate(file_info_sorted),
                                                                             total=len(file_info_sorted),
                                                                             desc="Processing files"):
        miller_index = re.search(r'^(\d+)_(111|001)_(\d+)_out_bridge_OH_(\d+)\.vasp$', vasp_file).group(2)

        CN, second_region_count, third_region_count = get_surfcate(miller_index)
        symbols, distances, srtindex, model, I_O = get_coord_atoms(vasp_file, 14)
        sites_type = str(symbols[0]) + str(symbols[1])

        sites_type = sites_type.replace('FePt', 'PtFe').replace('CoPt', 'PtCo').replace('NiPt', 'PtNi').replace('CuPt',
                                                                                                                'PtCu').replace(
            'ZnPt', 'PtZn') \
            .replace('CoFe', 'FeCo').replace('NiFe', 'FeNi').replace('CuFe', 'FeCu').replace('ZnFe', 'FeZn') \
            .replace('NiCo', 'CoNi').replace('CuCo', 'CoCu').replace('ZnCo', 'CoZn') \
            .replace('CuNi', 'NiCu').replace('ZnNi', 'NiZn') \
            .replace('ZnCu', 'CuZn')

        region_1, region_2, region_3 = classify_neighbors(srtindex, model, I_O, second_region_count, third_region_count)
        element_ratios = get_element_ratios(vasp_file)

        region_data['Vasp_File'].append(vasp_file)
        region_data['site_type'].append(sites_type)
        region_data['Coordination_Number'].append(CN)

        for elem in elements:
            region_data[f'{elem}1'].append(region_1.count(elem))
            region_data[f'{elem}2'].append(region_2.count(elem))
            region_data[f'{elem}3'].append(region_3.count(elem))
            region_data[f'{elem}_Ratio'].append(element_ratios[elem])

        pbar.update(1)

df = pd.DataFrame(region_data)
df.to_csv('pre_data.csv', index=False, header=True)

print("CSV file saved!")