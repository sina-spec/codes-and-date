import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman'

data = pd.read_csv('dataset_label_111.csv')

energies = data['label'].values
categories = data['site_type'].unique()

site_type_colors = {
    'PtFe': '#1f77b4',
    'PtCo': '#ff7f0e',
    'PtNi': '#2ca02c',
    'PtCu': '#d62728',
    'PtZn': '#9467bd',
    'FeFe': '#8c564b',
    'FeCo': '#e377c2',
    'FeNi': '#7f7f7f',
    'FeCu': '#bcbd22',
    'FeZn': '#17becf',
    'CoCo': '#f1c40f',
    'CoNi': '#e67e22',
    'CoCu': '#2ecc71',
    'CoZn': '#3498db',
    'NiNi': '#f39c12',
    'NiZn': '#c0392b',
    'NiCu': '#8e44ad',
    'CuCu': '#2c3e50',
    'CuZn': '#1abc9c',
    'ZnZn': '#9b59b6',
}

for cat in categories:
    if cat not in site_type_colors:
        print(f"Warning: No color specified for site type '{cat}', default color will be used")

colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#f1c40f', '#e67e22',
    '#2ecc71', '#3498db', '#f39c12', '#c0392b', '#8e44ad', '#2c3e50', '#1abc9c', '#9b59b6'
]

zorders = list(range(20, 0, -1))

energy_by_category = {cat: data[data['site_type'] == cat]['label'].values for cat in categories}

start, stop, spacing = energies.min() - 0.1, energies.max() + 0.1, 0.05
bins = np.arange(start, stop, spacing)

fig, ax = plt.subplots(figsize=(15, 11))

plt.hist(
    energies, bins=bins,
    facecolor='grey', ec='black', alpha=0.3,
    histtype='stepfilled', zorder=0, label='Total'
)

for i, cat in enumerate(categories):
    color = site_type_colors.get(cat, colors[i % len(colors)])

    plt.hist(
        energy_by_category[cat], bins=bins,
        facecolor=color, ec='black', alpha=0.6,
        histtype='stepfilled', zorder=zorders[i % len(zorders)], label=cat
    )

custom_order = [
    'Total',
    'PtFe', 'PtCo', 'PtNi', 'PtCu', 'PtZn',
    'FeFe', 'FeCo', 'FeNi', 'FeCu', 'FeZn',
    'CoCo', 'CoNi', 'CoCu', 'CoZn',
    'NiNi', 'NiZn', 'NiCu',
    'CuCu', 'CuZn',
    'ZnZn'
]

handles, labels = ax.get_legend_handles_labels()

label_to_handle = dict(zip(labels, handles))

sorted_handles = []
sorted_labels = []

for label in custom_order:
    if label in label_to_handle:
        sorted_handles.append(label_to_handle[label])
        sorted_labels.append(label)

plt.legend(sorted_handles, sorted_labels, fontsize=21, loc='best')

plt.xlabel('Adsorption Energy [eV]', fontsize=30, fontweight='bold')
plt.ylabel('Frequency', fontsize=30, fontweight='bold')

x_ticks = np.arange(-1.10, 1.01, 0.2)
plt.xlim(-1.12, 1.01)
plt.xticks(x_ticks, fontsize=20, fontweight='bold')

y_ticks = np.arange(0, 34, 5)
plt.ylim(0, 34)
plt.yticks(y_ticks, fontsize=20, fontweight='bold')

ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)

plt.grid(False)

plt.tight_layout()

plt.savefig('adsorption_energy_distribution_111.png', dpi=300, bbox_inches='tight')
plt.show()