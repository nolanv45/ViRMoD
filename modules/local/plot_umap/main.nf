process PLOT_UMAP {
    tag "nn${nn}_md${md_tag}"
    label "process_medium"
    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://community.wave.seqera.io/library/python_numpy_pandas_matplotlib_pillow:bc19f6df598f92e8' :
        'community.wave.seqera.io/library/python_numpy_pandas_matplotlib_pillow:352bfae9d89bced3' }"
    
    input:
        tuple val(nn), val(md_tag), path(coord_file), path(conns_file)
        path filtered_tsv    // TSV file mapping ORFs to genofeatures
        path umap_coloring_metadata   // Metadata file with colors/markers
        
    output:
        path "umap_nn${nn}_md${md_tag}.png", emit: plot

    script:
    """
#!/usr/bin/env python3
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd

GRAY = "#808080"

module_df = pd.read_csv("${filtered_tsv}", sep='\\t')
umap_coloring_metadata_df = pd.read_csv("${umap_coloring_metadata}", sep='\\t')
coord_df = pd.read_csv("${coord_file}", sep='\\t')
connections_df = pd.read_csv("${conns_file}", sep='\\t')
output_file = "umap_nn${nn}_md${md_tag}.png"

embedding_ids = coord_df['embedding_id'].tolist()
embedding_2d = coord_df[['x', 'y']].values
print(f"Loaded {len(embedding_ids)} points, shape {embedding_2d.shape}")

# Lookups built once
module_df["normalized_orf_id"] = module_df["orf_id"].str.replace(".", "-", regex=False)
orf_to_geno = (module_df.drop_duplicates("normalized_orf_id")
               .set_index("normalized_orf_id")["genofeature"].to_dict())
genofeature_styles = (
    umap_coloring_metadata_df.drop_duplicates("genofeature")
    .set_index("genofeature")[["color", "marker"]].to_dict("index")
)

# Style per point
styles = []
n_unstyled = 0
for eid in embedding_ids:
    m = genofeature_styles.get(orf_to_geno.get(eid))
    if m is None:
        n_unstyled += 1
        styles.append((GRAY, "."))
    else:
        color = m["color"] if pd.notna(m["color"]) else GRAY
        marker = m["marker"] if pd.notna(m["marker"]) else "."
        styles.append((color, marker))
print(f"{n_unstyled} points had no genofeature/metadata match (drawn gray)")

# Connections
id_to_idx = {eid: i for i, eid in enumerate(embedding_ids)}
pairs = [(id_to_idx[a], id_to_idx[b])
         for a, b in zip(connections_df['id1'], connections_df['id2'])
         if a in id_to_idx and b in id_to_idx]
print(f"Using {len(pairs)} of {len(connections_df)} connections")

fig, ax = plt.subplots(figsize=(10, 10))

if pairs:
    segs = embedding_2d[np.array(pairs)]  # (n_conns, 2, 2)
    ax.add_collection(LineCollection(segs, colors='#CCCCCC', alpha=0.5,
                                     linewidths=0.2, zorder=1))

# One scatter per (color, marker); gray first so colored points sit on top
groups = defaultdict(list)
for i, s in enumerate(styles):
    groups[s].append(i)

for (color, marker), idx in sorted(groups.items(), key=lambda kv: kv[0][0] != GRAY, reverse=False):
    pts = embedding_2d[idx]
    ax.scatter(pts[:, 0], pts[:, 1], color=color, marker=marker,
               s=10, alpha=0.7, zorder=2)

# removed autoscale since it can distort the aspect ratio
# Equal scale on both axes, centered in a square frame
x, y = embedding_2d[:, 0], embedding_2d[:, 1]
cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
half = max(x.max() - x.min(), y.max() - y.min()) / 2 * 1.05

ax.set_xlim(cx - half, cx + half)
ax.set_ylim(cy - half, cy + half)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)

fig.savefig(output_file, dpi=600, bbox_inches="tight")
plt.close(fig)
    """
}