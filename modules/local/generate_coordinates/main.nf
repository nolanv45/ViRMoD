process GENERATE_COORDINATES {
    tag "nn${nn}_md${md}"
    publishDir "${params.outdir}",
        mode: 'copy'
    label 'process_medium'
    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://community.wave.seqera.io/library/python_pytorch_numpy_pandas_pruned:02ddcbc0a6f7a925' :
        'community.wave.seqera.io/library/python_pytorch_numpy_pandas_pruned:80951f99909b8c30' }"

    cpus 1

    input:
        tuple path(embeddings_dirs, stageAs: 'embeddings/?/*'), val(excluded_genofeatures), val(nn), val(md), val(publish_subdir), path(combined_datasets_tsv)
        
    output:
        path "${publish_subdir}", emit: coordinates_files
        path "versions.yml", emit: versions

    script:
"""
# Enforce single-threading
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export NUMBA_NUM_THREADS=1
# export NUMBA_DEFAULT_NUM_THREADS=1
# export NUMBA_THREADING_LAYER=workqueue

# export NUMBA_DISABLE_INTEL_SVML=1
# export NUMBA_CPU_FEATURES="+sse,+sse2"   # tightest baseline; replaces generic's defaults
# export NUMBA_OPT=0   # slow-ish, but stops LLVM from reordering float ops


# export OPENBLAS_CORETYPE=Haswell   # AVX2 on both Intel and AMD

# export MKL_CBWR=AVX2
# export OPENBLAS_CORETYPE=Haswell

# Freeze numba to generic CPU vector logic
# export NUMBA_CPU_NAME=generic
# export NUMBA_CACHE_DIR="\${PWD}/.numba_cache"
# export MKL_ENABLE_INSTRUCTIONS=AVX2

echo "HOST: \$(hostname)"
echo "CPU: \$(lscpu | grep 'Model name' | sed 's/Model name: *//')"

# Pass hash seed to initialization of runtime process
PYTHONHASHSEED=42 python3 <<'PY'

import os
import random
from pathlib import Path
import platform

os.environ["NUMBA_CACHE_DIR"] = os.path.join(
    os.getcwd(),
    ".numba_cache"
)

os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import torch
import numpy as np
import pandas as pd
import umap

# Seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load embeddings
def load_embedding(file_path):
    try:
        embedding_data = torch.load(
            file_path,
            map_location='cpu'
        )

        embedding = (
            embedding_data["mean_representations"][36]
            .numpy()
            .astype(np.float32)
        )

        embedding_id = embedding_data.get("label")

        if embedding_id is not None:
            embedding_id = str(embedding_id)

        return embedding, embedding_id

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None


def find_pt_files(base_dir):
    pt_files = []
    for root, dirs, files in os.walk(base_dir, followlinks=True):
        dirs.sort()
        files.sort()
        for file in files:
            if file.endswith(".pt"):
                pt_files.append(
                    os.path.join(root, file)
                )
    return pt_files

base_dirs = [
    ${embeddings_dirs.collect { "'${it}'" }.join(',')}
]

pt_files = []
for base_dir in base_dirs:
    if os.path.isdir(base_dir):
        pt_files.extend(find_pt_files(base_dir))

print(f"Found {len(pt_files)} .pt files")

# Read metadata from combined datasets tsv for contig comparison
metadata_path = "${combined_datasets_tsv}"

metadata_df = pd.read_csv(metadata_path, sep="\\t", dtype=str)

metadata_df = metadata_df.dropna(
    subset=["contig_id", "orf_id"]
).copy()

metadata_df["contig_id"] = (
    metadata_df["contig_id"].str.strip()
)

metadata_df["orf_id"] = (
    metadata_df["orf_id"].str.strip()
)

metadata_df = metadata_df[
    (metadata_df["contig_id"] != "") &
    (metadata_df["orf_id"] != "")
]

# Each ORF must map to one unique contig
mapping_counts = (
    metadata_df.groupby("orf_id")["contig_id"].nunique()
)

ambiguous_orfs = mapping_counts[mapping_counts > 1]

if not ambiguous_orfs.empty:
    raise ValueError(
        "Some orf_id values map to multiple contig_id values. "
        f"Examples: {ambiguous_orfs.index[:10].tolist()}"
    )

orf_to_contig = (
    metadata_df
    .drop_duplicates(subset=["orf_id"])
    .set_index("orf_id")["contig_id"]
    .to_dict()
)


# Read embeddings
excluded_raw = "${excluded_genofeatures}".strip()

excluded_subdirs = []

if excluded_raw and excluded_raw.lower() not in ["null", "none"]:
    excluded_subdirs = [
        x.strip().lower()
        for x in excluded_raw.split(",")
        if x.strip()
    ]

embeddings = []
embedding_ids = []

for file_path in pt_files:

    genofeature = Path(file_path).parent.parent.name

    if genofeature.lower() in excluded_subdirs:
        continue

    embedding, embedding_id = load_embedding(file_path)

    if embedding is not None:
        if embedding_id is None or not embedding_id.strip():
            embedding_id = f"__unlabeled__:{file_path}"
            print(
                f"WARNING: Missing embedding label in {file_path}; "
                f"using synthetic ID {embedding_id}."
            )

        embeddings.append(embedding)
        embedding_ids.append(embedding_id.strip())

print(
    f"Processing {len(embeddings)} embeddings"
)


# Sort deterministically
sorted_pairs = sorted(
    zip(
        embedding_ids,
        embeddings
    ),
    key=lambda x: x[0]
)


embedding_ids, embeddings = zip(
    *sorted_pairs
)


embedding_ids = list(embedding_ids)

embeddings = np.stack(
    embeddings
).astype(np.float32)

if len(embedding_ids) != len(set(embedding_ids)):
    raise ValueError(
        "Duplicate embedding IDs detected. "
        "Each embedding ID must be unique."
    )

# UMAP
nn = int(${nn})
md = float(${md})

print(
    f"Generating UMAP nn={nn}, md={md}"
)

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=nn,
    min_dist=md,
    metric="cosine",
    random_state=42,
    transform_seed=42,
    n_epochs=200,
    low_memory=False,
    force_approximation_algorithm=False,
    angular_rp_forest=False
)

reducer.fit(embeddings)

coordinates = reducer.embedding_

# Save coordinates
md_int = int(md * 10)

coord_df = pd.DataFrame({
    "embedding_id": embedding_ids,
    "x": coordinates[:, 0],
    "y": coordinates[:, 1]
})


# Build connection table using metadata
groups = {}
unmatched_ids = []
skipped_identical_ids = []

for i, eid in enumerate(embedding_ids):

    contig_id = orf_to_contig.get(eid)

    if contig_id is None:
        unmatched_ids.append(eid)
        continue

    # Skip embeddings where contig_id and orf_id are identical.
    if contig_id == eid:
        skipped_identical_ids.append(eid)
        continue

    groups.setdefault(contig_id, []).append(i)

print(
    f"Contigs represented by mapped embeddings: {len(groups)}"
)

print(
    f"Embeddings skipped because contig_id == orf_id: "
    f"{len(skipped_identical_ids)}"
)

if unmatched_ids:
    print(
        f"WARNING: {len(unmatched_ids)} embedding IDs "
        "were not found in combined_datasets.tsv."
    )
    print(
        "Unmatched embedding ID examples:",
        unmatched_ids[:10]
    )

connections = []

for contig_id, indices in groups.items():

    if len(indices) > 1:
        for i in range(len(indices) - 1):
            for j in range(i + 1, len(indices)):
                connections.append([
                    embedding_ids[indices[i]],
                    embedding_ids[indices[j]]
                ])

connections_df = pd.DataFrame(
    connections,
    columns=["id1", "id2"]
)

print(f"Generated {len(connections_df)} connections")


# Create output directory
os.makedirs("${publish_subdir}", exist_ok=True)

coord_df.to_csv(
    "${publish_subdir}/coordinates_nn{}_md{}.tsv".format(
        nn,
        md_int
    ),
    sep="\\t",
    index=False
)

connections_df.to_csv(
    "${publish_subdir}/connections.tsv",
    sep="\\t",
    index=False
)

# Versions
import platform
with open("versions.yml", "w") as handle:
    handle.write(f'"${task.process}":\\n')
    handle.write(f"    python: {platform.python_version()}\\n")
    handle.write(f"    numpy: {np.__version__}\\n")
    handle.write(f"    pandas: {pd.__version__}\\n")
    handle.write(f"    torch: {torch.__version__}\\n")
    handle.write(f"    umap: {umap.__version__}\\n")
PY
"""
}