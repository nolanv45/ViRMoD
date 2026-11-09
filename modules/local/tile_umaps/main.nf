process TILE_UMAPS {
    publishDir "${params.outdir}/04_parameter_selection/umap", mode: 'copy'
    label "process_medium"
    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://community.wave.seqera.io/library/python_numpy_pandas_matplotlib_pillow:bc19f6df598f92e8' :
        'community.wave.seqera.io/library/python_numpy_pandas_matplotlib_pillow:352bfae9d89bced3' }"
    
    input:
        path 'plots/*'
        path filtered_tsv
        path metadata_file
        
    output:
        path "tiled_image.png", emit: tiled_image
        path "versions.yml", emit: versions

    script:
    """
#!/usr/bin/env python3

import os
import math

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from PIL import Image


def build_legend_handles(filtered_df, metadata_df):

    # Find genofeatures actually present in the filtered data
    present = set(
        filtered_df["genofeature"]
        .dropna()
        .unique()
    )

    # Keep only metadata for those genofeatures
    metadata = (
        metadata_df[
            metadata_df["genofeature"].isin(present)
        ]
        .drop_duplicates("genofeature")
        .sort_values("genofeature")
    )

    # Create one legend handle per genofeature
    return [
        mlines.Line2D(
            [],
            [],
            color=row.color,
            marker=row.marker,
            linestyle="None",
            markersize=8,
            label=row.display_name,
        )
        for row in metadata.itertuples()
    ]


def render_legend(legend_handles, height):

    fig_height = max(height / 100, 4)

    fig, ax = plt.subplots(
        figsize=(8, fig_height),
        dpi=100
    )

    ax.axis("off")

    ax.legend(
        handles=legend_handles,
        title="Genofeature",
        loc="center left",
        bbox_to_anchor=(0, 0.5),
        frameon=False,
        fontsize=30,
        title_fontsize=36,
        borderaxespad=2,
        labelspacing=2.0,
        handlelength=4,
        handleheight=4,
        markerscale=4,
    )

    fig.tight_layout()

    legend_path = "legend.png"

    fig.savefig(
        legend_path,
        bbox_inches="tight",
        transparent=True,
    )

    plt.close(fig)

    return Image.open(legend_path).convert("RGBA")


def tile_images(plot_dir, output_file, legend_handles):

    image_files = sorted(
        os.path.join(plot_dir, filename)
        for filename in os.listdir(plot_dir)
        if filename.endswith(".png")
    )

    if not image_files:
        raise RuntimeError("No UMAP plot images found.")

    images = [
        Image.open(path).convert("RGBA")
        for path in image_files
    ]

    # Make every tile the same size
    tile_width = max(img.width for img in images)
    tile_height = max(img.height for img in images)

    normalized = []

    for img in images:

        canvas = Image.new(
            "RGBA",
            (tile_width, tile_height),
            (255, 255, 255, 255),
        )

        x = (tile_width - img.width) // 2
        y = (tile_height - img.height) // 2

        canvas.paste(
            img,
            (x, y),
            img,
        )

        normalized.append(canvas)

    # Determine grid dimensions
    n_images = len(normalized)
    n_cols = math.ceil(math.sqrt(n_images))
    n_rows = math.ceil(n_images / n_cols)

    grid_width = n_cols * tile_width
    grid_height = n_rows * tile_height

    grid = Image.new(
        "RGBA",
        (grid_width, grid_height),
        (255, 255, 255, 255),
    )

    # Place each plot into the grid
    for i, img in enumerate(normalized):

        row = i // n_cols
        col = i % n_cols

        grid.paste(
            img,
            (
                col * tile_width,
                row * tile_height,
            ),
            img,
        )

    # Create one legend for the entire figure
    legend = render_legend(
        legend_handles,
        grid_height,
    )

    legend_width = legend.width
    padding = 50

    final = Image.new(
        "RGBA",
        (
            grid_width + legend_width + padding,
            grid_height,
        ),
        (255, 255, 255, 255),
    )

    final.paste(
        grid,
        (0, 0),
        grid,
    )

    # Center legend vertically
    legend_y = max(
        0,
        (grid_height - legend.height) // 2,
    )

    final.paste(
        legend,
        (
            grid_width + padding,
            legend_y,
        ),
        legend,
    )

    final.convert("RGB").save(
        output_file,
        dpi=(300, 300),
    )


def write_versions(output_file):

    with open(output_file, "w") as f:
        f.write("TILE_UMAPS:\\\\n")
        f.write(
            "  python: %s\\\\n"
            % __import__("sys").version.split()[0]
        )
        f.write(
            "  pandas: %s\\\\n"
            % pd.__version__
        )
        f.write(
            "  matplotlib: %s\\\\n"
            % matplotlib.__version__
        )


# Read input files
filtered_df = pd.read_csv(
    "${filtered_tsv}",
    sep="\\t",
)

metadata_df = pd.read_csv(
    "${metadata_file}",
    sep="\\t",
)

# Build legend from genofeatures actually present
legend_handles = build_legend_handles(
    filtered_df,
    metadata_df,
)

# Tile plots and add legend
tile_images(
    "plots",
    "tiled_image.png",
    legend_handles,
)

# Record software versions
write_versions(
    "versions.yml",
)
    """
}