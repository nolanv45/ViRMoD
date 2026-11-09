process TILE_UMAPS {
    publishDir "${params.outdir}/04_parameter_selection/umap",
        mode: 'copy',
        saveAs: { filename ->
            // individual UMAP plots are published as umap/plots/, next to the tiled image
            if (filename == "individual_plots") "plots"
            else if (filename.startsWith("individual_plots/")) "plots/" + filename.substring("individual_plots/".length())
            else filename
        }
    label "process_medium"
    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'oras://community.wave.seqera.io/library/python_numpy_pandas_matplotlib_pillow:bc19f6df598f92e8' :
        'community.wave.seqera.io/library/python_numpy_pandas_matplotlib_pillow:352bfae9d89bced3' }"

    input:
        path 'plots/*'
        path 'filtered.tsv'
        path 'metadata.tsv'

    output:
        path "tiled_image.png",  emit: tiled_image
        path "individual_plots", emit: plots
        path "versions.yml",    emit: versions

    script:
    """
#!/usr/bin/env python3

import os
import re
import shutil
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import PIL
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

PLOT_DIR = "plots"
OUTPUT_FILE = "tiled_image.png"
PLOT_OUT_DIR = "individual_plots"
PADDING = 50

# Matches umap_nn<NN>_md<MD>.png where MD is e.g. 0p1 (same convention as TILE_HDBSCAN)
FILENAME_RE = re.compile(r"umap_nn(\\d+)_md(\\d+p?\\d*)\\.png")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_font(size):
    # Prefer the DejaVu font bundled with matplotlib (always present in this container)
    candidates = [
        os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf"),
        "${projectDir}/fonts/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: {name} is missing required column(s): {', '.join(missing)}. "
            f"Columns found: {', '.join(map(str, df.columns))}"
        )


def collect_plots(plot_dir):
    plot_files = {}
    for filename in sorted(os.listdir(plot_dir)):
        match = FILENAME_RE.fullmatch(filename)
        if match is None:
            print(f"Skipping unrecognised file: {filename}")
            continue
        nn = int(match.group(1))
        md = float(match.group(2).replace("p", "."))
        plot_files[(nn, md)] = os.path.join(plot_dir, filename)

    if not plot_files:
        sys.exit("ERROR: No UMAP plot images (umap_nn*_md*.png) found in " + plot_dir)
    return plot_files


def copy_plots(plot_files, out_dir):
    # Copy (not link) the individual plots so the published directory holds real files
    os.makedirs(out_dir, exist_ok=True)
    for path in plot_files.values():
        shutil.copy2(path, os.path.join(out_dir, os.path.basename(path)))


def build_legend_handles(filtered_df, metadata_df):
    # Only genofeatures actually present in the filtered data get a legend entry
    present = set(filtered_df["genofeature"].dropna().unique())

    metadata = (
        metadata_df[metadata_df["genofeature"].isin(present)]
        .drop_duplicates("genofeature")
    )

    missing = present - set(metadata["genofeature"])
    if missing:
        print("WARNING: no metadata for genofeature(s): " + ", ".join(sorted(map(str, missing))))

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


def render_legend(legend_handles, target_height_px, font_size):
    # Make the figure tall enough for every entry (labelspacing=2.0 -> ~3.2 lines per entry)
    needed_in = (len(legend_handles) * font_size * 3.0 + font_size * 3) / 72
    fig_height = max(target_height_px / 100, needed_in, 4)

    # Scale the legend icons with the text (font_size is in points)
    for handle in legend_handles:
        handle.set_markersize(font_size)
        handle.set_markeredgewidth(font_size * 0.2)

    fig, ax = plt.subplots(figsize=(8, fig_height), dpi=100)
    ax.axis("off")

    ax.legend(
        handles=legend_handles,
        title="Genofeature",
        loc="center left",
        bbox_to_anchor=(0, 0.5),
        frameon=False,
        fontsize=font_size,
        title_fontsize=font_size,
        borderaxespad=2,
        labelspacing=1.0,
        handlelength=2,
        handleheight=1.5,
        handletextpad=0.8,
    )

    fig.savefig("legend.png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return Image.open("legend.png").convert("RGBA")


def draw_rotated_text(canvas, text, font, center, fill="black"):
    left, top, right, bottom = font.getbbox(text)
    w, h = right - left, bottom - top
    layer = Image.new("RGBA", (w + 20, h + 20), (255, 255, 255, 0))
    ImageDraw.Draw(layer).text((10 - left, 10 - top), text, fill=fill, font=font)
    layer = layer.rotate(90, expand=True)
    canvas.paste(
        layer,
        (center[0] - layer.width // 2, center[1] - layer.height // 2),
        layer,
    )


# ----------------------------------------------------------------------------
# Tiling
# ----------------------------------------------------------------------------
def tile_images(plot_files, legend_handles, output_file, padding=PADDING):
    nn_values = sorted({nn for nn, _ in plot_files})
    md_values = sorted({md for _, md in plot_files})
    images = {key: Image.open(path).convert("RGBA") for key, path in plot_files.items()}

    # Every cell is the size of the largest plot; smaller plots are centred in their cell
    tile_w = max(img.width for img in images.values())
    tile_h = max(img.height for img in images.values())

    n_cols = len(md_values)
    n_rows = len(nn_values)
    grid_w = n_cols * tile_w + (n_cols - 1) * padding
    grid_h = n_rows * tile_h + (n_rows - 1) * padding

    # The legend is sized in points at 100 dpi and PIL text in pixels, so convert
    # to make the axis titles and labels the same size as the legend text
    legend_pt = max(48, min(120, tile_w // 50))
    label_size = round(legend_pt * 100 / 72)
    font = load_font(label_size)

    # Margins: left holds rotated axis title + row labels, top holds axis title + column labels
    left_margin = label_size * 6
    top_margin = int(label_size * 4.5)
    bottom_margin = label_size

    grid_x = left_margin
    grid_y = top_margin

    canvas = Image.new(
        "RGBA",
        (left_margin + grid_w, top_margin + grid_h + bottom_margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    # Axis titles
    draw.text(
        (grid_x + grid_w // 2, label_size),
        "Minimum Distance (md)",
        fill="black",
        font=font,
        anchor="mm",
    )
    draw_rotated_text(
        canvas,
        "Nearest Neighbors (nn)",
        font,
        (label_size, grid_y + grid_h // 2),
    )

    # Column labels (md) and row labels (nn)
    for col, md in enumerate(md_values):
        draw.text(
            (grid_x + col * (tile_w + padding) + tile_w // 2, label_size * 3),
            f"md={md:.1f}",
            fill="black",
            font=font,
            anchor="mm",
        )
    for row, nn in enumerate(nn_values):
        draw.text(
            (int(label_size * 3.6), grid_y + row * (tile_h + padding) + tile_h // 2),
            f"nn={nn}",
            fill="black",
            font=font,
            anchor="mm",
        )

    # Plots: rows = nn (ascending, top to bottom), columns = md (ascending, left to right)
    for (nn, md), img in images.items():
        row = nn_values.index(nn)
        col = md_values.index(md)
        x = grid_x + col * (tile_w + padding) + (tile_w - img.width) // 2
        y = grid_y + row * (tile_h + padding) + (tile_h - img.height) // 2
        canvas.paste(img, (x, y), img)

    # Legend on the right, vertically centred on the plot grid
    if legend_handles:
        legend = render_legend(legend_handles, grid_h, legend_pt)
        final_w = canvas.width + padding + legend.width
        final_h = max(canvas.height, legend.height)
        final = Image.new("RGBA", (final_w, final_h), "white")
        final.paste(canvas, (0, 0), canvas)
        legend_y = max(0, grid_y + grid_h // 2 - legend.height // 2)
        final.paste(legend, (canvas.width + padding, legend_y), legend)
    else:
        print("WARNING: no legend entries found; writing tiles without a legend")
        final = canvas

    final.convert("RGB").save(output_file, dpi=(300, 300))
    print(f"Saved {output_file} ({final.width} x {final.height}px, {len(images)} tiles)")


def write_versions(output_file):
    with open(output_file, "w") as f:
        f.write('"${task.process}":\\n')
        f.write(f"    python: {sys.version.split()[0]}\\n")
        f.write(f"    pandas: {pd.__version__}\\n")
        f.write(f"    matplotlib: {matplotlib.__version__}\\n")
        f.write(f"    pillow: {PIL.__version__}\\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    filtered_df = pd.read_csv("filtered.tsv", sep="\\t")
    metadata_df = pd.read_csv("metadata.tsv", sep="\\t")

    require_columns(filtered_df, ["genofeature"], "filtered.tsv")
    require_columns(metadata_df, ["genofeature", "color", "marker", "display_name"], "metadata.tsv")

    legend_handles = build_legend_handles(filtered_df, metadata_df)
    plot_files = collect_plots(PLOT_DIR)
    copy_plots(plot_files, PLOT_OUT_DIR)

    tile_images(plot_files, legend_handles, OUTPUT_FILE)
    write_versions("versions.yml")


main()
    """
}