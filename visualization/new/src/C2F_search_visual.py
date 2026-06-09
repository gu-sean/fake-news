"""Create a coarse-to-fine iterative search visualization."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "img"

BLUE = "#2463EB"
RED = "#FF4B4B"
DARK = "#253047"
GRID = "#D9E1EF"
PANEL_BG = "#FAFCFF"


def draw_panel(ax, points, target, iteration, scale_label):
    """Draw one search iteration while preserving the reference point pattern."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.set_facecolor(PANEL_BG)

    for spine in ax.spines.values():
        spine.set_color(DARK)
        spine.set_linewidth(1.8)

    ax.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        s=27,
        color=BLUE,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    ax.scatter(
        target[0],
        target[1],
        s=155,
        color=RED,
        edgecolors="white",
        linewidths=1.5,
        zorder=4,
    )

    # The highlighted box indicates the local region selected for the next pass.
    box_width, box_height = 0.24, 0.22
    ax.add_patch(
        Rectangle(
            (target[0] - box_width / 2, target[1] - box_height / 2),
            box_width,
            box_height,
            fill=False,
            edgecolor=RED,
            linewidth=1.4,
            linestyle=(0, (4, 3)),
            alpha=0.75,
            zorder=2,
        )
    )

    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(length=0, labelbottom=False, labelleft=False)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.55)

    ax.set_title(f"Iteration {iteration}", fontsize=14, fontweight="bold", color=DARK, pad=13)
    ax.text(
        0.5,
        -0.115,
        scale_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.5,
        color="#657087",
    )


def add_flow_arrow(fig, left_ax, right_ax):
    """Connect adjacent panels with a horizontal process arrow."""
    left_box = left_ax.get_position()
    right_box = right_ax.get_position()
    start = (left_box.x1 + 0.008, (left_box.y0 + left_box.y1) / 2)
    end = (right_box.x0 - 0.008, (right_box.y0 + right_box.y1) / 2)
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color="#8793A8",
        zorder=5,
    )
    fig.add_artist(arrow)


def main():
    # Stage 1 mirrors the broad point arrangement from the supplied sketch.
    stage_1 = [
        (0.15, 0.90), (0.43, 0.90), (0.58, 0.90), (0.85, 0.90),
        (0.15, 0.63), (0.43, 0.63), (0.58, 0.63), (0.85, 0.63),
        (0.15, 0.47), (0.43, 0.47),               (0.85, 0.47),
        (0.15, 0.18), (0.43, 0.18), (0.58, 0.18), (0.85, 0.18),
    ]
    target_1 = (0.58, 0.46)

    # Later stages retain the same grid-like structure at progressively finer scales.
    stage_2 = [
        (0.37, 0.69), (0.50, 0.69), (0.56, 0.69), (0.69, 0.69),
        (0.37, 0.54), (0.50, 0.54), (0.56, 0.54),
        (0.37, 0.45), (0.50, 0.45), (0.56, 0.45), (0.69, 0.45),
        (0.37, 0.32), (0.50, 0.32), (0.56, 0.32), (0.69, 0.32),
    ]
    target_2 = (0.69, 0.55)

    stage_3 = [
        (0.48, 0.62), (0.56, 0.62), (0.64, 0.62), (0.72, 0.62),
        (0.48, 0.55), (0.56, 0.55),               (0.72, 0.55),
        (0.48, 0.49), (0.56, 0.49), (0.64, 0.49), (0.72, 0.49),
        (0.48, 0.43), (0.56, 0.43), (0.64, 0.43), (0.72, 0.43),
    ]
    target_3 = (0.64, 0.55)

    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.20, top=0.82, wspace=0.36)

    panels = [
        (stage_1, target_1, 1, "Broad search"),
        (stage_2, target_2, 2, "Refined local search"),
        (stage_3, target_3, 3, "Fine local search"),
    ]
    for ax, (points, target, iteration, scale_label) in zip(axes, panels):
        draw_panel(ax, points, target, iteration, scale_label)

    axes[0].set_ylabel("Feature y", fontsize=12, fontweight="bold", color=DARK, labelpad=14)
    for ax in axes:
        ax.set_xlabel("Feature x", fontsize=12, fontweight="bold", color=DARK, labelpad=10)

    add_flow_arrow(fig, axes[0], axes[1])
    add_flow_arrow(fig, axes[1], axes[2])

    fig.suptitle(
        "Iterative Coarse-to-Fine Search",
        x=0.075,
        y=0.925,
        ha="left",
        fontsize=24,
        fontweight="bold",
        color=DARK,
    )
    fig.text(
        0.075,
        0.875,
        "The search range narrows around the best candidate at each iteration.",
        ha="left",
        fontsize=12,
        color="#657087",
    )
    legend_items = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
               markeredgecolor="white", markersize=7, label="candidates"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED,
               markeredgecolor="white", markersize=10, label="best candidate"),
        Line2D([0], [0], color=RED, linewidth=1.4, linestyle=(0, (4, 3)),
               label="next search region"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=False,
        fontsize=10.5,
        labelcolor="#657087",
        handlelength=1.5,
        columnspacing=2.0,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "C2F_search_visual.png"
    svg_path = OUTPUT_DIR / "C2F_search_visual.svg"
    fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
