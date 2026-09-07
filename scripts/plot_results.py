#!/usr/bin/env python3
"""Render the seven submitted meshes for visual inspection (optional matplotlib)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hac_forward.stl import load_stl_triangles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("docs/submission_models.png"))
    args = parser.parse_args()
    manifest = json.loads((args.results / "manifest.json").read_text())
    fig = plt.figure(figsize=(14, 7.3), facecolor="white")
    for index, record in enumerate(manifest["records"]):
        ax = fig.add_subplot(2, 4, index + 1, projection="3d", proj_type="ortho")
        model = record["model_id"]
        triangles = load_stl_triangles(args.results / record["stl"])
        ax.add_collection3d(
            Poly3DCollection(
                triangles, facecolors="#5596a8", edgecolors="#244453", linewidths=0.25, shade=True
            )
        )
        bound = max(1, manifest["config"]["cylinder_radii"][str(model)]) * 1.05
        ax.set(xlim=(-bound, bound), ylim=(-bound, bound), zlim=(-bound, bound))
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=25, azim=-55)
        ax.set_axis_off()
        ax.set_title(
            f"Model {model} · {record['metrics']['effective_k']} facets\nR = {manifest['config']['cylinder_radii'][str(model)]:g}",
            fontsize=11,
            pad=-8,
        )
        assert np.all(np.isfinite(triangles))
    fig.text(
        0.765,
        0.285,
        "Final convex reconstructions\n\nEach panel has equal axis scaling.\nPanel magnification varies by model.\nAll models have height 2.\n\nShapes of Models 4–10 are secret;\nthese are inferred candidates.",
        fontsize=11,
        va="center",
        linespacing=1.6,
        color="#233346",
    )
    fig.suptitle(
        "HAC 2026 · Optimal transport convex submission", fontsize=18, x=0.035, ha="left", y=0.98
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.02, hspace=0.02)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
