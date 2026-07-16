#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Measure (and plot) the root-velocity discontinuity at choreography seams.

`/choreograph` writes the prompt-change frames into the .npz, so this needs
nothing but the file(s):

    # after only
    python scripts/seam_report.py outputs/choreo_stream.npz

    # before/after, one plot
    python scripts/seam_report.py outputs/choreo_stream.npz \
        --before outputs/choreo_stitch.npz --plot seams.png

Produce the pair from one service with the same seed -- mode is the only
difference:

    curl -s localhost:9600/choreograph -H 'content-type: application/json' \
      -d '{"seed":42,"mode":"stitch","steps":[{"prompt":"walk forward confidently","duration":3},
                                              {"prompt":"stop and turn around","duration":2}]}'
    curl -s localhost:9600/choreograph -H 'content-type: application/json' \
      -d '{"seed":42,"mode":"stream","steps":[{"prompt":"walk forward confidently","duration":3},
                                              {"prompt":"stop and turn around","duration":2}]}'

The number to read is `ratio`: how many times harder the root's velocity changes
at the seam than it does on a typical frame of the same clip. ~1 means the seam
is indistinguishable from ordinary motion. Large means a pop.

matplotlib is optional -- without it you still get the numbers.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from director_service.streaming import root_velocity, seam_report  # noqa: E402


def load(path):
    data = np.load(path)
    if "root_positions" not in data:
        raise SystemExit(f"{path}: no root_positions (not a director clip?)")
    root = np.asarray(data["root_positions"])
    if root.ndim == 3 and root.shape[0] == 1:  # tolerate an unsqueezed sample dim
        root = root[0]
    fps = float(data["fps"]) if "fps" in data else 30.0
    seams = [int(f) for f in np.atleast_1d(data["seam_frames"])] if "seam_frames" in data else []
    text = str(data["text"]) if "text" in data else ""
    return {"path": path, "root": root, "fps": fps, "seams": seams, "text": text}


def describe(clip, seams):
    report = seam_report(clip["root"], seams, clip["fps"])
    print(f"\n{os.path.basename(clip['path'])}  ({clip['root'].shape[0]} frames @ {clip['fps']:g} fps)")
    if clip["text"]:
        print(f"  prompts: {clip['text']}")
    if not report["seams"]:
        print("  no seams to measure (single-prompt clip?)")
        return report
    print(f"  typical frame-to-frame velocity change: {report['baseline_jump']:.4f} units/s")
    for row in report["seams"]:
        print(f"  seam @ frame {row['frame']:>4} ({row['time_s']:>5.2f}s): "
              f"jump {row['jump']:7.4f} units/s = {row['ratio']:6.1f}x typical")
    return report


def plot(after, report_after, before=None, report_before=None, out="seams.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- numbers only; `pip install matplotlib` for the plot)")
        return None

    clips = [(after, report_after, "stream (autoregressive)", "tab:green")]
    if before is not None:
        clips.insert(0, (before, report_before, "stitch (v0.1)", "tab:red"))

    # Shared axes: a before/after where each panel silently rescales its own y
    # makes a pop look the same size as smooth motion.
    fig, axes = plt.subplots(len(clips), 1, figsize=(10, 3.2 * len(clips)),
                             sharex=True, sharey=True, squeeze=False)
    for ax, (clip, report, label, color) in zip(axes[:, 0], clips):
        v = root_velocity(clip["root"], clip["fps"])
        speed = np.linalg.norm(v, axis=1)
        t = np.arange(len(speed)) / clip["fps"]
        ax.plot(t, speed, color=color, lw=1.2, label=f"{label} — root speed")
        for row in report["seams"]:
            ax.axvline(row["time_s"], color="k", ls="--", lw=0.9, alpha=0.7)
            ax.annotate(f"{row['ratio']:.0f}x", xy=(row["time_s"], ax.get_ylim()[1]),
                        xytext=(3, -12), textcoords="offset points", fontsize=8)
        ax.set_ylabel("root speed (units/s)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle("Root velocity at prompt changes (dashed = seam)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("after", help="clip to measure (a /choreograph .npz)")
    ap.add_argument("--before", help="optional clip to compare against (e.g. mode=stitch)")
    ap.add_argument("--seams", type=int, nargs="*",
                    help="seam frames, if the .npz does not carry them")
    ap.add_argument("--plot", nargs="?", const="seams.png",
                    help="write a root-velocity plot (default seams.png)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args()

    after = load(args.after)
    seams_after = args.seams if args.seams else after["seams"]
    report_after = describe(after, seams_after)

    before = report_before = None
    if args.before:
        before = load(args.before)
        report_before = describe(before, args.seams if args.seams else before["seams"])
        b, a = report_before["max_ratio"], report_after["max_ratio"]
        print(f"\nworst seam: {b:.1f}x -> {a:.1f}x", end="")
        print(f"  ({b / a:.1f}x smoother)" if a > 1e-9 and b > a else "")

    if args.plot:
        plot(after, report_after, before, report_before, args.plot)

    if args.json:
        print(json.dumps({"after": report_after, "before": report_before}, indent=2, default=float))


if __name__ == "__main__":
    main()
