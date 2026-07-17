#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""CLI: rig a Blender humanoid OBJ onto ARDY's core skeleton.

    python scripts/rig_avatar.py AVATAR.obj [OUT.npz] [--smooth] [--color] [--up z]

Emits an ARDY ``skin_standard.npz``. Drop it into a skeleton folder (next to a
``joints.p`` copied from ``ardy/assets/skeletons/cskel27/``) and point ARDY's
viewer at that folder -- the character now wears your mesh. Export the humanoid
from Blender in a neutral T-pose as OBJ with "Objects/Groups as OBJ groups"
enabled so each body part carries a label (``LeftArm``, ``upper_arm.L`` ...);
without labels the tool falls back to nearest-bone geometry.

Needs only numpy -- no GPU, no ARDY model install.
"""
import argparse
import os
import sys

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from director_service.avatar_rig import rig_avatar  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obj", help="input humanoid mesh (Wavefront .obj, T-pose)")
    ap.add_argument("out", nargs="?", help="output skin_standard.npz (default: alongside the obj)")
    ap.add_argument("--up", choices=["auto", "y", "z"], default="auto",
                    help="mesh up axis; 'z' for Blender, 'y' for ARDY-native, "
                         "'auto' picks the tallest axis (default: %(default)s)")
    ap.add_argument("--smooth", action="store_true",
                    help="blend the nearest bones by inverse distance for "
                         "bend-friendly seams (default: rigid per-part)")
    ap.add_argument("--color", action="store_true",
                    help="also write per-vertex flat limb colors")
    args = ap.parse_args(argv)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.obj)), "skin_standard.npz")
    report = rig_avatar(args.obj, out, up_axis=args.up, smooth=args.smooth, color=args.color)

    print(f"wrote {report['out']}")
    print(f"  vertices    : {report['vertices']}")
    print(f"  faces       : {report['faces']}")
    print(f"  skin mode   : {report['mode']}  (labeled mesh: {report['labeled']})")
    print(f"  joints bound: {report['joints_used']}/27")
    print(f"  colors      : {report['colors']}")
    print("next: back up ardy/assets/skeletons/cskel27/skin_standard.npz, swap "
          "this file in, then run ardy/scripts/visualize.py (see README).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
