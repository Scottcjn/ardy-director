#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""CLI: turn an ARDY director .npz clip into an FBX skeletal animation.

    python scripts/export_fbx.py CLIP.npz [OUT.fbx] [--scale 100]

The FBX imports into Unreal Engine 5 as a joint hierarchy + animation; from
there the IK Retargeter drives the UE5 Mannequin or a MetaHuman. See the
"ARDY -> Unreal" section of the repo README for the import recipe.

Needs only numpy -- no GPU, no ARDY model. Everything the export needs is
already baked into the clip the generator wrote.
"""
import argparse
import os
import sys

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from director_service.fbx_export import convert  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", help="input director .npz (from /generate or /choreograph)")
    ap.add_argument("out", nargs="?", help="output .fbx path (default: alongside the clip)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="length multiplier; ARDY is metric, pass 100 for cm "
                         "(Unreal-native). Default 1.0 -- you can also rescale "
                         "in UE's import dialog. (default: %(default)s)")
    args = ap.parse_args(argv)

    out = args.out or (os.path.splitext(args.clip)[0] + ".fbx")
    report = convert(args.clip, out, scale=args.scale)

    print(f"wrote {report['out']}")
    print(f"  bones      : {report['bones']}")
    print(f"  frames     : {report['frames']} @ {report['fps']} fps")
    print(f"  prompt     : {report['text']!r}")
    print(f"  rigid check: max rest-offset drift {report['offset_drift']:.2e} "
          f"(should be ~0 for a rigid skeleton)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
