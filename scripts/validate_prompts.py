#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Run the prompt library through a live Director service and report what happened.

The prompt library (examples/prompts.json) ships UNVERIFIED: the prompts are
written against ARDY's documented training distribution, but nobody has watched
them produce motion. This script is how they get confirmed — one command on a
host that can reach the Director service, instead of hand-checking 40 clips.

What it does NOT do: judge whether the motion *matches the prompt*. That needs
eyes. What it does is generate every prompt, then report cheap mechanical
signals (did it generate, how far the root travelled, how much of the clip had a
foot down) and flag the ones that contradict the library's own expectations —
a locomotion prompt that never moves, an out-of-distribution prompt that sailed
through. Those flags are where to point your eyes first, not a verdict.

Usage:
    # generate every prompt and write a report
    python scripts/validate_prompts.py --url http://192.168.0.136:9600

    # just one category, fixed seed, faster sampling
    python scripts/validate_prompts.py --category locomotion --seed 42 --diffusion-steps 8

    # fold the results back into examples/prompts.json (sets verified/observed)
    python scripts/validate_prompts.py --write-back

Exit status is 0 unless a prompt failed to generate at all (an HTTP/model error).
Drift flags do not fail the run — they are for a human to read.
"""
import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LIBRARY = os.path.join(REPO_ROOT, "examples", "prompts.json")
DEFAULT_URL = os.environ.get("DIRECTOR_URL", "http://192.168.0.136:9600")

# A root that moves less than this over the whole clip is "in place" for our
# purposes. ARDY core units are metres; a walk covers metres, jitter is cm.
DISPLACEMENT_EPS_M = 0.5


def load_library(path):
    with open(path) as fh:
        lib = json.load(fh)
    if not lib.get("prompts"):
        raise SystemExit(f"{path}: no prompts in library")
    return lib


def measure_npz(path):
    """Cheap mechanical signals from a generated clip. Best-effort by design:
    the report is still useful when the .npz is on a host we cannot read."""
    try:
        import numpy as np
    except ImportError:
        return {"measured": False, "why": "numpy not available here"}
    if not path or not os.path.exists(path):
        return {"measured": False, "why": "npz not readable from this host"}
    try:
        with np.load(path, allow_pickle=True) as data:
            out = {"measured": True}
            if "root_positions" in data:
                rp = np.asarray(data["root_positions"])
                if rp.ndim == 2 and rp.shape[0] > 1:
                    # travel in the ground plane (XZ), start to finish
                    delta = rp[-1, [0, 2]] - rp[0, [0, 2]]
                    out["displacement_m"] = round(float(np.linalg.norm(delta)), 3)
                    # path length catches "walks in a circle", where start≈end
                    steps = np.linalg.norm(np.diff(rp[:, [0, 2]], axis=0), axis=1)
                    out["path_length_m"] = round(float(steps.sum()), 3)
                    out["root_height_range_m"] = round(
                        float(rp[:, 1].max() - rp[:, 1].min()), 3)
            if "foot_contacts" in data:
                fc = np.asarray(data["foot_contacts"], dtype=float)
                if fc.size:
                    out["foot_contact_fraction"] = round(float(fc.mean()), 3)
            return out
    except Exception as e:  # a readable report beats a traceback
        return {"measured": False, "why": f"{type(e).__name__}: {e}"}


def flags_for(entry, result, signals):
    """Contradictions between what the library expects and what came out."""
    flags = []
    if not result.get("ok"):
        return ["generate_failed"]
    disp = signals.get("displacement_m")
    path_len = signals.get("path_length_m")
    if disp is not None:
        expects_travel = entry.get("expect_displacement")
        if expects_travel:
            # Path length, not net displacement: "walks in a circle" ends where
            # it started, so net travel is ~0 while the character clearly walked.
            went_somewhere = max(disp, path_len or 0.0) > DISPLACEMENT_EPS_M
            if not went_somewhere:
                flags.append("expected_travel_but_stayed_put")
        elif expects_travel is False:
            # Net displacement only. Path length is useless here: a standing
            # character's root still jitters every frame, and summing that over
            # a few hundred frames clears any sane threshold, so path length
            # would flag every in-place prompt in the library.
            if disp > DISPLACEMENT_EPS_M:
                flags.append("expected_in_place_but_travelled")
    if entry.get("expect_failure"):
        # An OOD prompt that generates cleanly is the interesting case: either
        # the collapse is subtle (needs eyes) or the prompt is not OOD after all.
        flags.append("ood_generated_check_by_eye")
    return flags


def run_one(session, url, entry, args):
    payload = {
        "prompt": entry["prompt"],
        "model": args.model,
        "duration": entry.get("duration", 4.0),
        "cfg_weight": args.cfg_weight,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.diffusion_steps is not None:
        payload["diffusion_steps"] = args.diffusion_steps

    t0 = time.time()
    try:
        r = session.post(f"{url}/generate", json=payload, timeout=args.timeout)
        elapsed = round(time.time() - t0, 1)
        if r.status_code != 200:
            return {"ok": False, "error": r.text[:300], "status": r.status_code,
                    "elapsed_s": elapsed}
        body = r.json()
        body["elapsed_s"] = elapsed
        return body
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.time() - t0, 1)}


def write_markdown(report, path):
    lib = report["library"]
    rows = report["results"]
    with open(path, "w") as fh:
        fh.write("# Prompt library validation report\n\n")
        fh.write(f"- Service: `{report['url']}`\n")
        fh.write(f"- Model: `{report['model']}`  ")
        fh.write(f"seed: `{report['seed']}`  ")
        fh.write(f"diffusion steps: `{report['diffusion_steps'] or 'model default'}`\n")
        fh.write(f"- Ran: {report['ran_at']}\n")
        fh.write(f"- {report['generated']}/{report['total']} generated, "
                 f"{report['failed']} failed, {report['flagged']} flagged for review\n\n")
        fh.write("`displacement` is start-to-finish travel in the ground plane; "
                 "`path` is total distance walked (they differ when the character "
                 "loops back). Flags mark a clip that contradicts the library's "
                 "expectation — a human still has to look.\n\n")
        fh.write("| id | category | prompt | ok | displacement (m) | path (m) | foot contact | flags |\n")
        fh.write("|----|----------|--------|----|------------------|----------|--------------|-------|\n")
        for r in rows:
            s = r["signals"]
            def cell(v):
                return "—" if v is None else v
            fh.write("| `{id}` | {cat} | {p} | {ok} | {d} | {pl} | {fc} | {fl} |\n".format(
                id=r["id"], cat=r["category"], p=r["prompt"].replace("|", "\\|")[:60],
                ok="yes" if r["result"].get("ok") else "**NO**",
                d=cell(s.get("displacement_m")), pl=cell(s.get("path_length_m")),
                fc=cell(s.get("foot_contact_fraction")),
                fl=", ".join(r["flags"]) or "—"))
        fh.write("\n")
        if lib.get("about", {}).get("phrasing"):
            fh.write("> " + lib["about"]["phrasing"] + "\n")


def write_back(lib, report, path):
    """Fold observations into the library so 'verified' stops being a promise."""
    by_id = {r["id"]: r for r in report["results"]}
    for entry in lib["prompts"]:
        r = by_id.get(entry["id"])
        if not r:
            continue
        ok = bool(r["result"].get("ok"))
        entry["verified"] = ok
        observed = dict(r["signals"])
        observed.pop("measured", None)
        observed.pop("why", None)
        if r["flags"]:
            observed["flags"] = r["flags"]
        if not ok:
            observed["error"] = r["result"].get("error", "")[:200]
        observed["ran_at"] = report["ran_at"]
        entry["observed"] = observed
    lib.setdefault("about", {}).setdefault("verification", {})["status"] = (
        f"Validated against {report['url']} on {report['ran_at']}: "
        f"{report['generated']}/{report['total']} generated, "
        f"{report['flagged']} flagged for review. "
        "'verified' means the prompt generated motion — not that a human judged it."
    )
    with open(path, "w") as fh:
        json.dump(lib, fh, indent=2)
        fh.write("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL, help=f"Director service (default {DEFAULT_URL})")
    ap.add_argument("--library", default=DEFAULT_LIBRARY, help="prompts.json to run")
    ap.add_argument("--category", action="append", help="only this category (repeatable)")
    ap.add_argument("--model", default="core")
    ap.add_argument("--seed", type=int, default=None, help="fixed seed for reproducibility")
    ap.add_argument("--cfg-weight", type=float, default=4.0)
    ap.add_argument("--diffusion-steps", type=int, default=None,
                    help="fewer steps = faster sweep, lower quality")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--report", default=None, help="report basename (default: reports/prompt-validation-<ts>)")
    ap.add_argument("--write-back", action="store_true",
                    help="update the library's verified/observed fields in place")
    ap.add_argument("--skip-ood", action="store_true",
                    help="skip out_of_distribution prompts (they are expected to fail)")
    args = ap.parse_args()

    try:
        import requests
    except ImportError:
        raise SystemExit("validate_prompts.py needs `requests` (pip install requests)")

    lib = load_library(args.library)
    entries = lib["prompts"]
    if args.category:
        entries = [e for e in entries if e["category"] in set(args.category)]
    if args.skip_ood:
        entries = [e for e in entries if e["category"] != "out_of_distribution"]
    if not entries:
        raise SystemExit("no prompts selected")

    session = requests.Session()
    try:
        h = session.get(f"{args.url}/health", timeout=15).json()
        print(f"Director up: device={h.get('device')} models={h.get('models_loaded')}")
    except Exception as e:
        raise SystemExit(f"Director unreachable at {args.url}: {type(e).__name__}: {e}\n"
                         "Start director_service/service.py on the ARDY host first.")

    ran_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results = []
    for i, entry in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {entry['id']}: {entry['prompt'][:60]} ... ", end="", flush=True)
        result = run_one(session, args.url, entry, args)
        signals = measure_npz(result.get("npz")) if result.get("ok") else {"measured": False}
        flags = flags_for(entry, result, signals)
        results.append({"id": entry["id"], "category": entry["category"],
                        "prompt": entry["prompt"], "duration": entry.get("duration"),
                        "result": result, "signals": signals, "flags": flags})
        if result.get("ok"):
            extra = f" [{', '.join(flags)}]" if flags else ""
            print(f"ok {result.get('frames')}f in {result['elapsed_s']}s{extra}")
        else:
            print(f"FAILED: {str(result.get('error'))[:80]}")

    failed = sum(1 for r in results if not r["result"].get("ok"))
    report = {
        "ran_at": ran_at, "url": args.url, "model": args.model, "seed": args.seed,
        "diffusion_steps": args.diffusion_steps, "library": lib,
        "total": len(results), "generated": len(results) - failed, "failed": failed,
        "flagged": sum(1 for r in results if r["flags"]),
        "results": results,
    }

    base = args.report or os.path.join(REPO_ROOT, "reports",
                                       f"prompt-validation-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
    slim = dict(report)
    slim["library"] = {"schema_version": lib.get("schema_version"), "model": lib.get("model")}
    with open(base + ".json", "w") as fh:
        json.dump(slim, fh, indent=2)
    write_markdown(report, base + ".md")
    print(f"\n{report['generated']}/{report['total']} generated, {failed} failed, "
          f"{report['flagged']} flagged")
    print(f"Report: {base}.md")

    if args.write_back:
        write_back(lib, report, args.library)
        print(f"Wrote observations back into {args.library}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
