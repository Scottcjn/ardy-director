# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Tests for the prompt preset library and its validator.

These run anywhere -- no GPU, no ARDY, no Director service. They cover what can
be checked without hardware: that the library is well-formed and honest about
its own state, and that the validator's drift flags fire on the right shapes.
Whether a prompt actually produces sensible motion is not testable here; that is
what scripts/validate_prompts.py is for.
"""
import collections
import importlib.util
import json
import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY_PATH = os.path.join(REPO_ROOT, "examples", "prompts.json")
README_PATH = os.path.join(REPO_ROOT, "README.md")

_spec = importlib.util.spec_from_file_location(
    "validate_prompts", os.path.join(REPO_ROOT, "scripts", "validate_prompts.py"))
validate_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validate_prompts)


@pytest.fixture(scope="module")
def library():
    with open(LIBRARY_PATH) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def prompts(library):
    return library["prompts"]


def test_library_is_valid_json_with_prompts(library):
    assert library["prompts"], "library must not be empty"
    assert library["model"] == "core"


def test_acceptance_at_least_30_usable_prompts(prompts):
    """The issue asks for at least 30 prompts. OOD entries are documentation of
    the model's edges, not recommendations, so they must not pad the count."""
    usable = [p for p in prompts if p["category"] != "out_of_distribution"]
    assert len(usable) >= 30, f"only {len(usable)} usable prompts"


def test_every_prompt_has_the_documented_fields(prompts):
    for p in prompts:
        missing = {"id", "category", "prompt", "duration", "note", "provenance",
                   "verified"} - set(p)
        assert not missing, f"{p.get('id')} missing {missing}"
        assert p["prompt"].strip(), f"{p['id']} has an empty prompt"
        assert p["note"].strip(), f"{p['id']} has an empty note"


def test_ids_are_unique(prompts):
    ids = [p["id"] for p in prompts]
    assert len(ids) == len(set(ids)), "duplicate prompt ids"


def test_prompts_are_unique(prompts):
    texts = [p["prompt"].lower() for p in prompts]
    assert len(texts) == len(set(texts)), "duplicate prompt text"


def test_categories_are_declared(library, prompts):
    declared = set(library["categories"])
    used = {p["category"] for p in prompts}
    assert used <= declared, f"undeclared categories: {used - declared}"
    assert declared == used, f"declared but unused: {declared - used}"


def test_every_category_is_populated(library, prompts):
    for category in library["categories"]:
        assert any(p["category"] == category for p in prompts), f"{category} is empty"


def test_durations_are_within_the_service_bounds(prompts):
    """director_service GenerateReq: duration = Field(gt=0.1, le=30.0)."""
    for p in prompts:
        assert 0.1 < p["duration"] <= 30.0, f"{p['id']} duration {p['duration']} rejected by /generate"


def test_provenance_values_are_declared(library, prompts):
    declared = set(library["about"]["provenance_values"])
    for p in prompts:
        assert p["provenance"] in declared, f"{p['id']}: unknown provenance {p['provenance']}"


def test_upstream_presets_are_quoted_verbatim(prompts):
    """The upstream-attributed prompts must match nv-tlabs/ardy's PRESET_PROMPTS
    exactly. If someone 'tidies' the phrasing, the attribution becomes a lie and
    the author-vetted status is lost -- that is the whole value of these entries.
    Source: scripts/interactive_demo/common.py (PRESET_PROMPTS) and the
    scripts/generate.py docstring examples."""
    upstream = {
        "A person is walking.",
        "A person jumps backwards.",
        "A person side steps to the right.",
        "A person is walking backwards.",
        "A person is kicking with their right leg.",
        "A person is standing.",
        "A young lady walks forward elegantly.",
        "A person bows down and then stands upright.",
        "A ballet dancer, performs a forward, turn joining feet, in a repeating loop",
        "a performer gives high bow, with arms to the side, right leg crossed behind the left",
    }
    upstream_doc = {"A person walks in a circle.", "A person jumps."}
    for p in prompts:
        if p["provenance"] == "upstream_preset":
            assert p["prompt"] in upstream, f"{p['id']} claims upstream_preset but is not verbatim"
        elif p["provenance"] == "upstream_doc":
            assert p["prompt"] in upstream_doc, f"{p['id']} claims upstream_doc but is not verbatim"


def test_library_ships_unverified(prompts):
    """Nothing here has been run on ARDY hardware. If a prompt claims otherwise
    without an 'observed' block from the validator, the claim is unbacked."""
    for p in prompts:
        if p["verified"]:
            assert p.get("observed"), f"{p['id']} claims verified with no observed data"


def test_ood_prompts_are_marked_as_expected_failures(prompts):
    for p in prompts:
        if p["category"] == "out_of_distribution":
            assert p.get("expect_failure") is True, f"{p['id']} is OOD but not flagged"


def test_only_ood_prompts_expect_failure(prompts):
    for p in prompts:
        if p.get("expect_failure"):
            assert p["category"] == "out_of_distribution", \
                f"{p['id']} expects failure but is offered as a usable prompt"


def test_readme_category_table_matches_the_library(prompts):
    """The README table states a count per category. Counting by hand is how a
    README starts lying -- pin it to the JSON instead."""
    readme = open(README_PATH).read()
    documented = {m.group(1): int(m.group(2))
                  for m in re.finditer(r"^\| `(\w+)` \| (\d+) \|", readme, re.M)}
    actual = collections.Counter(p["category"] for p in prompts)
    assert documented, "README category table not found"
    assert documented == dict(actual), \
        f"README says {documented}, library has {dict(actual)}"


# --- validator drift flags ---------------------------------------------------

OK = {"ok": True}


def test_flag_fires_when_a_travel_prompt_stays_put():
    entry = {"expect_displacement": True}
    flags = validate_prompts.flags_for(entry, OK, {"displacement_m": 0.01, "path_length_m": 0.02})
    assert "expected_travel_but_stayed_put" in flags


def test_no_flag_when_a_travel_prompt_travels():
    entry = {"expect_displacement": True}
    assert validate_prompts.flags_for(entry, OK, {"displacement_m": 4.2, "path_length_m": 4.5}) == []


def test_circular_path_counts_as_travel_despite_returning_home():
    """'A person walks in a circle' ends near where it started, so start-to-end
    displacement is ~0 while the character clearly walked. Path length is what
    saves this one from a false flag."""
    entry = {"expect_displacement": True}
    flags = validate_prompts.flags_for(entry, OK, {"displacement_m": 0.2, "path_length_m": 12.0})
    assert flags == []


def test_flag_fires_when_an_in_place_prompt_wanders():
    entry = {"expect_displacement": False}
    flags = validate_prompts.flags_for(entry, OK, {"displacement_m": 3.0, "path_length_m": 3.0})
    assert "expected_in_place_but_travelled" in flags


def test_in_place_prompt_is_judged_on_net_travel_not_accumulated_jitter():
    """A standing character's root jitters every frame. Summed over a few
    hundred frames that easily clears the threshold, so judging 'in place' by
    path length flags the entire standing/gesture/idle half of the library.
    Net displacement is the only honest measure here."""
    entry = {"expect_displacement": False}
    signals = {"displacement_m": 0.03, "path_length_m": 1.4}  # wobbled, went nowhere
    assert validate_prompts.flags_for(entry, OK, signals) == []


def test_ood_prompt_that_generates_is_flagged_for_eyes():
    entry = {"expect_failure": True}
    flags = validate_prompts.flags_for(entry, OK, {"displacement_m": 5.0})
    assert "ood_generated_check_by_eye" in flags


def test_generate_failure_is_the_only_flag():
    entry = {"expect_displacement": True}
    flags = validate_prompts.flags_for(entry, {"ok": False, "error": "boom"}, {})
    assert flags == ["generate_failed"]


def test_no_flags_without_measurements():
    """The npz lives on the ARDY host and may not be readable from the machine
    running the sweep. No measurement means no claim, not a false flag."""
    entry = {"expect_displacement": True}
    assert validate_prompts.flags_for(entry, OK, {"measured": False}) == []


def test_every_library_entry_survives_the_flagger(prompts):
    """A well-behaved run -- everything generates and moves as expected -- must
    produce no flags for any in-distribution prompt."""
    for p in prompts:
        if p["category"] == "out_of_distribution":
            continue
        signals = {"displacement_m": 3.0 if p.get("expect_displacement") else 0.05,
                   "path_length_m": 3.0 if p.get("expect_displacement") else 0.05}
        assert validate_prompts.flags_for(p, OK, signals) == [], f"{p['id']} flagged on a clean run"
