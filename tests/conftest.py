# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Elyan Labs LLC
"""Test fixtures.

The director service imports ARDY, which only exists inside the GPU host's
`ardy` venv. Stub those imports so the service's own staging/camera/HTTP logic
can be tested anywhere. Nothing here fakes ARDY behaviour -- the stubs exist
purely to satisfy import time; every test monkeypatches the model access it
needs. `director_service.staging` and `director_service.camera` import no ARDY
at all, so their geometry is tested against the real code.
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _unavailable(*_a, **_k):  # pragma: no cover - must never run in tests
    raise AssertionError("real ARDY must not be called from tests")


class _RecordingRoot2DConstraintSet:
    """Stands in for ardy.constraints.Root2DConstraintSet.

    Mirrors the real constructor signature (skeleton, frame_indices, root_2d,
    global_root_heading=None, to_crop=False) and records what it was handed, so
    tests can assert the service builds the constraint ARDY expects.
    """

    name = "root2d"

    def __init__(self, skeleton, frame_indices, root_2d, global_root_heading=None, to_crop=False):
        assert len(root_2d) == len(frame_indices), "real ARDY asserts this too"
        self.skeleton = skeleton
        self.frame_indices = frame_indices
        self.root_2d = root_2d
        self.global_root_heading = global_root_heading
        self.to_crop = to_crop


_stub("ardy")
_stub("ardy.model")
_stub("ardy.model.load_model", load_model=_unavailable, load_text_encoder=_unavailable)
_stub("ardy.model.registry", resolve_model_name=lambda n: n)
_stub("ardy.postprocess", post_process_motion=_unavailable)
_stub("ardy.constraints", Root2DConstraintSet=_RecordingRoot2DConstraintSet)
