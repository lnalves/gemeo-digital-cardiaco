"""Quadro 5 of the TC: full truth table of the decision rules."""

import itertools

import pytest

from cardiac_twin.config import load_config
from cardiac_twin.decision.rules import (
    ALERT,
    ATTENTION,
    CRITICAL,
    NORMAL,
    VRunCounter,
    decide_beat,
    decide_window,
)

VT = load_config().decision.vt_consecutive_v
L = 1.0
KL = {"above": 2.0, "below": 0.5, "equal": L, "none": None}


def expected(cls, v_run, kl_case, use_profile, calibrating):
    if cls == "V" and v_run >= VT:
        return CRITICAL
    if cls == "N" or calibrating:
        return NORMAL
    if not use_profile:
        return ALERT
    return ALERT if kl_case == "above" else ATTENTION


CASES = list(itertools.product("NSVFQ", [0, 1, VT - 1, VT, VT + 4], KL, [True, False], [True, False]))


@pytest.mark.parametrize("cls, v_run, kl_case, use_profile, calibrating", CASES)
def test_truth_table(cls, v_run, kl_case, use_profile, calibrating):
    kl = KL[kl_case]
    threshold = None if kl is None else L
    got = decide_beat(
        cls, v_run, kl, threshold, use_profile=use_profile, calibrating=calibrating, vt_consecutive_v=VT
    )
    assert got == expected(cls, v_run, kl_case, use_profile, calibrating)


@pytest.mark.parametrize("calibrating", [True, False])
@pytest.mark.parametrize("use_profile", [True, False])
def test_three_v_in_a_row_is_critical_even_below_threshold(calibrating, use_profile):
    counter = VRunCounter()
    levels = []
    for cls in ["N", "V", "V", "V", "V", "N", "V"]:
        run = counter.update(cls)
        levels.append(
            decide_beat(
                cls, run, 0.1, L, use_profile=use_profile, calibrating=calibrating, vt_consecutive_v=VT
            )
        )
    assert levels[3] == levels[4] == CRITICAL
    assert CRITICAL not in levels[:3] and levels[6] != CRITICAL


def test_configuration_a_ignores_kl():
    for kl in (0.0, 0.5, 2.0, None):
        assert decide_beat("S", 0, kl, L, use_profile=False, calibrating=False, vt_consecutive_v=VT) == ALERT


def test_v_run_counter_resets_on_any_other_class():
    c = VRunCounter()
    assert [c.update(x) for x in "VVQVVVN"] == [1, 2, 0, 1, 2, 3, 0]


@pytest.mark.parametrize(
    "has_non_n, kl, level",
    [
        (False, 2.0, ATTENTION),
        (False, 0.5, NORMAL),
        (False, L, NORMAL),
        (True, 2.0, NORMAL),
        (True, 0.5, NORMAL),
    ],
)
def test_window_rule(has_non_n, kl, level):
    assert decide_window(has_non_n, kl, L) == level
