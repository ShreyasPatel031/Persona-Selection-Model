"""Mutually exclusive gate toggles (apply_gate_toggles + client normalization)."""

from app.persona.gate_planner import apply_gate_toggles, normalize_exclusive_gates


def test_turn_on_clears_others() -> None:
    g = [True, True, True, True]
    apply_gate_toggles(g, [(4, True)])
    assert g == [False, False, False, True]


def test_turn_on_from_all_off() -> None:
    g = [False, False, False, False]
    apply_gate_toggles(g, [(2, True)])
    assert g == [False, True, False, False]


def test_switch_active_gate() -> None:
    g = [False, False, False, True]
    apply_gate_toggles(g, [(1, True)])
    assert g == [True, False, False, False]


def test_turn_off_only() -> None:
    g = [False, True, False, False]
    apply_gate_toggles(g, [(2, False)])
    assert g == [False, False, False, False]


def test_multiple_toggles_in_order() -> None:
    g = [False, False, False, False]
    apply_gate_toggles(g, [(1, True), (3, True)])
    assert g == [False, False, True, False]


def test_normalize_client_multiple_true() -> None:
    assert normalize_exclusive_gates([True, True, False, False]) == [True, False, False, False]
    assert normalize_exclusive_gates([False, False, False, True]) == [False, False, False, True]
    assert normalize_exclusive_gates([False, False, False, False]) == [False, False, False, False]
