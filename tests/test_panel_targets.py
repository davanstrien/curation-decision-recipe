import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("pt", Path(__file__).resolve().parents[1] / "data/panel_targets.py")
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)

KEYS = [f"level_{i}" for i in range(5)]


def _votes(per_rater):
    """{rater: [level per item]} -> {item: {rater: {"q": level}}}"""
    out = {}
    for m, levels in per_rater.items():
        for i, x in enumerate(levels):
            out.setdefault(i, {})[m] = {"q": x}
    return out


def test_identical_raters_map_to_identity():
    v = _votes({"a": [0, 1, 2, 3, 4, 2], "b": [0, 1, 2, 3, 4, 2]})
    maps = pt.rank_maps(v, "q", 5)
    for m in ("a", "b"):
        assert all(maps[m][k] == {k: 1.0} for k in range(5))


def test_shared_level_stays_mostly_shared():
    # rater a piles mass on 3, rater b on 2/3: when both say 3, the target keeps most mass on 3 (the midpoint
    # snap of the first version sent a's 3 to 4 and b's 3 to 2, leaving no mass on 3)
    v = _votes({"a": [3] * 8 + [4] * 2, "b": [2] * 4 + [3] * 4 + [4] * 2})
    maps = pt.rank_maps(v, "q", 5)
    tgt, n = pt.target_for("scale", "q", KEYS, {"a": {"q": 3}, "b": {"q": 3}}, maps, 2)
    assert n == 2 and abs(sum(tgt.values()) - 1) < 1e-3
    assert tgt["level_3"] == max(tgt.values())


def test_multi_majority_and_min_raters():
    item = {"a": {"d": ["missing", "ambiguous"]}, "b": {"d": ["missing"]}, "c": {"d": []}}
    tgt, n = pt.target_for("multi", "d", ["missing", "ambiguous", "figure"], item, None, 2)
    assert n == 3 and tgt == {"missing": 1.0, "ambiguous": 0.0, "figure": 0.0}
    tag, _ = pt.target_for("multi", "d", None, item, None, 2, tag="ambiguous")
    assert tag == {"false": 1.0}
    none, n1 = pt.target_for("choice", "x", ["A", "B"], {"a": {"x": "A"}, "b": {"x": None}}, None, 2)
    assert none is None and n1 == 1


def test_choice_vote_share():
    tgt, _ = pt.target_for("choice", "x", ["A", "B", "C"], {"a": {"x": "A"}, "b": {"x": "A"}, "c": {"x": "B"}},
                           None, 2)
    assert tgt == {"A": 0.6667, "B": 0.3333}


def test_scale_hard_median():
    item = {"a": {"q": 1}, "b": {"q": 3}, "c": {"q": 4}}
    tgt, n = pt.target_for("scale", "q", KEYS, item, None, 2, scale_mode="hard")
    assert n == 3 and tgt == {"level_3": 1.0}
    tgt2, _ = pt.target_for("scale", "q", KEYS, {"a": {"q": 1}, "b": {"q": 2}}, None, 2, scale_mode="hard")
    assert tgt2 == {"level_1": 1.0}
