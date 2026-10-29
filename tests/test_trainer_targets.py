"""CPU checks of the trainer's target routing and conditional RPS.
run all tests: uv run --python 3.12 --with torch,numpy,pytest,openai,huggingface_hub,polars,pyyaml pytest -q tests/"""

import importlib.util
import sys
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("tr", Path(__file__).parents[1] / "train/gliclass_ub_train.py")
tr = importlib.util.module_from_spec(spec)
sys.modules["tr"] = tr
spec.loader.exec_module(tr)


def row(labels, targets, **kw):
    return {"text": "x", "all_labels": labels, "true_labels": dict(zip(labels, targets)), **kw}


def test_multi_label_one_positive_not_exclusive():
    rows = [row(["a", "b", "c"], [0, 1, 0], multi_label=True), row(["a", "b", "c"], [0, 1, 0])]
    tr.mark_targets(rows)
    assert rows[0]["exclusive"] == 0
    assert rows[1]["exclusive"] == 1


def test_decision_multi_label_zero_vector_kept_and_flagged():
    item = {"id": "x", "state": "t", "question": {"type": "choice", "instructions": "i",
            "criteria": {"a": "A", "b": "B"}}, "target": {"a": 0.0, "b": 0.0}, "multi_label": True}
    g, why = tr.decision_to_gliclass(item, "engine")
    assert why is None and g["multi_label"] and list(g["true_labels"].values()) == [0.0, 0.0]


def test_ordinal_guards():
    ok = row(["l0", "l1", "l2"], [0, 1, 0], ordinal_order=["l0", "l1", "l2"])
    missing = row(["l0", "l2"], [0, 1], ordinal_order=["l0", "l1", "l2"])
    multi = row(["l0", "l1", "l2"], [1, 1, 0], ordinal_order=["l0", "l1", "l2"])
    gold_abstain = row(["l0", "l1", "Indeterminate"], [0, 0, 1], ordinal_order=["l0", "l1"])
    st = tr.mark_targets([ok, missing, multi, gold_abstain])
    assert st["ordinal"] == 1 and st["ordinal_level_missing"] == 1
    assert st["ordinal_not_exclusive"] == 1 and st["ordinal_gold_not_level"] == 1
    assert "ordinal_order" in ok and "ordinal_order" not in missing


def test_conditional_rps():
    # ranks [0, 1, -1], p [.25, .25, .5], gold [0, 1, 0]: conditional p over levels [.5, .5] -> RPS 0.25
    probs = torch.tensor([[0.25, 0.25, 0.5]])
    target = torch.tensor([[0.0, 1.0, 0.0]])
    ranks = torch.tensor([[0, 1, -1]])
    assert abs(tr.rps_per_row(probs.log(), target, ranks).item() - 0.25) < 1e-6
    # gold on the non-level slot -> 0
    assert tr.rps_per_row(probs.log(), torch.tensor([[0.0, 0.0, 1.0]]), ranks).item() == 0.0
    # shuffled slots: same value as ordered
    p2 = torch.tensor([[0.5, 0.25, 0.25]])
    t2 = torch.tensor([[0.0, 1.0, 0.0]])
    r2 = torch.tensor([[-1, 1, 0]])
    assert abs(tr.rps_per_row(p2.log(), t2, r2).item() - 0.25) < 1e-6
    # mixed batch with a padded / non-ordinal row
    pb = torch.tensor([[0.25, 0.25, 0.5], [0.3, 0.7, 1e-9]])
    tb = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    rb = torch.tensor([[0, 1, -1], [-1, -1, -1]])
    out = tr.rps_per_row(pb.log(), tb, rb)
    assert abs(out[0].item() - 0.25) < 1e-6 and out[1].item() == 0.0


def test_level_keys_give_ordinal_order_any_wording():
    item = {"id": "x", "state": "t", "question": {"type": "choice", "instructions": "i",
            "criteria": {"level_0": "Primary School level", "level_2": "University", "level_1": "High school"}},
            "target": {"level_1": 1.0}}
    g, _ = tr.decision_to_gliclass(item, "engine")
    assert g["ordinal_order"] == ["Primary School level", "High school", "University"]
    plain = {"id": "y", "state": "t", "question": {"type": "choice", "instructions": "i",
             "criteria": {"a": "A", "b": "B"}}, "target": {"a": 1.0}}
    assert "ordinal_order" not in tr.decision_to_gliclass(plain, "engine")[0]


def test_soft_single_label_targets_renormalised_and_exclusive():
    item = {"id": "w", "state": "t", "question": {"type": "choice", "instructions": "i",
            "criteria": {"a": "A", "b": "B", "c": "C"}}, "target": {"a": 0.7, "b": 0.2, "c": 0.0985}}
    g, _ = tr.decision_to_gliclass(item, "engine")
    st = tr.mark_targets([g])
    assert g["exclusive"] == 1 and st["single_label_renormalised"] == 1
    assert abs(sum(g["true_labels"].values()) - 1.0) < 1e-9


def test_conditional_rps_stable_under_dominant_nonlevel_logit():
    logits = torch.tensor([[0.0, 0.0, 30.0]])
    out = tr.rps_per_row(logits, torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([[0, 1, -1]]))
    assert abs(out.item() - 0.25) < 1e-6


def test_conversion_keeps_provenance():
    item = {"id": "p", "src": "nvidia/HelpSteer3", "teacher": "human", "licence_class": "sa", "revision": "abc",
            "state": "t", "question": {"type": "choice", "instructions": "i", "criteria": {"a": "A", "b": "B"}},
            "target": {"a": 1.0}}
    g, _ = tr.decision_to_gliclass(item, "engine")
    assert (g["src"], g["teacher"], g["licence_class"], g["revision"]) == ("nvidia/HelpSteer3", "human", "sa", "abc")
