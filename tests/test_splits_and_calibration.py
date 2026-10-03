"""Tests for the split + conformal machinery every paper number flows through:
grouped splits (no group straddles train/test) and the max-over-steps conformal
eval (hand-computed qhat/prevention/FAR/delay).

Runnable two ways:
    pytest tests/test_experiments.py
    python tests/test_experiments.py         # no pytest needed
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiments.train_scorer import within_split, conformal_report, eval_traces
from caval.scorer import grouped_four_way_split
from analysis.paper_tables import _running_stat, _trace_metrics


def test_within_split_grouped_and_disjoint():
    groups = np.array([f"g{i // 4}" for i in range(200)])   # 50 groups of 4
    tr, va, ca, te = within_split(groups, seed=1)
    parts = [tr, va, ca, te]
    assert sum(len(p) for p in parts) == 200
    assert len(np.unique(np.concatenate(parts))) == 200      # disjoint, complete
    for a in range(4):                                       # no group straddles
        for b in range(a + 1, 4):
            assert not set(groups[parts[a]]) & set(groups[parts[b]])


def test_grouped_four_way_holdout_forced_to_test():
    labels = np.array([0, 1] * 50)
    groups = np.array([f"g{i // 2}" for i in range(100)])
    hold = np.zeros(100, dtype=bool)
    hold[:10] = True
    tr, va, ca, te = grouped_four_way_split(labels, groups, seed=3, holdout_mask=hold)
    assert set(range(10)) <= set(te)                         # forced rows in test
    assert not set(range(10)) & (set(tr) | set(va) | set(ca))


def _traces(entries):
    """entries: {trace: [(t, score)], meta (y, t_star)} via conformal_report rows."""
    rows, scores = [], []
    for tr, (steps, y, t_star) in entries.items():
        for t, s in steps:
            rows.append(dict(trace=tr, t=t, y_trace=y, t_star=t_star))
            scores.append(s)
    return conformal_report(rows, scores, None)


def test_eval_traces_hand_computed():
    # cal: 4 benign traces, max scores .1 .2 .3 .4; alpha=.2, n=4 ->
    # k=ceil(5*.8)=4 -> qhat = .4 (largest benign max)
    cal = _traces({i: ([(0, m)], 0, None) for i, m in enumerate([.1, .2, .3, .4])})
    test = _traces({
        10: ([(0, .1), (1, .5)], 1, 1),   # alarm t=1 == t* -> prevented
        11: ([(0, .1), (1, .5)], 1, 0),   # alarm t=1 > t*=0 -> late, delay 1
        12: ([(0, .1), (1, .2)], 1, 0),   # never crosses -> missed
        13: ([(0, .9)], 0, None),         # benign, alarms -> false alarm
        14: ([(0, .1)], 0, None),         # benign, quiet
    })
    r = eval_traces(cal, test, alpha=0.2)
    assert r["qhat"] == .4 and r["n_cal_benign"] == 4
    assert r["prevention"] == 1 / 3 and r["missed"] == 1
    assert r["per_trace_far"] == 0.5 and r["mean_delay"] == 1.0


def test_eval_traces_infinite_qhat_when_cal_too_small():
    cal = _traces({0: ([(0, .5)], 0, None)})                 # n=1, alpha=.05 -> k=2>n
    test = _traces({1: ([(0, .99)], 1, 0)})
    r = eval_traces(cal, test, alpha=0.05)
    assert r["qhat"] is None and r["prevention"] == 0.0      # inf threshold: no alarms


# ---- running maximum and trace-level metrics (tables.py) ----
def test_running_max_and_trace_metrics():
    assert _running_stat([.2, .8, .4, .6], "max") == [.2, .8, .8, .8]
    te = {0: [(0, .9, 0, None, "s")], 1: [(0, .1, 0, None, "s")],
          2: [(0, .3, 1, 0, "s")], 3: [(0, .7, 1, 0, "s")]}
    m = _trace_metrics(te, .5)          # one of two benign traces alarms, one of two compromised traces is caught in time
    assert m["far"] == 0.5 and m["prev"] == 0.5 and m["auc"] == 0.5


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"OK: {len(fns)} tests passed")
