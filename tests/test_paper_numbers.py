"""The paper is the receipt: every number below is printed in the paper, and each is checked against the read-out that
produces it. Run scripts/reproduce_paper_results.sh first (it writes results/), then: uv run python tests/test_paper_numbers.py
Percentages are compared at the paper's one decimal, rates at three decimals."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
J = lambda f: json.loads((R / f).read_text())
FAIL = []


def check(where, got, want, digits=1):
    if round(got, digits) != round(want, digits):
        FAIL.append(f"{where}: read-out {got:.{digits + 2}f}, paper {want}")


def test_table2_seven_suites():
    e = J("seven_suite_comparison.json")
    for k, want in (("benign", 47.0), ("util_attack", 28.8), ("asr_injected", 5.3), ("asr_direct", 26.7)):
        check(f"Table 2 CAVAL {k}", e["recovery_scorers"][k]["mean"], want)
    for k, want in (("benign", 55.0), ("util_attack", 34.2), ("injected", 38.4), ("direct", 80.6)):
        check(f"Table 2 undefended {k}", e["recovery_ref_undefended_draws"][k]["pct"], want)


def test_table2_agentdojo_and_agentdyn():
    cols = ("benign_utility", "utility_under_attack", "asr_injected", "asr_direct")
    d = J("agentdojo_defenses.json")["pooled"]
    for row, want in (("undefended (our harness)", (67.0, 34.4, 31.8, 68.6)), ("tool_filter (fork harness)", (62.9, 60.5, 6.8, 65.7)),
                      ("progent (fork harness)", (66.0, 58.4, 11.6, 74.3)), ("camel (fork harness)", (33.0, 41.9, 0.2, 65.7)),
                      ("DRIFT (our harness)", (61.9, 59.5, 2.3, 65.7)), ("CAVAL (our harness)", (60.0, 36.8, 5.2, 29.1))):
        for c, w in zip(cols, want): check(f"Table 2 AgentDojo {row} {c}", d[row][c], w)
    a = J("agentdyn_defenses.json")["pooled"]
    for row, want in (("undefended", (46.7, 35.4, 51.1, 100.0)), ("tool_filter", (6.7, 5.4, 5.9, 53.6)), ("progent", (6.7, 3.8, 10.7, 53.6)),
                      ("drift", (18.3, 19.1, 3.6, 60.7)), ("CAVAL (ours, same tasks)", (26.0, 23.8, 3.9, 23.6))):
        for c, w in zip(cols, want): check(f"Table 2 AgentDyn {row} {c}", a[row][c], w)


def test_table11_agentdyn_full():
    a = J("agentdyn_defenses.json")["pooled"]
    cols = ("benign_utility", "utility_under_attack", "asr_injected")
    for row, want in (("repeat_user_prompt", (50.0, 38.4, 34.8)), ("spotlighting_with_delimiting", (36.7, 35.2, 48.2)),
                      ("transformers_pi_detector", (1.7, 0.9, 1.4)), ("piguard_detector", (16.7, 3.2, 1.4)),
                      ("prompt_guard_2_detector", (45.0, 19.6, 35.5)), ("undefended_camel_subset", (47.9, 41.9, 37.5)), ("camel", (0.0, 0.0, 0.0))):
        for c, w in zip(cols, want): check(f"Table 11 {row} {c}", a[row][c], w)


def test_full_coverage_and_budget_text():
    f = J("full_coverage.json")["recovery_scorers"]
    for k, m, s in (("benign", 47.0, 3.6), ("util_attack", 31.2, 1.8), ("asr_injected", 4.9, 1.0), ("asr_direct", 26.7, 6.6)):
        check(f"full coverage {k} mean", f[k]["mean"], m)
    b = J("budget_alpha025.json")["stop_scorers"]
    check("budget 0.25 benign", b["benign"]["mean"], 50.6); check("budget 0.25 direct", b["asr_direct"]["mean"], 28.6)
    check("online benign stops 0.25", J("online_stops_alpha025.json")["benign stop rate"]["candidate"], 8.4)
    check("online benign stops 0.30", J("online_stops_alpha030.json")["benign stop rate"]["candidate"], 12.9)


def test_table5_recovery():
    r = J("recovery.json")
    for m, stop, rec in (("utility under attack", 23.5, 36.1), ("injected ASR", 7.6, 9.3), ("benign utility", 58.9, 58.6), ("direct ASR", 37.1, 37.1)):
        check(f"recovery {m} stop", r[m]["baseline"], stop); check(f"recovery {m} recovery", r[m]["candidate"], rec)


def test_table7_components():
    c = J("components.json")
    for arm, b, bs, d, ds in (("rules_only", 47.4, 2.2, 65.4, 1.9), ("stop", 47.1, 3.2, 22.9, 6.3), ("recovery", 47.0, 4.5, 26.7, 5.9), ("rules_and_caval", 44.3, 2.5, 25.1, 2.7)):
        S = c[f"{arm}_scorers"]
        check(f"components {arm} benign", S["benign"]["mean"], b); check(f"components {arm} benign sd", S["benign"]["sd"], bs)
        check(f"components {arm} direct", S["asr_direct"]["mean"], d); check(f"components {arm} direct sd", S["asr_direct"]["sd"], ds)


def test_tables3_4_calibration():
    c = J("calibration_rules.json")
    want = {"rgcn": {0.05: (0.017, 0.009, 0.185, 0.042, 0.024, 0.370), 0.10: (0.054, 0.032, 0.458, 0.115, 0.061, 0.671),
                     0.20: (0.147, 0.077, 0.771, 0.300, 0.155, 0.939), 0.30: (0.254, 0.132, 0.921, 0.496, 0.266, 0.984)},
            "gatv2": {0.10: (0.047, 0.024, 0.393, 0.117, 0.061, 0.622)}, "seq_lstm": {0.10: (0.078, 0.038, 0.433, 0.146, 0.074, 0.708)},
            "mlp_pool": {0.10: (0.040, 0.024, 0.259, 0.095, 0.054, 0.506)}, "gcn": {0.10: (0.046, 0.025, 0.232, 0.091, 0.047, 0.387)}}
    keys = ("far_any", "work_any", "prev_any", "far_work", "work_work", "prev_work")
    for m, per in want.items():
        for a, vals in per.items():
            for k, w in zip(keys, vals): check(f"calibration {m} {a} {k}", c[f"{m}_{a}"][k], w, 3)


def test_table6_generalization():
    a = J("offline_architectures.json")["R-GCN"]
    for k, w in (("auc", 0.933), ("prev_010", 0.458), ("far_010", 0.054), ("prev_030", 0.921), ("far_030", 0.254)):
        check(f"in distribution {k}", a[k][0], w, 3)
    h = J("offline_heldout_objectives.json")
    for k, w in (("auc", 0.830), ("prev_010", 0.342), ("far_010", 0.087)): check(f"held-out objectives {k}", h[k][0], w, 3)
    t = J("unseen_wording.json")
    for tmpl, vals in (("ignore_previous", (0.927, 0.063, 0.010, 0.794, 0.099)), ("tool_knowledge", (0.954, 0.281, 0.003, 0.770, 0.051))):
        for k, w in zip(("auc", "prev_0.1", "far_0.1", "prev_0.3", "far_0.3"), vals): check(f"{tmpl} {k}", t[tmpl][k][0], w, 3)


def test_table8_architectures_and_ablations():
    a = J("offline_architectures.json")
    for m, auc, sd in (("R-GCN", 0.933, 0.018), ("GATv2", 0.924, 0.031), ("LSTM", 0.918, 0.030), ("MLP", 0.915, 0.026), ("GCN", 0.902, 0.033)):
        check(f"{m} AUC", a[m]["auc"][0], auc, 3); check(f"{m} AUC sd", a[m]["auc"][1], sd, 3)
    for m, p in (("GCN", 0.006), ("MLP", 0.020), ("LSTM", 0.049)): check(f"Wilcoxon R-GCN vs {m}", a[m]["wilcoxon_p_vs_rgcn"], p, 3)
    b = J("offline_ablations.json")
    for v, auc, prev in (("No content embeddings", 0.865, 0.260), ("With utility objective", 0.923, 0.427), ("No data-flow edges", 0.925, 0.410),
                         ("No call-return edges", 0.933, 0.455), ("No temporal edges", 0.927, 0.453),
                         ("Data-flow overlap >= 1", 0.929, 0.504), ("Data-flow overlap >= 3", 0.928, 0.406)):
        check(f"ablation {v} AUC", b[v]["auc"][0], auc, 3); check(f"ablation {v} prevention", b[v]["prev_010"][0], prev, 3)
    star = {v: b[v]["p_auc"] < 0.05 for v in b}
    if not (star["No content embeddings"] and star["With utility objective"] and b["No content embeddings"]["p_prev"] < 0.05):
        FAIL.append("ablation stars differ from Table 8b")


def test_figure5_adaptive():
    ad = J("verifier_directed_attack.json")
    three = lambda lbl, f: 100 * sum(ad[lbl]["per_suite"][s][f] for s in ("banking", "travel", "slack")) / sum(ad[lbl]["per_suite"][s]["n"] for s in ("banking", "travel", "slack"))
    l = J("adversary_knowledge.json")
    check("plain template CAVAL", l["no_names"]["refs"]["CAVAL deployed, same scorer, plain template"]["pct"], 5.9)
    check("verifier-directed CAVAL", three("CAVAL", "adaptive_k"), 6.4)
    check("verifier-directed undefended (plain)", three("Undefended", "plain_k"), 46.0)
    check("verifier-directed undefended", three("Undefended", "adaptive_k"), 44.0)
    for v, und, cav in (("no_names", 42.4, 6.9), ("wrong_user_name", 28.3, 4.4)):
        check(f"{v} undefended", l[v]["POOLED"]["undef_asr"]["pct"], und); check(f"{v} CAVAL", l[v]["POOLED"]["caval_asr"]["pct"], cav)
    check("rewrite recovered after five rounds", J("score_aware_attack.json")["POOLED"]["by_round"][5]["pct"], 11.7)


def test_table9_overhead():
    o = J("overhead.json")
    for name, u, a, tok, sec, e in (("Undefended", 62.5, 52.1, 7031, 4.2, 0.35), ("Tool filter", 50.0, 16.0, 2895, 3.3, 3.56),
                                    ("Progent", 43.8, 27.8, 6458, 5.3, 0.47), ("CaMeL", 31.2, 0.0, 46220, 20.4, 0.03),
                                    ("DRIFT", 50.0, 6.9, 16103, 19.8, 0.14), ("CAVAL", 46.2, 12.9, 8041, 4.7, 0.88)):
        check(f"overhead {name} utility", o[name]["benign_utility"], u); check(f"overhead {name} ASR", o[name]["asr_injected"], a)
        check(f"overhead {name} tokens", o[name]["tokens"], tok, 0); check(f"overhead {name} seconds", o[name]["seconds"], sec)
        check(f"overhead {name} efficiency", o[name]["efficiency"], e, 2)


def test_figure4_calibration_curve():
    c = J("calibration_curve.json"); far = dict(zip(c["alpha"], c["far_mean"]))
    check("offline FAR at 0.25", 100 * far[0.25], 19.8); check("offline FAR at 0.30", 100 * far[0.3], 25.4)
    if not all(f <= a for a, f in far.items()):
        FAIL.append("mean offline false-alarm rate exceeds the budget at some alpha")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print("\n".join(FAIL) if FAIL else f"OK: every checked paper number reproduced ({len(tests)} groups)")
    sys.exit(1 if FAIL else 0)
