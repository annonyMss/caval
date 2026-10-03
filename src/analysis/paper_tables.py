"""Read-outs behind every table and figure of the CAVAL paper.

Each subcommand reads the episode logs (data/runs, data/benchmarks/AgentDyn/runs) or the offline score files
(results/offline) and writes a JSON file with every value and a CSV with the table as printed, both to results/.
Usage: uv run python src/analysis/paper_tables.py <subcommand> [args]; scripts/reproduce_paper_results.sh runs them all.

Episode conventions used throughout: an injected episode has an injection_task_id; a direct episode has none and a
user task that is an injection task (its attack success is `utility`, because `security` is fixed to True there);
a benign episode has none and a user task that is a user task. Comparisons are on matched (suite, user task,
injection task) keys.
"""
import csv
import json
import math
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from caval.trace_graph import build_tool_vocab_matrix
from caval.scorer import build_model, load_state_compat
from experiments.train_scorer import build_prefix_data

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


OUT_NAME = None     # set by --out <name>; reproduce_paper_results.sh names every result after its paper element


def _write(name, obj, header=None, rows=None):
    """results/<name>.json with every value, and results/<name>.csv with the table as printed (name = --out if given)."""
    name = OUT_NAME or name
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    if header is not None:
        with open(RESULTS / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"-> results/{name}.json" + (f", results/{name}.csv" if header is not None else ""))


def _wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def _episodes(*dirs, suites=None):
    """Every episode JSON under the given run directories, keyed by
    (suite, user_task_id, injection_task_id). Later directories win on duplicate keys."""
    out = {}
    for rd in dirs:
        for f in Path(rd).rglob("*.json"):
            try: d = json.loads(f.read_text())
            except Exception: continue
            if "user_task_id" not in d: continue
            suite = d.get("suite_name") or f.parts[-4]
            if suites and suite not in suites: continue
            out[(suite, d["user_task_id"], d.get("injection_task_id"))] = d
    return out


def _stopped(d):
    return any(e.get("enforced") == "block" for e in d.get("verifier_log") or [])


def _e0_traces(rows, idx, p):
    """E0_scores rows -> {trace: sorted [(t, score, y_trace, t_star)]}."""
    d = defaultdict(list)
    for j, sc in zip(idx, p):
        r = rows[j]
        d[r["trace"]].append((r["t"], float(sc), r["y_trace"], r["t_star"]))
    return {tr: sorted(v) for tr, v in d.items()}


def _lam_any(cal, alpha):
    m = sorted(max(x[1] for x in v) for v in cal.values() if v[0][2] == 0)
    k = math.ceil((len(m) + 1) * (1 - alpha))
    return m[k - 1] if k <= len(m) else float("inf")


def cmd_online_comparison():
    """Online comparison on matched episodes (Table 2 seven-suite block, Table 7, the budget and full-coverage text).
    --arms <name>=<run prefix>,... reads each arm from <prefix>_s<k> (injected), <prefix>_s<k>_direct and <prefix>_s<k>_benign,
    k = 1..5. --undefended <prefix> reads our undefended draws <prefix>_r<k>_{benign,direct,injected}; the AgentDyn injected
    episodes come from the runs released with the benchmark. --shared restricts every arm to the episodes DRIFT covers.
    Per scorer, mean +- sd over scorers, and pooled (Wilson 95%); the first two arms are also compared on paired episodes."""
    DOJO = Path("data/runs/gpt-4o-mini-2024-07-18"); ATK, BEN = DOJO / "attack_important_instructions", DOJO / "no_attack"
    DYN = Path("data/benchmarks/AgentDyn/runs")
    custom = sys.argv[sys.argv.index("--arms") + 1] if "--arms" in sys.argv else None
    undef = sys.argv[sys.argv.index("--undefended") + 1] if "--undefended" in sys.argv else None
    shared = "--shared" in sys.argv          # restrict every arm and reference to the episodes DRIFT covers (the matched headline)
    slug = ("" if custom is None else "_" + custom.replace("/", "_").replace("=", "-").replace(",", "+")) + ("_undef" if "--undefended" in sys.argv else "") + ("_shared" if "--shared" in sys.argv else "")
    dojo, adyn = ("banking", "slack", "travel", "workspace"), ("dailylife", "github", "shopping")
    refs = {"undefended": {"injected": [ATK / "undefended_injected", ATK / "undefended_injected_slack"] + [DYN / "gpt-4o-mini-2024-07-18" / s for s in adyn],
                           "direct": [DYN / "gpt-4o-mini-2024-07-18" / s for s in adyn],
                           "benign": [BEN / "undefended_benign", BEN / "undefended_benign_slack"] + [DYN / "gpt-4o-mini-2024-07-18" / s for s in adyn]},
            # later directories win on duplicate keys: the paper's runs (dojo/slack, our adyn benign) come last
            "DRIFT": {"injected": [ATK / "drift_injected", ATK / "drift_injected_banking_travel", ATK / "drift_injected_slack"] + [DYN / "gpt-4o-mini-2024-07-18-drift" / s for s in adyn],
                      "direct": [BEN / "drift_direct"] + [DYN / "gpt-4o-mini-2024-07-18-drift" / s for s in adyn],
                      "benign": [DYN / "gpt-4o-mini-2024-07-18-drift" / s for s in adyn] + [BEN / "drift_benign", BEN / "drift_benign_slack", BEN / "drift_benign_agentdyn"]}}
    assert custom, "give --arms <name>=<run prefix>,..."
    arms = {}
    for item in custom.split(","):
        name, prefix = item.split("=")
        arms[name] = (prefix + "_scorer{k}", prefix + "_scorer{k}_direct", prefix + "_scorer{k}_benign")
    base_arm, cand_arm = list(arms)[0], (list(arms)[1] if len(arms) > 1 else None)
    kinds = {"injected": lambda d: bool(d.get("injection_task_id")),
             "direct": lambda d: d.get("injection_task_id") is None and str(d["user_task_id"]).startswith("injection_task"),
             "benign": lambda d: d.get("injection_task_id") is None and str(d["user_task_id"]).startswith("user_task")}

    def load(dirs, kind):
        eps = _episodes(*dirs)
        return {k: d for k, d in eps.items() if kinds[kind](d)}

    def rej_stats(d):
        """(rejections in the episode, episode ended by the rejection cap). Under fail-stop a block is one rejection that ends the episode."""
        log = d.get("verifier_log") or []
        rej = sum(1 for e in log if e.get("rejected")) if d.get("enforce") in ("reject", "closed", "isolate", "redact") else sum(1 for e in log if e.get("enforced") == "block")
        cap = d.get("reject_cap") or 1
        return rej, rej >= cap

    data = {}
    for arm, (inj, dr, bn) in arms.items():
        for k in range(1, 6):
            per = {"injected": load([ATK / inj.format(k=k)], "injected"), "direct": load([BEN / dr.format(k=k)], "direct"),
                   "benign": load([BEN / bn.format(k=k)], "benign")}
            if any(per.values()):
                data[(arm, k)] = per
    if not data:
        print("no arm episodes found"); return
    ref = {name: {kind: load(dirs, kind) for kind, dirs in kinds_dirs.items()} for name, kinds_dirs in refs.items()}
    if shared:
        for (arm, k), per in data.items():
            for kind in kinds:
                per[kind] = {kk_: d for kk_, d in per[kind].items() if kk_ in ref["DRIFT"][kind]}
        print("shared set: every row restricted to the episodes DRIFT covers")
    draws = {}          # undefended draws: kind -> [dict per draw]; each draw is matched to the arm's pairs on its own
    if undef:
        for kind, sub in (("benign", "_benign"), ("direct", "_direct"), ("injected", "_injected")):
            base_dir = ATK if kind == "injected" else BEN
            ds = [load([base_dir / f"{undef}_draw{k}{sub}"], kind) for k in range(1, 6)]
            ds = [d for d in ds if d]
            if ds: draws[kind] = ds
        print(f"undefended draws found: { {k: len(v) for k, v in draws.items()} } (AgentDyn injected keeps the authors' run)")

    def rate(eps, keys, field):
        vals = [eps[k].get(field) is True for k in keys]; n = len(vals); kk = sum(vals); lo, hi = _wilson(kk, n)
        return dict(k=kk, n=n, pct=100 * kk / n if n else float("nan"), lo=lo, hi=hi)

    fmt = lambda r: f"{r['pct']:5.1f} [{r['lo']:4.1f},{r['hi']:4.1f}] n={r['n']:<4d}"
    out = {}
    # 1. each arm on the pairs it has, per seed and pooled, with references on the same pairs
    for arm in arms:
        seeds = [k for (a, k) in data if a == arm]
        if not seeds: continue
        print(f"\n=== arm {arm} (seeds {seeds}) ===")
        print(f"{'seed':>4s} | {'benign utility':>26s} | {'util under attack':>26s} | {'injected ASR':>26s} | {'direct ASR':>26s} | rej/ep  at-cap%  benign>=1rej%")
        pool = {kind: {} for kind in kinds}; pool_meta = []
        for k in seeds:
            per = data[(arm, k)]; row = {}
            keys_i, keys_d, keys_b = sorted(per["injected"]), sorted(per["direct"]), sorted(per["benign"])
            row["benign"] = rate(per["benign"], keys_b, "utility"); row["util_attack"] = rate(per["injected"], keys_i, "utility")
            row["asr_injected"] = rate(per["injected"], keys_i, "security"); row["asr_direct"] = rate(per["direct"], keys_d, "utility")
            allep = list(per["injected"].values()) + list(per["direct"].values()) + list(per["benign"].values())
            rs = [rej_stats(d) for d in allep]
            row["rejections_per_episode"] = sum(r for r, _ in rs) / max(len(rs), 1)
            row["at_cap_pct"] = 100 * sum(1 for _, ab in rs if ab) / max(len(rs), 1)
            brs = [rej_stats(d)[0] for d in per["benign"].values()]
            row["benign_ge1_rejection_pct"] = 100 * sum(1 for r in brs if r >= 1) / max(len(brs), 1)
            print(f"{k:4d} | {fmt(row['benign'])} | {fmt(row['util_attack'])} | {fmt(row['asr_injected'])} | {fmt(row['asr_direct'])} | {row['rejections_per_episode']:5.2f}  {row['at_cap_pct']:5.1f}   {row['benign_ge1_rejection_pct']:5.1f}")
            out[f"{arm}_scorer{k}"] = row
            for kind in kinds:
                for kk_, d in per[kind].items(): pool[kind][(k,) + kk_] = d
        # spread over scorers: mean +- sd of the per-scorer rates (the honest error bar, scorers are trainings not tasks)
        vals = {m: [out[f"{arm}_scorer{k}"][m]["pct"] for k in seeds if out[f"{arm}_scorer{k}"][m]["n"]] for m in ("benign", "util_attack", "asr_injected", "asr_direct")}
        sd = {m: ((statistics.mean(v), statistics.pstdev(v)) if v else (float("nan"), float("nan"))) for m, v in vals.items()}
        print(f"{'mean':>4s} | " + " | ".join(f"{sd[m][0]:5.1f} +- {sd[m][1]:4.1f}{'':16s}" for m in ("benign", "util_attack", "asr_injected", "asr_direct")) + " |  (over scorers)")
        out[f"{arm}_scorers"] = {m: dict(mean=sd[m][0], sd=sd[m][1]) for m in sd}
        P = {kind: rate(pool[kind], sorted(pool[kind]), "utility" if kind != "injected" else "security") for kind in kinds}
        P["util_attack"] = rate(pool["injected"], sorted(pool["injected"]), "utility")
        print(f"{'POOL':>4s} | {fmt(P['benign'])} | {fmt(P['util_attack'])} | {fmt(P['injected'])} | {fmt(P['direct'])} |")
        out[f"{arm}_pooled"] = dict(benign=P["benign"], util_attack=P["util_attack"], asr_injected=P["injected"], asr_direct=P["direct"])
        # references on the union of this arm's pairs (seed-1 keys)
        k1 = seeds[0]
        for name in refs:
            if name == "undefended" and draws: continue      # our own undefended draws replace the released runs below
            r = {}
            for kind, field in (("benign", "utility"), ("direct", "utility"), ("injected", "security")):
                keys = [kk_ for kk_ in data[(arm, k1)][kind] if kk_ in ref[name][kind]]
                r[kind] = rate(ref[name][kind], keys, field)
            r["util_attack"] = rate(ref[name]["injected"], [kk_ for kk_ in data[(arm, k1)]["injected"] if kk_ in ref[name]["injected"]], "utility")
            print(f"{name:>4s} | {fmt(r['benign'])} | {fmt(r['util_attack'])} | {fmt(r['injected'])} | {fmt(r['direct'])} |  (same pairs as seed {k1})")
            out[f"{arm}_ref_{name}"] = r
        if draws:
            def pooled(kind, field):
                dds = draws.get(kind)
                if not dds:                                           # no own draw for this surface: the stored reference
                    keys = [kk_ for kk_ in data[(arm, k1)][kind] if kk_ in ref["undefended"][kind]]
                    return rate(ref["undefended"][kind], keys, field)
                merged = {}
                for i, dd in enumerate(dds):
                    for kk_ in data[(arm, k1)][kind]:
                        if kk_ in dd: merged[(i,) + kk_] = dd[kk_]
                    if kind == "injected":                            # AgentDyn suites: the authors' undefended run, once
                        for kk_ in data[(arm, k1)][kind]:
                            if kk_ not in dd and kk_ in ref["undefended"][kind] and i == 0: merged[(i,) + kk_] = ref["undefended"][kind][kk_]
                return rate(merged, sorted(merged), field)
            r = {kind: pooled(kind, field) for kind, field in (("benign", "utility"), ("direct", "utility"), ("injected", "security"))}
            r["util_attack"] = pooled("injected", "utility")
            print(f"{'undefended, own draws':>4s} | {fmt(r['benign'])} | {fmt(r['util_attack'])} | {fmt(r['injected'])} | {fmt(r['direct'])} |")
            out[f"{arm}_ref_undefended_draws"] = r
    # 2. paired comparison stop vs reject on the intersection of pairs, same seeds
    common = [k for k in range(1, 6) if cand_arm and (base_arm, k) in data and (cand_arm, k) in data]
    if common:
        print(f"\n=== PAIRED {base_arm} vs {cand_arm}, seeds {common}, intersection of pairs per surface ===")
        cmp = {}
        for kind, field in (("injected", "security"), ("injected", "utility"), ("benign", "utility"), ("direct", "utility")):
            label = "util_attack" if (kind, field) == ("injected", "utility") else ("asr_injected" if kind == "injected" else kind)
            ps, pr = {}, {}
            for k in common:
                keys = set(data[(base_arm, k)][kind]) & set(data[(cand_arm, k)][kind])
                for kk_ in keys:
                    ps[(k,) + kk_] = data[(base_arm, k)][kind][kk_]; pr[(k,) + kk_] = data[(cand_arm, k)][kind][kk_]
            rs, rr = rate(ps, sorted(ps), field), rate(pr, sorted(pr), field)
            overlap = not (rs["hi"] < rr["lo"] or rr["hi"] < rs["lo"])
            cmp[label] = {base_arm: rs, cand_arm: rr, "overlap": overlap, "delta": rr["pct"] - rs["pct"]}
            print(f"{label:13s} {base_arm} {fmt(rs)}  {cand_arm} {fmt(rr)}  delta {rr['pct']-rs['pct']:+5.1f}  intervals overlap: {overlap}")
        out["paired"] = cmp
    rows = []
    for arm in arms:
        if f"{arm}_scorers" in out:
            S = out[f"{arm}_scorers"]
            rows.append([arm] + [f"{S[m]['mean']:.1f} +- {S[m]['sd']:.1f}" for m in ("benign", "util_attack", "asr_injected", "asr_direct")])
        if f"{arm}_ref_undefended_draws" in out:
            U = out[f"{arm}_ref_undefended_draws"]
            if not any(r[0] == "undefended" for r in rows):
                rows.append(["undefended"] + [f"{U[m]['pct']:.1f}" for m in ("benign", "util_attack", "injected", "direct")])
    _write("enforce" + slug, out, ["arm", "benign utility", "utility under attack", "ASR injected", "ASR direct"], rows)


def cmd_five_scorer_summary():
    """Five-scorer read-out of one configuration, pooled over scorers and suites (Wilson 95%) and as the per-scorer mean.
    Usage: paper_tables.py five_scorer_summary <tag> <suite|comma list|all> [<baseline tag>]. Runs are <tag>_scorer<k>{,_benign,_direct};
    with a baseline tag both are read on the same episodes and compared (sign test on paired episodes)."""
    from scipy.stats import binomtest
    tag = sys.argv[2]; suite = sys.argv[3] if len(sys.argv) > 3 else "all"
    btag = sys.argv[4] if len(sys.argv) > 4 else None     # optional: another configuration as the baseline (same scorers)
    DOJO = Path("data/runs/gpt-4o-mini-2024-07-18"); ATK, BEN = DOJO / "attack_important_instructions", DOJO / "no_attack"
    ALL = ("banking", "slack", "travel", "workspace", "dailylife", "github", "shopping")
    suites = ALL if suite == "all" else tuple(suite.split(","))     # one suite, a comma list (the panel), or all
    is_dir = lambda d: d.get("injection_task_id") is None and str(d["user_task_id"]).startswith("injection_task")
    is_ben = lambda d: d.get("injection_task_id") is None and str(d["user_task_id"]).startswith("user_task")
    is_inj = lambda d: bool(d.get("injection_task_id"))
    load = lambda root, pred: {k: d for k, d in _episodes(root, suites=suites).items() if pred(d)}
    metrics = [("benign utility", "benign", lambda d: d.get("utility") is True), ("benign stop rate", "benign", _stopped),
               ("direct ASR", "direct", lambda d: d.get("utility") is True), ("injected ASR", "injected", lambda d: d.get("security") is True),
               ("utility under attack", "injected", lambda d: d.get("utility") is True)]
    pooled = {m: [] for m, _, _ in metrics}; per_seed = {m: [] for m, _, _ in metrics}
    for k in range(1, 6):
        cand = {"benign": load(BEN / f"{tag}_scorer{k}_benign", is_ben), "direct": load(BEN / f"{tag}_scorer{k}_direct", is_dir),
                "injected": load(ATK / f"{tag}_scorer{k}", is_inj)}
        base = ({"benign": load(BEN / f"{btag}_scorer{k}_benign", is_ben), "direct": load(BEN / f"{btag}_scorer{k}_direct", is_dir), "injected": load(ATK / f"{btag}_scorer{k}", is_inj)}
                if btag else cand)
        for m, kind, f in metrics:
            keys = sorted(set(cand[kind]) & set(base[kind]))
            if not keys: continue
            c = [f(cand[kind][x]) for x in keys]; b = [f(base[kind][x]) for x in keys]
            pooled[m] += list(zip(c, b)); per_seed[m].append((100 * sum(c) / len(c), 100 * sum(b) / len(b), len(keys)))
    print(f"tag {tag} ({suite}): candidate vs baseline {btag or 'none'}, seeds with data, matched pairs")
    print(f"{'metric':22s} {'pairs':>6s} {'baseline':>20s} {'candidate':>20s} {'delta':>7s} {'sign p':>7s} | per-seed mean±sd base -> cand")
    out = {}
    for m, _, _ in metrics:
        pr = pooled[m]; n = len(pr)
        if not n: continue
        kc, kb = sum(c for c, _ in pr), sum(b for _, b in pr)
        lc, hc = _wilson(kc, n); lb, hb = _wilson(kb, n)
        up = sum(1 for c, b in pr if c and not b); dn = sum(1 for c, b in pr if b and not c)
        pv = binomtest(up, up + dn).pvalue if up + dn else float("nan")
        cs = [x[0] for x in per_seed[m]]; bs = [x[1] for x in per_seed[m]]
        sd = lambda v: statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"{m:22s} {n:6d} {100*kb/n:6.1f} [{lb:4.1f},{hb:4.1f}] {100*kc/n:6.1f} [{lc:4.1f},{hc:4.1f}] {100*(kc-kb)/n:+6.1f} {pv:7.3f} | {statistics.mean(bs):5.1f}±{sd(bs):3.1f} -> {statistics.mean(cs):5.1f}±{sd(cs):3.1f}")
        out[m] = dict(n=n, baseline=100*kb/n, candidate=100*kc/n, ci_base=(lb, hb), ci_cand=(lc, hc), delta=100*(kc-kb)/n, p=pv,
                      per_seed_base=bs, per_seed_cand=cs)
    name = f"five_scorer_summary_{tag}_{suite}" + (f"_vs_{btag}" if btag else "")
    _write(name, out, ["metric", "episodes", "baseline", "candidate"], [[m, v["n"], round(v["baseline"], 1), round(v["candidate"], 1)] for m, v in out.items()])


def cmd_agentdojo_defenses():
    """AgentDojo defense comparison (Table 2 AgentDojo block, Figure 3, the ASR cells of Table 9): tool filter, Progent and
    CaMeL run in AgentDyn's copy of the AgentDojo harness, next to the undefended agent, DRIFT and CAVAL from our harness,
    on the intersection of the episodes every row has. Usage: paper_tables.py agentdojo_defenses --caval <run prefix> --undefended <prefix>
    (CAVAL = five scorers <prefix>_s<k>; undefended = our draws <prefix>_r<k>)."""
    FORK = Path("data/benchmarks/AgentDyn/runs"); DOJO = Path("data/runs/gpt-4o-mini-2024-07-18")
    caval = sys.argv[sys.argv.index("--caval") + 1]
    # --undefended <prefix>: our own undefended draws <prefix>_r<k>_{benign,direct,injected} replace the August single
    # draw; benign and direct are the mean over the draws found, injected is draw 1 (the only injected draw). With --caval,
    # CAVAL injected is the mean over the five scorers <prefix>_s<k>, like benign and direct.
    undef = sys.argv[sys.argv.index("--undefended") + 1]
    slug = ("" if caval is None else "_" + caval.replace("/", "_")) + ("" if undef is None else "_undef")
    ATK, BEN = DOJO / "attack_important_instructions", DOJO / "no_attack"
    suites = ("banking", "slack", "travel", "workspace")

    def split(eps):
        kinds = {"benign": {}, "injected": {}, "direct": {}}
        for k, d in eps.items():
            if d.get("injection_task_id"): kinds["injected"][k] = d
            elif d["user_task_id"].startswith("injection_task"): kinds["direct"][k] = d
            elif d["user_task_id"].startswith("user_task"): kinds["benign"][k] = d
        return kinds

    rows = {}
    for p in sorted(FORK.glob("gpt-4o-mini-2024-07-18-*")):
        if any((p / s).exists() for s in suites):
            rows[f"{p.name[len('gpt-4o-mini-2024-07-18-'):]} (fork harness)"] = split(_episodes(p, suites=suites))
    ours = {"undefended (our harness)": (ATK / f"{undef}_draw1_injected", BEN / f"{undef}_draw1_direct", BEN / f"{undef}_draw1_benign"),
            "DRIFT (our harness)": (ATK / "drift_injected_banking_travel", ATK / "drift_injected_slack", ATK / "drift_injected",
                                    BEN / "drift_direct", BEN / "drift_benign", BEN / "drift_benign_slack"),
            "CAVAL (our harness)": (ATK / f"{caval}_scorer1", BEN / f"{caval}_scorer1_direct", BEN / f"{caval}_scorer1_benign")}
    for name, dirs in ours.items():
        rows[name] = split(_episodes(*dirs, suites=suites))
    seed_dir = lambda kind, k: BEN / f"{caval}_scorer{k}_{kind}"
    caval_seeds = {kind: [split(_episodes(seed_dir(kind, k), suites=suites))[kind] for k in range(1, 6)] for kind in ("benign", "direct")}
    caval_seeds["injected"] = [split(_episodes(ATK / f"{caval}_scorer{k}", suites=suites))["injected"] for k in range(1, 6)]
    undef_draws = {kind: [e for k in range(1, 6) if (BEN / f"{undef}_draw{k}_{kind}").exists()
                          for e in [split(_episodes(BEN / f"{undef}_draw{k}_{kind}", suites=suites))[kind]]]
                   for kind in ("benign", "direct")} if undef else {}
    if undef: print("undefended draws found:", {k: len(v) for k, v in undef_draws.items()}, "(injected: draw 1)")
    keys = {kind: set.intersection(*(set(r[kind]) for r in rows.values() if r[kind])) for kind in ("benign", "injected", "direct")}
    print("matched pairs per column (intersection over rows):", {k: len(v) for k, v in keys.items()})
    if not any(keys.values()): print("no fork episodes on the AgentDojo suites yet"); return
    pct = lambda eps, ks, f: 100.0 * sum(eps[k].get(f) is True for k in ks) / len(ks) if ks else float("nan")
    out, lines = {}, []
    hdr = f"{'defense':32s} {'benign util':>11s} {'util@attack':>11s} {'ASR inj':>8s} {'ASR direct':>10s}"
    for suite in [None] + list(suites):
        sel = {kind: sorted(k for k in ks if suite is None or k[0] == suite) for kind, ks in keys.items()}
        if not any(sel.values()): continue
        print(f"\n=== AgentDojo suites, {'pooled' if suite is None else suite} (n benign/injected/direct = {len(sel['benign'])}/{len(sel['injected'])}/{len(sel['direct'])}) ===")
        print(hdr)
        for name, r in rows.items():
            res = dict(benign_utility=pct(r["benign"], sel["benign"], "utility"), utility_under_attack=pct(r["injected"], sel["injected"], "utility"),
                       asr_injected=pct(r["injected"], sel["injected"], "security"), asr_direct=pct(r["direct"], sel["direct"], "utility"))
            def over(draws, kind, field, key):          # mean +- sd over scorers or draws, each matched to the pairs it has
                vals = [pct(e, [k for k in sel[kind] if k in e], field) for e in draws if e]
                if vals and sel[kind]: res[key], res[key + "_sd"] = statistics.mean(vals), statistics.pstdev(vals)
            if name.startswith("CAVAL"):
                over(caval_seeds["benign"], "benign", "utility", "benign_utility"); over(caval_seeds["direct"], "direct", "utility", "asr_direct")
                if "injected" in caval_seeds:
                    over(caval_seeds["injected"], "injected", "security", "asr_injected"); over(caval_seeds["injected"], "injected", "utility", "utility_under_attack")
            if name.startswith("undefended") and undef_draws:
                over(undef_draws["benign"], "benign", "utility", "benign_utility"); over(undef_draws["direct"], "direct", "utility", "asr_direct")
            out.setdefault(suite or "pooled", {})[name] = res
            print(f"{name:32s} {res['benign_utility']:11.1f} {res['utility_under_attack']:11.1f} {res['asr_injected']:8.1f} {res['asr_direct']:10.1f}")
            if suite is None:
                lines.append([name.replace(' (fork harness)', '').replace(' (our harness)', ''), round(res['benign_utility'], 1), round(res['utility_under_attack'], 1), round(res['asr_injected'], 1), round(res['asr_direct'], 1)])
    out["n"] = {k: len(v) for k, v in keys.items()}; out["caval_runs"] = caval
    _write("dojo_defenses" + slug, out, ["defense", "benign utility", "utility under attack", "ASR injected", "ASR direct"], lines)
    print("Note: fork-harness rows are the defenses' default configurations in AgentDyn's copy of AgentDojo; the other rows come from our harness on the same tasks, model and attack. CAVAL values are means over five scorers.")


def cmd_agentdyn_defenses():
    """AgentDyn comparison (Table 2 AgentDyn block, Table 11): the nine defenses AgentDyn released as GPT-4o-mini runs on its
    three suites, read as released, next to CAVAL on the same episodes (injected = scorer 1, benign and direct = mean over
    five scorers). CaMeL has no DailyLife run, so it and a second undefended row use CaMeL's subset.
    Usage: paper_tables.py agentdyn_defenses --caval <run prefix> [--suites]"""
    base = Path("data/benchmarks/AgentDyn/runs")
    model = "gpt-4o-mini-2024-07-18"
    per_suite = "--suites" in sys.argv

    def load(dirname):
        eps = {"benign": {}, "injected": {}, "direct": {}}
        for f in (base / dirname).glob("*/*/*/*.json"):
            d = json.loads(f.read_text())
            if d["suite_name"] not in ("dailylife", "github", "shopping"):
                continue            # the same folders also hold the defenses' AgentDojo runs (dojo_defenses reads those)
            key = (d["suite_name"], d["user_task_id"], d.get("injection_task_id"))
            if d.get("injection_task_id") and d.get("attack_type") == "important_instructions":
                eps["injected"][key] = (d.get("utility") is True, d.get("security") is True)
            elif d["user_task_id"].startswith("injection_task"):
                eps["direct"][key] = (d.get("utility") is True, None)
            elif d.get("injection_task_id") is None and d["user_task_id"].startswith("user_task"):
                eps["benign"][key] = (d.get("utility") is True, None)
        return eps

    def summarize(eps, keys=None, suite=None):
        def sel(kind):
            items = eps[kind].items()
            if keys is not None:
                items = [(k, v) for k, v in items if k in keys[kind]]
            if suite is not None:
                items = [(k, v) for k, v in items if k[0] == suite]
            return [v for _, v in items]
        ben, inj, dr = sel("benign"), sel("injected"), sel("direct")
        pct = lambda xs: 100.0 * sum(xs) / len(xs) if xs else float("nan")
        return dict(benign_utility=pct([u for u, _ in ben]), n_benign=len(ben),
                    utility_under_attack=pct([u for u, _ in inj]), asr_injected=pct([s for _, s in inj]), n_injected=len(inj),
                    asr_direct=pct([u for u, _ in dr]), n_direct=len(dr))

    defenses = ["repeat_user_prompt", "spotlighting_with_delimiting", "tool_filter", "transformers_pi_detector",
                "piguard_detector", "prompt_guard_2_detector", "camel", "progent", "drift"]
    runs = {"undefended": load(model)}
    for d in defenses:
        runs[d] = load(f"{model}-{d}")
    ref_keys = {k: set(v) for k, v in runs["undefended"].items()}
    # CAVAL on the same three suites from OUR run directories (same fork harness, verifier added):
    # injected = the deployed scorer at the online operating point (one scorer), direct and benign =
    # the five deployed scorers, reported as the mean over scorers as in the paper.
    ours = Path("data/runs/gpt-4o-mini-2024-07-18")
    adyn_suites = ("dailylife", "github", "shopping")
    prefix = sys.argv[sys.argv.index("--caval") + 1]
    slug = "_" + prefix.replace("/", "_")
    inj_dir = f"attack_important_instructions/{prefix}_scorer1"
    caval = {"benign": {}, "injected": {}, "direct": {}}
    for f in (ours / inj_dir).glob("*/*/*/injection_task_*.json"):
        d = json.loads(f.read_text())
        if d["suite_name"] in adyn_suites and d.get("injection_task_id"):
            caval["injected"][(d["suite_name"], d["user_task_id"], d["injection_task_id"])] = (d.get("utility") is True, d.get("security") is True)
    per_seed = {"benign": [], "direct": []}
    for sd in range(1, 6):
        for kind, rid, pat in (("benign", f"{prefix}_scorer{sd}_benign", "user_task_*"),
                               ("direct", f"{prefix}_scorer{sd}_direct", "injection_task_*")):
            vals = {}
            for f in (ours / "no_attack" / rid).glob(f"*/{pat}/none/none.json"):
                d = json.loads(f.read_text())
                if d["suite_name"] in adyn_suites:
                    vals[(d["suite_name"], d["user_task_id"], None)] = (d.get("utility") is True, None)
            per_seed[kind].append(vals)
    # store seed-1 keys for n, and the per-seed means for the rate
    caval["benign"], caval["direct"] = per_seed["benign"][0], per_seed["direct"][0]
    runs["CAVAL (ours, same tasks)"] = caval
    caval_seed_means = {k: [100.0 * sum(u for u, _ in v.values()) / len(v) for v in per_seed[k] if v] for k in per_seed}
    matched = {k: len(set(caval[k]) & ref_keys[k]) for k in caval}
    print(f"CAVAL episodes matched to the AgentDyn set: {matched} (injected one scorer, direct/benign five scorers)")
    out, lines = {}, []
    suites = [None] + (["dailylife", "github", "shopping"] if per_suite else [])
    hdr = f"{'defense':30s} {'benign util':>11s} {'util@attack':>11s} {'ASR inj':>8s} {'ASR direct':>10s}   n(ben/inj/dir)"
    for suite in suites:
        print(f"\n=== AgentDyn harness, {model}, {'pooled over 3 suites' if suite is None else suite} ===")
        print(hdr)
        for name, eps in runs.items():
            keys = None
            if name == "camel":
                keys = {k: set(v) for k, v in eps.items()}
            r = summarize(eps, keys, suite)
            if name.startswith("CAVAL") and suite is None:
                r["benign_utility"] = statistics.mean(caval_seed_means["benign"]); r["benign_utility_sd"] = statistics.pstdev(caval_seed_means["benign"])
                r["asr_direct"] = statistics.mean(caval_seed_means["direct"]); r["asr_direct_sd"] = statistics.pstdev(caval_seed_means["direct"])
                r["n_benign"] = f"{r['n_benign']}x5"; r["n_direct"] = f"{r['n_direct']}x5"
            out.setdefault(suite or "pooled", {})[name] = r
            print(f"{name:30s} {r['benign_utility']:11.1f} {r['utility_under_attack']:11.1f} {r['asr_injected']:8.1f} {r['asr_direct']:10.1f}   {r['n_benign']}/{r['n_injected']}/{r['n_direct']}")
            if suite is None:
                lines.append([name.replace('_', ' '), round(r['benign_utility'], 1), round(r['utility_under_attack'], 1), round(r['asr_injected'], 1), round(r['asr_direct'], 1)])
            if name == "camel":
                camel_keys = {k: set(v) for k, v in eps.items()}
                r2 = summarize(runs["undefended"], camel_keys, suite)
                out[suite or "pooled"]["undefended_camel_subset"] = r2
                print(f"{'undefended (CaMeL subset)':30s} {r2['benign_utility']:11.1f} {r2['utility_under_attack']:11.1f} {r2['asr_injected']:8.1f} {r2['asr_direct']:10.1f}   {r2['n_benign']}/{r2['n_injected']}/{r2['n_direct']}")
                if suite is None:
                    lines.append(["undefended (CaMeL subset)", round(r2['benign_utility'], 1), round(r2['utility_under_attack'], 1), round(r2['asr_injected'], 1), round(r2['asr_direct'], 1)])
    out["caval_runs"] = prefix
    _write("adyn_defenses" + slug, out, ["defense", "benign utility", "utility under attack", "ASR injected", "ASR direct"], lines)
    print("Note: the nine defenses are the runs released with AgentDyn (default configurations); CAVAL is our run on the same tasks in the same harness with the verifier added.")


def cmd_adversary_knowledge():
    """Adversary-knowledge variants (Figure 5, left): AgentDojo's important_instructions with the user and model names removed
    or a wrong user name, on Banking, Travel and Slack, for the undefended agent and CAVAL (scorer 1, deployed configuration),
    next to both on the plain template. Usage: paper_tables.py adversary_knowledge --caval "caval_deployed_{v}" --plain <plain run id>"""
    DOJO = Path("data/runs/gpt-4o-mini-2024-07-18"); ATK = DOJO / "attack_important_instructions"
    suites = ("banking", "travel", "slack")
    variants = sorted(p.name[len("attack_important_instructions_"):] for p in DOJO.glob("attack_important_instructions_*"))
    cav_prefix = sys.argv[sys.argv.index("--caval") + 1]      # run id = <prefix> with {v} = variant tag
    plain_id = sys.argv[sys.argv.index("--plain") + 1]         # the same scorer on the plain template
    slug = "_" + cav_prefix.replace("/", "_")
    refs = {"undefended (plain template)": _episodes(ATK / "undefended_injected", ATK / "undefended_injected_slack", suites=suites),
            "CAVAL deployed, same scorer, plain template": _episodes(ATK / plain_id, suites=suites)}

    def rate(eps, keys, field):
        vals = [eps[k].get(field) is True if field != "stopped" else _stopped(eps[k]) for k in keys]
        k = sum(vals); n = len(vals); lo, hi = _wilson(k, n)
        return dict(k=k, n=n, pct=100 * k / n if n else float("nan"), lo=lo, hi=hi)

    fmt = lambda r: f"{r['pct']:5.1f} [{r['lo']:4.1f},{r['hi']:4.1f}] n={r['n']:<3d}"
    out, lines = {}, []
    for v in variants:
        und = _episodes(DOJO / f"attack_important_instructions_{v}" / f"undefended_{v}", suites=suites)
        cav_id = cav_prefix.format(v=v) if "{v}" in cav_prefix else f"{cav_prefix}_{v}"   # e.g. caval_deployed_{v}
        cav = _episodes(DOJO / f"attack_important_instructions_{v}" / cav_id, suites=suites)
        keys_all = set(und) & set(cav)
        if not keys_all: print(f"{v}: no matched episodes yet"); continue
        print(f"\n=== variant {v} (matched pairs {len(keys_all)}) ===")
        print(f"{'suite':8s} | {'undefended ASR':>26s} | {'CAVAL ASR':>26s} | {'CAVAL util@attack':>26s} | {'CAVAL stop rate':>26s}")
        for suite in list(suites) + ["POOLED"]:
            keys = sorted(k for k in keys_all if suite == "POOLED" or k[0] == suite)
            if not keys: continue
            r = dict(undef_asr=rate(und, keys, "security"), caval_asr=rate(cav, keys, "security"),
                     caval_util=rate(cav, keys, "utility"), caval_stop=rate(cav, keys, "stopped"))
            print(f"{suite:8s} | {fmt(r['undef_asr'])} | {fmt(r['caval_asr'])} | {fmt(r['caval_util'])} | {fmt(r['caval_stop'])}")
            out.setdefault(v, {})[suite] = r
            if suite == "POOLED":
                lines.append([v.replace('_', ' '), round(r['undef_asr']['pct'], 1), round(r['caval_asr']['pct'], 1), round(r['caval_util']['pct'], 1), round(r['caval_stop']['pct'], 1)])
        # references on the same pairs, pooled, and the rule
        pooled = out[v]["POOLED"]; keys = sorted(keys_all)
        print(f"{'':8s}   references on the same {len(keys)} pairs, plain important_instructions:")
        for name, eps in refs.items():
            kk = [k for k in keys if k in eps]
            if not kk: continue
            rr = rate(eps, kk, "security"); out[v].setdefault("refs", {})[name] = rr
            probe = pooled["undef_asr"] if name.startswith("undefended") else pooled["caval_asr"]
            inside = rr["lo"] <= probe["pct"] <= rr["hi"]
            print(f"{'':8s}   {name:38s} {fmt(rr)}  variant {probe['pct']:.1f} inside its 95% interval: {inside}")
    _write("ladder" + slug, out, ["variant", "undefended ASR", "CAVAL ASR", "CAVAL utility under attack", "CAVAL stop rate"], lines)
    print("ASR = security True on attacked episodes; stop rate = attacked episodes with an enforced block; Wilson 95% intervals.")


def cmd_score_aware_attack():
    """Score-aware rewriting (Figure 5, right). Usage: paper_tables.py score_aware_attack <tag>. Reads data/runs/score_aware_attack/<tag>/state.json
    (written by src/experiments/score_aware_attack.py): per suite and pooled, the cumulative share of initially stopped pairs the
    rewriter recovers by round (Wilson 95%), and the mean verifier score of the still-rejected action by round."""
    tag = sys.argv[2]
    st = json.loads(Path(f"data/runs/score_aware_attack/{tag}/state.json").read_text())
    suites = sorted({k.split("/")[0] for k in st}); R = max((h["round"] for v in st.values() for h in v["history"]), default=0)
    out, lines = {}, []
    print(f"tag {tag}: {len(st)} pairs, {R} rounds run")
    print(f"{'suite':8s} {'n':>4s} | " + " | ".join(f"round {r:d} cumulative ASR" for r in range(0, R + 1)))
    for su in suites + ["POOLED"]:
        keys = [k for k in st if su == "POOLED" or k.startswith(su + "/")]; n = len(keys)
        cells = []
        for r in range(0, R + 1):
            k_ = sum(1 for k in keys if st[k]["closed_round"] is not None and st[k]["closed_round"] <= r)
            lo, hi = _wilson(k_, n); cells.append(dict(k=k_, n=n, pct=100 * k_ / n if n else float("nan"), lo=lo, hi=hi))
        scores = {r: [h["rejected"][1] for k in keys for h in st[k]["history"] if h["round"] == r and h.get("rejected")] for r in range(1, R + 1)}
        out[su] = dict(by_round=cells, mean_rejected_score={r: (sum(v) / len(v) if v else None) for r, v in scores.items()})
        print(f"{su:8s} {n:4d} | " + " | ".join(f"{c['pct']:5.1f} [{c['lo']:4.1f},{c['hi']:4.1f}]" for c in cells))
        lines.append([su.capitalize() if su != 'POOLED' else 'All', n] + [round(c['pct'], 1) for c in cells])
    if R:
        print("mean score of the rejected action by round (pooled): " + ", ".join(f"r{r}={out['POOLED']['mean_rejected_score'][r]:.3f}" if out['POOLED']['mean_rejected_score'][r] is not None else f"r{r}=n/a" for r in range(1, R + 1)))
        last = out["POOLED"]["by_round"][R]; first = out["POOLED"]["by_round"][0]
        overlap = last["lo"] <= first["hi"]
        out["verdict"] = dict(overlap=overlap, round=R)
    _write(f"rewrite_{tag}", out, ["suite", "pairs"] + [f"round {r}" for r in range(R + 1)], lines)


def cmd_unseen_wording():
    """Unseen injection wording (Table 6, last two rows): the five deployed scorers, trained on important_instructions
    (and tool_knowledge on Travel and Shopping), scored offline on the traces of the other AgentDojo templates in
    data/corpus/trace_graphs_unseen_wording.pkl. Per template: AUC, prevention and per-trace false-alarm rate at
    alpha 0.10 and 0.30, mean +- sd over the scorers. Usage: paper_tables.py unseen_wording --ckpt-prefix results/checkpoints/caval_scorer_seed"""
    pay = pickle.loads(Path("data/corpus/trace_graphs_unseen_wording.pkl").read_bytes())
    rows = pickle.loads(Path("data/corpus/action_labels_unseen_wording.pkl").read_bytes())["rows"]
    recs = pay["records"]
    tool_emb = build_tool_vocab_matrix(pay["tool_vocab"])[0]
    cmats = np.load("data/corpus/content_embeddings_unseen_wording.npz")
    TEMPLATES = {"ignore_previous": None, "system_message": None, "tool_knowledge": ("workspace",)}
    sel = [j for j, r in enumerate(rows) if recs[r["trace"]].get("attack_kind") in TEMPLATES
           and (TEMPLATES[recs[r["trace"]]["attack_kind"]] is None or recs[r["trace"]]["suite"] in TEMPLATES[recs[r["trace"]]["attack_kind"]])]
    data = build_prefix_data(pay, [rows[j] for j in sel], tool_emb, cmats)
    dev = torch.device("cpu")
    prefix = sys.argv[sys.argv.index("--ckpt-prefix") + 1]
    ckpts = [f"{prefix}{i}.pt" for i in range(1, 6)]
    res = defaultdict(lambda: defaultdict(list))
    for ck in ckpts:
        L = torch.load(ck, weights_only=False)
        model = build_model(L["model_name"], in_dim=L["in_dim"], hidden=L["hidden"], dropout=L["dropout"],
                            emb_dim=L["emb_dim"], proj_dim=L["proj_dim"]).to(dev)
        load_state_compat(model, L["model_state"]); model.eval()
        calM = sorted(L["cal_M"]); n = len(calM)
        lam = {a: (calM[math.ceil((n + 1) * (1 - a)) - 1] if math.ceil((n + 1) * (1 - a)) <= n else float("inf")) for a in (0.10, 0.30)}
        ps = []
        with torch.no_grad():
            for b in DataLoader(data, batch_size=128):
                lg, _ = model(b.to(dev)); ps.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
        p = np.concatenate(ps)
        per = defaultdict(list)
        for j, s_ in zip(sel, p):
            r = rows[j]; per[r["trace"]].append((r["t"], float(s_), r["y_trace"], r["t_star"]))
        by_t = defaultdict(list)
        for tr, v in per.items():
            by_t[recs[tr]["attack_kind"]].append(sorted(v))
        for tmpl, traces in by_t.items():
            M = np.array([max(x[1] for x in v) for v in traces]); y = np.array([v[0][2] for v in traces])
            r = res[tmpl]
            r["n"].append(len(traces)); r["n_comp"].append(int(y.sum()))
            r["auc"].append(float(roc_auc_score(y, M)) if 0 < y.sum() < len(y) else float("nan"))
            for a, lm in lam.items():
                ben = M[y == 0]; r[f"far_{a}"].append(float((ben > lm).mean()) if len(ben) else float("nan"))
                prev = []
                for v in traces:
                    if v[0][2] != 1 or v[0][3] is None: continue
                    hit = next((x[0] for x in v if x[1] > lm), None)
                    prev.append(int(hit is not None and hit <= v[0][3]))
                r[f"prev_{a}"].append(float(np.mean(prev)) if prev else float("nan"))
    out = {}
    print(f"{'template':16s} {'traces':>6s} {'comp':>5s} {'AUC':>14s} {'prev@0.30':>14s} {'FAR@0.30':>14s} {'prev@0.10':>14s} {'FAR@0.10':>14s}")
    for tmpl, r in res.items():
        f = lambda k: f"{np.nanmean(r[k]):.3f}+-{np.nanstd(r[k]):.3f}"
        out[tmpl] = {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in r.items()}
        print(f"{tmpl:16s} {r['n'][0]:6d} {r['n_comp'][0]:5d} {f('auc'):>14s} {f('prev_0.3'):>14s} {f('far_0.3'):>14s} {f('prev_0.1'):>14s} {f('far_0.1'):>14s}")
    _write("template_shift", out, ["template", "AUC", "prevention 0.10", "FAR 0.10", "prevention 0.30", "FAR 0.30"],
           [[k, round(v["auc"][0], 3), round(v["prev_0.1"][0], 3), round(v["far_0.1"][0], 3), round(v["prev_0.3"][0], 3), round(v["far_0.3"][0], 3)] for k, v in out.items()])


def cmd_calibration_rules():
    """Any-alarm (Proposition 1) and disruption (Theorem 1, conformal risk control) calibration, Tables 3 and 4. Thresholds
    from benign calibration traces, false-alarm rate, disruption and prevention on test, mean over ten seeds.
    Usage: paper_tables.py calibration_rules --scores <scores.npz> --index <action_labels.pkl> [model ...] (default: all five)."""
    a = sys.argv[2:]
    scores, index = a[a.index("--scores") + 1], a[a.index("--index") + 1]
    z = np.load(scores)
    rows = pickle.loads(Path(index).read_bytes())["rows"]
    for flag in ("--scores", "--index"):
        if flag in a: i = a.index(flag); del a[i:i + 2]
    models = a or ["rgcn", "gatv2", "seq_lstm", "mlp_pool", "gcn"]
    saved = {}
    alphas = (0.05, 0.10, 0.20, 0.30)

    def traces(idx, p):
        d = defaultdict(list)
        for j, s in zip(idx, p):
            r = rows[j]
            d[r["trace"]].append((r["t"], float(s), r["y_trace"], r["t_star"]))
        return {tr: (np.array([x[1] for x in sorted(v)]), np.array([x[0] for x in sorted(v)]),
                     v[0][2], v[0][3]) for tr, v in d.items()}

    def work_lost(scores, lam):
        hit = np.nonzero(scores > lam)[0]
        return 0.0 if len(hit) == 0 else (len(scores) - hit[0]) / len(scores)

    def evaluate(test, lam):
        ben = [v for v in test.values() if v[2] == 0]
        com = [v for v in test.values() if v[2] == 1 and v[3] is not None]
        far = np.mean([v[0].max() > lam for v in ben])
        work = np.mean([work_lost(v[0], lam) for v in ben])
        prev = np.mean([(v[0] > lam).any() and v[1][np.nonzero(v[0] > lam)[0][0]] <= v[3]
                        for v in com])
        return far, work, prev

    print("Theorem 1 check: thresholds from benign calibration traces; test means over seeds.")
    print(f"{'model':8s} {'alpha':>5s} | {'lam_conf':>8s} {'lam_crc':>8s} | "
          f"any-alarm rule: FAR  work  prev | work-lost rule: FAR  work  prev | valid")
    for m in models:
        for alpha in alphas:
            acc = []
            for s in range(1, 11):
                cal = traces(z[f"{m}_s{s}_ca_idx"], z[f"{m}_s{s}_ca_p"])
                test = traces(z[f"{m}_s{s}_te_idx"], z[f"{m}_s{s}_te_p"])
                calb = [v[0] for v in cal.values() if v[2] == 0]
                n = len(calb)
                M = np.sort([c.max() for c in calb])
                k = int(np.ceil((n + 1) * (1 - alpha)))
                lam_conf = M[k - 1] if k <= n else np.inf
                grid = np.unique(np.concatenate([np.concatenate(calb), [0.0, 1.0]]))
                risk = np.array([(sum(work_lost(c, g) for c in calb) + 1.0) / (n + 1) for g in grid])
                ok = np.nonzero(risk <= alpha)[0]
                lam_crc = grid[ok[0]] if len(ok) else np.inf
                acc.append((lam_conf, lam_crc) + evaluate(test, lam_conf) + evaluate(test, lam_crc))
            a = np.mean(acc, axis=0)
            saved[f"{m}_{alpha}"] = dict(zip(("lam_conf", "lam_crc", "far_any", "work_any", "prev_any", "far_work", "work_work", "prev_work"), map(float, a)))
            print(f"{m:8s} {alpha:5.2f} | {a[0]:8.4f} {a[1]:8.4f} | "
                  f"{a[2]:.3f} {a[3]:.3f} {a[4]:.3f} | {a[5]:.3f} {a[6]:.3f} {a[7]:.3f} | "
                  f"{'work<=alpha' if a[6] <= alpha else 'VIOLATED'}")
        print()
    print("Read: the two columns budget DIFFERENT events (any alarm on a benign trace vs")
    print("expected fraction of benign work lost); never present them as the same guarantee.")
    _write("crc", saved, ["model", "alpha", "FAR any-alarm", "disruption any-alarm", "prevention any-alarm", "FAR disruption rule", "disruption disruption rule", "prevention disruption rule"],
           [[k.rsplit("_", 1)[0], k.rsplit("_", 1)[1]] + [round(v[x], 3) for x in ("far_any", "work_any", "prev_any", "far_work", "work_work", "prev_work")] for k, v in saved.items()])


def cmd_latency():
    """Verifier latency per decision from the latency_ms field the verifier writes into every decision record.
    Usage: paper_tables.py latency <run_dir> [<run_dir> ...]"""
    dirs = [Path(d) for d in sys.argv[2:]]
    assert dirs, "give at least one run directory"
    per_dev = defaultdict(lambda: dict(lat=[], per_ep=[], share=[], n_dec=[], eps=0, no_field=0))
    for rd in dirs:
        for f in rd.rglob("*.json"):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            log = d.get("verifier_log") or []
            if not log:
                continue
            lats = [e["latency_ms"] for e in log if "latency_ms" in e]
            dev = next((e.get("device") for e in log if e.get("device")), "unknown")
            r = per_dev[dev]; r["eps"] += 1
            if not lats:
                r["no_field"] += 1; continue
            r["lat"] += lats; r["n_dec"].append(len(lats)); r["per_ep"].append(sum(lats))
            if d.get("duration"):
                r["share"].append(sum(lats) / 1000.0 / d["duration"])
    out = {}
    for dev, r in per_dev.items():
        if not r["lat"]:
            print(f"{dev}: {r['eps']} episodes, none carry latency_ms (run predates the timer)"); continue
        lat = np.array(r["lat"])
        out[dev] = dict(episodes=r["eps"], decisions=int(len(lat)), decisions_per_episode=float(np.mean(r["n_dec"])),
                        ms_mean=float(lat.mean()), ms_median=float(np.median(lat)), ms_p95=float(np.percentile(lat, 95)),
                        ms_per_episode=float(np.mean(r["per_ep"])), share_of_episode=float(np.mean(r["share"])) if r["share"] else None)
        o = out[dev]
        print(f"{dev}: {o['episodes']} episodes, {o['decisions']} decisions ({o['decisions_per_episode']:.1f}/episode) | "
              f"per decision mean {o['ms_mean']:.1f} ms, median {o['ms_median']:.1f}, p95 {o['ms_p95']:.1f} | "
              f"per episode {o['ms_per_episode']:.0f} ms" + (f" = {100*o['share_of_episode']:.1f}% of episode wall time" if o['share_of_episode'] is not None else ""))
    _write("latency", out, ["device", "episodes", "decisions", "median ms", "mean ms", "p95 ms"],
           [[k, v["episodes"], v["decisions"], round(v["ms_median"], 1), round(v["ms_mean"], 1), round(v["ms_p95"], 1)] for k, v in out.items()])


def _dump_traces(z, rows, model, seed, part):
    """{trace: [(t, p, y_trace, t_star, suite)] sorted by t} from a train.py --dump-scores npz."""
    per = defaultdict(list)
    for i, pp in zip(z[f"{model}_s{seed}_{part}_idx"], z[f"{model}_s{seed}_{part}_p"]):
        r = rows[i]; per[r["trace"]].append((r["t"], float(pp), r["y_trace"], r["t_star"], r["suite"]))
    return {k: sorted(v) for k, v in per.items()}


def _lam_alpha(ms, alpha):
    m = sorted(ms); n = len(m); k = math.ceil((n + 1) * (1 - alpha))
    return m[k - 1] if k <= n else float("inf")


def _running_stat(scores, stat, beta=0.7):
    """Causal per-step statistic over the step scores of one trace; the paper uses max, the running maximum."""
    out, run = [], None
    for i, s in enumerate(scores):
        if stat == "max":
            run = s if run is None else max(run, s)
        elif stat == "top2":
            top = sorted(scores[: i + 1], reverse=True)[:2]; run = sum(top) / len(top)
        elif stat == "ewma":
            run = s if run is None else beta * run + (1 - beta) * s
        else:
            raise ValueError(stat)
        out.append(run)
    return out


def _stat_traces(traces, stat, beta=0.7):
    """Replace each step's score by the running statistic (t order is preserved)."""
    return {k: [(x[0], r) + tuple(x[2:]) for x, r in zip(v, _running_stat([x[1] for x in v], stat, beta))]
            for k, v in traces.items()}


def _trace_metrics(te, L):
    """Pooled per-trace FAR, prevention (alarm at some t <= t*), trace AUC (max of the statistic)."""
    neg = [v for v in te.values() if v[0][2] == 0]; pos = [v for v in te.values() if v[0][2] == 1]
    far = float(np.mean([max(x[1] for x in v) > L for v in neg])) if neg else float("nan")
    prev = float(np.mean([any(x[1] > L for x in v if x[0] <= x[3]) for v in pos])) if pos else float("nan")
    ys = [v[0][2] for v in te.values()]; ms = [max(x[1] for x in v) for v in te.values()]
    auc = float(roc_auc_score(ys, ms)) if 0 < sum(ys) < len(ys) else float("nan")
    return dict(far=far, prev=prev, auc=auc, n_neg=len(neg), n_pos=len(pos), lam=float(L))


def cmd_offline_models():
    """Offline results over ten seeds (Tables 6 and 8, the architecture paragraph): trace-level AUC, prevention and
    per-trace false-alarm rate at alpha 0.10 and 0.30 for the five scorers, paired Wilcoxon tests of the R-GCN against
    each, the R-GCN ablations, and the held-out-objective row. Reads results/offline/*.json and ten_seed_scores.npz.
    Usage: paper_tables.py offline_models"""
    from scipy.stats import wilcoxon
    R = Path("results/offline")
    rows = pickle.loads(Path("data/corpus/action_labels.pkl").read_bytes())["rows"]
    z = np.load(R / "ten_seed_scores.npz")
    load = lambda f: json.loads((R / f).read_text())
    per_seed = lambda recs, m, key: {r["seed"]: r[key] for r in recs if r["model"] == m}
    conf = lambda recs, m, a, key: {r["seed"]: next(c[key] for c in r["conformal"] if abs(c["alpha"] - a) < 1e-9) for r in recs if r["model"] == m}
    ms = lambda d: (float(np.mean(list(d.values()))), float(np.std(list(d.values()))))
    main = load("ten_seed_models.json"); out = {"architectures": {}, "ablations": {}}
    names = {"rgcn": "R-GCN", "gatv2": "GATv2", "seq_lstm": "LSTM", "mlp_pool": "MLP", "gcn": "GCN"}
    arch_rows = []
    for m, nm in names.items():
        auc = per_seed(main, m, "auc_trace")
        at30 = [_trace_metrics(_stat_traces(_dump_traces(z, rows, m, s, "te"), "max"),
                               _lam_alpha([max(x[1] for x in v) for v in _stat_traces(_dump_traces(z, rows, m, s, "ca"), "max").values() if v[0][2] == 0], 0.30))
                for s in sorted(auc)]
        r = dict(auc=ms(auc), prev_010=ms(conf(main, m, 0.10, "prevention")), far_010=ms(conf(main, m, 0.10, "per_trace_far")),
                 prev_030=(float(np.mean([x["prev"] for x in at30])), float(np.std([x["prev"] for x in at30]))),
                 far_030=(float(np.mean([x["far"] for x in at30])), float(np.std([x["far"] for x in at30]))))
        if m != "rgcn":
            base = per_seed(main, "rgcn", "auc_trace"); seeds = sorted(set(base) & set(auc))
            r["wilcoxon_p_vs_rgcn"] = float(wilcoxon([base[s] - auc[s] for s in seeds]).pvalue)
        out["architectures"][nm] = r
        arch_rows.append([nm] + [f"{r[k][0]:.3f} +- {r[k][1]:.3f}" for k in ("auc", "prev_010", "far_010", "prev_030", "far_030")] + [f"{r['wilcoxon_p_vs_rgcn']:.3f}" if "wilcoxon_p_vs_rgcn" in r else ""])
        print(f"{nm:6s} AUC {r['auc'][0]:.3f}+-{r['auc'][1]:.3f} | 0.10 prev {r['prev_010'][0]:.3f} FAR {r['far_010'][0]:.3f} | 0.30 prev {r['prev_030'][0]:.3f} FAR {r['far_030'][0]:.3f}"
              + (f" | Wilcoxon p vs R-GCN {r['wilcoxon_p_vs_rgcn']:.3f}" if "wilcoxon_p_vs_rgcn" in r else ""))
    _write("offline_architectures", out["architectures"], ["model", "AUC", "prevention 0.10", "FAR 0.10", "prevention 0.30", "FAR 0.30", "Wilcoxon p vs R-GCN"], arch_rows)
    full_auc, full_prev = per_seed(main, "rgcn", "auc_trace"), conf(main, "rgcn", 0.10, "prevention")
    abl_rows = [["Full model", f"{ms(full_auc)[0]:.3f} +- {ms(full_auc)[1]:.3f}", "", f"{ms(full_prev)[0]:.3f} +- {ms(full_prev)[1]:.3f}", ""]]
    for f, nm in (("ablation_no_content.json", "No content embeddings"), ("ablation_utility_objective.json", "With utility objective"),
                  ("ablation_no_data_flow.json", "No data-flow edges"), ("ablation_no_call_return.json", "No call-return edges"),
                  ("ablation_no_temporal.json", "No temporal edges"), ("ablation_dataflow_overlap1.json", "Data-flow overlap >= 1"),
                  ("ablation_dataflow_overlap3.json", "Data-flow overlap >= 3")):
        recs = load(f); a, p = per_seed(recs, "rgcn", "auc_trace"), conf(recs, "rgcn", 0.10, "prevention")
        sa, sp = sorted(set(a) & set(full_auc)), sorted(set(p) & set(full_prev))
        pa = float(wilcoxon([full_auc[s] - a[s] for s in sa]).pvalue); pp = float(wilcoxon([full_prev[s] - p[s] for s in sp]).pvalue) if any(full_prev[s] != p[s] for s in sp) else 1.0
        out["ablations"][nm] = dict(auc=ms(a), prev_010=ms(p), p_auc=pa, p_prev=pp)
        abl_rows.append([nm, f"{ms(a)[0]:.3f} +- {ms(a)[1]:.3f}", f"{pa:.3f}", f"{ms(p)[0]:.3f} +- {ms(p)[1]:.3f}", f"{pp:.3f}"])
        print(f"{nm:24s} AUC {ms(a)[0]:.3f} (p {pa:.3f}) prev@0.10 {ms(p)[0]:.3f} (p {pp:.3f})")
    _write("offline_ablations", out["ablations"], ["variant", "AUC", "Wilcoxon p (AUC)", "prevention 0.10", "Wilcoxon p (prevention)"], abl_rows)
    ho = load("heldout_objectives.json")
    h = dict(auc=ms(per_seed(ho, "rgcn", "auc_trace")), prev_010=ms(conf(ho, "rgcn", 0.10, "prevention")), far_010=ms(conf(ho, "rgcn", 0.10, "per_trace_far")))
    print(f"held-out objectives: AUC {h['auc'][0]:.3f} prev@0.10 {h['prev_010'][0]:.3f} FAR {h['far_010'][0]:.3f}")
    _write("offline_heldout_objectives", h, ["model", "AUC", "prevention 0.10", "FAR 0.10"], [["R-GCN", f"{h['auc'][0]:.3f}", f"{h['prev_010'][0]:.3f}", f"{h['far_010'][0]:.3f}"]])


def cmd_overhead():
    """Efficiency on AgentDojo Banking (Table 9): benign utility and injected ASR (from results/agentdojo_defenses.json),
    tokens and wall time per benign episode, and the defense efficiency E = (utility - ASR) / (thousand tokens x seconds),
    Equation 1. Tokens and time: our harness records them per episode (undefended draws, DRIFT, CAVAL's five scorers);
    for the defenses run in AgentDyn's harness, tokens come from data/runs/baseline_tokens/<defense>.json (every model call
    counted by scripts/count_baseline_tokens.sh over the 16 Banking tasks) and time from their episode files.
    Usage: paper_tables.py overhead (run agentdojo_defenses first)"""
    BEN = Path("data/runs/gpt-4o-mini-2024-07-18/no_attack"); FORK = Path("data/benchmarks/AgentDyn/runs")
    dd = json.loads((RESULTS / "agentdojo_defenses.json").read_text())["banking"]
    def ours(*dirs):
        eps = [json.loads(f.read_text()) for d in dirs for f in (BEN / d / "banking").glob("user_task_*/none/none.json")]
        return len(eps), float(np.mean([e["total_tokens"] for e in eps])), float(np.mean([e["duration"] for e in eps]))
    def fork(d):
        t = json.loads(Path(f"data/runs/baseline_tokens/{d}.json").read_text())
        eps = [json.loads(f.read_text()) for f in (FORK / f"gpt-4o-mini-2024-07-18-{d}" / "banking").glob("user_task_*/none/none.json")]
        return len(eps), (t["prompt"] + t["completion"]) / len(eps), float(np.mean([e["duration"] for e in eps]))
    rows = {"Undefended": ("undefended (our harness)", ours(*[f"undefended_draw{k}_benign" for k in range(1, 6)])),
            "Tool filter": ("tool_filter (fork harness)", fork("tool_filter")), "Progent": ("progent (fork harness)", fork("progent")),
            "CaMeL": ("camel (fork harness)", fork("camel")), "DRIFT": ("DRIFT (our harness)", ours("drift_benign")),
            "CAVAL": ("CAVAL (our harness)", ours(*[f"caval_deployed_scorer{k}_benign" for k in range(1, 6)]))}
    out, table = {}, []
    print(f"{'defense':12s} {'benign util':>11s} {'ASR inj':>8s} {'tokens':>8s} {'time s':>7s} {'E':>6s}  episodes")
    for name, (key, (n, tok, sec)) in rows.items():
        u, a = dd[key]["benign_utility"], dd[key]["asr_injected"]
        e = (round(u, 1) - round(a, 1)) / (round(tok) / 1000 * round(sec, 1))   # from the columns as printed in Table 9
        out[name] = dict(benign_utility=u, asr_injected=a, tokens=tok, seconds=sec, efficiency=e, episodes=n)
        table.append([name, round(u, 1), round(a, 1), round(tok), round(sec, 1), round(e, 2)])
        print(f"{name:12s} {u:11.1f} {a:8.1f} {tok:8.0f} {sec:7.1f} {e:6.2f}  {n}")
    _write("overhead", out, ["defense", "benign utility", "ASR injected", "tokens", "time (s)", "efficiency"], table)


def cmd_verifier_directed_attack():
    """Verifier-directed injection (Figure 5, left): the same injected pairs under the plain important_instructions
    text and under the text that addresses the verifier, for CAVAL, the undefended agent and DRIFT, per suite and
    pooled. Usage: paper_tables.py verifier_directed_attack --caval-runs <plain run id>,<adaptive run id> --undef-adaptive <run id>"""
    DOJO = Path("data/runs/gpt-4o-mini-2024-07-18"); ADAPT = Path("data/runs/gpt-4o-mini-2024-07-18-adaptive_attack")
    ATK = "attack_important_instructions"
    plain_id, adaptive_id = sys.argv[sys.argv.index("--caval-runs") + 1].split(",")
    suites = ("banking", "travel", "slack", "workspace")
    PLAIN = {"DRIFT": {"banking": "drift_injected_banking_travel", "travel": "drift_injected_banking_travel", "slack": "drift_injected_slack", "workspace": "drift_injected"},
             "CAVAL": {s: plain_id for s in suites},
             "Undefended": {"banking": "undefended_injected", "travel": "undefended_injected", "slack": "undefended_injected_slack", "workspace": "undefended_injected"}}
    ADAPTIVE = {"DRIFT": "drift_verifier_directed", "CAVAL": adaptive_id,
                "Undefended": sys.argv[sys.argv.index("--undef-adaptive") + 1]}

    def load(root):
        out = {}
        for f in sorted(Path(root).rglob("*.json")):
            d = json.loads(f.read_text())
            if d.get("injection_task_id") is None: continue
            out[(str(d.get("user_task_id")), str(d["injection_task_id"]))] = d.get("security") is True
        return out

    pooled = {lbl: [0, 0, 0] for lbl in PLAIN}; per_suite = {}
    print(f"{'suite':10s} {'defense':12s} {'n':>4s} {'plain ASR':>10s} {'adaptive ASR':>13s}")
    for s in suites:
        for lbl in PLAIN:
            P, A = load(DOJO / ATK / PLAIN[lbl][s] / s), load(ADAPT / ATK / ADAPTIVE[lbl] / s)
            keys = set(P) & set(A); n = len(keys)
            if not n: continue
            kp = sum(P[k] for k in keys); ka = sum(A[k] for k in keys)
            pooled[lbl][0] += n; pooled[lbl][1] += kp; pooled[lbl][2] += ka
            per_suite.setdefault(lbl, {})[s] = dict(n=n, plain_k=kp, adaptive_k=ka)
            print(f"{s:10s} {lbl:12s} {n:4d} {100*kp/n:10.1f} {100*ka/n:13.1f}")
    out, rows = {}, []
    for lbl, (n, kp, ka) in pooled.items():
        if not n: continue
        out[lbl] = dict(n=n, plain_k=kp, adaptive_k=ka, plain_pct=100 * kp / n, adaptive_pct=100 * ka / n, per_suite=per_suite.get(lbl, {}))
        rows.append([lbl, n, round(100 * kp / n, 1), round(100 * ka / n, 1)])
        print(f"{'POOLED':10s} {lbl:12s} {n:4d} {100*kp/n:10.1f} {100*ka/n:13.1f}")
    _write("adaptive", out, ["defense", "pairs", "plain ASR", "verifier-directed ASR"], rows)


_COMMANDS = {"online_comparison": cmd_online_comparison, "five_scorer_summary": cmd_five_scorer_summary, "agentdojo_defenses": cmd_agentdojo_defenses,
             "agentdyn_defenses": cmd_agentdyn_defenses, "verifier_directed_attack": cmd_verifier_directed_attack, "adversary_knowledge": cmd_adversary_knowledge,
             "score_aware_attack": cmd_score_aware_attack, "unseen_wording": cmd_unseen_wording, "offline_models": cmd_offline_models,
             "calibration_rules": cmd_calibration_rules, "latency": cmd_latency, "overhead": cmd_overhead}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        print("usage: python src/analysis/paper_tables.py <" + "|".join(_COMMANDS) + "> [args]")
        sys.exit(1)
    if "--out" in sys.argv:
        i = sys.argv.index("--out"); OUT_NAME = sys.argv[i + 1]; del sys.argv[i:i + 2]
    _COMMANDS[sys.argv[1]]()
