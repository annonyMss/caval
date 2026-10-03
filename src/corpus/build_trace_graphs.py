# ==== merged from agentdyn_adapter.py ====
"""Reshape AgentDyn `messages` logs into the local `conversations` schema that
trace_to_graph expects. This is instance #1 of a benchmark adapter; the field
mismatches it handles are the requirements for a canonical trace spec (see
project-benchmark-ingestion memory).

Known gaps vs the local schema (documented, not bugs):
  - AgentDyn has no `tool_permission` -> privilege feature is all-zero here.
  - AgentDyn has no `initial_trajectory` -> in_initial_traj always False (leaky, dropped).
  - content is 'None' or a stringified [{'type':'text','content':X}] list.
"""


"""Build the combined UNDEFENDED compromised-label graph dataset across 4 suites:
local slack (buckets C/D) + AgentDyn gpt-4o-mini (dailylife/github/shopping).

One GLOBAL tool vocab so cross-suite tool identity is consistent; records carry
`suite` for leave-one-suite-out (D4). Privilege zeroed everywhere for consistency
(AgentDyn ships no tool_permission). Label = compromised (attack_success is True).
"""
import argparse
import json
import pickle
from pathlib import Path
import ast
from json_repair import loads
import re

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))


from caval import trace_graph as _ttg
from caval.trace_graph import (trace_to_graph, build_tool_vocab, graph_targets,
                            collect_slack_poc_paths, _bucket_for_path, load_log)

ADYN_SUITES = ("dailylife", "github", "shopping")
DOJO_SUITES = ("banking", "travel", "workspace")
EXTRA_ATTACK_MODES = {"travel": ["attack_tool_knowledge"]}

def _unwrap(content):
    """'None' or "[{'type':'text','content':X}]" -> plain text."""
    s = str(content)
    if s == "None":
        return ""
    try:
        parts = ast.literal_eval(s)
        if isinstance(parts, list):
            return " ".join(p.get("content", "") for p in parts if isinstance(p, dict))
    except (ValueError, SyntaxError):
        pass
    return s


def to_conversations(log):
    """Return `log` with a `conversations` key in the local schema. Top-level keys
    (attack_type, security, utility, suite_name, user_task_id) pass through, so
    graph_targets / labels / grouping work unchanged.

    Some AgentDyn-sourced logs (e.g. the Shopping tool_knowledge round, generated
    via AgentDyn's own copy of the DRIFT harness) already write `conversations`
    directly and have NO `messages` key at all. Rebuilding from `messages` in that
    case silently overwrote the real data with an empty list (bug found
    that path, those traces became graphs with no nodes).
    Pass those through unchanged instead of reprocessing them."""
    if "messages" not in log:
        return log
    msgs = log.get("messages", [])
    text_idx = [i for i, m in enumerate(msgs)
                if m.get("role") == "assistant" and _unwrap(m.get("content"))]
    last = text_idx[-1] if text_idx else None
    conv = []
    for i, m in enumerate(msgs):
        r = m.get("role")
        if r == "user":
            conv.append({"role": "user", "content": _unwrap(m.get("content"))})
        elif r == "assistant":
            txt = _unwrap(m.get("content"))
            tag = "final_answer" if i == last else "function_thought"
            conv.append({"role": "assistant",
                         "content": f"<{tag}>{txt}</{tag}>" if txt else "",
                         "tool_calls": m.get("tool_calls") or []})                         
        elif r == "tool":
            err = m.get("error")
            conv.append({"role": "tool", "content": _unwrap(m.get("content")),
                         "tool_call": m.get("tool_call") or {},
                         "tool_call_id": m.get("tool_call_id"),
                         "error": None if err in (None, "None") else err})
    out = dict(log)
    out["conversations"] = conv
    return out



def collect_local_dojo_undef(base_dir, extra_modes=()):
    """Undefended AgentDojo traces of Banking, Travel and Workspace, injected and benign."""
    base = Path(base_dir)
    out = []
    for suite in DOJO_SUITES:
        modes = list(dict.fromkeys(["attack_important_instructions"]
                                   + EXTRA_ATTACK_MODES.get(suite, []) + list(extra_modes)))
        for mode in modes:
            root = base / mode / "undefended_injected" / suite
            for p in sorted(root.glob("user_task_*/*/injection_task_*.json")):
                log = load_log(p)
                log["tool_permission"] = {}   
                out.append((log, suite))
        broot = base / "no_attack" / "undefended_benign" / suite
        for p in sorted(broot.glob("user_task_*/none/none.json")):
            log = load_log(p)
            log["tool_permission"] = {}
            out.append((log, suite))
    return out


def collect_local_slack_undef(base_dir):
    """Local slack buckets C/D (defended==False), privilege zeroed."""
    out = []
    for p in collect_slack_poc_paths(base_dir):
        _, _, defended = _bucket_for_path(p)
        if defended is False:
            log = load_log(p)
            log["tool_permission"] = {}         
            out.append((log, "slack"))
    return out


def collect_adyn_undef(adyn_base):
    out = []
    for suite in ADYN_SUITES:
        for p in sorted((Path(adyn_base) / suite).rglob("*.json")):
            out.append((to_conversations(json.loads(p.read_text())), suite))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slack-base", default="data/runs/gpt-4o-mini-2024-07-18")
    ap.add_argument("--adyn-base", default="data/benchmarks/AgentDyn/runs/gpt-4o-mini-2024-07-18")
    ap.add_argument("--output", default="data/corpus/trace_graphs.pkl")
    ap.add_argument("--extra-attack-modes", nargs="*", default=[],
                    help="other AgentDojo injection wordings to add, e.g. attack_ignore_previous "
                         "attack_system_message attack_tool_knowledge (the unseen-wording rows)")
    ap.add_argument("--vocab-from", default=None,
                    help="reuse the tool_vocab of an existing graph pkl (tools not in it "
                         "map to id 0) so a variant dataset stays feature-compatible")
    ap.add_argument("--df-min-overlap", type=int, default=None,
                    help="override the data-flow token-overlap threshold (data-flow ablation)")
    args = ap.parse_args()
    if args.df_min_overlap is not None:
        _ttg.DATA_FLOW_MIN_OVERLAP = args.df_min_overlap
        print(f"data_flow min-overlap OVERRIDDEN to {args.df_min_overlap}")

    tagged = (collect_local_slack_undef(args.slack_base) + collect_adyn_undef(args.adyn_base)
              + collect_local_dojo_undef(args.slack_base, args.extra_attack_modes))
    logs = [l for l, _ in tagged]
    if args.vocab_from:
        with open(args.vocab_from, "rb") as f:
            vocab = pickle.load(f)["tool_vocab"]
        missing = sorted(set(build_tool_vocab(logs)) - set(vocab))
        print(f"vocab reused from {args.vocab_from} ({len(vocab)} tools); "
              f"{len(missing)} tools map to id 0: {missing}")
    else:
        vocab = build_tool_vocab(logs)

    graphs, records = [], []
    for log, suite in tagged:
        t = graph_targets(log)
        graphs.append(trace_to_graph(log, vocab))
        records.append(dict(
            suite=suite, user_task_id=log.get("user_task_id"),
            injection_task_id=log.get("injection_task_id"),
            attack_type=log.get("attack_type"), attack_kind=t["attack_kind"],
            is_attack_run=t["is_attack_run"], attack_success=t["attack_success"],
            utility_success=t["utility_success"],
            compromised=int(t["attack_success"] is True),
        ))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(dict(graphs=graphs, records=records, tool_vocab=vocab), f)

    print(f"built {len(graphs)} graphs | tool_vocab {len(vocab)} | -> {args.output}")
    print(f"{'suite':10s} {'n':>5s} {'compromised':>12s} {'clean':>6s}")
    for s in sorted({r["suite"] for r in records}):
        rs = [r for r in records if r["suite"] == s]
        pos = sum(r["compromised"] for r in rs)
        print(f"{s:10s} {len(rs):5d} {pos:12d} {len(rs)-pos:6d}")


if __name__ == "__main__":
    main()