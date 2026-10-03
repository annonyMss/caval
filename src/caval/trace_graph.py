import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
import math
import os
import pickle
import re
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from corpus.tool_descriptions import build_map, DEFAULT_ROOTS
import sys

NODE_TYPES = {
    'user_query': 0,
    'call_error': 1,
    'assistant_reasoning': 2,
    'tool_call': 3,
    'tool_return': 4,
    'final_answer': 5,
}

# Build a global tool vocabulary first - needed for the embedding layer later.
# Reserve 0 for unknown/no-tool so real tools do not collide with missing values.
def build_tool_vocab(all_logs):
    tools = set()
    for log in all_logs:
        for msg in log.get('conversations', []):
            for tc in (msg.get('tool_calls') or []):
                if tc.get('function'):
                    tools.add(tc.get('function'))
            tool_call = msg.get('tool_call') or {}
            if tool_call.get('function'):
                tools.add(tool_call.get('function'))
    return {t: i + 1 for i, t in enumerate(sorted(tools))}

PRIVILEGE_MAP = {None: 0, 'Read': 1, 'Write': 2, 'Execute': 3}

# data_flow edge threshold: an edge fires when >= DATA_FLOW_MIN_OVERLAP shared
# non-stopword tokens of >= 4 chars. Module-level so the data-flow threshold
# ablation (run_battery.py / build_multisuite.py --df-min-overlap) can vary it
# for a variant dataset build without touching call sites.
DATA_FLOW_MIN_OVERLAP = 2


def tokens(s, min_len=4):
    """Content tokens for data-flow edge detection."""
    toks = set(re.findall(r'\w+', str(s).lower()))
    return {t for t in toks if len(t) >= min_len}

STOPWORDS = {'true', 'false', 'null', 'none', 'this', 'that',
            'with', 'from', 'have', 'been', 'will'}

def load_log(path):
    return json.loads(Path(path).read_text())

def extract_tag(content, tag):
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    match = pattern.search(str(content))
    return match.group(1).strip() if match else ''

def is_call_error(content):
    text = str(content)
    return '[CALL ERROR]' in text or '<function_error>' in text

def has_injection_marker(content):
    text = str(content).lower()
    return '<information>' in text or '</information>' in text

def content_length_bin(content):
    length = len(str(content or ''))
    return int(math.log2(length + 1)) if length else 0

def graph_targets(log):
    """Return clear labels for utility and ASR.

    In AgentDojo result JSON, `security=True` means the injection task
    succeeded. For benign runs there is no injection task, so attack_success is
    intentionally None rather than using the placeholder security value.

    DIRECT-attack runs (AgentDyn: the user task IS the injection task, path
    injection_task_*/none/none.json, attack_type None) are attack runs too —
    security there means the attacker goal was achieved with the user as the
    adversary. These must not fall through to
    attack_success=None and were mislabeled clean.
    """
    direct = (log.get('injection_task_id') is None
              and str(log.get('user_task_id') or '').startswith('injection_task'))
    injected = log.get('attack_type') is not None
    is_attack_run = injected or direct
    return {
        'utility_success': bool(log.get('utility')),
        'attack_success': bool(log.get('security')) if is_attack_run else None,
        'is_attack_run': is_attack_run,
        'attack_kind': 'direct' if direct else log.get('attack_type'),
        # injection-present keeps its original meaning: an injection was PLACED
        # in the environment (buckets B/C). Direct runs have no injection.
        'injection_present': injected,
    }

def trace_to_graph(log, tool_vocab):
    """Build graph from  log."""
    G = nx.MultiDiGraph()
    initial_traj = set(log.get('initial_trajectory') or [])
    permissions = log.get('tool_permission') or {}
    targets = graph_targets(log)

    G.graph.update({
        'suite_name': log.get('suite_name'),
        'run_mode': log.get('run_mode'),
        'run_id': log.get('run_id'),
        'user_task_id': log.get('user_task_id'),
        'injection_task_id': log.get('injection_task_id'),
        'attack_type': log.get('attack_type'),
        **targets,
    })

    events = []
    t = 0
    for msg in log.get('conversations', []):
        role = msg.get('role')
        if role == 'user':
            content = msg.get('content', '')
            events.append({
                't': t,
                'type': 'call_error' if is_call_error(content) else 'user_query',
                'text': content,
                'content_length': len(str(content)),
                'content_length_bin': content_length_bin(content),
                'has_injection_marker': has_injection_marker(content),
            })
            t += 1
        elif role == 'assistant':
            content = msg.get('content', '')
            thought = extract_tag(content, 'function_thought')
            final_answer = extract_tag(content, 'final_answer')
            if thought or not (msg.get('tool_calls') or []):
                events.append({
                    't': t,
                    'type': 'assistant_reasoning',
                    'text': thought or content,
                    'content_length': len(str(thought or content)),
                    'content_length_bin': content_length_bin(thought or content),
                    'has_injection_marker': has_injection_marker(thought or content),
                })
                t += 1
            for tc in (msg.get('tool_calls') or []):
                tool = tc.get('function')
                args = tc.get('args', {}) or {}
                events.append({
                    't': t, 'type': 'tool_call',
                    'tool': tool,
                    'tool_id': tool_vocab.get(tool, 0),
                    'args': args,
                    'arg_count': len(args),
                    'call_id': tc.get('id'),
                    'in_initial_traj': tool in initial_traj,
                    'privilege': PRIVILEGE_MAP.get(permissions.get(tool, None), 0),
                    'content_length': len(json.dumps(args)),
                    'content_length_bin': content_length_bin(json.dumps(args)),
                    'has_injection_marker': has_injection_marker(args),
                })
                t += 1
            if final_answer:
                events.append({
                    't': t,
                    'type': 'final_answer',
                    'text': final_answer,
                    'content_length': len(str(final_answer)),
                    'content_length_bin': content_length_bin(final_answer),
                    'has_injection_marker': has_injection_marker(final_answer),
                })
                t += 1
        elif role == 'tool':
            tool_call_info = msg.get('tool_call') or {}
            output = str(msg.get('content', ''))
            events.append({
                't': t, 'type': 'tool_return',
                'tool': tool_call_info.get('function'),
                'tool_id': tool_vocab.get(tool_call_info.get('function'), 0),
                'output': output,
                'output_length': len(output),
                'content_length': len(output),
                'content_length_bin': content_length_bin(output),
                'call_id': msg.get('tool_call_id'),
                'has_injection_marker': has_injection_marker(output),
                'has_error': msg.get('error') is not None,
            })
            t += 1
    # Add nodes
    for e in events:
        e['type_id'] = NODE_TYPES.get(e['type'], -1)
        e['step_norm'] = e['t'] / max(len(events) - 1, 1)
        G.add_node(e['t'], **e)
    # Temporal edges
    for i in range(len(events) - 1):
        G.add_edge(events[i]['t'], events[i+1]['t'], kind='temporal')
    
    # Call-return edges
    open_calls = {}
    for e in events:
        if e['type'] == 'tool_call':
            open_calls[e['call_id']] = e['t']
        elif e['type'] == 'tool_return' and e['call_id'] in open_calls:
            G.add_edge(open_calls[e['call_id']], e['t'], kind='call_return')
    
    # Data-flow edges (token overlap)
    for i, later in enumerate(events):
        if later['type'] != 'tool_call':
            continue
        arg_toks = tokens(json.dumps(later.get('args', {})))
        arg_toks -= STOPWORDS
        for earlier in events[:i]:
            if earlier['type'] != 'tool_return':
                continue
            ret_toks = tokens(earlier.get('output', '')) - STOPWORDS
            overlap = arg_toks & ret_toks
            if len(overlap) >= DATA_FLOW_MIN_OVERLAP:
                G.add_edge(earlier['t'], later['t'],
                           kind='data_flow', overlap=list(overlap)[:5])
    
    return G

# Bucket spec: (bucket_id, run_id, run_mode, attacked, defended, glob_relative_to_suite)
# - attacked / defended are booleans the GNN dataset keeps as side-info; they
#   are NOT inputs to the model (we don't want the GNN to memorize "this is a
#   DRIFT-shaped trace" - it must learn structural attack signal).
SLACK_RUN_GROUPS = [
    ('A', 'drift_benign_slack',       'no_attack',                       False, True,  'user_task_*/none/none.json'),
    ('B', 'drift_injected_slack',     'attack_important_instructions',   True,  True,  'user_task_*/important_instructions/injection_task_*.json'),
    ('C', 'undefended_injected_slack','attack_important_instructions',   True,  False, 'user_task_*/important_instructions/injection_task_*.json'),
    ('D', 'undefended_benign_slack',  'no_attack',                       False, False, 'user_task_*/none/none.json'),
]

def collect_slack_poc_paths(base_dir):
    """Collect the four Slack run groups (A, B, C, D) in a stable order.

    A: benign + DRIFT       (defended)
    B: attacked + DRIFT     (defended, blocks most attacks)
    C: attacked + undefended (positive attack-success examples)
    D: benign + undefended  (utility upper bound, false-refusal control)
    """
    base = Path(base_dir)
    paths = []
    for _, run_id, run_mode, _, _, glob_pat in SLACK_RUN_GROUPS:
        suite_root = base / run_mode / run_id / 'slack'
        paths += sorted(suite_root.glob(glob_pat))
    return paths


def _bucket_for_path(path):
    s = str(path)
    for bucket_id, run_id, _, attacked, defended, _ in SLACK_RUN_GROUPS:
        if f'/{run_id}/' in s:
            return bucket_id, attacked, defended
    return None, None, None


# ---- Stage B: frozen content features ----
"""Frozen CONTENT features (Stage B; METHOD LOCK 12c, work-order step 2).

Embeds per-node content with the same frozen MiniLM as tool identity:
  tool_call   -> json.dumps(args, sort_keys=True)   (the proposed action's content)
  tool_return -> output text
  other nodes -> zero row (lock scope: action args + returns only)
Plus a scalar EFFECT-RETURN DIVERGENCE per call_return pair:
  div(return node) = 1 - cos(emb(args), emb(output))
a v1 proxy for Paper-B's declared-vs-actual signal; embeddings are L2-normalized
so cos = dot. Attacker text DOES reach the encoder here - deliberately: the
encoder is frozen and never decodes a token, so there is nothing to instruct
(we claim non-instructability, not adversarial-example robustness; same caveat
as Stage A).

Output: npz keyed g{i}, aligned to the graphs pkl; each entry is (N_i, 385)
float32 = [384 content emb | 1 divergence]. Prefix views index rows <= t
unchanged.
"""

CONTENT_DIM = 385  # 384 emb + 1 divergence


def node_content(d):
    if d.get("type") == "tool_call":
        return json.dumps(d.get("args", {}) or {}, sort_keys=True)
    if d.get("type") == "tool_return":
        return str(d.get("output", ""))
    return ""


def graph_content_matrix(graph, enc):
    nodes = sorted(graph.nodes())
    texts = [node_content(graph.nodes[n]) for n in nodes]
    mat = np.zeros((len(nodes), CONTENT_DIM), dtype=np.float32)
    nonempty = [i for i, t in enumerate(texts) if t]
    if nonempty:
        mat[nonempty, :384] = enc.encode([texts[i] for i in nonempty])
    idx = {n: i for i, n in enumerate(nodes)}
    for u, v, d in graph.edges(data=True):          # divergence on the RETURN node
        if d.get("kind") == "call_return" and u in idx and v in idx:
            a, r = mat[idx[u], :384], mat[idx[v], :384]
            if a.any() and r.any():
                mat[idx[v], 384] = 1.0 - float(a @ r)
    return mat


def main_content():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="data/corpus/trace_graphs.pkl")
    ap.add_argument("--output", default="data/corpus/content_embeddings.npz")
    args = ap.parse_args()
    pay = pickle.loads(Path(args.graphs).read_bytes())
    enc = ToolIdentityEncoder()                      # same frozen model + sha1 cache
    mats = {}
    for i, g in enumerate(pay["graphs"]):
        mats[f"g{i}"] = graph_content_matrix(g, enc)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pay['graphs'])} graphs encoded")
    np.savez_compressed(args.output, **mats)
    n = sum(m.shape[0] for m in mats.values())
    div = np.concatenate([m[:, 384] for m in mats.values()])
    print(f"{len(mats)} graphs, {n} nodes -> {args.output} | "
          f"divergence nonzero {int((div > 0).sum())}, mean {div[div > 0].mean():.3f}")


"""Frozen sentence-encoder as a DETERMINISTIC feature extractor (audit tag F5).

This is not an LLM in the verification path. It maps text -> a fixed vector and
never decodes a token, so there is no output an injected instruction can steer
(non-instructability). It is a differentiable public function, so an adaptive
white-box attacker retains an adversarial-example surface -- we claim
non-instructability, NOT robustness. See project-method-decision-gnn.

Stage A embeds tool identity ("name: description") - no attacker text. Stage B
(content_features.py, METHOD LOCK 12c) deliberately sends action args and tool
returns - attacker-controlled text - through this same frozen encoder: the
non-instructability argument is unchanged (nothing decodes tokens), only the
input scope widened.
"""

# ---- Stage A: frozen tool-identity encoder ----
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384


def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class ToolIdentityEncoder:
    """Deterministic, offline, CPU, frozen. Caches embeddings by sha1(text)."""

    def __init__(self, cache_path="data/corpus/minilm_embedding_cache.npz", use_cache=True):
        self.model_name = MODEL_NAME
        self.dim = EMB_DIM
        self.cache_path = Path(cache_path) if cache_path else None
        self.use_cache = use_cache and cache_path is not None
        self._model = None
        self._cache = {}
        if self.use_cache and self.cache_path.exists():
            self._load_cache()

    def _load_cache(self):
        with np.load(self.cache_path, allow_pickle=False) as z:
            keys = z["keys"]
            mat = z["mat"]
        self._cache = {str(k): mat[i] for i, k in enumerate(keys)}

    def _save_cache(self):
        if not self.use_cache or not self._cache:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._cache.keys())
        mat = np.stack([self._cache[k] for k in keys]).astype(np.float32)
        # write-then-rename so a parallel launcher never reads a half-written file (09-21: two startup retries)
        tmp = self.cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(tmp, keys=np.array(keys), mat=mat)
        os.replace(tmp, self.cache_path)

    def _lazy_model(self):
        if self._model is None:
            torch.manual_seed(0)
            m = SentenceTransformer(self.model_name, device="cpu")
            m.eval()
            self._model = m
        return self._model

    def encode(self, texts):
        """Return (N, 384) float32, L2-normalized. Deterministic; cache-transparent."""
        texts = [str(t) for t in texts]
        missing = [t for t in texts if _sha1(t) not in self._cache]
        if missing:
            uniq = sorted(set(missing))
            model = self._lazy_model()
            with torch.no_grad():
                vecs = model.encode(uniq, convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False)
            for t, v in zip(uniq, vecs.astype(np.float32)):
                self._cache[_sha1(t)] = v
            self._save_cache()
        return np.stack([self._cache[_sha1(t)] for t in texts]).astype(np.float32)

    def build_vocab_matrix(self, tool_vocab, identity_map):
        """(V+1, 384) matrix aligned to tool_id. Row 0 = zeros (unknown/no-tool slot).

        tool_vocab maps name -> id in 1..V (id 0 is reserved for unknown, see
        trace_to_graph.build_tool_vocab). identity_map maps name -> "name: desc".
        Non-tool nodes (user_query, reasoning, ...) index row 0 and get zeros.
        """
        v = max(tool_vocab.values()) if tool_vocab else 0
        mat = np.zeros((v + 1, self.dim), dtype=np.float32)
        names = sorted(tool_vocab, key=lambda n: tool_vocab[n])
        strings = [identity_map.get(n, str(n)) for n in names]
        embs = self.encode(strings) if strings else np.zeros((0, self.dim), np.float32)
        for n, e in zip(names, embs):
            mat[tool_vocab[n]] = e
        return mat


def build_tool_vocab_matrix(tool_vocab, roots=None, cache_path="data/corpus/minilm_embedding_cache.npz"):
    """Convenience: descriptions -> encoder -> (V+1, 384) matrix + coverage info."""
    identity_map, coverage = build_map(tool_vocab, roots or DEFAULT_ROOTS)
    enc = ToolIdentityEncoder(cache_path=cache_path)
    mat = enc.build_vocab_matrix(tool_vocab, identity_map)
    return mat, identity_map, coverage


if __name__ == "__main__":
    CMDS = {"content": main_content}
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        sys.exit(f"usage: encoders.py {{{'|'.join(CMDS)}}} [args] "
                 "(content = Stage-B node-content npz, tools = Stage-A tool-emb npz)")
    cmd = sys.argv.pop(1)   
    CMDS[cmd]()