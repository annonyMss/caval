"""Build a {tool_name -> "name: description"} map from the AgentDojo suite source.

The description is the first line of each tool function's docstring. We AST-parse
the suite `tools/` modules rather than importing the `agentdojo` package, so this
is immune to the package-name collision between stock AgentDojo and the AgentDyn
fork (both install as `agentdojo`).

The resulting identity string is what the frozen encoder embeds (Stage A). No tool
RETURN content and no attacker text is ever included here -- only the tool's own
name and its author-written docstring.
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import argparse
import ast
import json
from pathlib import Path
import pickle

DEFAULT_ROOTS = [
    "data/benchmarks/AgentDyn/src/agentdojo/default_suites",
]


def _first_doc_line(node):
    doc = ast.get_docstring(node) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def collect_tool_docs(roots=DEFAULT_ROOTS):
    """Scan `.../tools/*.py` modules for top-level function docstrings.

    Restricting to directories literally named `tools` keeps helper functions
    (parsers, formatters) from polluting the map. When the same tool name appears
    across suite versions, keep the LONGEST description (most informative).
    """
    docs = {}
    seen_files = 0
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for py in root.rglob("*.py"):
            if py.parent.name != "tools":
                continue
            try:
                tree = ast.parse(py.read_text())
            except (SyntaxError, UnicodeDecodeError):
                continue
            seen_files += 1
            for n in tree.body:  # top-level defs only
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    desc = _first_doc_line(n)
                    if desc and len(desc) > len(docs.get(n.name, "")):
                        docs[n.name] = desc
    return docs, seen_files


def identity_string(name, docs):
    """The string the encoder embeds. Falls back to the bare name if undocumented."""
    desc = docs.get(name)
    return f"{name}: {desc}" if desc else str(name)


def build_map(tool_vocab, roots=DEFAULT_ROOTS):
    """Return {tool_name: identity_string} for every name in tool_vocab, plus coverage."""
    docs, seen_files = collect_tool_docs(roots)
    out = {name: identity_string(name, docs) for name in tool_vocab}
    covered = sum(1 for name in tool_vocab if name in docs)
    coverage = {
        "n_tools": len(tool_vocab),
        "n_documented": covered,
        "n_undocumented": len(tool_vocab) - covered,
        "modules_scanned": seen_files,
        "docs_available": len(docs),
    }
    return out, coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="data/corpus/trace_graphs.pkl",
                    help="Pickle whose tool_vocab drives which tools to map.")
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    ap.add_argument("--output", default="data/corpus/tool_descriptions.json")
    args = ap.parse_args()

    vocab = pickle.loads(Path(args.graphs).read_bytes())["tool_vocab"]
    mapping, coverage = build_map(vocab, args.roots)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"identity_strings": mapping, "coverage": coverage}, indent=2, sort_keys=True))
    print(json.dumps(coverage, indent=2))
    undocumented = sorted(n for n in vocab if mapping.get(n, n) == n)
    if undocumented:
        print("UNDOCUMENTED (name-only fallback):", undocumented)


if __name__ == "__main__":
    main()
