"""Compromise-scoring models: the R-GCN used by CAVAL and the GATv2, GCN, LSTM and MLP baselines of Table 8.

All three models share the (in_dim, hidden, dropout, num_classes=2) signature
and produce two logits tensors: (attack_logits, utility_logits) of shape
(num_graphs, 2).  This lets the training loop stay model-agnostic.

- TraceGCN: 2-layer GCN + global mean pool. Captures graph structure
  (temporal + call_return + data_flow edges).
- MLPPool: per-node MLP, mean-pool over nodes, 2-head classifier. Ignores
  edges entirely. Strongest "graph-free" baseline given identical features.
- SeqLSTM: BiLSTM over time-ordered nodes, mean-pool over valid positions.
  Captures temporal order but not data-flow / call-return relations.

The three together let us isolate what graph structure contributes on top of
node features alone (MLPPool) and on top of temporal sequence (SeqLSTM).
"""

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATv2Conv, RGCNConv, global_mean_pool
from torch_geometric.utils import to_dense_batch
import re

def _readout(x, batch, mode):
    """mean: global mean pool (the original readout). mean_action: mean pool
    concatenated with the embedding of the proposed action a_t, which is the
    LAST node of every prefix graph (a prefix is nodes 0..t and node t is the
    scored tool call; the live gate selects the same node). Gives the head a
    view of the action itself instead of averaging it away."""
    pooled = global_mean_pool(x, batch.batch)
    if mode == "mean":
        return pooled
    last = batch.ptr[1:] - 1
    return torch.cat([pooled, x[last]], dim=-1)


def _head_dim(hidden, readout):
    return hidden * (2 if readout == "mean_action" else 1)


class _TwoHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attack_head = nn.Linear(hidden, 2)
        self.utility_head = nn.Linear(hidden, 2)

    def forward(self, h):
        return self.attack_head(h), self.utility_head(h)


class TraceGCN(nn.Module):
    name = "gcn"

    def __init__(self, in_dim, hidden=32, dropout=0.3, layers=2, readout="mean"):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.convs = nn.ModuleList(GCNConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout, self.readout = dropout, readout
        self.heads = _TwoHead(_head_dim(hidden, readout))

    def forward(self, batch):
        x = batch.x
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, batch.edge_index))
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.heads(_readout(x, batch, self.readout))


class MLPPool(nn.Module):
    """Per-node MLP, then mean-pool. Pure node-feature baseline; ignores edges."""
    name = "mlp_pool"

    def __init__(self, in_dim, hidden=32, dropout=0.3, layers=2, readout="mean"):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.fcs = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout, self.readout = dropout, readout
        self.heads = _TwoHead(_head_dim(hidden, readout))

    def forward(self, batch):
        x = batch.x
        for i, fc in enumerate(self.fcs):
            x = F.relu(fc(x))
            if i < len(self.fcs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.heads(_readout(x, batch, self.readout))


class SeqLSTM(nn.Module):
    """BiLSTM over time-ordered node features, mean-pool over valid positions.

    Nodes are inserted in `trace_to_graph` keyed by their integer timestep `t`,
    so a per-graph sort by node id (default in `graph_to_data`) yields
    chronological order. `to_dense_batch` preserves that per-graph order while
    padding to the max length in the batch.
    """
    name = "seq_lstm"

    def __init__(self, in_dim, hidden=32, dropout=0.3, layers=1, readout="mean"):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden // 2,
            num_layers=max(1, layers - 1),   # layers counts like the GNNs: 2 -> one LSTM layer
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 2 else 0.0,
        )
        self.dropout, self.readout = dropout, readout
        self.heads = _TwoHead(_head_dim(hidden, readout))

    def forward(self, batch):
        x, mask = to_dense_batch(batch.x, batch.batch)
        h, _ = self.lstm(x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        if self.readout == "mean":
            return self.heads(pooled)
        last = mask.sum(dim=1) - 1           # last valid position = node t = a_t
        h_last = h[torch.arange(h.size(0), device=h.device), last]
        return self.heads(torch.cat([pooled, h_last], dim=-1))

class TraceGATv2(nn.Module):
    name = "gatv2"

    def __init__(self, in_dim, hidden=32, dropout=0.3, heads=8, layers=2, readout="mean"):
        super().__init__()
        self.dropout, self.readout = dropout, readout
        h1 = int(max(heads/4,8))
        convs = [GATv2Conv(in_dim, hidden, heads=h1, dropout=dropout)]
        prev = h1 * hidden
        for _ in range(layers - 1):
            convs.append(GATv2Conv(prev, hidden, heads=1, concat=True, dropout=dropout))
            prev = hidden
        self.convs = nn.ModuleList(convs)
        self.heads = _TwoHead(_head_dim(hidden, readout))

    def forward(self, batch):
        x = batch.x
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, batch.edge_index))
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.heads(_readout(x, batch, self.readout))


class TraceRGCN(nn.Module):
    """Typed-edge R-GCN: a separate learned transform per relation
    (temporal / call_return / data_flow x forward/reverse = 6). This is the model
    the 'typed trace graph' thesis is actually about; the plain GCN discards types."""
    name = "rgcn"

    def __init__(self, in_dim, hidden=32, dropout=0.3, num_relations=6, layers=2, readout="mean"):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.convs = nn.ModuleList(RGCNConv(a, b, num_relations) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout, self.readout = dropout, readout
        self.heads = _TwoHead(_head_dim(hidden, readout))

    def forward(self, batch):
        x = batch.x
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, batch.edge_index, batch.edge_type))
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.heads(_readout(x, batch, self.readout))


MODELS = {
    TraceGCN.name: TraceGCN,
    MLPPool.name: MLPPool,
    SeqLSTM.name: SeqLSTM,
    TraceGATv2.name: TraceGATv2,
    TraceRGCN.name: TraceRGCN,
}


class EmbedProject(nn.Module):
    """Learn-project the trailing emb_dim node features to proj_dim, then run the
    base model. The projection is a matrix -> no instruction-following surface."""

    def __init__(self, base_cls, struct_dim, emb_dim, proj_dim, hidden, dropout,
                 layers=2, readout="mean"):
        super().__init__()
        self.struct_dim = struct_dim
        self.proj = nn.Linear(emb_dim, proj_dim)
        self.base = base_cls(in_dim=struct_dim + proj_dim, hidden=hidden, dropout=dropout,
                             layers=layers, readout=readout)

    def forward(self, batch):
        s, e = batch.x[:, :self.struct_dim], batch.x[:, self.struct_dim:]
        batch.x = torch.cat([s, self.proj(e)], dim=1)   # fresh batch per loader step
        return self.base(batch)



def load_state_compat(model, state):
    """Load a state_dict whose layers use the names conv1/conv2 and fc1/fc2 (as the released checkpoints do) into
    the ModuleList layout convs/fcs."""
    remapped = {re.sub(r"\.(conv|fc)(\d+)\.", lambda m: f".{m.group(1)}s.{int(m.group(2)) - 1}.", k): v
                for k, v in state.items()}
    model.load_state_dict(remapped)
    return model

def build_model(name, in_dim, hidden=32, dropout=0.3, emb_dim=0, proj_dim=32,
                layers=2, readout="mean"):
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODELS)}")
    if emb_dim:
        return EmbedProject(MODELS[name], in_dim - emb_dim, emb_dim, proj_dim, hidden, dropout,
                            layers=layers, readout=readout)
    return MODELS[name](in_dim=in_dim, hidden=hidden, dropout=dropout, layers=layers, readout=readout)


# ---- features, splits, training loop ----
NUM_NODE_TYPES = 6
NUM_PRIVILEGES = 4

# Fixed before deployment: a step's position is normalized against this
# constant, never against a trace's own final length (which would leak
# information not available when G_t is constructed pre-execution). Chosen
# from the corpus's step-count distribution (median 19, p90 31, p99 43, max
# 55), with headroom. graph['step_norm'] (trace_to_graph.py) is NOT used
# here for exactly this reason -- it divides by the completed trace's final
# length and is not available at decision time.
MAX_TRACE_HORIZON = 60

# Edge relations for the typed R-GCN: 3 kinds x {forward, reverse} = 6.
KIND_ID = {"temporal": 0, "call_return": 1, "data_flow": 2}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot(index, size):
    out = [0.0] * size
    if 0 <= int(index) < size:
        out[int(index)] = 1.0
    return out


# Leaky features (OFF by default, ablation-only): in_initial_traj (DRIFT's
# reference trajectory), has_injection_marker (the <INFORMATION> template
# delimiter, ~equivalent to the label), has_error. They let the model shortcut
# the label instead of learning execution structure.
def node_features(attrs, tool_vocab_size, drop_leaky=True):
    tool_id = float(attrs.get("tool_id") or 0) / max(tool_vocab_size, 1)
    # Causal position: t / a FIXED horizon, never t / this trace's own final
    # length. Deliberately ignores the stored (non-causal) step_norm attribute.
    step_pos = min(float(attrs.get("t", 0)) / MAX_TRACE_HORIZON, 1.0)
    base = (
        one_hot(attrs.get("type_id", -1), NUM_NODE_TYPES)
        + one_hot(attrs.get("privilege", 0), NUM_PRIVILEGES)
        + [
            tool_id,
            float(attrs.get("arg_count", 0)),
            float(attrs.get("content_length_bin", 0)),
            float(attrs.get("output_length", 0) > 0),
            step_pos,
        ]
    )
    if drop_leaky:
        return base
    return base + [
        float(attrs.get("in_initial_traj", False)),
        float(attrs.get("has_injection_marker", False)),
        float(attrs.get("has_error", False)),
    ]


def graph_to_data(graph, record, tool_vocab_size, drop_leaky=True, tool_emb=None,
                  content_emb=None, drop_kinds=()):
    # Feature layout: [struct (+divergence) | tool emb 384 | content emb 384].
    # The divergence scalar lives in the struct block; EmbedProject projects the
    # trailing emb block(s) with one linear, so emb_dim = 384*(tool)+384*(content).
    nodes = sorted(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    feats = []
    for i, n in enumerate(nodes):
        a = graph.nodes[n]
        f = node_features(a, tool_vocab_size, drop_leaky=drop_leaky)
        if content_emb is not None:
            f = f + [float(content_emb[i, 384])]     # effect-return divergence
        if tool_emb is not None:
            f = f + tool_emb[int(a.get("tool_id") or 0)].tolist()  # zero row for non-tool nodes
        if content_emb is not None:
            f = f + content_emb[i, :384].tolist()    # frozen content emb (Stage B)
        feats.append(f)
    x = torch.tensor(feats, dtype=torch.float)

    edges, etypes = [], []
    for u, v, d in graph.edges(data=True):
        if d.get("kind") in drop_kinds:              # relation-ablation switch
            continue
        if u in node_to_idx and v in node_to_idx:
            k = KIND_ID.get(d.get("kind"), 0)
            edges.append((node_to_idx[u], node_to_idx[v])); etypes.append(k)      # forward
            edges.append((node_to_idx[v], node_to_idx[u])); etypes.append(k + 3)  # reverse = distinct relation
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(etypes, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)

    # y_label = the PRIMARY detection target: COMPROMISED (attack_success is
    # True). Negatives = benign OR attacked-but-ignored. Label-agnostic name so
    # a future label pivot touches only this line.
    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y_label=torch.tensor(int(record.get("attack_success") is True), dtype=torch.long),
        y_utility=torch.tensor(int(record["utility_success"]), dtype=torch.long),
    )


def class_weights(labels, device):
    counts = np.bincount(labels, minlength=2).astype(float)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float, device=device)


def run_epoch(model, loader, optimizer, label_weights, utility_weights, device,
              aux_weight=1.0):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        label_logits, utility_logits = model(batch)
        loss = F.cross_entropy(label_logits, batch.y_label, weight=label_weights)
        if aux_weight:
            loss = loss + aux_weight * F.cross_entropy(
                utility_logits, batch.y_utility, weight=utility_weights)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * batch.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    label_logits_all, utility_logits_all = [], []
    label_y, utility_y = [], []
    for batch in loader:
        batch = batch.to(device)
        label_logits, utility_logits = model(batch)
        label_logits_all.append(label_logits.cpu())
        utility_logits_all.append(utility_logits.cpu())
        label_y.extend(batch.y_label.cpu().tolist())
        utility_y.extend(batch.y_utility.cpu().tolist())

    label_logits = torch.cat(label_logits_all)
    utility_logits = torch.cat(utility_logits_all)
    label_pred = label_logits.argmax(dim=-1).numpy()
    utility_pred = utility_logits.argmax(dim=-1).numpy()

    return {
        "label_acc": accuracy_score(label_y, label_pred),
        "label_f1": f1_score(label_y, label_pred, zero_division=0),
        "utility_acc": accuracy_score(utility_y, utility_pred),
        "utility_f1": f1_score(utility_y, utility_pred, zero_division=0),
        "label_y": np.array(label_y),
        "utility_y": np.array(utility_y),
        "label_logits": label_logits.numpy(),
        "utility_logits": utility_logits.numpy(),
    }


def grouped_four_way_split(labels, groups, seed,
                           val_frac=0.15, cal_frac=0.175, test_frac=0.175,
                           holdout_mask=None):
    """Split indices into train / val / cal / test, GROUPED by `groups`.

    Grouping (e.g. by user_task_id) keeps the four A/B/C/D bucket-siblings of one
    task on the same side, so a near-duplicate graph cannot straddle train/test
    (the D3 leakage fix). Model selection uses `val`; conformal calibration uses
    `cal`; both are disjoint from `test`, which restores the exchangeability the
    conformal guarantee assumes (the old code selected the best epoch on the
    calibration set, biasing qhat and undercovering).

    If `holdout_mask` (bool array over all indices) is given, those rows are
    forced entirely into `test` and removed from the fit pool -- this is how a
    leave-one-attack-type-out fold is built (step 5). Grouping is by `groups`;
    label balance is not stratified because the group constraint takes priority.
    """
    idx = np.arange(len(labels))
    forced_test = idx[holdout_mask] if holdout_mask is not None else np.array([], dtype=int)
    pool = idx[~np.isin(idx, forced_test)]

    def gsplit(pool_idx, frac):
        gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        a, b = next(gss.split(pool_idx, labels[pool_idx], groups[pool_idx]))
        return pool_idx[a], pool_idx[b]

    rest, test_g = gsplit(pool, test_frac)
    test_idx = np.concatenate([test_g, forced_test])
    rest, cal_idx = gsplit(rest, cal_frac / (1.0 - test_frac))
    train_idx, val_idx = gsplit(rest, val_frac / (1.0 - test_frac - cal_frac))
    return train_idx, val_idx, cal_idx, test_idx


def train_one(
    data,
    train_idx,
    val_idx,
    cal_idx,
    test_idx,
    model_name,
    hidden,
    dropout,
    epochs,
    batch_size,
    lr,
    weight_decay,
    seed,
    device,
    emb_dim=0,
    aux_weight=1.0,
    layers=2,
    readout="mean",
):
    """Train a single model; select on val, report cal + test logits/metrics.

    aux_weight scales the auxiliary utility-head loss (0 in the paper's model; the ablation uses 1. At 0 it is disabled
    it; epoch selection then uses label F1 alone, since selecting on a head
    that receives no gradient would be noise)."""
    set_seed(seed)
    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    cal_data = [data[i] for i in cal_idx]
    test_data = [data[i] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    cal_loader = DataLoader(cal_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    in_dim = data[0].x.shape[1]
    model = build_model(model_name, in_dim=in_dim, hidden=hidden, dropout=dropout, emb_dim=emb_dim,
                        layers=layers, readout=readout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    labels = np.array([int(d.y_label) for d in data])
    utility_labels = np.array([int(d.y_utility) for d in data])
    label_weights = class_weights(labels[train_idx], device)
    utility_weights = class_weights(utility_labels[train_idx], device)

    best_state, best_score = None, -1.0
    history = []
    for epoch in range(1, epochs + 1):
        loss = run_epoch(model, train_loader, optimizer, label_weights, utility_weights,
                         device, aux_weight=aux_weight)
        val_metrics = evaluate(model, val_loader, device)   # SELECT on val, never cal/test
        score = val_metrics["label_f1"] + (val_metrics["utility_f1"] if aux_weight else 0.0)
        history.append({
            "epoch": epoch,
            "loss": loss,
            "val_label_f1": val_metrics["label_f1"],
            "val_utility_f1": val_metrics["utility_f1"],
        })
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    val_metrics = evaluate(model, val_loader, device)
    cal_metrics = evaluate(model, cal_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    return {
        "model_name": model_name,
        "seed": seed,
        "in_dim": in_dim,
        "hidden": hidden,
        "layers": layers,
        "readout": readout,
        "best_state": best_state,
        "history": history,
        "val": val_metrics,
        "cal": cal_metrics,
        "test": test_metrics,
        "splits": {
            "train": len(train_data),
            "val": len(val_data),
            "cal": len(cal_data),
            "test": len(test_data),
        },
        "device": str(device),
    }
