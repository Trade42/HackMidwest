"""The second head: one probability per fault mechanism of the decoders' problem, for this shot.

The trunk (neural/model.py, a trained X decoder) reads the shot's 1,728 detector bits and keeps
a vector of 128 numbers on every check in every round (`features`). The head turns those
vectors into one logit per column of the kit's basis problem H (8,784 for X memory): "did this
fault mechanism fire in this shot?".

For one column k, in plain words:
1. Its detectors. Column k flips 2 to 6 detectors of the basis; each sits on one check in one
   round, where the trunk keeps a vector. The head shrinks every such vector to 32 numbers.
2. Two averages. It averages those vectors over the column's detectors that FIRED in this shot,
   and separately over the ones that did not. A fault that fired lights all its detectors
   unless another fault cancels one, so "which of my detectors are lit" is the main evidence;
   the trunk's vectors add what the network learned about the neighbourhood.
3. What kind of mechanism it is. The 8,784 columns fall into 122 shapes, each repeated at the
   72 positions of the torus (a shape is the column's pattern of detectors, up to a shift).
   The head learns one small vector per shape, and one per (detectors, of which lit) count.
4. A small network shared by all columns turns all of that into a correction to the kit's own
   log-odds: logit_k = log(prior_k / (1 - prior_k)) + correction_k.
   The last layer starts at zero, so an untrained head gives exactly the kit's priors.

Everything the head reads is the trunk's vectors, the shot's basis detector bits and the kit's
fixed priors: no truth. Every map is the same at every torus position, so shifting the
syndrome on the torus shifts the head's output in the same way (tested).

Why these choices:
- Averages, not one weight per detector slot: the order of a column's detectors is not the
  same across a shape's 72 copies unless it is rebuilt from the shift, and the lit/dark split
  already carries what a slot would. Cheap: the averages are one sparse product with H.
- A correction to the kit's log-odds, not a fresh logit: almost every column is off in every
  shot (about 47 of 8,784 fire at p = 0.004). Starting from the prior handles that imbalance,
  so the loss can be the plain binary cross-entropy, a proper score: the head's probabilities
  are then calibrated, which is what BP needs from a prior.
- The kit's log-odds is also an input, so one head serves several noise strengths.
"""
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

from neural.graph import build_graph


def column_shapes(H, rounds, checks, ell, m):
    """[columns] int: the shape of every column (its detectors up to a torus shift), numbered 0, 1, ...

    rounds[i], checks[i]: round and check (0 .. n2 - 1, x = check // m, y = check % m) of row i of H.
    """
    H = sp.csc_matrix(H)
    shapes, numbers = {}, np.zeros(H.shape[1], np.int64)
    for k in range(H.shape[1]):
        rows = H.indices[H.indptr[k]:H.indptr[k + 1]]
        best = None
        for dx in range(ell):
            for dy in range(m):
                moved = tuple(sorted((int(rounds[i]), int(((checks[i] // m + dx) % ell) * m + (checks[i] % m + dy) % m))
                                     for i in rows))
                if best is None or moved < best:
                    best = moved
        numbers[k] = shapes.setdefault(best, len(shapes))
    return numbers


class PriorHead(nn.Module):
    """Trunk features and basis detector bits -> one logit per column of the basis problem."""

    def __init__(self, H, basis_rows, code="gross", basis="X", rounds=None, trunk_width=128, project=32, hidden=64,
                 shape_width=16):
        super().__init__()
        graph = build_graph(code, basis, rounds, "kit")
        H = sp.csc_matrix(H)
        basis_rows = np.asarray(basis_rows)
        row_round, row_check = graph.detector_round[basis_rows], graph.detector_check[basis_rows]
        basis_type = "XZ".index(basis)
        if not (row_check // graph.n2 == basis_type).all():
            raise ValueError("a row of H is not a detector on a check of the memory basis")
        checks = 2 * graph.n2
        weights = np.diff(H.indptr)
        if weights.min() < 1 or weights.max() > 6:
            raise ValueError("every column must flip 1 to 6 detectors")
        shapes = column_shapes(H, row_round, row_check % graph.n2, graph.ell, graph.m)
        self.config = {"code": code, "basis": basis, "rounds": graph.rounds, "trunk_width": trunk_width,
                       "project": project, "hidden": hidden, "shape_width": shape_width,
                       "columns": int(H.shape[1]), "rows": int(H.shape[0]), "shapes": int(shapes.max() + 1)}
        incidence = H.T.tocoo()                                                   # [columns, rows]
        self.register_buffer("incidence_index", torch.from_numpy(np.vstack([incidence.row, incidence.col]).astype(np.int64)),
                             persistent=False)
        self.register_buffer("node_of_row", torch.from_numpy((row_round * checks + row_check).astype(np.int64)),
                             persistent=False)
        self.register_buffer("weight", torch.from_numpy(weights.astype(np.int64)), persistent=False)
        self.register_buffer("shape", torch.from_numpy(shapes), persistent=False)
        self.num_columns, self.num_rows = int(H.shape[1]), int(H.shape[0])

        self.norm = nn.LayerNorm(trunk_width)
        self.project = nn.Linear(trunk_width, project)
        self.shape_embedding = nn.Embedding(int(shapes.max()) + 1, shape_width)
        self.count_embedding = nn.Embedding(7 * 7, shape_width)                  # (detectors, of which lit)
        self.mlp = nn.Sequential(nn.Linear(2 * project + 2 * shape_width + 1, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _incidence(self, device, dtype):
        """The sparse [columns, rows] 0/1 matrix H transposed."""
        values = torch.ones(self.incidence_index.shape[1], device=device, dtype=dtype)
        return torch.sparse_coo_tensor(self.incidence_index, values, (self.num_columns, self.num_rows),
                                       check_invariants=False).coalesce()      # the index comes from the kit's H

    def _per_column(self, incidence, per_row):
        """[shots, rows, c] -> [shots, columns, c]: the sum over each column's rows."""
        shots, rows, width = per_row.shape
        flat = per_row.permute(1, 0, 2).reshape(rows, shots * width)
        return torch.sparse.mm(incidence, flat).reshape(self.num_columns, shots, width).permute(1, 0, 2)

    def forward(self, features, syndrome, prior_logit):
        """features [shots, rounds, checks, width] (the trunk's), syndrome [shots, rows] 0/1 (the basis bits, H's row
        order), prior_logit [columns] or [shots, columns] (the kit's log-odds). Returns logits [shots, columns]."""
        shots = features.shape[0]
        dtype = self.project.weight.dtype                    # float32, also under the trunk's mixed precision
        with torch.autocast(features.device.type, enabled=False):
            x = features.to(dtype).reshape(shots, -1, features.shape[-1])[:, self.node_of_row]  # [shots, rows, width]
            u = self.project(F.silu(self.norm(x)))                                            # [shots, rows, project]
            lit = syndrome.to(dtype)[..., None]
            incidence = self._incidence(features.device, dtype)
            sums = self._per_column(incidence, torch.cat([u * lit, u * (1 - lit), lit], dim=-1))
            project = u.shape[-1]
            lit_count = sums[..., -1].round().long()                                           # [shots, columns]
            weight = self.weight[None].expand_as(lit_count)
            lit_mean = sums[..., :project] / lit_count.clamp(min=1)[..., None]
            dark_mean = sums[..., project:2 * project] / (weight - lit_count).clamp(min=1)[..., None]
            prior_logit = prior_logit.to(dtype)
            if prior_logit.dim() == 1:
                prior_logit = prior_logit[None].expand(shots, -1)
            inputs = torch.cat([lit_mean, dark_mean,
                                self.shape_embedding(self.shape)[None].expand(shots, -1, -1),
                                self.count_embedding(weight * 7 + lit_count),
                                (prior_logit + 5.5)[..., None]], dim=-1)
            return prior_logit + self.mlp(inputs).squeeze(-1)


def prior_logit(priors):
    """The kit's priors as log-odds, a float32 tensor."""
    priors = np.asarray(priors, np.float64)
    return torch.from_numpy(np.log(priors / (1 - priors)).astype(np.float32))


def build_head(problem_H, basis_rows, trunk_config, **sizes):
    """A head that fits a trunk of neural.train (its config dict)."""
    return PriorHead(problem_H, basis_rows, trunk_config["code"], trunk_config["basis"], trunk_config["rounds"],
                     trunk_config["width"], **sizes)
