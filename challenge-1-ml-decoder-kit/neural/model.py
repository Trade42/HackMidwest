"""A convolutional decoder that follows the code's own connections, after arXiv:2604.08358 ("Cascade").

Input: the detection events of one memory experiment. Output: one logit per logical
observable; a positive logit predicts that the observable flipped.

The network keeps a vector of `width` numbers on every check, in every round.
1. Embedding. Every detection event (0 or 1) becomes a learned vector at its check and round.
   A place without a detector gets a third learned vector.
2. `depth` residual blocks. A block narrows the vectors to width / 4, lets every data qubit
   collect from its six checks, lets every check collect from its six data qubits, and widens
   the vectors again. A collecting step (`Hop`) has one weight matrix per neighbour and per time
   offset, used at every position of the torus: a convolution on the code's graph.
3. Readout. One more step collects on the data qubits. The average over the data qubits that a
   logical observable reads, and over time, goes through one small network that all observables
   share.

All tensors keep the channel last: [shots, rounds, nodes, channels].
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from neural.graph import build_graph, check_parity_matrix

EVENT_CODES = 3   # per check type: no event, event, no detector at this place


class Norm(nn.BatchNorm1d):
    """Batch normalisation over every axis but the last."""

    def forward(self, x):
        return super().forward(x.reshape(-1, x.shape[-1])).reshape(x.shape)


class Gather(torch.autograd.Function):
    """index_select along the node axis, with a backward pass that also only gathers.

    PyTorch's own backward pass of index_select scatters with atomic additions, which are slow
    on a GPU in half precision. Here every source is picked the same number of times, so its
    gradient is the sum of that many known places of the incoming gradient: a gather and a sum.
    """

    @staticmethod
    def forward(ctx, x, index, places):
        ctx.places = places            # [sources, copies]: where each source went in the result
        return x.index_select(-2, index)

    @staticmethod
    def backward(ctx, gradient):
        places = ctx.places
        picked = gradient.index_select(-2, places.reshape(-1))
        return picked.reshape(*gradient.shape[:-2], *places.shape, gradient.shape[-1]).sum(dim=-2), None, None


class Hop(nn.Module):
    """One step of message passing between checks and data qubits.

    Every target node adds up a linear map of each of its neighbours. The maps differ by
    neighbour slot, by time offset and by the target's block (X or Z check, left or right data
    qubit), and are the same at every position of the torus.

    There are two time offsets. The "same" maps act between a data qubit in cycle c and a check
    in round c, the "after" maps between cycle c and round c + 1. One Linear layer per block of
    targets reads the six neighbours side by side and returns both results side by side, so it
    holds 6 x 2 weight matrices.
    """

    def __init__(self, neighbours, width_in, width_out):
        super().__init__()
        targets, slots = neighbours.shape
        neighbours = neighbours.reshape(2, targets // 2 * slots)     # one row per block of targets
        sources = int(neighbours.max()) + 1
        if any(len(set(torch.bincount(row, minlength=sources).tolist())) != 1 for row in neighbours):
            raise ValueError("within a block of targets, every source must be a neighbour equally often")
        # places[block][source]: where that source appears in the block's row of neighbours.
        places = torch.stack([torch.argsort(row, stable=True).reshape(sources, -1) for row in neighbours])
        self.register_buffer("neighbours", neighbours, persistent=False)
        self.register_buffer("places", places, persistent=False)
        self.targets_per_block = targets // 2
        self.linears = nn.ModuleList(nn.Linear(slots * width_in, 2 * width_out) for _ in range(2))

    def messages(self, x):
        """[..., sources, c] -> (same, after), each [..., targets, c']."""
        results = []
        for neighbours, places, linear in zip(self.neighbours, self.places, self.linears, strict=True):
            side_by_side = Gather.apply(x, neighbours, places).reshape(*x.shape[:-2], self.targets_per_block, -1)
            results.append(linear(side_by_side))
        return torch.cat(results, dim=-2).chunk(2, dim=-1)

    def checks_to_data(self, checks):
        """[shots, R + 1, checks, c] -> [shots, R, data, c']: cycle c collects from rounds c and c + 1."""
        same, after = self.messages(checks)
        return same[:, :-1] + after[:, 1:]

    def data_to_checks(self, data):
        """[shots, R, data, c] -> [shots, R + 1, checks, c']: round r collects from cycles r and r - 1."""
        same, after = self.messages(data)
        # Round 0 has no earlier cycle and round R has no cycle of its own.
        return torch.cat([same[:, :1], same[:, 1:] + after[:, :-1], after[:, -1:]], dim=1)


class Block(nn.Module):
    """A bottleneck residual block. Normalisation and SiLU come before every linear step."""

    def __init__(self, check_neighbours, data_neighbours, width, bottleneck, branch_scale, scale_skip):
        super().__init__()
        narrow = width // bottleneck
        self.branch_scale, self.scale_skip = branch_scale, scale_skip
        self.norm_in, self.down = Norm(width), nn.Linear(width, narrow)
        self.norm_checks, self.to_data = Norm(narrow), Hop(data_neighbours, narrow, narrow)
        self.norm_data, self.to_checks = Norm(narrow), Hop(check_neighbours, narrow, narrow)
        self.norm_out, self.up = Norm(narrow), nn.Linear(narrow, width)

    def forward(self, checks):
        x = self.down(F.silu(self.norm_in(checks)))
        x = self.to_data.checks_to_data(F.silu(self.norm_checks(x)))
        x = self.to_checks.data_to_checks(F.silu(self.norm_data(x)))
        x = F.silu(self.norm_out(x))
        if self.scale_skip:
            x = self.up(x)
            return self.branch_scale * checks.to(x.dtype) + x
        # x + f(x) / sqrt(2 depth): the factor multiplies the small weight matrix, not the large result.
        x = F.linear(x, self.up.weight * self.branch_scale, self.up.bias * self.branch_scale)
        return checks.to(x.dtype) + x   # under mixed precision the whole stream is then half precision


class CascadeDecoder(nn.Module):
    """Detection events -> one logit per logical observable.

    detectors="all": the input is every detector of the circuit, in circuit order (1,728 for
    the Gross code). detectors="basis": the input is the detectors on checks of the memory basis,
    in the order of circuit.detector_ids (936). The network is the same in both cases: with
    "basis", the other checks carry the "no detector" vector in every round.

    residual="branch" computes x + f(x) / sqrt(2 depth). residual="skip" computes
    x / sqrt(2 depth) + f(x), the other reading of the paper's block figure.

    logicals="kit": the 12 logits are the kit's 12 observables. logicals="light": they are 12
    lighter logical operators of the same basis (graph.py); `kit_bits` turns their bits into
    the kit's.
    """

    def __init__(self, code="gross", basis="X", rounds=None, detectors="all", width=128, depth=8,
                 bottleneck=4, residual="branch", logicals="kit"):
        super().__init__()
        if detectors not in ("all", "basis") or residual not in ("branch", "skip"):
            raise ValueError("detectors is 'all' or 'basis'; residual is 'branch' or 'skip'")
        if width % bottleneck:
            raise ValueError("width must be a multiple of the bottleneck factor")
        graph = build_graph(code, basis, rounds, logicals)
        self.config = {"code": code, "basis": basis, "rounds": graph.rounds, "detectors": detectors,
                       "width": width, "depth": depth, "bottleneck": bottleneck, "residual": residual,
                       "logicals": logicals}
        self.rounds, self.checks = graph.rounds + 1, 2 * graph.n2

        inputs = torch.arange(graph.num_detectors) if detectors == "all" else torch.from_numpy(graph.basis_detectors)
        place = torch.from_numpy(graph.detector_round * self.checks + graph.detector_check)
        check_type = torch.arange(self.checks) // graph.n2
        self.num_inputs = len(inputs)
        self.register_buffer("input_place", place[inputs], persistent=False)                       # where every input bit goes
        self.register_buffer("input_code", EVENT_CODES * check_type[place[inputs] % self.checks], persistent=False)
        self.register_buffer("empty_code", (EVENT_CODES * check_type + 2).repeat(self.rounds), persistent=False)
        support = torch.from_numpy(graph.observable_support).float()
        self.register_buffer("pool", support / support.sum(dim=1, keepdim=True), persistent=False)   # [observables, data]
        # kit bits = kit_from_model x model bits + kit_from_inputs x detector bits (mod 2). Small exact counts, kept as floats.
        from_inputs = (graph.kit_from_checks.astype(np.int64) @ check_parity_matrix(graph, inputs.numpy())) % 2
        self.register_buffer("kit_from_model", torch.from_numpy(graph.kit_from_model).float(), persistent=False)
        self.register_buffer("kit_from_inputs", torch.from_numpy(from_inputs).float(), persistent=False)

        check_neighbours = torch.from_numpy(graph.check_neighbours)
        data_neighbours = torch.from_numpy(graph.data_neighbours)
        self.embedding = nn.Embedding(2 * EVENT_CODES, width)
        self.blocks = nn.ModuleList(
            Block(check_neighbours, data_neighbours, width, bottleneck, (2 * depth) ** -0.5, residual == "skip")
            for _ in range(depth))
        self.norm = Norm(width)
        self.to_data = Hop(data_neighbours, width, width)
        self.head = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, 1))

    def embed(self, detectors):
        """[shots, inputs] of 0/1 -> [shots, rounds, checks, width]."""
        if detectors.shape[1] != self.num_inputs:
            raise ValueError(f"expected {self.num_inputs} detector bits per shot, got {detectors.shape[1]}")
        codes = self.empty_code.repeat(len(detectors), 1)
        codes[:, self.input_place] = self.input_code + detectors.long()
        return self.embedding(codes).reshape(len(detectors), self.rounds, self.checks, -1)

    def features(self, detectors):
        """The vectors on the checks after the last block: [shots, rounds, checks, width]."""
        x = self.embed(detectors)
        for block in self.blocks:
            x = block(x)
        return x

    def readout(self, features):
        """[shots, rounds, checks, width] -> [shots, observables] logits.

        The paper scatters to the data qubits and then averages over a logical operator's
        support. Both steps are linear, so the average over time is taken first: the result is
        the same and the scatter runs on one time slice instead of all of them.
        """
        x = F.silu(self.norm(features))
        averaged = torch.stack([x[:, :-1].mean(dim=1), x[:, 1:].mean(dim=1)], dim=1)   # the rounds c, and the rounds c + 1
        data = self.to_data.checks_to_data(averaged).squeeze(1)             # [shots, data, width]
        pooled = torch.matmul(self.pool.to(data.dtype), data)              # [shots, observables, width]
        return self.head(pooled).squeeze(-1)

    def forward(self, detectors):
        return self.readout(self.features(detectors))

    def kit_bits(self, detectors, bits):
        """The kit's 12 observable bits, from the model's 12 bits (0/1, [shots, 12]) and the shots' detector bits.

        With logicals="kit" the bits come back unchanged.
        """
        if self.config["logicals"] == "kit":
            return bits.to(torch.uint8)
        counts = bits.float() @ self.kit_from_model.T + detectors.float() @ self.kit_from_inputs.T
        return counts.remainder(2).to(torch.uint8)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def mirrored_weights(state):
    """The weights of a decoder of one memory basis, turned into weights for the other basis.

    The code looks the same with X and Z checks exchanged, the two data blocks exchanged and
    every torus position negated. In the network's tables that exchanges the two blocks of
    every kind of node and moves neighbour slot s to slot s + 3 (mod 6). The network with
    these weights, reading a syndrome, computes exactly what the original computes on the
    mirror image of that syndrome (tested). The two memory circuits are not mirror images of
    each other, so the result is a starting point for training, not a finished decoder.
    """
    swapped = {".linears.0.": ".linears.1.", ".linears.1.": ".linears.0."}
    result = {}
    for name, tensor in state.items():
        if name == "embedding.weight":
            tensor = tensor.roll(EVENT_CODES, dims=0)                    # the codes of the two check types change places
        for block, other in swapped.items():
            if block in name:
                tensor = state[name.replace(block, other)]              # the two blocks of targets change places
                if name.endswith("weight"):
                    tensor = tensor.roll(tensor.shape[1] // 2, dims=1)  # slots 0 1 2 and 3 4 5 change places
                break
        result[name] = tensor.clone()
    return result
