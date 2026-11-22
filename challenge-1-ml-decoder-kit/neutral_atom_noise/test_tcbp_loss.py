"""GPU-free tests of tcbp_loss.py's prior combination: the flag raise on a base vector, and the 0.5 cap."""
import numpy as np

from loss.lossmap import _combine
from neutral_atom_noise.tcbp_loss import PRIOR_CAP, combine, shot_factors


def test_raise_matches_loss_combine_on_any_base():
    """The shipped factors F = prod(1 - 2 v) give loss/'s XOR combination on a matched and on a head-like base."""
    rng = np.random.default_rng(1)
    columns = np.array([3, 7, 7, 12])                     # column 7 gets two bumps
    bumps = np.array([0.5, 0.1, 0.2, 0.5 / 6])
    unique = np.unique(columns)
    alone = _combine(np.zeros(20), columns, bumps, 1.0)   # the bumps on a zero base, uncapped, as convert does
    factors = 1 - 2 * alone[unique]
    for base in (rng.uniform(1e-9, 0.05, 20), rng.uniform(1e-4, 0.6, 20)):
        np.testing.assert_allclose(combine(base, unique, factors), _combine(base, columns, bumps, PRIOR_CAP),
                                   rtol=0, atol=1e-15)


def test_raise_touches_only_flagged_columns_and_caps():
    base = np.array([0.01, 0.7, 0.02, 0.3, 1e-9])
    out = combine(base, np.array([2, 4]), np.array([0.0, 0.8]))
    assert out[0] == 0.01 and out[3] == 0.3               # columns without a bump keep their value
    assert out[1] == PRIOR_CAP                            # an unbumped value above 0.5 is capped
    assert out[2] == 0.5                                  # a 0.5 bump (F = 0) sets the column to 0.5
    np.testing.assert_allclose(out[4], (1 - (1 - 2e-9) * 0.8) / 2)
    assert base[2] == 0.02                                # the base vector is not changed in place


def test_shot_factors_reads_csr_rows():
    flags = {"indptr": np.array([0, 2, 2, 3]), "columns": np.array([5, 9, 1]), "factors": np.array([0.0, 0.9, 0.5])}
    cols, f = shot_factors(flags, 0)
    assert cols.tolist() == [5, 9] and f.tolist() == [0.0, 0.9]
    assert len(shot_factors(flags, 1)[0]) == 0
    assert shot_factors(flags, 2)[0].tolist() == [1]
