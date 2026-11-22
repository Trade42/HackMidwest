"""The frozen router accepts equivalent matrices across sparse/floating representations."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from evaluation.predecoder_layout import matrix_identity, matrix_semantic_identity
from evaluation.router import load_frozen_router


def _problem(priors):
    H = csc_matrix(np.array([[1, 0], [0, 1]], dtype=np.uint8))
    L = csc_matrix(np.array([[0, 1]], dtype=np.uint8))
    return SimpleNamespace(basis="X", p=.003, H=H, L=L, priors=np.array(priors))


def test_portable_fingerprint_accepts_only_same_ordered_problem(tmp_path):
    mac = _problem([.003795837179896183, .0015999999999997986])
    linux = _problem([.003795837179896184, .0015999999999997984])
    for matrix in (linux.H, linux.L):
        matrix.indptr = matrix.indptr.astype(np.int64)
        matrix.indices = matrix.indices.astype(np.int64)
    assert matrix_identity(mac.H, mac.L, mac.priors) != matrix_identity(linux.H, linux.L, linux.priors)
    assert matrix_semantic_identity(mac.H, mac.L, mac.priors) == matrix_semantic_identity(
        linux.H, linux.L, linux.priors)

    path = tmp_path / "model.json"
    path.write_text(json.dumps({"basis": "X", "p": .003, "threshold": .1,
                                "matrix_sha256": matrix_identity(mac.H, mac.L, mac.priors),
                                "matrix_semantic_sha256": matrix_semantic_identity(mac.H, mac.L, mac.priors)}))
    assert load_frozen_router(path, linux)[0]["threshold"] == .1
    with pytest.raises(ValueError, match="matrix identity"):
        load_frozen_router(path, _problem([.003795837179896183, .0016000001]))
    reordered = _problem(mac.priors[::-1])
    with pytest.raises(ValueError, match="matrix identity"):
        load_frozen_router(path, reordered)


def test_old_router_needs_exact_fingerprint(tmp_path):
    mac = _problem([.003795837179896183, .0015999999999997986])
    linux = _problem([.003795837179896184, .0015999999999997984])
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"basis": "X", "p": .003, "threshold": .1,
                                "matrix_sha256": matrix_identity(mac.H, mac.L, mac.priors)}))
    assert load_frozen_router(path, mac)[0]["threshold"] == .1
    with pytest.raises(ValueError, match="matrix identity"):
        load_frozen_router(path, linux)
