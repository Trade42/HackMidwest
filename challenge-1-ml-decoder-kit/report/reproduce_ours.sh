#!/bin/sh
# From a fresh clone to our rows of the comparison table. REPRODUCE.md explains every stage,
# says how long it takes, and says which stages were run where.
#
#   report/reproduce_ours.sh env                 build the Python environment and PriorTilt
#   report/reproduce_ours.sh test                the tests of report/ and hierarchy/
#   report/reproduce_ours.sh smoke               the decoder through the team's evaluator on 24 training shots
#   report/reproduce_ours.sh checkpoints         fetch the two networks and check their sha256
#   report/reproduce_ours.sh row <p> <row>       one row on the saved test shots (hours at p = 0.005)
#   report/reproduce_ours.sh collect             copy the rows you made into report/data/test/
#   report/reproduce_ours.sh report              the tables and the figures
#
# <p> is 0.003, 0.004 or 0.005. <row> is bposd, relay-bp, router, hierarchy-weight or hierarchy.
# Settings, all optional: PYTHON (default .venv-ours/bin/python), PRIORTILT_REPO (default: a folder
# PriorTiltDecoder beside this repository), OUT (default runs/clean), DEVICE (cpu), PY312 (a Python 3.12
# interpreter for the env stage, default python3.12; with uv: PY312=$(uv python find 3.12)).
set -eu
cd "$(dirname "$0")/.."
VENV=${VENV:-.venv-ours}
PYTHON=${PYTHON:-$VENV/bin/python}
PRIORTILT_REPO=${PRIORTILT_REPO:-$(cd ../.. && pwd)/PriorTiltDecoder}
PRIORTILT_COMMIT=8eefd37d694b5942fe60ec6d5c82ce97f3d2a450
OUT=${OUT:-runs/clean}
CHECKPOINTS=${CHECKPOINTS:-runs/checkpoints}
DEVICE=${DEVICE:-cpu}
# The frozen networks, each pinned to one commit of a data-only branch of this repository: branch, commit, path, sha256.
CKPT_X_BRANCH=laptop-artifacts
CKPT_X_COMMIT=2f752e9367f80e1180711ded6b6ca854a5c0c147
CKPT_X_PATH=cascade_X_s1/checkpoint_final_step120000.pt
CKPT_X_SHA256=d8d1c70f552233300f1c10997b69292092042eb8517f5b6526b607c1f6dc4933
CKPT_Z_BRANCH=${CKPT_Z_BRANCH:-PENDING}
CKPT_Z_COMMIT=${CKPT_Z_COMMIT:-PENDING}
CKPT_Z_PATH=${CKPT_Z_PATH:-PENDING}
CKPT_Z_SHA256=${CKPT_Z_SHA256:-PENDING}
export PRIORTILT_REPO PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
stage=${1:-}

valid_p() {
  case "$1" in 0.003|0.004|0.005) ;; *) echo "p must be 0.003, 0.004 or 0.005" >&2; exit 2;; esac
}

options() {  # write the plugin's options for a rule: "network" or "weight"
  mkdir -p "$OUT"
  if [ "$1" = network ]; then
    printf '{"rule": "network", "checkpoints": {"X": "%s", "Z": "%s"}, "seeds": [1, 1001], "device": "%s", "torch_threads": 1}\n' \
      "$CHECKPOINTS/cascade_X.pt" "$CHECKPOINTS/cascade_Z.pt" "$DEVICE" > "$OUT/hierarchy_options.json"
    echo "$OUT/hierarchy_options.json"
  else
    printf '{"rule": "weight", "seeds": [1, 1001]}\n' > "$OUT/hierarchy-weight_options.json"
    echo "$OUT/hierarchy-weight_options.json"
  fi
}

case "$stage" in
  env)
    # Python 3.12, the kit's requirements, torch (CPU build), and PriorTilt from its own repository.
    # relay-bp and PriorTilt build from source: a Rust toolchain (cargo, https://rustup.rs) is needed.
    PY312=${PY312:-python3.12}
    command -v "$PY312" > /dev/null || { echo "no Python 3.12 found as $PY312: set PY312 (with uv: PY312=\$(uv python find 3.12))" >&2; exit 1; }
    "$PY312" -m venv "$VENV"
    "$VENV/bin/pip" install -r requirements.txt pytest matplotlib
    "$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu
    # PriorTilt (IonQ, arXiv:2611.13373) is licensed CC BY-NC-SA 4.0: none of its code is in this repository.
    [ -d "$PRIORTILT_REPO" ] || git clone https://github.com/ionq-publications/PriorTiltDecoder.git "$PRIORTILT_REPO"
    git -C "$PRIORTILT_REPO" checkout "$PRIORTILT_COMMIT"
    "$VENV/bin/pip" install "$PRIORTILT_REPO/decoder"
    echo "environment in $VENV; PriorTilt at $PRIORTILT_REPO ($PRIORTILT_COMMIT)"
    ;;
  test)
    "$PYTHON" -m pytest -p no:cacheprovider report hierarchy
    ;;
  smoke)
    # The whole path on a file small enough for a few minutes: the first 24 shots of the committed
    # TRAINING files at p = 0.004 (never the test shots), our decoder and relay-BP through the team's evaluator.
    S=${SMOKE_OUT:-runs/smoke}
    rm -rf "$S"; mkdir -p "$S"
    "$PYTHON" - "$S" <<'EOF'
import sys
import numpy as np
for basis in "XZ":
    with np.load(f"data/p0.004/train_{basis}.npz") as file:
        np.savez_compressed(f"{sys.argv[1]}/small_{basis}.npz", p=file["p"], seed=file["seed"], basis=file["basis"],
                            detectors=file["detectors"][:24], observables=file["observables"][:24],
                            basis_detectors=file["basis_detectors"])
EOF
    if [ -f "$CHECKPOINTS/cascade_X.pt" ] && [ -f "$CHECKPOINTS/cascade_Z.pt" ]; then
      printf '{"rule": "network", "checkpoints": {"X": "%s", "Z": "%s"}, "device": "%s", "torch_threads": 1}\n' \
        "$CHECKPOINTS/cascade_X.pt" "$CHECKPOINTS/cascade_Z.pt" "$DEVICE" > "$S/options.json"
    else
      echo "no checkpoints in $CHECKPOINTS: the smoke run uses the tie-break without a network"
      printf '{"rule": "weight"}\n' > "$S/options.json"
    fi
    "$PYTHON" -m evaluation.evaluate --test-x "$S/small_X.npz" --test-z "$S/small_Z.npz" \
      --decoder hierarchy.plugin:make_decoder --decoder relay-bp --plugin-options "$S/options.json" --out "$S/eval"
    cut -d, -f1,7-10,18 "$S/eval/summary.csv"
    ;;
  checkpoints)
    # The two frozen networks are kept on data-only branches of this repository, not in the source tree.
    if [ "$CKPT_Z_COMMIT" = PENDING ]; then
      echo "the Z checkpoint's location is not filled in yet (see REPRODUCE.md)" >&2; exit 1
    fi
    mkdir -p "$CHECKPOINTS"
    git fetch origin "$CKPT_X_BRANCH" "$CKPT_Z_BRANCH"
    git show "$CKPT_X_COMMIT:$CKPT_X_PATH" > "$CHECKPOINTS/cascade_X.pt"
    git show "$CKPT_Z_COMMIT:$CKPT_Z_PATH" > "$CHECKPOINTS/cascade_Z.pt"
    printf '%s  %s\n%s  %s\n' "$CKPT_X_SHA256" "$CHECKPOINTS/cascade_X.pt" "$CKPT_Z_SHA256" "$CHECKPOINTS/cascade_Z.pt" | sha256sum -c -
    ;;
  row)
    p=${2:-}; row=${3:-}
    valid_p "$p"
    destination="$OUT/eval_p${p}_$row"
    if [ -e "$destination" ]; then echo "refusing to overwrite $destination" >&2; exit 1; fi
    mkdir -p "$OUT"
    case "$row" in
      bposd|relay-bp) decoder=$row; extra="" ;;
      router)
        printf '{"router_models":{"X":"results/router_p%s/model_X.json","Z":"results/router_p%s/model_Z.json"}}\n' "$p" "$p" > "$OUT/router_options_p$p.json"
        decoder=evaluation.router:make_tiered_decoder; extra="$OUT/router_options_p$p.json" ;;
      hierarchy) decoder=hierarchy.plugin:make_decoder; extra=$(options network) ;;
      hierarchy-weight) decoder=hierarchy.plugin:make_decoder; extra=$(options weight) ;;
      *) echo "row must be bposd, relay-bp, router, hierarchy-weight or hierarchy" >&2; exit 2 ;;
    esac
    # One process, one thread, the team's evaluator; the timer wraps the whole decode of a shot.
    if [ -n "$extra" ]; then
      "$PYTHON" -m evaluation.evaluate --test-x "data/p$p/test_X.npz" --test-z "data/p$p/test_Z.npz" --decoder "$decoder" --plugin-options "$extra" --out "$destination"
    else
      "$PYTHON" -m evaluation.evaluate --test-x "data/p$p/test_X.npz" --test-z "data/p$p/test_Z.npz" --decoder "$decoder" --out "$destination"
    fi
    ;;
  collect)
    # Put your own rows where the tables read them. This replaces the committed copies of the same rows.
    for folder in "$OUT"/eval_p*; do
      [ -f "$folder/summary.json" ] || continue
      mkdir -p "report/data/test/$(basename "$folder")"
      cp "$folder"/summary.json "$folder"/summary.csv "$folder"/*_shots.npz "report/data/test/$(basename "$folder")/"
      echo "copied $(basename "$folder")"
    done
    ;;
  report)
    "$PYTHON" -m report.make_tables
    "$PYTHON" -m report.make_plots
    ;;
  *)
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
