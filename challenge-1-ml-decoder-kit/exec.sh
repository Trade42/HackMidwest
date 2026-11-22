.venv312/bin/python -m evaluation.router_features \
  --split validation --basis Z \
  --data data/p0.005/validation_Z.npz \
  --out results/router_p0.005/validation_Z.npz

MAX_MISSED_WRONG=0 ./reproduce_eval.sh router-train 0.003
MAX_MISSED_WRONG=0 ./reproduce_eval.sh router-train 0.005