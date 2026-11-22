cd /home/blakew3/HackMidwest/challenge-1-ml-decoder-kit
git pull --ff-only origin main
MIDQ_PY="$HOME/envs/decoder-py312/bin/python3.12"

"$MIDQ_PY" -m evaluation.evaluate \
  --test-x data/p0.003/test_X.npz --test-z data/p0.003/test_Z.npz \
  --decoder evaluation.router:make_tiered_decoder \
  --plugin-options results/router_p0.003/eval_options.json \
  --out results/icrn/router_eval_p0.003_portable

"$MIDQ_PY" -m evaluation.compare_gain \
  --reference-results results/icrn/eval_p0.003 \
  --candidate-results results/icrn/router_eval_p0.003_portable \
  --candidate-decoder evaluation.router:make_tiered_decoder \
  --out results/icrn/gain_p0.003_portable