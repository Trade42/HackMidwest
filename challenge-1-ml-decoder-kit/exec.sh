# Prepare the other two noise rates.
for p in 0.003 0.005; do ./reproduce_eval.sh dataset "$p"; done
for p in 0.003 0.005; do ./reproduce_eval.sh router-features "$p"; done

# Freeze one router per noise rate, using validation data only.
# Zero allows no incorrect hard candidates to be accepted on validation.
for p in 0.003 0.004 0.005; do
  MAX_MISSED_WRONG=0 ./reproduce_eval.sh router-train "$p"
done

for p in 0.003 0.004 0.005; do
  ./reproduce_eval.sh evaluate "$p"

  printf '{"router_models":{"X":"results/router_p%s/model_X.json","Z":"results/router_p%s/model_Z.json"}}\n' \
    "$p" "$p" > "results/router_p$p/eval_options.json"

  .venv312/bin/python -m evaluation.evaluate \
    --test-x "data/p$p/test_X.npz" --test-z "data/p$p/test_Z.npz" \
    --decoder evaluation.router:make_tiered_decoder \
    --plugin-options "results/router_p$p/eval_options.json" \
    --out "results/router_eval_p$p"

  .venv312/bin/python -m evaluation.compare_gain \
    --reference-results "results/eval_p$p" \
    --candidate-results "results/router_eval_p$p" \
    --candidate-decoder evaluation.router:make_tiered_decoder \
    --out "results/gain_p$p"
done
