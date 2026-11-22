#!/bin/bash
# ctl 110: THE ONE RUN of the frozen learned selector (branch learned-selector, frozen b8608de, model
# selector/models/p0.005.json sha256 696bf544...698ce6; fresh review APPROVE) on the held-out TEST members' files at
# p = 0.005, as its README gives the command: no joint file (p = 0.005 has none), chunk-mode candidates. It only
# applies the frozen model (it checks --expect-sha256); it fits nothing. A light numpy job (seconds), niced.
# It starts a background waiter and returns: the waiter waits for the test run's cand_p0.005 and tcbp_p0.005 steps
# to have ended with exit 0, then runs once. Printout runs/selector_test.out; files files/selector_test/.
B=$HOME/mlBranchMidwest; SHA=b8608de24f55b11126c26580c035eb47b7d1f354; W=$B/selector-b8608de
if [ ! -d "$W" ]; then
  git -C "$B/repo" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$B/repo" fetch -q origin learned-selector
  git -C "$B/repo" worktree add -q --detach "$W" "$SHA" || { echo "the checkout could not be made"; exit 1; }
fi
[ "$(git -C "$W" rev-parse HEAD)" = "$SHA" ] || { echo "the checkout is not at $SHA"; exit 1; }
echo "checkout: $(git -C "$W" log --oneline -1)"
mkdir -p "$B/jobs"
cat > "$B/jobs/selector_test.sh" <<'EOS'
#!/bin/bash
B=$HOME/mlBranchMidwest; W=$B/selector-b8608de; T=$B/runs/test; O=$B/runs/selector_test; F=$B/logsync/files/selector_test
echo "$(date -u +%FT%TZ) waiting for the test run's cand_p0.005 and tcbp_p0.005 steps"
until [ -e "$T/done/cand_p0.005.rc" ] && [ -e "$T/done/tcbp_p0.005.rc" ]; do sleep 60; done
for s in cand_p0.005 tcbp_p0.005; do
  [ "$(cat "$T/done/$s.rc")" = 0 ] || { echo "$(date -u +%FT%TZ) $s ended with exit $(cat "$T/done/$s.rc"): the selector is not run"; exit 1; }
done
for f in "$T/cand_p0.005/test_X.chunks.npz" "$T/cand_p0.005/test_Z.chunks.npz" "$T/tcbp_p0.005/test_X.tcbp.npz" "$T/tcbp_p0.005/test_Z.tcbp.npz"; do
  [ -s "$f" ] || { echo "missing $f: the selector is not run"; exit 1; }
done
[ -e "$O/selector_test_p0.005.json" ] && { echo "already run: $O"; exit 0; }
mkdir -p "$O" "$F"
cd "$W/challenge-1-ml-decoder-kit" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
sha256sum selector/models/p0.005.json
echo "$(date -u +%FT%TZ) START selector test p = 0.005"
nice -n 15 python -m selector.evaluate --model selector/models/p0.005.json \
  --expect-sha256 696bf5449a5468177dc2c00664e8557e7d019419c858a44052eea05521698ce6 \
  --label "test, p = 0.005" \
  --x "$T/cand_p0.005/test_X.chunks.npz" "$T/tcbp_p0.005/test_X.tcbp.npz" \
  --z "$T/cand_p0.005/test_Z.chunks.npz" "$T/tcbp_p0.005/test_Z.tcbp.npz" \
  --save "$O/selector_test_p0.005"
rc=$?
echo "$(date -u +%FT%TZ) END selector test p = 0.005 exit $rc"
cp "$O"/* "$F/" 2>/dev/null
exit $rc
EOS
chmod +x "$B/jobs/selector_test.sh"
nohup "$B/jobs/selector_test.sh" > "$B/runs/selector_test.out" 2>&1 < /dev/null &
echo "started the waiter, pid $!"; sleep 3; cat "$B/runs/selector_test.out"
