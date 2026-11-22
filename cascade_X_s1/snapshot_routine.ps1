# The per-snapshot routine for dq-orch-2 (steps 2 to 4 of its scoring task), for one kept snapshot:
#  2. score checkpoint_step<Step>.pt on the six team X files, one at a time, each only with >= 2 GiB RAM available
#     (tools\cascade_eval_one.ps1; stops at the first file that does not run cleanly);
#  3. analyze_network_select.py <Step> on the p = 0.004 selector files;
#  4. copy the snapshot, the eval files and the table to the laptop-artifacts checkout, update SHA256SUMS,
#     SCORES.md and train.log, commit and push (a fast-forward, never forced).
# Exit codes: 0 done; 2 not enough RAM for a file (nothing committed); 1 anything else.
param([Parameter(Mandatory = $true)][int]$Step)
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\salav\Salavat\hackMidwest'
$A = "$root\repo-artifacts"
$C = "$root\runs\cascade_X_s1"
foreach ($f in 'a', 'd', 'v3', 't3', 'v5', 't5') {
    $o = & powershell -NoProfile -ExecutionPolicy Bypass -File "$root\tools\cascade_eval_one.ps1" -Step $Step -File $f
    $code = $LASTEXITCODE
    "== $f exit $code"
    $o | Where-Object { $_ -match 'available RAM|^exit |failures \d+ of|logical error per shot' }
    if ($code -eq 2) { 'stopping: not enough RAM'; exit 2 }
    if ($code -ne 0 -or -not ($o -match '^exit 0')) { "stopping: $f did not run cleanly"; $o | Select-Object -Last 15; exit 1 }
}
$table = & "$root\.venv-cpu\Scripts\python.exe" -I -B "$root\runs\priortilt\analyze_network_select.py" $Step 2>&1 | ForEach-Object { "$_" }
$table | Set-Content -Encoding utf8 "$root\runs\priortilt\results_network_select_step$Step.txt"
$table
Copy-Item "$C\checkpoint_step$Step.pt" "$A\cascade_X_s1\"
Copy-Item "$C\train.log" "$A\cascade_X_s1\train.log" -Force
Copy-Item "$root\runs\priortilt\results_network_select_step$Step.txt" "$A\priortilt\"
Get-ChildItem "$C\eval" -File | Where-Object { $_.Name -match "^step${Step}_.*\.(json|log|npz)$" -and $_.Name -notmatch '\.err\.log$' } | Copy-Item -Destination "$A\cascade_X_s1\eval\"
$sums = Get-ChildItem "$A\cascade_X_s1" -Filter 'checkpoint_step*.pt' | Sort-Object { [int]($_.BaseName -replace '\D', '') } |
    ForEach-Object { "$((Get-FileHash $_.FullName).Hash.ToLower())  $($_.Name)" }
[IO.File]::WriteAllText("$A\cascade_X_s1\SHA256SUMS", (($sums -join "`n") + "`n"))
& "$root\.venv-cpu\Scripts\python.exe" -I -B "$root\tools\scores_table.py" $A | Out-Null
Copy-Item "$root\tools\scores_table.py", "$root\tools\snapshot_routine.ps1", "$root\tools\cascade_eval_one.ps1" "$A\cascade_X_s1\" -Force
& git -C $A add -A
$msgFile = "$env:TEMP\laptop_artifacts_msg_$Step.txt"
$msg = "Cascade step ${Step}: snapshot, Stim scores on six team files, selection table`n`n" +
    "- cascade_X_s1/checkpoint_step$Step.pt (sha256 in SHA256SUMS), eval/step${Step}_* on train_X and validation_X`n" +
    "  at p = 0.003, 0.004 and 0.005, SCORES.md, and train.log.`n" +
    "- priortilt/results_network_select_step$Step.txt: the network as a selector at step $Step.`n`n" +
    "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`n" +
    "Claude-Session: https://claude.ai/code/session_01AptVDq1GzMDFNwVfYqnuE5`n"
[IO.File]::WriteAllText($msgFile, $msg)
& git -C $A commit -q -F $msgFile 2>&1 | Where-Object { $_ -notmatch 'LF will be replaced' }
& git -C $A push origin laptop-artifacts 2>&1 | ForEach-Object { "$_" }
if ($LASTEXITCODE -ne 0) { 'push failed'; exit 1 }
"commit $(& git -C $A rev-parse HEAD)"
"remote $(& git -C $A ls-remote --heads origin laptop-artifacts)"
"checkpoint sha256 $((Get-FileHash "$C\checkpoint_step$Step.pt").Hash.ToLower())"
