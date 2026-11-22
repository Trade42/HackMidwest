# One check, one run, exit: score a kept cascade snapshot on ONE kit shot file with neural.evaluate_stim.
# Refuses to start (exit 2) if available RAM is under -MinMiB, (exit 3) if the snapshot's report has not finished,
# (exit 4) on a data sha256 mismatch or a held-out file. Otherwise runs in the foreground and prints the output.
#   powershell -File tools\cascade_eval_one.ps1 -Step 20000 -File a      (a = p0.004 validation_X, b = p0.003 seed 9203, c = p0.005 seed 9205)
param(
    [Parameter(Mandatory = $true)][int]$Step,
    [Parameter(Mandatory = $true)][ValidateSet('a', 'b', 'c', 'd')][string]$File,   # d = p0.004 train_X
    [int]$MinMiB = 2048
)
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\salav\Salavat\hackMidwest'
$out = "$root\runs\cascade_X_s1"
$kit = "$root\repo-cascade-measure\challenge-1-ml-decoder-kit"
$py = "$root\.venv\Scripts\python.exe"
$files = @{
    a = @{ label = 'validation_X_p0.004'; path = "$root\repo-main\challenge-1-ml-decoder-kit\data\p0.004\validation_X.npz"; sha = 'b00ee09c35ed088a12bcba60b8b704b528c47fb156c7d09cf0f171f90a181f3d' }
    b = @{ label = 'gen_X_seed9203_p0.003'; path = "$root\runs\priortilt\data\p0.003\gen_X_seed9203.npz"; sha = '4625e704682b3a6ef5c465eec33479d04f3d6f9a62912d4e4ce176e51bf7725e' }
    c = @{ label = 'gen_X_seed9205_p0.005'; path = "$root\runs\priortilt\data\p0.005\gen_X_seed9205.npz"; sha = '45a9761e688088c0e502e80a5569258985242e40231eea7f474f7887188213eb' }
    d = @{ label = 'train_X_p0.004'; path = "$root\repo-main\challenge-1-ml-decoder-kit\data\p0.004\train_X.npz"; sha = '98c0afeb2a0b6b27f3b689e832160a060a4c0aeb37d229f84daa1995a9a0f7cc' }
}
$f = $files[$File]
$avail = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue
"$(Get-Date -Format HH:mm:ss) available RAM $avail MiB (need $MinMiB)"
if ($avail -lt $MinMiB) { 'not starting: RAM'; exit 2 }
$lines = @(Get-Content -LiteralPath "$out\train.log"); $i = -1
for ($k = 0; $k -lt $lines.Count; $k++) { if ($lines[$k] -like "step $Step of *") { $i = $k } }
$done = $false; if ($i -ge 0) { for ($k = $i + 1; $k -lt $lines.Count; $k++) { if ($lines[$k] -like 'speed: *') { $done = $true } } }
$ck = "$out\checkpoint_step$Step.pt"
if (-not $done -or -not (Test-Path -LiteralPath $ck)) { 'not starting: snapshot not complete'; exit 3 }
if ((Split-Path $f.path -Leaf) -match '^(test_|go_)') { 'refusing a held-out file'; exit 4 }
$h = (Get-FileHash -LiteralPath $f.path).Hash.ToLower()
if ($h -ne $f.sha) { "sha256 mismatch: $h"; exit 4 }
"snapshot $ck $((Get-Item $ck).Length) bytes sha256 $((Get-FileHash -LiteralPath $ck).Hash.ToLower()); data sha256 ok"
New-Item -ItemType Directory -Force -Path "$out\eval" | Out-Null
$prefix = "$out\eval\step${Step}_$($f.label)"
$env:PYTHONDONTWRITEBYTECODE = '1'
$t0 = Get-Date
$p = Start-Process -FilePath $py -WorkingDirectory $kit -NoNewWindow -PassThru `
    -ArgumentList "-m neural.evaluate_stim --checkpoint $ck --data $($f.path) --save $prefix" `
    -RedirectStandardOutput "$prefix.log" -RedirectStandardError "$prefix.err.log"
$null = $p.Handle
$p.WaitForExit()
"exit $($p.ExitCode), wall $([math]::Round(((Get-Date) - $t0).TotalSeconds, 1)) s"
Get-Content -LiteralPath "$prefix.log"
$err = Get-Content -LiteralPath "$prefix.err.log"
if ($err) { '--- stderr:'; $err }
