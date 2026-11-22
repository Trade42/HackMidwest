# Step 9: generate the p = 0.005 and p = 0.003 shot files with the kit's make_dataset.py (repo-main, unchanged),
# record their sha256, and write the decode job list. Runs the generators at BelowNormal priority on CPUs 0-11.
$ErrorActionPreference = 'Stop'
$root = 'C:\Users\salav\Salavat\hackMidwest'
$kit = "$root\repo-main\challenge-1-ml-decoder-kit"
$py = "$root\.venv-cpu\Scripts\python.exe"
$data = "$root\runs\priortilt\data"
$specs = @(
    @{ p = '0.005'; basis = 'X'; seed = 9205; shots = 20000 },
    @{ p = '0.005'; basis = 'Z'; seed = 9206; shots = 20000 },
    @{ p = '0.003'; basis = 'X'; seed = 9203; shots = 50000 },
    @{ p = '0.003'; basis = 'Z'; seed = 9204; shots = 50000 }
)
$env:PYTHONDONTWRITEBYTECODE = '1'   # the kit tracks __pycache__/*.pyc in git
$procs = foreach ($s in $specs) {
    $dir = "$data\p$($s.p)"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $out = "$dir\gen_$($s.basis)_seed$($s.seed).npz"
    if (Test-Path $out) { throw "$out exists" }
    $p = Start-Process -FilePath $py -WorkingDirectory $kit -NoNewWindow -PassThru `
        -ArgumentList "make_dataset.py --basis $($s.basis) --p $($s.p) --shots $($s.shots) --seed $($s.seed) --out $out" `
        -RedirectStandardOutput "$dir\gen_$($s.basis)_seed$($s.seed).log" -RedirectStandardError "$dir\gen_$($s.basis)_seed$($s.seed).err.log"
    try { $p.PriorityClass = 'BelowNormal'; $p.ProcessorAffinity = [IntPtr]0x0FFF } catch {}
    $p
}
$procs | ForEach-Object { $_.WaitForExit() }
$lines = foreach ($s in $specs) {
    $f = Get-Item "$data\p$($s.p)\gen_$($s.basis)_seed$($s.seed).npz"
    "p=$($s.p) $($s.basis) seed $($s.seed) shots $($s.shots): $($f.FullName) $($f.Length) bytes sha256 $((Get-FileHash $f.FullName).Hash.ToLower())"
}
$lines | Set-Content -Encoding ascii "$data\data_sha256_step9.txt"
$lines
# Job list: p = 0.005 first, then p = 0.003; relay-BP and PriorTilt interleaved per chunk.
$jobs = @('# step 9: p = 0.005 then p = 0.003')
foreach ($s in $specs) {
    $f = "$data\p$($s.p)\gen_$($s.basis)_seed$($s.seed).npz"
    for ($c = 0; $c -lt ($s.shots / 500); $c++) { $jobs += "$f relay $c"; $jobs += "$f priortilt $c" }
}
$jobs | Set-Content -Encoding ascii "$root\runs\priortilt\jobs_step9.txt"
"jobs: $($jobs.Count - 1)"
