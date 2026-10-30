# Sweep PTQ1_0 PT mat-vec tunables (rebuild + microbench per config). usage: pwsh pt-sweep.ps1 "R34,M34,R58,M58" ...
$f = 'C:\git\0\bonsai2\llama.cpp\ggml\src\ggml-cuda\mmvq-ptq1_0.cuh'
foreach ($cfg in $args) {
    $r34, $m34, $r58, $m58 = $cfg.Split(',')
    $s = Get-Content $f -Raw
    $s = $s -replace '#define PTQ1_0_PT_ROWS_34 \d+', "#define PTQ1_0_PT_ROWS_34 $r34"
    $s = $s -replace '#define PTQ1_0_PT_MINB_34 \d+', "#define PTQ1_0_PT_MINB_34 $m34"
    $s = $s -replace '#define PTQ1_0_PT_ROWS_58 \d+', "#define PTQ1_0_PT_ROWS_58 $r58"
    $s = $s -replace '#define PTQ1_0_PT_MINB_58 \d+', "#define PTQ1_0_PT_MINB_58 $m58"
    [IO.File]::WriteAllText($f, $s)
    cmd /c C:\git\0\bonsai2\build-v2.cmd 2>&1 | Select-String -Pattern " error |error:|FAILED" | Select-Object -First 5
    "### rows34=$r34 minb34=$m34 rows58=$r58 minb58=$m58"
    python C:\git\0\bonsai2\mv_perf.py
}
