# Sweep the GQA-6 24-column tile config (rebuild + FA microbench). usage: pwsh g6-sweep.ps1 "NTHREADS,OCC,NBK" ...
$f = 'C:\git\0\bonsai2\llama.cpp\ggml\src\ggml-cuda\fattn-tile.cuh'
$env:FOLDVAR = 'GGML_CUDA_FA_TILE_GQA'
foreach ($cfg in $args) {
    $nt, $oc, $nbk = $cfg.Split(',')
    $s = Get-Content $f -Raw
    $s = $s -replace '#define GGML_CUDA_FA_TILE_G6_24_NTHREADS \d+', "#define GGML_CUDA_FA_TILE_G6_24_NTHREADS $nt"
    $s = $s -replace '#define GGML_CUDA_FA_TILE_G6_24_OCC \d+', "#define GGML_CUDA_FA_TILE_G6_24_OCC $oc"
    $s = $s -replace '#define GGML_CUDA_FA_TILE_G6_24_NBK \d+', "#define GGML_CUDA_FA_TILE_G6_24_NBK $nbk"
    [IO.File]::WriteAllText($f, $s)
    cmd /c C:\git\0\bonsai2\build-v2.cmd 2>&1 | Select-String -Pattern " error |error:|FAILED" | Select-Object -First 5
    "### nthreads=$nt occ=$oc nbatch_K=$nbk"
    python C:\git\0\bonsai2\fa_perf.py 6 'hsk=256,hsv=256,nh=4,nr23=\[6,1\].*kv=16384,nb=[345].*type_K=q8_0'
}
