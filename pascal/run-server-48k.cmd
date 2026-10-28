@echo off
rem Bonsai 2 27B on GTX 1070, pascal-v3-fa build: no MTP, 48k context, q8_0 KV (~21 t/s decode)
if not defined BONSAI2 set BONSAI2=C:\git\0\bonsai2
set PATH=%BONSAI2%\cuda\bin;%PATH%
set GGML_CUDA_FORCE_GRAPHS=1
set GGML_CUDA_GDN_FUSION=1
"%~dp0..\build-v2\bin\llama-server.exe" ^
  -m %BONSAI2%\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf ^
  -a bonsai2-27b ^
  -ngl 99 -fa on -c 49152 -np 1 -b 512 -ub 512 -ctk q8_0 -ctv q8_0 ^
  --jinja --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 ^
  --host 0.0.0.0 --port 8080 --path %BONSAI2%\web %*
