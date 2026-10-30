@echo off
rem Bonsai 2 27B on GTX 1070, pascal-v3-fa build: MTP n-max 2, 24k context, q8_0 KV (~35.6 t/s short chat, ~31.5 t/s at 16.5k)
rem The model carries a 64k-token draft head (see README.md): +2-3% over the plain MTP GGUF.
rem 24k is the largest MTP context that does not spill to system RAM with the desktop on this card.
rem n-max 2 on purpose: the n-max 3 stop rule is +4% on chat but -4% at 16.5k tokens here (28-29 vs 30.2 t/s, any graph cache).
if not defined BONSAI2 set BONSAI2=C:\git\0\bonsai2
set PATH=%BONSAI2%\cuda\bin;%PATH%
set GGML_CUDA_FORCE_GRAPHS=1
set GGML_CUDA_GDN_FUSION=1
set LLAMA_ARG_SPEC_DRAFT_UBATCH=128
"%~dp0..\build-v2\bin\llama-server.exe" ^
  -m %BONSAI2%\models\Ternary-Bonsai-2-27B-PTQ1_0-MTP-Q4_K-dh64k.gguf ^
  -a bonsai2-27b ^
  -ngl 99 -fa on -c 24576 -np 1 -b 512 -ub 512 -ctk q8_0 -ctv q8_0 ^
  --spec-type draft-mtp --spec-draft-n-max 2 -ctkd q8_0 -ctvd q8_0 ^
  --jinja --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 ^
  --host 0.0.0.0 --port 8080 --path %BONSAI2%\web %*
