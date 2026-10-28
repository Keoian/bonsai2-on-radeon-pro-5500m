@echo off
rem Bonsai 2 27B on GTX 1070, pascal-v3-fa build: MTP n-max 3 + learned stop rule, 32k context, q8_0 K + q4_0 V
rem (~36 t/s short chat, ~24 t/s decode at 27k context; n-max 2 gave 34.6 / 23.2).
rem q4_0 V lets MTP reach 32k without spilling; since the FA kernel changes it decodes as fast as q8_0 V.
rem The stop rule decides early by default (LLAMA_SPEC_ISO_EARLY=0 = full rule); LLAMA_GRAPH_CACHE=8 keeps a graph per verify width.
if not defined BONSAI2 set BONSAI2=C:\git\0\bonsai2
set PATH=%BONSAI2%\cuda\bin;%PATH%
set GGML_CUDA_FORCE_GRAPHS=1
set GGML_CUDA_GDN_FUSION=1
set LLAMA_ARG_SPEC_DRAFT_UBATCH=128
set LLAMA_GRAPH_CACHE=8
set LLAMA_SPEC_PAD=0
"%~dp0..\build-v2\bin\llama-server.exe" ^
  -m %BONSAI2%\models\Ternary-Bonsai-2-27B-PTQ1_0-MTP-Q4_K.gguf ^
  -a bonsai2-27b ^
  -ngl 99 -fa on -c 32768 -np 1 -b 512 -ub 512 -ctk q8_0 -ctv q4_0 ^
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0 -ctkd q8_0 -ctvd q4_0 ^
  --spec-draft-stop-iso-file "%~dp0tables-n3-v6.txt" --spec-draft-stop-iso 0.5 ^
  --jinja --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 ^
  --host 0.0.0.0 --port 8080 --path %BONSAI2%\web %*
