@echo off
rem Build this tree into build-v2 for the GTX 1070 (sm_61); run from PowerShell or cmd (not `cmd //c` from bash).
if not defined BONSAI2 set BONSAI2=C:\git\0\bonsai2
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA_PATH=%BONSAI2%\cuda
set PATH=%CUDA_PATH%\bin;%PATH%
cd /d "%~dp0.."
if not exist build-v2\build.ninja cmake -B build-v2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe -DCUDAToolkit_ROOT=%CUDA_PATH% -DGGML_NATIVE=ON "-DGGML_CUDA_FA_QUANTS=q8_0-q8_0;q8_0-q4_0;q4_0-q4_0;f16-f16" -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF %EXTRA_CMAKE% || exit /b 1
cmake --build build-v2 -j 12 --target llama-server llama-bench llama-cli llama-perplexity test-backend-ops %EXTRA_TARGETS%
