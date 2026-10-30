#!/bin/bash
# Decode speed at several context depths, fold on vs off (same binary), ABBA order.
# usage: bash fa-depth.sh [depths] [ctv]
B=C:/git/0/bonsai2/llama.cpp/build-v2/bin
M=C:/git/0/bonsai2/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf
D=${1:-0,8192,16384,24576}
CTV=${2:-q8_0}
export PATH="/c/git/0/bonsai2/cuda/bin:$PATH" GGML_CUDA_FORCE_GRAPHS=1 GGML_CUDA_GDN_FUSION=1
run() { echo "## $1"; env $1 $B/llama-bench.exe -m $M -ngl 99 -fa 1 -ctk q8_0 -ctv $CTV -p 0 -n 64 -d $D -r 2 -o md 2>/dev/null | grep -E "tg64"; }
for arm in GGML_CUDA_FA_VEC_GQA=1 GGML_CUDA_FA_VEC_GQA=0 GGML_CUDA_FA_VEC_GQA=0 GGML_CUDA_FA_VEC_GQA=1; do run $arm; done
