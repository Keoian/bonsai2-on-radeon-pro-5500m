#!/bin/bash
# PTQ1_0 decode-shape MUL_MAT microbenchmark on the build-v2 tree. Env vars pass through (GGML_CUDA_*).
export PATH="/c/git/0/bonsai2/cuda/bin:$PATH"
/c/git/0/bonsai2/llama.cpp/build-v2/bin/test-backend-ops.exe perf -o MUL_MAT -b CUDA0 -p ptq1_0 2>&1 \
  | grep -E "runs -" | grep -E "n=${NCOLS:-1},k=" | sed -E 's/MUL_MAT\(type_a=ptq1_0,type_b=f32,//; s/,bs=.*\):\s+/ : /; s/ +/ /g; s/\x1b\[[0-9;]*m//g' | sort -u
