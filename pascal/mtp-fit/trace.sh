#!/bin/bash
# usage: bash trace.sh <nmax> <tag> [run_suite args...]   -- traces greedy MTP cycles with the rule off
N=$1; TAG=$2; shift 2
R=C:/git/0/bonsai2; F=$R/mtp-fit/runs
rm -f $F/trace-$TAG.jsonl
TR=(--spec-trace "$F/trace-$TAG.jsonl"); [ -n "$NOTRACE" ] && TR=()
PATH="/c/git/0/bonsai2/cuda/bin:$PATH" GGML_CUDA_FORCE_GRAPHS=1 GGML_CUDA_GDN_FUSION=1 LLAMA_ARG_SPEC_DRAFT_UBATCH=128 \
  $R/llama.cpp/build-v2/bin/llama-server.exe -m $R/models/Ternary-Bonsai-2-27B-PTQ1_0-MTP-Q4_K.gguf -ngl 99 -fa on -c 12288 \
  -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 -np 1 --jinja --port 8093 --no-webui \
  --spec-type draft-mtp --spec-draft-n-max $N --spec-draft-p-min 0 "${TR[@]}" $EXTRA > $F/server-$TAG.log 2>&1 &
SRV=$!
until curl -s localhost:8093/health | grep -q ok; do sleep 2; kill -0 $SRV 2>/dev/null || { echo "server died"; tail $F/server-$TAG.log; exit 1; }; done
python $R/mtp-fit/run_suite.py $F/rows-$TAG.jsonl --url http://127.0.0.1:8093 "$@"
kill $SRV; taskkill //F //IM llama-server.exe >/dev/null 2>&1
wc -l $F/trace-$TAG.jsonl
