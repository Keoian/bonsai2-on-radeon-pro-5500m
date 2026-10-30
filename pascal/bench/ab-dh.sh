#!/bin/bash
# ab-dh.sh: chat suite (test split, greedy) for MTP models with and without a trimmed draft head, 24k n2 config.
# Usage: ENVS="..." ./ab-dh.sh model1 model2 ...   (models are file names under models/)
cd /c/git/0/bonsai2
PORT=8094
for m in "$@"; do
  env $ENVS PATH="/c/git/0/bonsai2/cuda/bin:$PATH" GGML_CUDA_FORCE_GRAPHS=1 GGML_CUDA_GDN_FUSION=1 LLAMA_ARG_SPEC_DRAFT_UBATCH=128 \
    llama.cpp/build-v2/bin/llama-server.exe -m models/$m -ngl 99 -fa on -c 24576 -np 1 -b 512 -ub 512 -ctk q8_0 -ctv q8_0 \
    --spec-type draft-mtp --spec-draft-n-max ${NMAX:-2} -ctkd q8_0 -ctvd q8_0 $SERVER_ARGS --jinja --host 127.0.0.1 --port $PORT --no-webui \
    > results/ab-dh-server.log 2>&1 &
  srv=$!
  until curl -sf localhost:$PORT/health >/dev/null; do sleep 1; kill -0 $srv 2>/dev/null || { echo "$m: server died"; tail -5 results/ab-dh-server.log; continue 2; }; done
  head=$(grep -o "draft head[^,)]*\|MTP chain[^)]*)" results/ab-dh-server.log | head -1)
  python mtp-fit/run_suite.py results/ab-dh-$m.jsonl --url http://127.0.0.1:$PORT --sampling greedy --splits test > /dev/null
  python - "$m" "$head" <<'PY'
import json,sys
rs=[json.loads(l) for l in open(f"results/ab-dh-{sys.argv[1]}.jsonl")]
n=sum(r["predicted_n"] for r in rs); ms=sum(r["predicted_ms"] for r in rs); dn=sum(r["draft_n"] for r in rs); da=sum(r["draft_acc"] for r in rs)
print(f"{sys.argv[1]:45s} {n:5d} tok {n/ms*1000:6.2f} t/s acc {da/max(dn,1):.3f} ({da}/{dn})  {sys.argv[2]}")
PY
  kill $srv; wait $srv 2>/dev/null
done
