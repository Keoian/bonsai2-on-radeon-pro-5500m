#!/bin/bash
# Stop rule at long context: n2 vs full rule vs early-decision rule (LLAMA_SPEC_ISO_EARLY) at several thetas.
# usage: bash rule-early.sh [prompt_chars] [ctx]
cd /c/git/0/bonsai2
PC=${1:-30000}; CTX=${2:-12288}
T=C:/git/0/bonsai2/mtp-fit/tables-n3-v6.txt
export GEN=256 PROMPT_CHARS=$PC LLAMA_MTP_DRAFT_TIMING=1 LLAMA_GRAPH_CACHE=8 LLAMA_SPEC_PAD=0
run() {
    local tag=$1; shift
    echo -n "$tag: "; env "$@" python vram_probe.py $CTX q8_0 q8_0 mtp | sed 's/.*| prompt [0-9]* tok [0-9.]* t\/s, //'
    grep "mtp-draft-timing" results/vram_probe.log | tail -1 | sed 's/.*per call: /   draft per call: /'
}
run n2        NMAX=2
run rule      NMAX=3 SERVER_ARGS="--spec-draft-p-min 0 --spec-draft-stop-iso-file $T"
for th in 0.35 0.5 0.65 0.8; do
    run early$th NMAX=3 LLAMA_SPEC_ISO_EARLY=1 SERVER_ARGS="--spec-draft-p-min 0 --spec-draft-stop-iso-file $T --spec-draft-stop-iso $th"
done
run n2        NMAX=2
