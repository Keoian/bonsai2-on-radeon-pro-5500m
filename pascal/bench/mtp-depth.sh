#!/bin/bash
# MTP decode with a long prompt, old kernels (base) vs new (v3), ABBA. usage: bash mtp-depth.sh <prompt_chars> <ctx> <ctv> [nmax]
BASE='C:\git\0\bonsai2\llama.cpp\.claude\worktrees\agent-aca59943393cca510\build-cuda\bin'
V3='C:\git\0\bonsai2\llama.cpp\build-v2\bin'
for b in "$BASE" "$V3" "$V3" "$BASE"; do
  echo -n "$( [ "$b" = "$BASE" ] && echo base || echo v3 ) "
  BIN="$b" PROMPT_CHARS=$1 GEN=256 NMAX=${4:-1} python vram_probe.py $2 q8_0 $3 mtp
done
