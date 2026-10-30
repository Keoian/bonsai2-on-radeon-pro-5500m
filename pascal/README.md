# Bonsai 2 27B on a GTX 1070

Launchers for this branch's build. Paths outside the repo (CUDA toolkit, models, web UI) come from `BONSAI2`
(default `C:\git\0\bonsai2`): `%BONSAI2%\cuda`, `%BONSAI2%\models`, `%BONSAI2%\web`.

| Script | Setup | Decode t/s (desktop on the card) |
|---|---|---|
| `build-v2.cmd` | builds this tree into `build-v2` (sm_61) | |
| `run-server-mtp-32k.cmd` | MTP n-max 3 + stop rule, 32k, q8_0 K / q4_0 V | ~36 chat, 24.1 at 27k tokens |
| `run-server-mtp-24k.cmd` | MTP n-max 2 + 64k draft head, 24k, q8_0 KV | 35.6 chat, 31.5 at 16.5k tokens |
| `run-server-48k.cmd` | no MTP, 48k, q8_0 KV | ~21 |

`tables-n3-v6.txt` holds the stop-rule tables used by the 32k launcher. Extra arguments are passed to llama-server.

## Draft head (24k launcher)

The MTP head's logits over the full 248k vocabulary cost 1.8 ms per draft step. A draft head (FR-Spec style, already
supported by the loader) projects onto the 65536 most frequent tokens only (~0.5 ms); the target model still verifies
every draft, so output is unchanged and only acceptance can drop (held-out answers: 98.6% of tokens in the top 64k).

| n-max 2, 24k | plain MTP GGUF | + 64k draft head |
|---|---|---|
| chat suite, test split | 34.8 t/s, acc 0.690 | 35.6 t/s, acc 0.680 |
| 16.5k-token prompt | 30.5 t/s | 31.5 t/s |

No gain with the stop rule (n-max 3: 36.3 vs 36.2) and a 32k head loses too much acceptance (acc 0.654, 35.1 t/s).
Rebuild `Ternary-Bonsai-2-27B-PTQ1_0-MTP-Q4_K-dh64k.gguf` with:

```
python draft-head\add_draft_head.py %BONSAI2%\models\Ternary-Bonsai-2-27B-PTQ1_0-MTP-Q4_K.gguf %BONSAI2%\models\Ternary-Bonsai-2-27B-PTQ1_0-MTP-Q4_K-dh64k.gguf draft-head\ids-top64k.txt 65536
```

`ids-top64k.txt` came from `gen_texts.py` (suite dev/val answers, 3 seeds, temperature 0.7) and `rank_tokens.py`
(those answers at weight 4 plus prompts and `corpus.txt`; unseen ids follow in id order).

## Other folders

- `mtp-fit/`: the MTP stop rule's tooling (see its README): `run_suite.py` (72-prompt chat suite in
  `prompts-p100.json`, dev/val/test splits), `trace.sh` (traced runs), `fit_stop_rule.py` (fits the `tables-*.txt`;
  `tables-n3-v6.txt` in this folder is the one in use).
- `bench/`: the scripts behind the numbers in the commit messages. They hard-code `C:\git\0\bonsai2` and write to its
  `results\`. `vram_probe.py` loads a server, sends one long prompt and reports VRAM, spill ("shared": ~130-150 MiB of
  pinned memory is normal) and decode speed; `ab-dh.sh` runs the suite's test split per model; `rule-early.sh` compares
  stop-rule variants; `fa_perf.py`, `mv_perf.py`, `pt-sweep.ps1`, `g6-sweep.ps1` are kernel microbenchmarks/sweeps;
  `greedy_compare.py` checks greedy output against a reference build.
