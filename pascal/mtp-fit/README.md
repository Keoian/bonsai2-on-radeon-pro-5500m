# MTP learned stop rule: trace -> fit -> run

Port of the P100 fork's stop rule (Keoian/dual-p100-qwen-3-8-27b, `p100-docs/CHANGES.md` section 22, research branch
`mtp-adaptive-research`) to our llama.cpp branch `mtp-stop-rule`.

The MTP draft stops before adding token d when

    S(d) = prod_{i<d} iso_q(q_i) * iso_p1(p1_d) < theta

- `q_i`: the draft's probability of the token it drafted at step i (the top-1 p for greedy drafts)
- `p1_d`: the top-1 p of step d's candidates (top-10, T 1)
- `iso_q`, `iso_p1`: isotonic fits of the target's per-token acceptance on traced cycles

Step d's own drawn token is never looked at, so the output stays exactly the target's (greedy and sampled).

## Files

| file | what |
|---|---|
| `run_suite.py` | sends the prompt suite to a running server, one row per request (`task` id links it to the trace) |
| `fit_stop_rule.py` | traces (+ rows) -> table file; picks theta; plain Python 3, no numpy/sklearn |
| `prompts-p100.json` | the fork's 72-prompt suite: 8 categories x (3 dev, 3 val, 3 test) |

## Server switches (branch `mtp-stop-rule`)

| switch | env | default | meaning |
|---|---|---|---|
| `--spec-draft-stop-iso THETA` | `LLAMA_ARG_SPEC_DRAFT_STOP_ISO` | 0 (off) | rule on with this theta (built-in placeholder tables unless a file is given) |
| `--spec-draft-stop-iso-file F` | `LLAMA_ARG_SPEC_DRAFT_STOP_ISO_FILE` | none | tables from F; turns the rule on with F's theta (the flag above overrides it) |
| `--spec-trace F` | `LLAMA_ARG_SPEC_TRACE` | none | append one JSON line per verify cycle to F |
| - | `LLAMA_SPEC_PAD` | on iff the rule is on (`-np 1`) | 1/0: pad short drafts to n-max so the verify keeps one width |

With the rule on, p-min is not used. Chained drafts (`LLAMA_MTP_CHAIN`) are off with the rule or the trace.

## Steps

1. Trace (rule OFF, p-min 0, so every cycle drafts n-max tokens and every depth is labelled). Use the n-max you
   want to run with (3 or 4). Pick the sampling you will benchmark (greedy here); fit separate tables per mode
   (greedy / thinking-off sampling / thinking) because acceptance differs.

       llama-server -m <model> --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0 -np 1 ^
           --spec-trace C:\git\0\bonsai2\mtp-fit\runs\trace-n4.jsonl  <your usual flags>
       python run_suite.py runs\T-n4.jsonl --sampling greedy --exclude-cats longctx

   The trace adds a sampler pass per verify row and a write per cycle (well under a ms), so do not time with it on.
   `longctx` prompts are up to ~7k tokens; drop them if the context is tight. One seed of the 63 other prompts at
   300 tokens is ~6-8k cycles at n-max 4; add `--seeds 11,12` for more.

2. Fit:

       python fit_stop_rule.py --trace runs\trace-n4.jsonl --rows runs\T-n4.jsonl --out tables-n4.txt

   It prints the simulated t/s for every theta (val and test splits) and for p-min at the same width, the chosen
   theta (best on val) and its simulated test gain with a 95% bootstrap interval over prompts. If theta 0 wins, the
   rule does not pay on this setup and the file says theta 0 (the server then leaves it off).

3. Run with the tables (padded verify comes on with it):

       llama-server ... --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-stop-iso-file tables-n4.txt
       python run_suite.py runs\R-n4.jsonl --sampling greedy --exclude-cats longctx

   Compare against `--spec-draft-n-max 2` (current best, 33.6 t/s) and against the same n-max with
   `LLAMA_SPEC_PAD=0` / `=1` and no rule, to separate the padding effect from the rule.

## Trace format (one JSON line per verify cycle)

    {"task":41,"slot":0,"pos0":2531,"n_draft":4,"n_real":3,"temp":0,"rule":"greedy",
     "draft":[{"id":279,"q":0.93,"p1":0.93,"H":0.31,"t_id":279,"p":1,"a":1},
              {"id":3363,"q":0.71,"p1":0.71,"H":0.92,"t_id":3363,"p":1,"a":1},
              {"id":13,"q":0.55,"p1":0.55,"H":1.1,"t_id":11,"p":0,"a":0},
              {"id":198,"q":0.12,"p1":0.12,"H":2.9,"stop":"iso"}],
     "n_accepted":2,"draft_us":21000,"cycle_us":61000}

- `n_draft`: verified width (padding included); `n_real`: real drafted tokens. `pos0`: position of the last
  committed token (first verify row); drafted token i sits at `pos0+1+i`.
- `draft[i]`, i < n_real: `q`, `p1`, `H` (entropy over the draft's top-10) from the draft; `t_id`, `p`, `a` from the
  target, teacher-forced along the drafted tokens on a clone of the pre-verify sampler (so rows after the first
  rejection are labelled too). `p` = the target's post-sampler probability of the drafted token; `a` = acceptance
  probability: `min(1, p/q)` for sampled drafts (`rule` "dist"), `t_id == id` for a greedy target, else `p`.
- An entry at index n_real (no target fields) is the step that stopped drafting (`stop`: `p_min` or `iso`); `n_max`
  marks the last real step when n-max was reached.
- `draft_us`: the draft call; `cycle_us`: time since the previous record of the same task (absent on its first cycle).

## Table file format

    # comments
    theta 0.125
    x  0.000 0.020 ... 1.000     (increasing; the fork's 24 points)
    q  ...                        (iso_q at x)
    p1 ...                        (iso_p1 at x)

The server interpolates linearly between the points. Without a file it uses the fork's chat tables (Qwen3.8-27B,
thinking-off sampling, sampled drafts at T 1.0) as placeholders; they are not fitted for Bonsai-2-27B or greedy.

## How theta is chosen (as the fork's `policies.py`)

Tables are fit on the dev prompts. Every theta in 0, 0.025, ..., 0.95 is simulated on the traced cycles: a cycle
keeps k tokens (first d with S(d) < theta), decodes k+1 draft steps when it stops early, and commits
E = 1 + sum_{j<k} prod_{i<=j} a_i tokens in T = d0 + d1 * steps + rest us (draft cost from `draft_us`, verify and the
rest from `cycle_us - draft_us`, one constant because the verify is padded to one width). The theta with the best
sum E / sum T on the val prompts wins; the test prompts give the reported gain. The fork's cost model also had
per-width verify costs and the cost of skipped padded columns (their CUDA kernels skip them); ours has neither, see
the note on padding below.

## Padding note

The fork pads the verify to n-max+1 and its own CUDA kernels (hand-written mmvq-f16, q4p flash attention, through
`llama_set_n_active_tokens` / `ggml_backend_set_active_tokens`) then compute only the real columns. That part is
not ported: it lives in kernels our tree does not have. Here the padded columns are computed and rejected like any
wrong draft, so padding only removes the verify-graph rebuild when the draft length changes; the rule then saves
draft steps, not verify work. Measure `LLAMA_SPEC_PAD=0` against `=1`.
