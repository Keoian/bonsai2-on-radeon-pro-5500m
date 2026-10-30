# fit_stop_rule.py: turn llama-server --spec-trace records into a table file for --spec-draft-stop-iso-file.
# Plain Python 3 (no numpy / sklearn needed).
#
# The rule (P100 fork, CHANGES 22; research branch mtp-adaptive-research, analysis/heads.py, policies.py, sim.py):
#   S(d) = prod_{i<d} iso_q(q_i) * iso_p1(p1_d); do not add drafted token d when S(d) < theta.
# iso_q and iso_p1 are isotonic regressions of the per-token acceptance a_d (teacher-forced along the drafted prefix:
# min(1, p/q) for sampled drafts, 1/0 for a greedy target, p for a sampled target with greedy drafts) on the draft's
# q of the drafted token and on the top-1 p of the step, fit on the dev prompts (pool-adjacent-violators, the same fit
# as sklearn's IsotonicRegression, which the fork used). theta is chosen by simulated throughput on the val prompts
# (grid 0..0.95 step 0.025, as policies.py) and reported on the test prompts (paired bootstrap over prompts).
#
# The traces must come from a server with the stop rule OFF and --spec-draft-p-min 0, so every cycle drafts n-max
# tokens and every depth is labelled (the rule never changes which tokens are drafted, only how many are kept).
#
#   python fit_stop_rule.py --trace runs/trace-n4.jsonl --rows runs/T-n4.jsonl --out tables-n4.txt
import argparse, bisect, json, os, random, statistics, sys, zlib

ap = argparse.ArgumentParser()
ap.add_argument("--trace", nargs="+", required=True, help="--spec-trace JSONL file(s)")
ap.add_argument("--rows", nargs="*", default=[], help="run_suite.py output(s): task id -> prompt id and split")
ap.add_argument("--out", required=True, help="table file to write")
ap.add_argument("--temp-range", default="", help="lo,hi: keep only cycles whose request temperature is in [lo, hi]")
ap.add_argument("--draft-us", default="", help="d0,d1: draft cost = d0 + d1 * steps (us); default: fit from draft_us")
ap.add_argument("--rest-us", type=float, default=None, help="verify + other per-cycle cost (us); default: from cycle_us")
ap.add_argument("--rest-by-width", default="", help="w:us,... verify + rest per verify width (kept drafts + 1), for an unpadded "
                "verify whose cost depends on the width (e.g. 1:44600,2:52000,3:61000,4:63000); overrides --rest-us")
ap.add_argument("--thetas", default="", help="comma list (default 0..0.95 step 0.025)")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()

# the table points: the P100 fork's grid (iso_tables.py)
GRID = [0.0, 0.02, 0.05] + [round(0.1 + 0.05 * i, 3) for i in range(18)] + [0.98, 0.995, 1.0]
THETAS = [float(t) for t in a.thetas.split(",")] if a.thetas else [round(0.025 * i, 3) for i in range(39)]
PMINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
SPLITS = {"dev": 0, "val": 1, "test": 2}

# ---- load
rows = {}
for fn in a.rows:
    for l in open(fn, encoding="utf-8"):
        r = json.loads(l)
        if r.get("task") is not None:
            rows[r["task"]] = r
recs = []
for fn in a.trace:
    for l in open(fn, encoding="utf-8"):
        l = l.strip()
        if l:
            recs.append(json.loads(l))
if rows:
    recs = [r for r in recs if r["task"] in rows]  # drops the warm-up and anything not from the suite
if a.temp_range:
    lo, hi = (float(v) for v in a.temp_range.split(","))
    recs = [r for r in recs if lo <= r["temp"] <= hi]
if not recs:
    sys.exit("no trace records left")


class Cycle:
    pass


cyc = []
pids = {}
n_stopped = 0
for r in recs:
    c = Cycle()
    c.n_real = r["n_real"]
    c.steps = len(r["draft"])
    if rows:
        row = rows[r["task"]]
        key, sp = f'{row["id"]}/{row.get("seed")}', row["split"]
    else:  # no rows file: split by task id (about 50/25/25)
        key = str(r["task"])
        h = zlib.crc32(key.encode()) % 4
        sp = "dev" if h < 2 else "val" if h == 2 else "test"
    c.split = SPLITS[sp]
    c.pid = pids.setdefault(key, len(pids))
    c.q, c.p1, c.a = [], [], []
    for d, s in enumerate(r["draft"][:c.n_real]):
        c.q.append(s["q"]); c.p1.append(s["p1"]); c.a.append(s.get("a"))
    # a missing label (grammar stopped the teacher) ends the labelled prefix: treat as rejected from there on
    for d in range(len(c.a)):
        if c.a[d] is None:
            c.a[d:] = [0.0] * (len(c.a) - d)
            break
    stops = [s.get("stop") for s in r["draft"]]
    n_stopped += any(s in ("p_min", "iso") for s in stops)
    c.draft_us = r.get("draft_us")
    c.cycle_us = r.get("cycle_us")
    c.n_draft = r["n_draft"]
    cyc.append(c)
if n_stopped:
    print(f"WARNING: {n_stopped} cycles stopped on p-min or the stop rule; trace with the rule off and p-min 0", file=sys.stderr)

N = max(c.n_draft for c in cyc)  # the padded width - 1, i.e. the traced n-max
print(f"{len(cyc)} cycles, {len(pids)} requests, traced n-max {N}, "
      f"{100 * sum(c.n_real == N for c in cyc) / len(cyc):.1f}% full drafts; "
      f"dev/val/test cycles {[sum(c.split == s for c in cyc) for s in range(3)]}")

# ---- isotonic fits on dev pairs (every real drafted depth)


def pav(xs, ys):
    """non-decreasing least-squares fit; returns (unique x, fitted y), predict by linear interpolation (sklearn's)."""
    agg = {}
    for x, y in zip(xs, ys):
        s = agg.setdefault(x, [0.0, 0.0])
        s[0] += y; s[1] += 1.0
    ux = sorted(agg)
    blocks = []  # [mean, weight, count]
    for x in ux:
        s, w = agg[x]
        blocks.append([s / w, w, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, c2 = blocks.pop()
            m1, w1, c1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    fy = []
    for m, _, cnt in blocks:
        fy += [min(1.0, max(0.0, m))] * cnt
    return ux, fy


def interp(v, xs, ys):
    if v <= xs[0]:
        return ys[0]
    if v >= xs[-1]:
        return ys[-1]
    j = bisect.bisect_right(xs, v)
    t = (v - xs[j - 1]) / (xs[j] - xs[j - 1])
    return ys[j - 1] + t * (ys[j] - ys[j - 1])


dq, dp, da = [], [], []
for c in cyc:
    if c.split == 0:
        dq += c.q; dp += c.p1; da += c.a
if len(da) < 200:
    sys.exit(f"only {len(da)} labelled dev pairs; trace more prompts")
fq = pav(dq, da)
fp = pav(dp, da)
tab_q = [interp(x, *fq) for x in GRID]
tab_p = [interp(x, *fp) for x in GRID]
print(f"{len(da)} dev pairs, mean acceptance {statistics.fmean(da):.3f}")

# the server interpolates linearly between the table points: simulate with that
lk = lambda tab, v: interp(v, GRID, tab)

# ---- cost model (us per cycle). Padded verify: one width, so verify + the rest is one constant
if a.draft_us:
    d0, d1 = (float(v) for v in a.draft_us.split(","))
else:
    pts = [(c.steps, c.draft_us) for c in cyc if c.draft_us is not None]
    if len({s for s, _ in pts}) > 1:
        mx = statistics.fmean(s for s, _ in pts); my = statistics.fmean(t for _, t in pts)
        d1 = sum((s - mx) * (t - my) for s, t in pts) / sum((s - mx) ** 2 for s, _ in pts)
        d0 = my - d1 * mx
    else:  # every cycle decoded the same number of steps: no intercept to separate
        d0, d1 = 0.0, statistics.fmean(t / max(s, 1) for s, t in pts)
        print("NOTE: every traced cycle decoded the same number of draft steps; draft cost taken as proportional "
              "(--draft-us d0,d1 refines it)")
if a.rest_us is not None:
    rest = a.rest_us
else:
    rest = statistics.median(c.cycle_us - c.draft_us for c in cyc
                             if c.n_real == N and c.cycle_us is not None and c.draft_us is not None)
print(f"cost model: draft {d0:.0f} + {d1:.0f} us/step, verify+rest {rest:.0f} us (includes the trace's own overhead)")
REST_W = {int(w): float(u) for w, u in (kv.split(":") for kv in a.rest_by_width.split(",") if kv)}
if REST_W:
    print(f"verify+rest by width (unpadded): {REST_W}")

# ---- simulation (sim.py): k = drafted tokens kept, steps = draft decodes; E = 1 + sum_{j<k} prod_{i<=j} a_i
for c in cyc:
    s, c.S = 1.0, []
    for x in c.a:
        s *= x
        c.S.append(s)
    post = [lk(tab_q, x) for x in c.q]
    c.score, cp = [], 1.0
    for d in range(c.n_real):
        c.score.append(cp * lk(tab_p, c.p1[d]))
        cp *= post[d]


def simulate(stop):
    """stop(c, d) True: token d is not added. Returns per-cycle (E, T, k)."""
    out = []
    for c in cyc:
        k = c.n_real
        for d in range(c.n_real):
            if stop(c, d):
                k = d
                break
        stp = k + 1 if k < c.n_real else max(c.steps, c.n_real)
        out.append((1.0 + sum(c.S[:k]), d0 * (stp > 0) + d1 * stp + REST_W.get(k + 1, rest), k))
    return out


def rate(ET, split, w=None):
    se = st = 0.0
    for c, (E, T, _) in zip(cyc, ET):
        if c.split == split:
            wc = 1.0 if w is None else w.get(c.pid, 0)
            se += wc * E; st += wc * T
    return se / st * 1e6 if st > 0 else float("nan")


base = simulate(lambda c, d: False)
best = None
print("\ntheta  val t/s  test t/s  kept/cycle(val)  accepted(val)")
for th in THETAS:
    ET = simulate(lambda c, d: c.score[d] < th)
    rv = rate(ET, 1)
    if best is None or rv > best[0]:
        best = (rv, th, ET)
    kv = [k for c, (_, _, k) in zip(cyc, ET) if c.split == 1]
    ev = [E - 1 for c, (E, _, _) in zip(cyc, ET) if c.split == 1]
    print(f"{th:5.3f}  {rv:7.2f}  {rate(ET, 2):8.2f}  {statistics.fmean(kv) if kv else 0:15.2f}  {statistics.fmean(ev) if ev else 0:13.2f}")
print("\np-min (same width)  val t/s  test t/s")
for pm in PMINS:
    ET = simulate(lambda c, d: c.p1[d] < pm)
    print(f"{pm:19.2f}  {rate(ET, 1):7.2f}  {rate(ET, 2):8.2f}")

_, theta, best_ET = best
g = rate(best_ET, 2) / rate(base, 2) - 1
test_pids = sorted({c.pid for c in cyc if c.split == 2})
rnd = random.Random(0)
gs = []
for _ in range(a.boot if test_pids else 0):
    w = {}
    for p in rnd.choices(test_pids, k=len(test_pids)):
        w[p] = w.get(p, 0) + 1
    gs.append(rate(best_ET, 2, w) / rate(base, 2, w) - 1)
gs.sort()
ci = (gs[int(0.025 * len(gs))], gs[int(0.975 * len(gs)) - 1]) if gs else (float("nan"), float("nan"))
print(f"\nchosen on val: theta {theta:.3f}; SIMULATED test gain vs no stop rule at n-max {N} (same padded width): "
      f"{g * 100:+.1f}% [{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}]")
if theta == 0.0:
    print("theta 0 won on val: the rule does not pay on these traces (theta 0 in the file leaves it off)")

with open(a.out, "w", encoding="utf-8") as f:
    f.write(f"# MTP stop rule tables, fit_stop_rule.py: {', '.join(os.path.basename(t) for t in a.trace)}\n")
    f.write(f"# {len(cyc)} cycles, n-max {N}, {len(da)} dev pairs; theta chosen on val; simulated test gain {g * 100:+.1f}%\n")
    f.write(f"# use with: --spec-type draft-mtp --spec-draft-n-max {N} --spec-draft-p-min 0 --spec-draft-stop-iso-file {os.path.basename(a.out)}\n")
    f.write(f"theta {theta:.4f}\n")  # 0: the server leaves the rule off
    f.write("x  " + " ".join(f"{v:.3f}" for v in GRID) + "\n")
    f.write("q  " + " ".join(f"{v:.4f}" for v in tab_q) + "\n")
    f.write("p1 " + " ".join(f"{v:.4f}" for v in tab_p) + "\n")
print(f"wrote {a.out}")
