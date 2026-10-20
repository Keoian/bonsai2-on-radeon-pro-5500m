"""Benchmark harness for Bonsai 2 27B PTQ1_0 on the Radeon Pro 5500M (Vulkan).

  python bench.py bench  --label L [--tests tg0,tg16k,...] [--reps 3] [--cool 60] [--bin DIR]
  python bench.py heartbeat --label L [--reps 3] [--cool 60] [--bin DIR]

Every timed run is a separate process, followed by a cooldown. Results (all reps + median) are
appended to runs/<label>.json. VRAM is sampled from the Windows "GPU Adapter Memory" counter
(adapter-wide dedicated usage, i.e. includes other apps); the pre-run idle level is recorded too.
"""
import argparse, json, os, statistics, subprocess, sys, time, csv, re, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# this folder is <repo>/bonsai2-5500m/tuning; the kit root (models/, Bonsai-demo/) is the repo's parent
REPO = os.path.dirname(os.path.dirname(HERE))
KIT = os.environ.get("BONSAI_KIT", os.path.dirname(REPO))
MODEL_DIR = os.environ.get("BONSAI_MODEL_DIR", os.path.join(KIT, "models"))
MODEL = os.path.join(MODEL_DIR, "Ternary-Bonsai-2-27B-PTQ1_0.gguf")
MMPROJ = os.path.join(MODEL_DIR, "Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf")
DEF_BIN = os.path.join(REPO, "build", "bin")
KV = ["-ctk", "q8_0", "-ctv", "q4_0"]

TESTS = {  # name: (n_prompt, n_gen, depth)
    "tg0": (0, 128, 0), "tg8k": (0, 128, 8192), "tg16k": (0, 128, 16384), "tg32k": (0, 128, 32768),
    "pp0": (512, 0, 0), "pp16k": (512, 0, 16384),
}


def corpus_text():
    """Benchmark text: the concatenated Markdown files listed in corpus-files.txt (paths relative to the kit root)."""
    p = os.path.join(HERE, "corpus.txt")
    if not os.path.exists(p):
        names = [l.strip() for l in open(os.path.join(HERE, "corpus-files.txt"), encoding="utf-8") if l.strip()]
        names = [n.replace("llama.cpp/", os.path.basename(REPO) + "/", 1) if n.startswith("llama.cpp/") else n for n in names]
        with open(p, "wb") as out:
            for n in names:
                with open(os.path.join(KIT, n), "rb") as f:
                    out.write(f.read())
    return open(p, encoding="utf-8").read()


def amd_counter():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
        "Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' | Select-Object -Expand CounterSamples | "
        "Sort-Object CookedValue -Descending | Select-Object -First 1 -Expand Path"],
        capture_output=True, text=True).stdout.strip()
    return out.replace("\\\\" + os.environ.get("COMPUTERNAME", "").lower(), "") or None


class VramSampler:
    """Adapter-wide and per-process GPU memory (MB), sampled by vramsampler.ps1."""
    def __init__(self, path, procname):
        self.path, self.procname, self.proc = path, procname, None

    def idle(self):
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
            "$i=((Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples | Sort-Object CookedValue -Descending | Select-Object -First 1).InstanceName;"
            "(Get-Counter \"\\GPU Adapter Memory($i)\\Dedicated Usage\",\"\\GPU Adapter Memory($i)\\Shared Usage\").CounterSamples | ForEach-Object { [int]($_.CookedValue/1MB) }"],
            capture_output=True, text=True).stdout.split()
        return {"ded": int(out[0]), "shared": int(out[1])} if len(out) == 2 else {}

    def __enter__(self):
        if os.path.exists(self.path): os.remove(self.path)
        self.proc = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(HERE, "vramsampler.ps1"),
                                      "-Out", self.path, "-ProcName", self.procname],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        return self

    def __exit__(self, *a):
        self.proc.kill(); self.proc.wait()
        time.sleep(0.5)

    def peaks(self):
        try:
            with open(self.path, newline="") as f:
                rows = list(csv.DictReader(f))
            return {k: max(float(r[k] or 0) for r in rows) for k in ("adapter_ded", "adapter_shared", "proc_ded", "proc_shared")}
        except Exception as e:
            return {"err": str(e)}


def save(label, key, data):
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    p = os.path.join(HERE, "runs", f"{label}.json")
    d = json.load(open(p)) if os.path.exists(p) else {}
    d[key] = data
    json.dump(d, open(p, "w"), indent=1)


def run_bench(args):
    exe = os.path.join(args.bin, "llama-bench.exe")
    for name in args.tests.split(","):
        npp, ntg, depth = TESTS[name]
        reps = []
        for r in range(args.reps):
            s = VramSampler(os.path.join(HERE, "runs", "_vram.csv"), "llama-bench")
            idle = s.idle()
            cmd = [exe, "-m", MODEL, "-ngl", "99", "-fa", "1", *KV, "-r", "1", "-o", "json",
                   "-p", str(npp), "-n", str(ntg), "-d", str(depth)] + args.extra
            t0 = time.time()
            with s:
                p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **args.env})
            wall = time.time() - t0
            try:
                j = json.loads(p.stdout)
                ts = j[0]["avg_ts"]
            except Exception:
                print(p.stdout[-2000:], p.stderr[-3000:]); ts = float("nan")
            pk = s.peaks()
            reps.append({"ts": ts, "vram": pk, "idle": idle, "wall_s": wall})
            print(f"  {name} rep{r}: {ts:.2f} t/s  vram {pk} idle {idle}  wall {wall:.0f}s", flush=True)
            if r < args.reps - 1 or name != args.tests.split(",")[-1]:
                time.sleep(args.cool)
        med = statistics.median(x["ts"] for x in reps)
        res = {"median_ts": med, "reps": reps}
        save(args.label, name, res)
        print(f"{name}: median {med:.2f} t/s", flush=True)


# ---------------------------------------------------------------- server helpers
CTX_FALLBACK = [32768, 28672, 24576, 20480, 17408]


def start_server(cmd, log, args, base):
    """Start llama-server; if the KV cache does not fit, retry with the next smaller context."""
    ci = cmd.index("-c") + 1
    for ctx in [c for c in CTX_FALLBACK if c <= int(cmd[ci])]:
        cmd[ci] = str(ctx)
        log.write(f"=== starting with -c {ctx}\n"); log.flush()
        srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **args.env})
        for _ in range(600):
            try:
                if http(base + "/health").get("status") == "ok":
                    print(f"  server up with -c {ctx}", flush=True)
                    return srv, ctx
            except Exception: pass
            if srv.poll() is not None: break
            time.sleep(1)
        srv.kill(); srv.wait(); time.sleep(3)
        print(f"  server failed with -c {ctx}", flush=True)
    raise RuntimeError("server did not start at any context size")


def http(url, body=None, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream_completion(url, body):
    body = dict(body, stream=True)
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; toks = []; final = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"): continue
            ev = json.loads(line[5:])
            if ttft is None and (ev.get("content") or ev.get("tokens")): ttft = time.time() - t0
            toks += ev.get("tokens") or []
            if ev.get("stop"): final = ev
    return ttft, time.time() - t0, toks, final


def run_heartbeat(args):
    port = 8099
    base = f"http://127.0.0.1:{port}"
    cmd = [os.path.join(args.bin, "llama-server.exe"), "-m", MODEL, "--port", str(port), "-ngl", "99", "-fa", "on",
           "-c", str(args.ctx), "-np", "1", *KV, "--mmproj", MMPROJ, "--no-mmproj-offload", "--jinja"] + args.extra
    log = open(os.path.join(HERE, "runs", f"{args.label}-server.log"), "w")
    s = VramSampler(os.path.join(HERE, "runs", "_vram.csv"), "llama-server")
    idle = s.idle()
    s.__enter__()
    srv, ctx_used = start_server(cmd, log, args, base)
    try:
        text = corpus_text()
        toks = http(base + "/tokenize", {"content": text[:200000]})["tokens"]
        prefix = toks[:args.depth]
        pos = args.depth + 1000
        t0 = time.time()
        http(base + "/completion", {"prompt": prefix, "n_predict": 1, "temperature": 0, "cache_prompt": True})
        print(f"  prefix {len(prefix)} tokens cached in {time.time()-t0:.0f}s", flush=True)
        seq = list(prefix)
        reps = []
        for r in range(args.reps):
            time.sleep(args.cool)
            append = toks[pos:pos + 300]; pos += 300
            seq = seq + append
            ttft, total, gen, final = stream_completion(base + "/completion", {
                "prompt": seq, "n_predict": 64, "temperature": 0, "cache_prompt": True,
                "ignore_eos": True, "return_tokens": True})
            tm = final.get("timings", {})
            reps.append({"ttft_s": ttft, "total_s": total, "prompt_n": tm.get("prompt_n"),
                         "prompt_ms": tm.get("prompt_ms"), "predicted_n": tm.get("predicted_n"),
                         "predicted_per_second": tm.get("predicted_per_second")})
            print(f"  heartbeat rep{r}: ttft {ttft:.2f}s total {total:.2f}s prompt_n {tm.get('prompt_n')} "
                  f"pp {tm.get('prompt_per_second', 0):.1f} t/s tg {tm.get('predicted_per_second', 0):.2f} t/s", flush=True)
            seq = seq + gen[:-1] if gen else seq  # the last sampled token is not yet in the cache
    finally:
        srv.terminate(); srv.wait(); s.__exit__(); log.close()
    res = {"median_ttft_s": statistics.median(x["ttft_s"] for x in reps),
           "median_total_s": statistics.median(x["total_s"] for x in reps),
           "reps": reps, "vram": s.peaks(), "idle": idle, "ctx": ctx_used}
    save(args.label, "heartbeat", res)
    print(f"heartbeat: ttft {res['median_ttft_s']:.2f}s total {res['median_total_s']:.2f}s  "
          f"vram {res['vram']} idle {idle}", flush=True)


# ---------------------------------------------------------------- server protocol (per experiment)
# One llama-server with the real config (32K ctx, q8_0 K / q4_0 V, -np 1). Depth comes from restoring a
# saved slot (16K / 32K prefix of the corpus) instead of re-prefilling, then appending a few new tokens so
# only those are evaluated. Every timed run restores the slot again, so all reps start from the same state.
SLOTDIR = os.path.join(HERE, "slots")
DEPTHS = {"16k": 16384, "32k": 32000}


def run_srv(args):
    port = 8099
    base = f"http://127.0.0.1:{port}"
    os.makedirs(SLOTDIR, exist_ok=True)
    cmd = [os.path.join(args.bin, "llama-server.exe"), "-m", MODEL, "--port", str(port), "-ngl", "99", "-fa", "on",
           "-c", str(args.ctx), "-np", "1", *KV, "--mmproj", MMPROJ, "--no-mmproj-offload", "--jinja",
           "--slot-save-path", SLOTDIR] + args.extra
    log = open(os.path.join(HERE, "runs", f"{args.label}-srv.log"), "w")
    s = VramSampler(os.path.join(HERE, "runs", "_vram.csv"), "llama-server")
    idle = s.idle()
    s.__enter__()
    srv, ctx_used = start_server(cmd, log, args, base)
    out = {"ctx": ctx_used}
    try:
        text = corpus_text()
        toks = http(base + "/tokenize", {"content": text[:400000]})["tokens"]
        tail = toks[40000:]  # tokens appended after a restored prefix (never part of a prefix)

        depths = {"16k": DEPTHS["16k"], "32k": min(DEPTHS["32k"], ctx_used - 768)}
        out["depth32"] = depths["32k"]

        def restore(d):
            fn = f"p{depths[d]}.bin"
            if not os.path.exists(os.path.join(SLOTDIR, fn)):
                t0 = time.time()
                http(base + "/completion", {"prompt": toks[:depths[d]], "n_predict": 1, "temperature": 0, "cache_prompt": True})
                http(base + "/slots/0?action=save", {"filename": fn})
                print(f"  built slot {fn} in {time.time()-t0:.0f}s", flush=True)
            else:
                http(base + "/slots/0?action=restore", {"filename": fn})

        def gen(prompt, n, stream=False):
            body = {"prompt": prompt, "n_predict": n, "temperature": 0, "cache_prompt": True, "ignore_eos": True}
            if stream:
                ttft, total, _, final = stream_completion(base + "/completion", body)
                return ttft, total, final["timings"]
            t0 = time.time(); r = http(base + "/completion", body)
            return None, time.time() - t0, r["timings"]

        plan = [t for t in args.tests.split(",")]
        first = True
        for name in plan:
            reps = []
            for r in range(args.reps):
                if not first: time.sleep(args.cool)
                first = False
                if name == "tg0":
                    http(base + "/slots/0?action=erase", {})
                    _, tot, tm = gen(tail[:16], 128)
                    val = tm["predicted_per_second"]
                elif name in ("tg16k", "tg32k"):
                    d = name[2:]; restore(d)
                    _, tot, tm = gen(toks[:depths[d]] + tail[:16], 128)
                    val = tm["predicted_per_second"]
                elif name == "pp16k":
                    restore("16k")
                    _, tot, tm = gen(toks[:DEPTHS["16k"]] + tail[:512], 1)
                    val = tm["prompt_per_second"]
                elif name == "hb":
                    restore("16k")
                    ttft, tot, tm = gen(toks[:DEPTHS["16k"]] + tail[:300], 64, stream=True)
                    val = {"ttft_s": ttft, "total_s": tot}
                reps.append({"val": val, "prompt_n": tm.get("prompt_n"), "predicted_n": tm.get("predicted_n"),
                             "wall_s": tot})
                print(f"  {name} rep{r}: {val} (prompt_n {tm.get('prompt_n')}, gen {tm.get('predicted_n')})", flush=True)
            if name == "hb":
                med = {"ttft_s": statistics.median(x["val"]["ttft_s"] for x in reps),
                       "total_s": statistics.median(x["val"]["total_s"] for x in reps)}
            else:
                med = statistics.median(x["val"] for x in reps)
            out[name] = {"median": med, "reps": reps}
            print(f"{name}: median {med}", flush=True)
    finally:
        srv.terminate(); srv.wait(); s.__exit__(); log.close()
    out["vram"] = s.peaks(); out["idle"] = idle
    for k, v in out.items(): save(args.label, "srv_" + k, v)
    print(f"srv vram {out['vram']} idle {idle}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bench", "heartbeat", "srv"])
    ap.add_argument("--label", required=True)
    ap.add_argument("--tests", default="tg0,tg16k,tg32k,pp16k")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cool", type=int, default=60)
    ap.add_argument("--bin", default=DEF_BIN)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--depth", type=int, default=16384)
    ap.add_argument("--env", default="", help="K=V,K=V")
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    a.env = dict(kv.split("=", 1) for kv in a.env.split(",") if kv)
    {"bench": run_bench, "heartbeat": run_heartbeat, "srv": run_srv}[a.mode](a)
