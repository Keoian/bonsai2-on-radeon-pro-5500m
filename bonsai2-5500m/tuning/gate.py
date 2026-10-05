"""Correctness gates for the 5500M tuning run.

  python gate.py kld-base            reference logits (start build, f16 KV, ctx 1024, 12 chunks)
  python gate.py kld  --label L      KLD of --bin build (real config q8_0 K / q4_0 V) vs the reference
  python gate.py prompts --label L   8 fixed prompts, greedy, thinking on; checks final answers
  python gate.py slot --label L      slot save -> restart -> restore -> greedy continue == continuous run
Results go to runs/<label>.json under "gate_*".
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bench import MODEL, REPO, corpus_text  # noqa
DEF_BIN = os.path.join(REPO, "build", "bin")
# the KLD reference build: binaries of tag 5500m-start copied to build/bin-5500m-start
START_BIN = os.path.join(REPO, "build", "bin-5500m-start")
KLD_BASE = os.path.join(HERE, "kld-base.bin")
KLD_TEXT = os.path.join(HERE, "kld-text.txt")
KV = ["-ctk", "q8_0", "-ctv", "q4_0"]
PORT = 8098

sys.path.insert(0, HERE)
from bench import save  # noqa


def kld_text():
    if not os.path.exists(KLD_TEXT):
        t = corpus_text()
        open(KLD_TEXT, "w", encoding="utf-8").write(t[300000:420000])
    return KLD_TEXT


def run_ppl(binp, extra, env=None):
    cmd = [os.path.join(binp, "llama-perplexity.exe"), "-m", MODEL, "-ngl", "99", "-fa", "on", "-c", "1024", "-ub", "256",
           "--chunks", "12", "-f", kld_text()] + extra
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, **(env or {})})
    return p.stdout + p.stderr


def kld_base(a):
    out = run_ppl(START_BIN, ["-ctk", "f16", "-ctv", "f16", "--kl-divergence-base", KLD_BASE])
    open(os.path.join(HERE, "runs", "kld-base.log"), "w", encoding="utf-8").write(out)
    print(re.findall(r"Final estimate.*", out))


def kld(a):
    out = run_ppl(a.bin, KV + ["--kl-divergence-base", KLD_BASE, "--kl-divergence"], a.env)
    open(os.path.join(HERE, "runs", f"{a.label}-kld.log"), "w", encoding="utf-8").write(out)
    def grab(name):
        m = re.search(name + r"\s*:\s*([-0-9.]+)\s*±\s*([0-9.]+)", out)
        return [float(m.group(1)), float(m.group(2))] if m else None
    res = {"mean_kld": grab("Mean    KLD"), "ppl": grab(r"Mean PPL\(Q\)"), "same_top": grab(r"Same top p"),
           "rms_dp": grab(r"RMS Δp")}
    save(a.label, "gate_kld", res)
    print("kld:", res)


# ------------------------------------------------------------------ server helpers
def http(path, body=None, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Server:
    def __init__(self, binp, extra, env, log):
        self.cmd = [os.path.join(binp, "llama-server.exe"), "-m", MODEL, "--port", str(PORT), "-ngl", "99", "-fa", "on",
                    "-c", "17408", "-np", "1", *KV, "--jinja"] + extra
        self.env, self.log = env, log

    def __enter__(self):
        self.f = open(self.log, "a", encoding="utf-8")
        self.p = subprocess.Popen(self.cmd, stdout=self.f, stderr=subprocess.STDOUT, env={**os.environ, **self.env})
        for _ in range(600):
            try:
                if http("/health").get("status") == "ok": return self
            except Exception: pass
            if self.p.poll() is not None: raise RuntimeError("server died")
            time.sleep(1)
        raise RuntimeError("server did not start")

    def __exit__(self, *a):
        self.p.terminate(); self.p.wait(); self.f.close(); time.sleep(2)


PROMPTS = [  # (prompt, regex the final answer must match)
    ("What is 17 * 23? Reply with just the number at the end.", r"\b391\b"),
    ("A train travels 180 km in 2.5 hours. What is its average speed in km/h? Give the number.", r"\b72\b"),
    ("What is the sum of the first 20 positive integers? Give just the number at the end.", r"\b210\b"),
    ("Write a Python function `is_prime(n)` that returns True if n is prime. Output only the code in a code block.",
     r"def is_prime\s*\(\s*n"),
    ("What does this Python print?\n```python\nprint(sorted([3, 1, 2])[::-1])\n```\nAnswer with the exact output.",
     r"\[3,\s*2,\s*1\]"),
    ("Translate to French: 'The cat is sleeping on the table.' Give only the translation.",
     r"(?i)le chat dort sur la table"),
    ("What is the capital of Australia? Answer in one word.", r"(?i)canberra"),
    ("What is the chemical symbol for gold? Answer with just the symbol.", r"\bAu\b"),
]


def prompts(a):
    res = []
    with Server(a.bin, [], a.env, os.path.join(HERE, "runs", f"{a.label}-prompts-server.log")):
        for q, rx in PROMPTS:
            t0 = time.time()
            r = http("/v1/chat/completions", {"messages": [{"role": "user", "content": q}], "max_tokens": 6000,
                                              "temperature": 0, "seed": 1})
            msg = r["choices"][0]["message"]
            content = msg.get("content") or ""
            ok = bool(re.search(rx, content))
            res.append({"q": q[:40], "ok": ok, "finish": r["choices"][0]["finish_reason"],
                        "n": r["usage"]["completion_tokens"], "s": round(time.time() - t0), "answer": content[-160:]})
            print(res[-1], flush=True)
    passed = sum(x["ok"] for x in res)
    save(a.label, "gate_prompts", {"passed": passed, "total": len(res), "detail": res})
    print(f"prompts: {passed}/{len(res)}")


def slot(a):
    slotdir = os.path.join(HERE, "runs", "slots")
    shutil.rmtree(slotdir, ignore_errors=True); os.makedirs(slotdir)
    text = corpus_text()[50000:62000]
    log = os.path.join(HERE, "runs", f"{a.label}-slot-server.log")
    extra = ["--slot-save-path", slotdir]
    # P = cached prefix, S = new suffix. Continuous: P+S in one request. Restored: P cached, saved, server
    # restarted, slot restored, then P+S (only S is evaluated on top of the restored state).
    with Server(a.bin, extra, a.env, log):
        toks = http("/tokenize", {"content": text})["tokens"]
        P, S = toks[:-40], toks[-40:]
        body = {"prompt": P + S, "n_predict": 64, "temperature": 0, "cache_prompt": True}
        cont = http("/completion", body)["content"]
    with Server(a.bin, extra, a.env, log):
        http("/completion", {"prompt": P, "n_predict": 1, "temperature": 0, "cache_prompt": True})
        http("/slots/0?action=save", {"filename": "s.bin"})
    with Server(a.bin, extra, a.env, log):
        http("/slots/0?action=restore", {"filename": "s.bin"})
        r = http("/completion", body)
        rest = r["content"]
        reused = r.get("timings", {}).get("prompt_n")
    ok = cont == rest
    save(a.label, "gate_slot", {"identical": ok, "prompt_n_after_restore": reused, "cont": cont[:80], "restored": rest[:80]})
    print(f"slot save/restore identical: {ok} (prompt tokens processed after restore: {reused})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["kld-base", "kld", "prompts", "slot"])
    ap.add_argument("--label", default="x")
    ap.add_argument("--bin", default=DEF_BIN)
    ap.add_argument("--env", default="")
    a = ap.parse_args()
    a.env = dict(kv.split("=", 1) for kv in a.env.split(",") if kv)
    {"kld-base": kld_base, "kld": kld, "prompts": prompts, "slot": slot}[a.mode](a)
