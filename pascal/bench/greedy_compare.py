"""Greedy-decode a few prompts through llama-server and save the replies, to compare builds or env switches.

usage: python greedy_compare.py <bin_dir> <out.json> [ENV=VAL ...]
"""
import json, os, subprocess, sys, time, urllib.request

ROOT = r"C:\git\0\bonsai2"
MODEL = ROOT + r"\models\Ternary-Bonsai-2-27B-PTQ1_0.gguf"
PORT = 8091
PROMPTS = [
    "Explain how a hash map handles collisions, with a short C example.",
    "Write a haiku about Pascal GPUs, then explain its imagery.",
    "List five prime numbers greater than 100 and show how you checked each one.",
    "Summarize the causes of the French Revolution in one paragraph.",
]

def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

def main():
    bin_dir, out = sys.argv[1], sys.argv[2]
    env = dict(os.environ)
    env["PATH"] = ROOT + r"\cuda\bin;" + env["PATH"]
    for kv in sys.argv[3:]:
        k, v = kv.split("=", 1)
        env[k] = v
    extra = env.pop("SERVER_ARGS", "").split()
    model = env.pop("MODEL_OVERRIDE", MODEL)
    cmd = [os.path.join(bin_dir, "llama-server.exe"), "-m", model, "-ngl", "99", "-fa", "on", "-c", "8192",
           "-ctk", "q8_0", "-ctv", "q8_0", "-np", "1", "--jinja", "--port", str(PORT), "--no-webui"] + extra
    log = open(out + ".log", "w")
    srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            try:
                if json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        res = []
        for p in PROMPTS:
            r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "temperature": float(env.get("SAMPLE_TEMP", "0")), "top_p": 0.8, "top_k": 20, "min_p": 0, "max_tokens": 256,
                                              "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}})
            t = r.get("timings", {})
            res.append({"prompt": p, "text": r["choices"][0]["message"].get("content", ""),
                        "tps": t.get("predicted_per_second"), "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted")})
            print(f"{t.get('predicted_per_second', 0):6.2f} t/s  draft {t.get('draft_n')}/{t.get('draft_n_accepted')}  {p[:40]}")
        json.dump(res, open(out, "w"), indent=1)
    finally:
        srv.terminate()
        srv.wait()

if __name__ == "__main__":
    main()
