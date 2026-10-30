"""Load llama-server at a given context / KV type, push a ~3k token prompt, report VRAM and decode speed.

usage: python vram_probe.py <ctx> <ctk> <ctv> [mtp]
"""
import json, os, subprocess, sys, time, urllib.request

ROOT = r"C:\git\0\bonsai2"
PORT = 8092

def smi():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().split()[0])

def shared(pid):
    """MiB of this process's GPU allocations that Windows placed in system RAM (includes CUDA pinned host buffers)."""
    try:
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command",
            rf"(Get-Counter '\GPU Process Memory(pid_{pid}*)\Shared Usage').CounterSamples | Measure-Object CookedValue -Sum | % Sum"]).decode().strip()
        return int(float(out) / 2**20)
    except Exception:
        return -1

def main():
    ctx, ctk, ctv = sys.argv[1], sys.argv[2], sys.argv[3]
    mtp = len(sys.argv) > 4 and sys.argv[4] == "mtp"
    model = os.environ.get("MODEL") or ROOT + r"\models\Ternary-Bonsai-2-27B-PTQ1_0" + ("-MTP-Q4_K" if mtp else "") + ".gguf"
    env = dict(os.environ, PATH=ROOT + r"\cuda\bin;" + os.environ["PATH"], GGML_CUDA_FORCE_GRAPHS="1", GGML_CUDA_GDN_FUSION="1",
               LLAMA_ARG_SPEC_DRAFT_UBATCH="128")
    cmd = [os.path.join(os.environ.get("BIN", ROOT + r"\llama.cpp\build-v2\bin"), "llama-server.exe"), "-m", model, "-ngl", "99", "-fa", "on", "-c", ctx,
           "-ctk", ctk, "-ctv", ctv, "-np", "1", "-b", "512", "-ub", "512", "--jinja", "--port", str(PORT), "--no-webui"] + os.environ.get("SERVER_ARGS", "").split()
    if mtp:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", os.environ.get("NMAX", "1"), "-ctkd", ctk, "-ctvd", ctv]
    base = smi()
    srv = subprocess.Popen(cmd, env=env, stdout=open(ROOT + r"\results\vram_probe.log", "w"), stderr=subprocess.STDOUT)
    try:
        ok = False
        for _ in range(180):
            if srv.poll() is not None:
                break
            try:
                ok = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()).get("status") == "ok"
                if ok:
                    break
            except Exception:
                pass
            time.sleep(1)
        if not ok:
            print(f"ctx={ctx} {ctk}/{ctv} mtp={mtp}: FAILED to load (see results/vram_probe.log)")
            return
        loaded = smi()
        text = open(ROOT + r"\corpus.txt", encoding="utf-8", errors="ignore").read()[:int(os.environ.get("PROMPT_CHARS", "12000"))]
        body = {"messages": [{"role": "user", "content": "Summarize this text in three sentences:\n\n" + text}], "temperature": 0,
                "max_tokens": int(os.environ.get("GEN", "128")), "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        t = r.get("timings", {})
        print(f"ctx={ctx:>6} {ctk}/{ctv} mtp={mtp!s:5}: desktop {base} MiB, loaded {loaded} MiB, after prompt {smi()} MiB, "
              f"free {8192 - smi()} MiB | prompt {t.get('prompt_n')} tok {t.get('prompt_per_second', 0):.1f} t/s, "
              f"shared {shared(srv.pid)} MiB | decode {t.get('predicted_per_second', 0):.2f} t/s, draft {t.get('draft_n')}/{t.get('draft_n_accepted')}")
    finally:
        srv.terminate()
        srv.wait()

if __name__ == "__main__":
    main()
