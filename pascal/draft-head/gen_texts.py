# gen_texts.py OUT.jsonl: generate answers to the suite's dev/val prompts on a running llama-server and save the text
# (prompts: mtp-fit/prompts-p100.json next to the repo; token frequencies for a trimmed MTP draft head; the test split stays held out).
import argparse, json, os, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--splits", default="dev,val")
ap.add_argument("--seeds", default="11,12,13")
ap.add_argument("--max-tokens", type=int, default=400)
a = ap.parse_args()

suite = [p for p in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "mtp-fit", "prompts-p100.json"), encoding="utf-8"))
         if p["split"] in a.splits.split(",")]
with open(a.out, "w", encoding="utf-8") as out:
    for seed in [int(s) for s in a.seeds.split(",")]:
        for p in suite:
            b = dict(messages=[{"role": "user", "content": p["prompt"]}], max_tokens=a.max_tokens, seed=seed,
                     temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, chat_template_kwargs={"enable_thinking": False})
            if "tools" in p:
                b["tools"] = p["tools"]
            req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
            m = json.loads(urllib.request.urlopen(req, timeout=1800).read())["choices"][0]["message"]
            text = (m.get("content") or "") + "".join(json.dumps(tc) for tc in (m.get("tool_calls") or []))
            out.write(json.dumps(dict(id=p["id"], seed=seed, prompt=p["prompt"], text=text)) + "\n")
            out.flush()
            print(p["id"], seed, len(text), flush=True)
