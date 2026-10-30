# run_suite.py OUT.jsonl: send the prompt suite to a running llama-server (one request at a time) and write one JSON
# row per request: prompt id, category, split, task id (links the row to the --spec-trace records), timings.
# Ported from the P100 fork's p100-docs/mtp-research/scripts/run_suite.py (no API key, local server, our defaults).
#
#   python run_suite.py runs/T-n4.jsonl --url http://127.0.0.1:8080 --sampling greedy --splits dev,val,test
import argparse, json, statistics, time, urllib.request, collections, os

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--url", default="http://127.0.0.1:8080")
ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts-p100.json"))
ap.add_argument("--splits", default="dev,val,test")
ap.add_argument("--cats", default="", help="comma list of categories (default: all)")
ap.add_argument("--exclude-cats", default="", help="e.g. longctx (prompts up to ~7k tokens)")
ap.add_argument("--seeds", default="11")
ap.add_argument("--max-tokens", type=int, default=300)
ap.add_argument("--sampling", default="greedy", choices=["greedy", "nothink", "think"])
ap.add_argument("--think", default="off", choices=["off", "on"], help="chat_template_kwargs.enable_thinking")
ap.add_argument("--temp", type=float, default=None, help="override the sampling set's temperature")
a = ap.parse_args()

S = {"greedy":  dict(temperature=0.0, top_k=1),
     "nothink": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5),
     "think":   dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0)}[a.sampling]
if a.temp is not None:
    S["temperature"] = a.temp
splits = a.splits.split(",")
cats = set(a.cats.split(",")) - {""}
excl = set(a.exclude_cats.split(",")) - {""}
suite = [p for p in json.load(open(a.prompts, encoding="utf-8"))
         if p["split"] in splits and (not cats or p["category"] in cats) and p["category"] not in excl]

H = {"Content-Type": "application/json"}


def post(path, body):
    req = urllib.request.Request(a.url + path, json.dumps(body).encode(), H)
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())


def slot_task():
    try:
        sl = json.loads(urllib.request.urlopen(a.url + "/slots", timeout=30).read())
        return sl[0].get("id_task")
    except Exception:
        return None


os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
rows = []
with open(a.out, "w", encoding="utf-8") as out:
    for seed in [int(s) for s in a.seeds.split(",")]:
        for p in suite:
            b = dict(messages=[{"role": "user", "content": p["prompt"]}], max_tokens=a.max_tokens, seed=seed, **S,
                     chat_template_kwargs={"enable_thinking": a.think == "on"})
            if "tools" in p:
                b["tools"] = p["tools"]
            t0 = time.time()
            r = post("/v1/chat/completions", b)
            wall = time.time() - t0
            t = r.get("timings", {})
            row = dict(id=p["id"], category=p["category"], split=p["split"], seed=seed, wall_s=wall, task=slot_task(),
                       prompt_n=t.get("prompt_n"), predicted_n=t.get("predicted_n"), predicted_ms=t.get("predicted_ms"),
                       draft_n=t.get("draft_n", 0), draft_acc=t.get("draft_n_accepted", 0),
                       finish=r["choices"][0].get("finish_reason"))
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{p['id']:14s} {row['predicted_n']:5d} tok {row['predicted_n'] / max(row['predicted_ms'], 1e-9) * 1000:6.2f} t/s "
                  f"acc {row['draft_acc']}/{row['draft_n']}", flush=True)


def summ(rs):
    n = sum(r["predicted_n"] for r in rs); ms = sum(r["predicted_ms"] for r in rs)
    dn = sum(r["draft_n"] for r in rs); da = sum(r["draft_acc"] for r in rs)
    med = statistics.median([r["predicted_n"] / r["predicted_ms"] * 1000 for r in rs if r["predicted_n"] >= 8] or [0])
    return f"{n:6d} tok {n / ms * 1000:6.2f} t/s (median {med:6.2f}) acc {da / max(dn, 1):.3f} ({da}/{dn})"


by = collections.defaultdict(list)
for r in rows:
    by[r["category"]].append(r)
for c, rs in sorted(by.items()):
    print(f"{c:9s} {summ(rs)}")
print(f"{'ALL':9s} {summ(rows)}")
