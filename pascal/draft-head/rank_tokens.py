# rank_tokens.py OUT_IDS.txt TRAIN.jsonl[,...] [--corpus FILE] [--heldout HELD.jsonl]: count token ids of generated
# answers (gen_texts.py rows; prompts count at weight 1, answers at --answer-weight) through a running llama-server's
# /tokenize, rank by count, then the unseen ids in id order (BPE ids roughly follow merge order). With --heldout, print
# the share of held-out answer tokens inside the top K for a few K.
import argparse, collections, json, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("train")
ap.add_argument("--corpus", default="")
ap.add_argument("--heldout", default="")
ap.add_argument("--answer-weight", type=int, default=4)
ap.add_argument("--n-vocab", type=int, default=248320)
ap.add_argument("--url", default="http://127.0.0.1:8080")
a = ap.parse_args()


def tok(text):
    req = urllib.request.Request(a.url + "/tokenize", json.dumps({"content": text}).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["tokens"]


cnt = collections.Counter()
prompts = set()
for path in a.train.split(","):
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        for t in tok(row["text"]):
            cnt[t] += a.answer_weight
        if row["prompt"] not in prompts:
            prompts.add(row["prompt"])
            cnt.update(tok(row["prompt"]))
if a.corpus:
    text = open(a.corpus, encoding="utf-8", errors="ignore").read()
    for i in range(0, len(text), 100000):
        cnt.update(tok(text[i:i + 100000]))

ranked = [t for t, _ in cnt.most_common()]
seen = set(ranked)
ranked += [t for t in range(a.n_vocab) if t not in seen]
open(a.out, "w").write("\n".join(map(str, ranked)) + "\n")
print(f"{len(seen)} distinct ids seen, {sum(cnt.values())} weighted tokens -> {a.out}")

if a.heldout:
    held = [t for line in open(a.heldout, encoding="utf-8") for t in tok(json.loads(line)["text"])]
    rank = {t: i for i, t in enumerate(ranked)}
    for k in (8192, 16384, 32768, 65536):
        print(f"top {k:6d}: {sum(rank[t] < k for t in held) / len(held):.4f} of {len(held)} held-out answer tokens")
