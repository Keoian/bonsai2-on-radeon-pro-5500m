"""Run test-backend-ops FLASH_ATTN_EXT perf at the Bonsai shape for several GGML_CUDA_FA_VEC_GQA folds and tabulate us/run."""
import os, re, subprocess, sys
BIN = r"C:\git\0\bonsai2\llama.cpp\build-v2\bin\test-backend-ops.exe"
folds = sys.argv[1].split(",") if len(sys.argv) > 1 else ["1", "2", "3", "6"]
pat = sys.argv[2] if len(sys.argv) > 2 else r"hsk=256,hsv=256,nh=4,nr23=\[6,1\].*type_K=q8_0"
res = {}
for f in folds:
    env = dict(os.environ, **{os.environ.get("FOLDVAR", "GGML_CUDA_FA_VEC_GQA"): f}, PATH=r"C:\git\0\bonsai2\cuda\bin;" + os.environ["PATH"])
    out = subprocess.run([BIN, "perf", "-o", "FLASH_ATTN_EXT", "-b", "CUDA0", "-p", pat], env=env, capture_output=True, text=True).stdout
    name = None
    for line in out.splitlines():
        m = re.search(r"FLASH_ATTN_EXT\((.*?)\):", line)
        if m:
            name = m.group(1); k = 0
        u = re.search(r"([0-9.]+) us/run", line)
        if u and name:
            kv = re.search(r"kv=(\d+),nb=(\d+)", name); tv = re.search(r"type_V=(\w+)", name)
            key = (int(kv.group(1)), int(kv.group(2)), tv.group(1), k); k += 1
            res.setdefault(key, {})[f] = float(u.group(1))
print("kv     nb V    var | " + " ".join(f"fold{f:>3}" for f in folds) + " | best vs fold1")
for key in sorted(res):
    r = res[key]
    vals = " ".join(f"{r.get(f, float('nan')):8.1f}" for f in folds)
    print(f"{key[0]:6d} {key[1]}  {key[2]} {key[3]}   | {vals} | {r[folds[0]] / min(r.values()):.2f}x")
