"""PTQ1_0 MUL_MAT perf at the Bonsai projection shapes by column count (test-backend-ops perf)."""
import os, re, subprocess
out = subprocess.run([r"C:\git\0\bonsai2\llama.cpp\build-v2\bin\test-backend-ops.exe", "perf", "-o", "MUL_MAT", "-b", "CUDA0", "-p", "type_a=ptq1_0"],
                     capture_output=True, text=True, env=dict(os.environ, PATH=r"C:\git\0\bonsai2\cuda\bin;" + os.environ["PATH"])).stdout
res = {}
for l in out.splitlines():
    m = re.search(r'm=(\d+),n=(\d+),k=(\d+),bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1,src_overlap=0\).*?([0-9.]+) us/run', l)
    if m:
        res[(int(m[1]), int(m[3]), int(m[2]))] = float(m[4])
ns = range(1, 7)
print('     m x     k ' + ''.join(f'    n={n}' for n in ns))
for a, b in sorted({(a, b) for a, b, _ in res}):
    print(f'{a:6d} x {b:5d} ' + ''.join(f'{res.get((a, b, n), float("nan")):8.1f}' for n in ns))
