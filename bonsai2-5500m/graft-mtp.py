"""Add the ProCreations MTP head to the Bonsai 2 27B PTQ1_0 GGUF, for start-server.ps1 -Mtp.

The head is the 15 blk.64 tensors (Q8_0, 451 MB) of ProCreations/Ternary-Bonsai-2-27B-MTP. Only those bytes and the
file header are downloaded (HTTP range requests); the PTQ1_0 trunk is copied byte for byte. With --head-type q4_k
(the default) llama-quantize then requantizes only the head: 202 MB less VRAM, same draft acceptance, 48k context.

usage: python graft-mtp.py [--base Ternary-Bonsai-2-27B-PTQ1_0.gguf] [--head-type q4_k|q8_0] [--out OUT.gguf]
"""
import argparse
import hashlib
import io
import os
import shutil
import struct
import subprocess
import sys
import urllib.request

DONOR_URL = 'https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP/resolve/main/Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf'
DONOR_HEAD_SHA256 = '38e09459ec912a9932d9808af7d0ebcd9042a90c8f98ddded5db35dbc4630c07'
DONOR_HEADER_BYTES = 25_000_000

SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
GGUF_TYPE_UINT32 = 4


class Reader:
    def __init__(self, f):
        self.f = f

    def rd(self, fmt):
        return struct.unpack('<' + fmt, self.f.read(struct.calcsize('<' + fmt)))

    def rstr(self):
        (n,) = self.rd('Q')
        return self.f.read(n).decode('utf-8')

    def skip_val(self, t):
        if t in SCALARS:
            self.f.read(struct.calcsize('<' + SCALARS[t]))
        elif t == 8:
            (n,) = self.rd('Q')
            self.f.read(n)
        else:
            (at,) = self.rd('I')
            (n,) = self.rd('Q')
            if at in SCALARS:
                self.f.read(n * struct.calcsize('<' + SCALARS[at]))
            else:
                for _ in range(n):
                    self.skip_val(at)

    def header(self):
        magic, ver = self.rd('4sI')
        if magic != b'GGUF':
            raise ValueError('not a GGUF file')
        nt, nkv = self.rd('QQ')
        kvs = []  # (key, type, raw bytes of the whole entry, scalar value or None)
        for _ in range(nkv):
            start = self.f.tell()
            k = self.rstr()
            (t,) = self.rd('I')
            vstart = self.f.tell()
            self.skip_val(t)
            end = self.f.tell()
            self.f.seek(start)
            raw = self.f.read(end - start)
            val = None
            if t in SCALARS:
                self.f.seek(vstart)
                val = self.rd(SCALARS[t])[0]
            self.f.seek(end)
            kvs.append((k, t, raw, val))
        tensors = []
        for _ in range(nt):
            name = self.rstr()
            (nd,) = self.rd('I')
            dims = self.rd('Q' * nd)
            (tt,) = self.rd('I')
            (off,) = self.rd('Q')
            tensors.append((name, dims, tt, off))
        return ver, kvs, tensors, self.f.tell()


def gstr(s):
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def kv_u32(key, v):
    return gstr(key) + struct.pack('<II', GGUF_TYPE_UINT32, v)


def tensor_info(name, dims, tt, off):
    return gstr(name) + struct.pack('<I', len(dims)) + struct.pack('<' + 'Q' * len(dims), *dims) + struct.pack('<IQ', tt, off)


def align(x, a):
    return (x + a - 1) // a * a


def fetch(first, last, out=None):
    req = urllib.request.Request(DONOR_URL, headers={'Range': f'bytes={first}-{last}', 'User-Agent': 'graft-mtp'})
    with urllib.request.urlopen(req) as r:
        if r.status != 206:
            raise RuntimeError(f'server ignored the range request (HTTP {r.status})')
        if out is None:
            return r.read(), int(r.headers['Content-Range'].rsplit('/', 1)[1])
        h = hashlib.sha256()
        n, total = 0, last - first + 1
        while chunk := r.read(8 << 20):
            out.write(chunk)
            h.update(chunk)
            n += len(chunk)
            print(f'\r  {n >> 20} / {total >> 20} MiB', end='', flush=True)
        print()
        if n != total:
            raise RuntimeError(f'short read: {n} of {total} bytes')
        return h.hexdigest()


def graft(base_path, out_path):
    with open(base_path, 'rb') as f:
        ver, kvs, tensors, info_end = Reader(f).header()
    kv_map = {k: val for k, t, raw, val in kvs}
    if kv_map.get('qwen35.block_count') != 64 or 'qwen35.nextn_predict_layers' in kv_map:
        raise ValueError(f'{base_path} is not the 64-layer Bonsai 2 27B file (or already has a head)')
    alignment = kv_map.get('general.alignment') or 32
    base_data_start = align(info_end, alignment)
    base_data_len = os.path.getsize(base_path) - base_data_start

    print('reading the donor header ...')
    data, donor_size = fetch(0, DONOR_HEADER_BYTES - 1)
    _, dkvs, dtensors, dinfo_end = Reader(io.BytesIO(data)).header()
    dalign = {k: val for k, t, raw, val in dkvs}.get('general.alignment') or 32
    head = sorted([t for t in dtensors if t[0].startswith('blk.64.')], key=lambda t: t[3])
    if len(head) != 15:
        raise ValueError(f'expected 15 head tensors in the donor, found {len(head)}')
    # the head is the tail of the donor's data section
    head_first = align(dinfo_end, dalign) + head[0][3]
    head_len = donor_size - head_first

    meta = b''
    for k, t, raw, val in kvs:
        meta += kv_u32(k, 65) if k == 'qwen35.block_count' else raw
    meta += kv_u32('qwen35.nextn_predict_layers', 1)
    head_base = align(base_data_len, alignment)
    infos = b''.join(tensor_info(*t) for t in tensors)
    infos += b''.join(tensor_info(name, dims, tt, head_base + (off - head[0][3])) for name, dims, tt, off in head)
    hdr = struct.pack('<4sIQQ', b'GGUF', ver, len(tensors) + len(head), len(kvs) + 1) + meta + infos

    print(f'copying the trunk to {out_path} ...')
    with open(out_path, 'wb') as out, open(base_path, 'rb') as src:
        out.write(hdr + b'\0' * (align(len(hdr), alignment) - len(hdr)))
        src.seek(base_data_start)
        shutil.copyfileobj(src, out, 64 << 20)
        out.write(b'\0' * (head_base - base_data_len))
        print(f'downloading the head ({head_len >> 20} MiB) ...')
        sha = fetch(head_first, head_first + head_len - 1, out)
    if sha != DONOR_HEAD_SHA256:
        os.remove(out_path)
        raise RuntimeError(f'head checksum mismatch ({sha}); the donor file changed upstream')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--base', default=os.path.join(here, 'models', 'Ternary-Bonsai-2-27B-PTQ1_0.gguf'))
    ap.add_argument('--head-type', default='q4_k', choices=['q4_k', 'q5_k', 'q6_k', 'q8_0'])
    ap.add_argument('--out')
    ap.add_argument('--quantize', default=os.path.join(os.path.dirname(here), 'build', 'bin', 'llama-quantize.exe'))
    args = ap.parse_args()

    stem = os.path.join(os.path.dirname(os.path.abspath(args.base)), 'Ternary-Bonsai-2-27B-PTQ1_0-MTP-')
    out = args.out or stem + args.head_type.upper() + '.gguf'
    q8 = out if args.head_type == 'q8_0' else stem + 'Q8_0.gguf.tmp'
    graft(args.base, q8)
    if args.head_type != 'q8_0':
        # --pure keeps every trunk tensor at PTQ1_0 (copied as is); ssm_alpha/beta must stay bf16
        cmd = [args.quantize, '--allow-requantize', '--pure', '--tensor-type', f'blk\\.64\\..*={args.head_type}',
               '--tensor-type', 'ssm_alpha=bf16', '--tensor-type', 'ssm_beta=bf16', q8, out, 'PTQ1_0']
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)
        os.remove(q8)
    print(f'wrote {out}')


if __name__ == '__main__':
    sys.exit(main())
