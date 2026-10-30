# add_draft_head.py IN.gguf OUT.gguf IDS.txt K: copy a Qwen3.5 MTP GGUF and add a trimmed draft head (FR-Spec style):
# blk.N.nextn.draft_head.weight = rows IDS[:K] of output.weight (same quant type, rows are independent blocks) and
# blk.N.nextn.draft_head_ids.weight = those token ids (I32). The head is added to prism.hadamard.weight_names so its
# input gets the same rotation as output.weight. IDS.txt: one token id per line, most frequent first.
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import gguf

src, dst, ids_path, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
r = gguf.GGUFReader(src)
arch = r.fields["general.architecture"].contents()
il = r.fields[f"{arch}.block_count"].contents() - 1  # the MTP block is the last one
ids = np.array([int(x) for x in open(ids_path).read().split()[:k]], dtype=np.int32)
assert len(ids) == k and len(set(ids.tolist())) == k
out_t = next(t for t in r.tensors if t.name == "output.weight")
head = np.ascontiguousarray(out_t.data[ids])
name_h, name_i = f"blk.{il}.nextn.draft_head.weight", f"blk.{il}.nextn.draft_head_ids.weight"
print(f"{arch}: MTP block {il}, output.weight {out_t.tensor_type.name} {out_t.data.shape} -> draft head {head.shape} ({head.nbytes / 2**20:.1f} MiB)")

w = gguf.GGUFWriter(dst, arch, endianess=r.endianess)
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]
    st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    val = f.contents()
    if f.name == "prism.hadamard.weight_names":
        val = list(val) + [name_h]
    w.add_key_value(f.name, val, vt, sub_type=st)

tensors = [(t.name, t.data, t.tensor_type) for t in r.tensors] + [(name_h, head, out_t.tensor_type), (name_i, ids, gguf.GGMLQuantizationType.I32)]
for name, data, tt in tensors:
    w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, tt)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for name, data, tt in tensors:
    w.write_tensor_data(data, tensor_endianess=r.endianess)
w.close()
print("wrote", dst)
