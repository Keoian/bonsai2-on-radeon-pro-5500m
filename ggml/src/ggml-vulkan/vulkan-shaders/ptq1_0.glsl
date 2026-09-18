#ifndef PTQ1_0_GLSL
#define PTQ1_0_GLSL

// Shared PTQ1_0 trit accessor. Lives in its own header because two consumers need it
// from different include chains: dequant_funcs.glsl (mul_mat_vec, get_rows,
// copy_from_quant) and mul_mm.comp (via mul_mm_funcs.glsl), and mul_mm.comp does not
// include dequant_funcs.glsl. Duplicating it would leave two copies that must stay in
// step with the CPU codec in ggml-quants.c, where a divergence shows up as wrong
// matmul results rather than a build error.
//
// Element order is not positional: a 16-byte chunk of qs where byte j carries
// elements t*16+j, then an 8-byte chunk carrying 80 + t*8 + (j-16), then qh at four
// trits per byte carrying 120 + t*2 + h. Trits come out by the base-3 remainder
// recurrence t = (v*3)>>8, v = (v*3)&0xFF.

// FADI-OPT: 3^n mod 256 has period 64 (3^64 == 1 mod 256), so the serial
// recurrence loop (up to 119 multiplies per weight) collapses to one table
// lookup + one multiply. Values are exact; output is bit-identical.
const uint POW3_MOD256[64] = uint[64](
    1u, 3u, 9u, 27u, 81u, 243u, 217u, 139u,
    161u, 227u, 169u, 251u, 241u, 211u, 121u, 107u,
    65u, 195u, 73u, 219u, 145u, 179u, 25u, 75u,
    225u, 163u, 233u, 187u, 49u, 147u, 185u, 43u,
    129u, 131u, 137u, 155u, 209u, 115u, 89u, 11u,
    33u, 99u, 41u, 123u, 113u, 83u, 249u, 235u,
    193u, 67u, 201u, 91u, 17u, 51u, 153u, 203u,
    97u, 35u, 105u, 59u, 177u, 19u, 57u, 171u
);

float ptq1_0_trit(uint ib, uint a_offset, uint e) {
    uint b;
    uint n;
    if (e < 80u) {
        b = uint(data_a[a_offset + ib].qs[e & 15u]);
        n = e >> 4u;
    } else if (e < 120u) {
        const uint t = e - 80u;
        b = uint(data_a[a_offset + ib].qs[16u + (t & 7u)]);
        n = t >> 3u;
    } else {
        const uint t = e - 120u;
        b = uint(data_a[a_offset + ib].qh[t & 1u]);
        n = t >> 1u;
    }

    const uint v = (b * POW3_MOD256[n & 63u]) & 0xFFu;
    return float(int((v * 3u) >> 8u) - 1);
}

#endif // PTQ1_0_GLSL
