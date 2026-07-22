/* =====================================================================
 *  npu_driver.h — board-side driver for archway_npu core rev 4 (Tier-1)
 *
 *  Matches the bitstream AT1051_SoC_tier1_npu_rev4_explore.bit
 *  (built 2026-07-16). HARDWARE SEMANTICS (all verified in RTL):
 *
 *   - Activation layout is HWC everywhere:  byte addr = (h*W + w)*C + c
 *     (layer_ctrl.v e_act_addr/n_act_addr == post_proc writer layout).
 *     The host loads the input image in HWC order; layers chain through
 *     the ping-pong banks with NO CPU repacking.
 *   - Host activation addressing is 16-bit: bit15 = PHYSICAL bank (0/1),
 *     bits[14:0] = full 32 KB offset. (cnn_top.v inverts bit15 into the
 *     engine's bank_sel override and act_buffer.rd_data_ext un-inverts it,
 *     so host bit15==N always touches physical bank N for BOTH rd and wr.)
 *   - Engine ping-pong: after reset bank_sel=0 -> a layer READS physical
 *     bank bank_sel and WRITES the other; bank_sel flips at every
 *     layer_done. It is NOT reset between runs, so npu_probe_bank() must
 *     be called once at program start to discover the current parity.
 *   - Completion: reg 0x34 read = {30'd0, layer_running, done_sticky}.
 *     bit0 is STICKY done (set at layer_done, cleared by the next start
 *     rising edge) — plain polling works now; no pp_busy heuristics.
 *   - post_proc requant: out = clamp((acc * qscale) >>> qshift,
 *     relu ? 0 : -128, 127). Arithmetic shift = floor. ReLU max is 127
 *     (Tier-1 fix): all activations live in [0,127].
 *   - post_proc drains only Cout lanes (Tier-1 fix): no garbage tail; a
 *     tensor may fill a bank exactly (32768 B).
 *   - Weight BRAM: 512 rows x 128 int8. Row r = one (cin,kh,kw) MAC step,
 *     r = (kh*K + kw)*Cin + cin  (cin fastest — layer_ctrl.v counters).
 *     Write port is 32-bit words: word (r*32 + sub) carries output
 *     channels sub*4 .. sub*4+3 in bytes [7:0],[15:8],[23:16],[31:24].
 *     Rows restart at 0 every layer (weights are re-loaded per layer).
 *   - Act writes are gated by safe_to_write (= idle); only write between
 *     layers. Act reads are NOT gated — never read act space mid-layer.
 *
 *  Include order in the model program:
 *      #include "model_arch.h"   // defines layer_config_t + the tables
 *      #include "npu_driver.h"
 *
 *  Build for the board (VEGA AT1051, rv32ima, soft-float):
 *      riscv32-unknown-linux-gnu-gcc -O2 -march=rv32ima -mabi=ilp32 ...
 *  Needs root (mmap /dev/mem).
 * ===================================================================== */
#ifndef NPU_DRIVER_H
#define NPU_DRIVER_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_FPGA
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>

#define NPU_BASE      0x10140000u
#define NPU_SPAN      (256u * 1024u)
#define NPU_OFF_WEIGHT 0x10000u              /* awaddr[17:16] = 01 */
#define NPU_OFF_ACT    0x20000u              /* awaddr[17:16] = 10 */
#define NPU_ACT_BANK(b) ((b) ? 0x8000u : 0u) /* bit15 = physical bank */

/* config register byte offsets */
#define R_CIN    0x00
#define R_COUT   0x04
#define R_HIN    0x08
#define R_WIN    0x0C
#define R_HOUT   0x10
#define R_WOUT   0x14
#define R_K      0x18
#define R_STRIDE 0x1C
#define R_PAD    0x20
#define R_QSCALE 0x24
#define R_QSHIFT 0x28
#define R_RELU   0x2C
#define R_START  0x30
#define R_STATUS 0x34   /* read: bit0 = done_sticky, bit1 = layer_running */
#define R_PPBUSY 0x38   /* read: bit0 = pp_busy (debug only)              */
#define R_STREAM 0x3C   /* bit0 stream->acts, bit1 addr rst, bit2 firehose */

static volatile uint8_t *npu_base;
/* physical bank the engine will READ on the next layer start */
static int npu_cur_bank = 0;

static inline void npu_w32(uint32_t off, uint32_t v)
{
    *(volatile uint32_t *)(npu_base + off) = v;
}
static inline uint32_t npu_r32(uint32_t off)
{
    return *(volatile uint32_t *)(npu_base + off);
}

static int npu_map(void)
{
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("/dev/mem (need root)"); return -1; }
    void *p = mmap(NULL, NPU_SPAN, PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd, NPU_BASE);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap NPU"); return -1; }
    npu_base = (volatile uint8_t *)p;
    return 0;
}

static double npu_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Poll the sticky done bit (reg 0x34 bit0). Set by hardware at
 * layer_done, cleared automatically by the next start rising edge. */
static int npu_wait_done(int timeout_ms)
{
    double t0 = npu_now_ms();
    for (;;) {
        uint32_t s = npu_r32(R_STATUS);
        if (s & 1u) return 0;
        if (npu_now_ms() - t0 > (double)timeout_ms) {
            fprintf(stderr, "[NPU] timeout: status=0x%08x (running=%u)\n",
                    s, (s >> 1) & 1u);
            return -1;
        }
    }
}

/* Load one layer's weights from an OIHW int8 blob into the weight BRAM.
 * Row r = (kh*K + kw)*Cin + cin; only ceil(out_ch/4) sub-words per row
 * are written — PE lanes >= out_ch are never drained (Tier-1), so their
 * weights don't matter. */
static void npu_load_weights(const int8_t *w, int in_ch, int out_ch, int k)
{
    int need_sub = (out_ch + 3) >> 2;
    for (int kh = 0; kh < k; ++kh)
        for (int kw = 0; kw < k; ++kw)
            for (int cin = 0; cin < in_ch; ++cin) {
                int r = (kh * k + kw) * in_ch + cin;
                volatile uint32_t *row =
                    (volatile uint32_t *)(npu_base + NPU_OFF_WEIGHT
                                          + (uint32_t)r * 128u);
                for (int sub = 0; sub < need_sub; ++sub) {
                    uint32_t word = 0;
                    for (int b = 0; b < 4; ++b) {
                        int oc = sub * 4 + b;
                        int8_t wv = 0;
                        if (oc < out_ch)
                            wv = w[((oc * in_ch + cin) * k + kh) * k + kw];
                        word |= ((uint32_t)(uint8_t)wv) << (8 * b);
                    }
                    row[sub] = word;
                }
            }
}

/* Activations are HWC bytes, linear copy. bank = PHYSICAL bank 0/1.
 * The act write port is 8-bit (S00_AXI host_wr_data = WDATA[7:0]) — one
 * byte per AXI transaction is a hardware property, not a driver choice. */
static void npu_write_acts(const uint8_t *buf, uint32_t n, int bank)
{
    volatile uint8_t *ab = npu_base + NPU_OFF_ACT + NPU_ACT_BANK(bank);
    const uint8_t *end = buf + n;
    /* 8x unroll: on the 40MHz rv32ima every cycle of loop overhead is
     * visible next to the MMIO store itself */
    while (buf + 8 <= end) {
        ab[0] = buf[0]; ab[1] = buf[1]; ab[2] = buf[2]; ab[3] = buf[3];
        ab[4] = buf[4]; ab[5] = buf[5]; ab[6] = buf[6]; ab[7] = buf[7];
        ab += 8; buf += 8;
    }
    while (buf < end) *ab++ = *buf++;
}
static void npu_read_acts(uint8_t *buf, uint32_t n, int bank)
{
    volatile uint8_t *ab = npu_base + NPU_OFF_ACT + NPU_ACT_BANK(bank);
    /* Priming read: the act read-enable is qualified by the PREVIOUS
     * transaction's latched address, so the first act read after any
     * register/status access returns stale data. One throwaway read
     * re-arms the pipeline; all subsequent reads are correct. */
    (void)ab[0];
    for (uint32_t i = 0; i < n; ++i) buf[i] = ab[i];
}

/* Sparse head readout: prime once, then read single act bytes at will.
 * All reads between npu_prime_acts() and the next register access are
 * back-to-back act reads, so only the first needs the throwaway. */
static inline void npu_prime_acts(int bank)
{
    (void)*(npu_base + NPU_OFF_ACT + NPU_ACT_BANK(bank));
}
static inline uint8_t npu_read_act_byte(uint32_t off, int bank)
{
    return *(npu_base + NPU_OFF_ACT + NPU_ACT_BANK(bank) + off);
}

/* =====================================================================
 *  Prepared weights: weights.bin is repacked ONCE into the exact 32-bit
 *  words the weight BRAM wants (row r = (kh*K+kw)*Cin+cin, word sub =
 *  output channels sub*4..sub*4+3). Per layer run the load is then a
 *  tight pointer loop of MMIO stores — this matters when the same model
 *  runs on many tiles per image.
 * ===================================================================== */
typedef struct {
    uint32_t *words;        /* rows * need_sub packed words */
    int rows, need_sub;
} npu_wprep_t;

static void npu_prepare_weights(const void *layers_v, int n_layers,
                                const int8_t *weights_bin, npu_wprep_t *wp)
{
    const layer_config_t *layers = (const layer_config_t *)layers_v;
    for (int l = 0; l < n_layers; ++l) {
        const layer_config_t *L = &layers[l];
        const int8_t *w = weights_bin + L->weight_offset;
        int k = L->k, in_ch = L->in_ch, out_ch = L->out_ch;
        int rows = k * k * in_ch, need_sub = (out_ch + 3) >> 2;
        uint32_t *dst = malloc((size_t)rows * need_sub * 4);
        wp[l].words = dst;
        wp[l].rows = rows;
        wp[l].need_sub = need_sub;
        for (int kh = 0; kh < k; ++kh)
            for (int kw = 0; kw < k; ++kw)
                for (int cin = 0; cin < in_ch; ++cin) {
                    uint32_t *row = dst + (size_t)((kh * k + kw) * in_ch
                                                   + cin) * need_sub;
                    for (int sub = 0; sub < need_sub; ++sub) {
                        uint32_t word = 0;
                        for (int b = 0; b < 4; ++b) {
                            int oc = sub * 4 + b;
                            int8_t wv = 0;
                            if (oc < out_ch)
                                wv = w[((oc * in_ch + cin) * k + kh) * k
                                       + kw];
                            word |= ((uint32_t)(uint8_t)wv) << (8 * b);
                        }
                        row[sub] = word;
                    }
                }
    }
}

static void npu_load_weights_prepared(const npu_wprep_t *wl)
{
    const uint32_t *src = wl->words;
    int ns = wl->need_sub;
    for (int r = 0; r < wl->rows; ++r) {
        volatile uint32_t *row =
            (volatile uint32_t *)(npu_base + NPU_OFF_WEIGHT
                                  + (uint32_t)r * 128u);
        for (int s = 0; s < ns; ++s) row[s] = src[s];
        src += ns;
    }
}

/* npu_exec_layer with pre-packed weights (see npu_exec_layer below for
 * the semantics — input in bank npu_cur_bank, output in the flipped one) */
static int npu_exec_layer_prepared(const void *Lv, const npu_wprep_t *wl)
{
    const layer_config_t *L = (const layer_config_t *)Lv;
    npu_load_weights_prepared(wl);
    npu_w32(R_CIN, L->in_ch);   npu_w32(R_COUT, L->out_ch);
    npu_w32(R_HIN, L->in_h);    npu_w32(R_WIN, L->in_w);
    npu_w32(R_HOUT, L->out_h);  npu_w32(R_WOUT, L->out_w);
    npu_w32(R_K, L->k);         npu_w32(R_STRIDE, L->stride);
    npu_w32(R_PAD, L->pad);
    npu_w32(R_QSCALE, L->qscale); npu_w32(R_QSHIFT, L->qshift);
    npu_w32(R_RELU, L->relu);
    npu_w32(R_START, 0);
    npu_w32(R_START, 1);
    int rc = npu_wait_done(3000);
    npu_w32(R_START, 0);
    npu_cur_bank ^= 1;
    return rc;
}

/* Discover the engine's current bank_sel parity. The register isn't
 * readable, and it persists across program runs (flips once per layer
 * ever executed since reset). Trick: mark both banks, run a 1x1x1
 * "times two" layer, and see which mark got doubled into the other bank.
 *   bank0[0]=3, bank1[0]=5, w=2, qscale=1, qshift=0:
 *     engine read bank0 -> bank1[0] becomes 6 -> next layer reads bank1
 *     engine read bank1 -> bank0[0] becomes 10 -> next layer reads bank0
 */
static int npu_probe_bank(void)
{
    uint32_t s = npu_r32(R_STATUS);
    if ((s >> 1) & 1u) {
        fprintf(stderr, "[NPU] probe: engine busy (status=0x%08x)?\n", s);
        return -1;
    }
    uint8_t m0 = 3, m1 = 5;
    npu_write_acts(&m0, 1, 0);
    npu_write_acts(&m1, 1, 1);
    *(volatile uint32_t *)(npu_base + NPU_OFF_WEIGHT) = 2u; /* row0 oc0 = 2 */
    npu_w32(R_CIN, 1);  npu_w32(R_COUT, 1);
    npu_w32(R_HIN, 1);  npu_w32(R_WIN, 1);
    npu_w32(R_HOUT, 1); npu_w32(R_WOUT, 1);
    npu_w32(R_K, 1);    npu_w32(R_STRIDE, 1); npu_w32(R_PAD, 0);
    npu_w32(R_QSCALE, 1); npu_w32(R_QSHIFT, 0); npu_w32(R_RELU, 1);
    npu_w32(R_START, 0);
    npu_w32(R_START, 1);
    if (npu_wait_done(200) != 0) { npu_w32(R_START, 0); return -1; }
    npu_w32(R_START, 0);
    uint8_t b0, b1;
    npu_read_acts(&b0, 1, 0);
    npu_read_acts(&b1, 1, 1);
    if (b1 == 6 && b0 == 3)       npu_cur_bank = 1; /* was 0, flipped */
    else if (b0 == 10 && b1 == 5) npu_cur_bank = 0; /* was 1, flipped */
    else {
        fprintf(stderr, "[NPU] probe inconclusive: bank0[0]=%u bank1[0]=%u\n",
                b0, b1);
        if (b0 == b1)
            fprintf(stderr, "[NPU] both banks read IDENTICAL: host bank bit "
                    "(bit15) is aliasing, which is pre-rev4 behavior. The "
                    "FPGA is running an OLD bitstream — program "
                    "AT1051_SoC_tier1_npu_rev4_explore.bit and retry.\n");
        return -1;
    }
    printf("[NPU] bank probe OK — next layer reads physical bank %d\n",
           npu_cur_bank);
    return 0;
}

/* Run one layer. Input must already sit in physical bank npu_cur_bank
 * (HWC). On return the output sits in the flipped bank and npu_cur_bank
 * points at it. */
static int npu_exec_layer(const void *Lv, const int8_t *weights_bin)
{
    const layer_config_t *L = (const layer_config_t *)Lv;
    npu_load_weights(weights_bin + L->weight_offset,
                     L->in_ch, L->out_ch, L->k);
    npu_w32(R_CIN, L->in_ch);   npu_w32(R_COUT, L->out_ch);
    npu_w32(R_HIN, L->in_h);    npu_w32(R_WIN, L->in_w);
    npu_w32(R_HOUT, L->out_h);  npu_w32(R_WOUT, L->out_w);
    npu_w32(R_K, L->k);         npu_w32(R_STRIDE, L->stride);
    npu_w32(R_PAD, L->pad);
    npu_w32(R_QSCALE, L->qscale); npu_w32(R_QSHIFT, L->qshift);
    npu_w32(R_RELU, L->relu);
    npu_w32(R_START, 0);
    npu_w32(R_START, 1);
    int rc = npu_wait_done(3000);
    npu_w32(R_START, 0);
    npu_cur_bank ^= 1;
    return rc;
}
#endif /* USE_FPGA */

/* =====================================================================
 *  Bit-exact CPU reference of one layer (HWC in / HWC out).
 *  This IS the hardware math:
 *    acc  = sum( act * weight )                 int32
 *    v    = (acc * qscale) >> qshift            arithmetic shift (floor)
 *    out  = clamp(v, relu ? 0 : -128, 127)
 *  Used as the software fallback (no -DUSE_FPGA) and as the on-board
 *  self-test (-DNPU_SELFTEST=1). When the self-test fails, BELIEVE IT.
 * ===================================================================== */
static void npu_ref_layer(const uint8_t *in_hwc, const void *Lv,
                          const int8_t *weights_bin, uint8_t *out_hwc)
{
    const layer_config_t *L = (const layer_config_t *)Lv;
    const int8_t *w = weights_bin + L->weight_offset;
    int K = L->k, S = L->stride, P = L->pad;
    int32_t lo = L->relu ? 0 : -128;
    for (int oy = 0; oy < L->out_h; ++oy)
        for (int ox = 0; ox < L->out_w; ++ox)
            for (int oc = 0; oc < L->out_ch; ++oc) {
                int32_t acc = 0;
                for (int kh = 0; kh < K; ++kh) {
                    int iy = oy * S + kh - P;
                    if (iy < 0 || iy >= L->in_h) continue;
                    for (int kw = 0; kw < K; ++kw) {
                        int ix = ox * S + kw - P;
                        if (ix < 0 || ix >= L->in_w) continue;
                        for (int c = 0; c < L->in_ch; ++c) {
                            /* hardware multiplies the act byte as SIGNED;
                             * activations are always <= 127 so int8 cast
                             * is an identity, kept for exactness */
                            int32_t px = (int32_t)(int8_t)
                                in_hwc[(iy * L->in_w + ix) * L->in_ch + c];
                            int32_t wv = (int32_t)
                                w[((oc * L->in_ch + c) * K + kh) * K + kw];
                            acc += px * wv;
                        }
                    }
                }
                int64_t v = ((int64_t)acc * (int64_t)L->qscale) >> L->qshift;
                if (v < lo)  v = lo;
                if (v > 127) v = 127;
                out_hwc[(oy * L->out_w + ox) * L->out_ch + oc] = (uint8_t)v;
            }
}

/* =====================================================================
 *  Image helpers (integer grayscale + bilinear resize — the VEGA core
 *  is rv32ima soft-float, so the heavy loops stay in fixed point).
 * ===================================================================== */

/* RGB (stb output, 3ch) -> 8-bit gray, ITU-R 601 integer approximation */
static void img_rgb_to_gray(const uint8_t *rgb, int w, int h, uint8_t *gray)
{
    for (int i = 0; i < w * h; ++i)
        gray[i] = (uint8_t)((77 * rgb[3 * i] + 150 * rgb[3 * i + 1]
                             + 29 * rgb[3 * i + 2]) >> 8);
}

/* Bilinear resize, single channel, 16.16 fixed point.
 *
 * OPTIMIZED (bit-identical to the reference below): the source-x coordinate
 * depends only on the output column, so the per-pixel 64-bit divide/multiply
 * that the naive form ran dw*dh times is hoisted into a per-column table built
 * ONCE per call (dw divides instead of dw*dh). The inner loop is then pure
 * 32-bit: every product fits in int32 (255*65536 < 2^31), so the rv32ima core
 * uses single-instruction `mul` instead of the __muldi3/__divdi3 soft-64
 * routines that dominated the old ~1390 cycles/pixel. The lerp form
 *   a + (((b - a) * w) >> 16)  ==  (a*(65536-w) + b*w) >> 16   (exactly, for
 * arithmetic shift) halves the multiplies again. Output bytes are unchanged.
 *
 * Reference (kept for provenance):
 *   fy = ((2y+1)*sh<<15)/dh - (1<<15); y0=fy>>16; wy=fy&0xFFFF; ... 64-bit lerp
 */
static void img_resize_gray(const uint8_t *src, int sw, int sh,
                            uint8_t *dst, int dw, int dh)
{
    if (dw <= 0 || dh <= 0) return;
    /* per-column source index x0[] and fractional weight wx[] (0..65535) */
    int     *x0a = (int *)malloc((size_t)dw * sizeof(int));
    int32_t *wxa = (int32_t *)malloc((size_t)dw * sizeof(int32_t));
    if (!x0a || !wxa) {          /* fall back to the exact reference path */
        free(x0a); free(wxa);
        for (int y = 0; y < dh; ++y) {
            int32_t fy = (int32_t)(((int64_t)(2 * y + 1) * sh << 15) / dh)
                         - (1 << 15);
            if (fy < 0) fy = 0;
            int y0 = fy >> 16, y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
            int32_t wy = fy & 0xFFFF;
            for (int x = 0; x < dw; ++x) {
                int32_t fx = (int32_t)(((int64_t)(2 * x + 1) * sw << 15) / dw)
                             - (1 << 15);
                if (fx < 0) fx = 0;
                int x0 = fx >> 16, x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
                int32_t wx = fx & 0xFFFF;
                int64_t p00 = src[y0 * sw + x0], p01 = src[y0 * sw + x1];
                int64_t p10 = src[y1 * sw + x0], p11 = src[y1 * sw + x1];
                int64_t top = (p00 * (65536 - wx) + p01 * wx) >> 16;
                int64_t bot = (p10 * (65536 - wx) + p11 * wx) >> 16;
                dst[y * dw + x] =
                    (uint8_t)((top * (65536 - wy) + bot * wy) >> 16);
            }
        }
        return;
    }
    for (int x = 0; x < dw; ++x) {
        int32_t fx = (int32_t)(((int64_t)(2 * x + 1) * sw << 15) / dw)
                     - (1 << 15);
        if (fx < 0) fx = 0;
        x0a[x] = fx >> 16;
        wxa[x] = fx & 0xFFFF;
    }
    for (int y = 0; y < dh; ++y) {
        int32_t fy = (int32_t)(((int64_t)(2 * y + 1) * sh << 15) / dh)
                     - (1 << 15);
        if (fy < 0) fy = 0;
        int y0 = fy >> 16, y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
        int32_t wy = fy & 0xFFFF;
        const uint8_t *r0 = src + (size_t)y0 * sw;
        const uint8_t *r1 = src + (size_t)y1 * sw;
        uint8_t *drow = dst + (size_t)y * dw;
        for (int x = 0; x < dw; ++x) {
            int x0 = x0a[x];
            int x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
            int32_t wx = wxa[x];
            int a0 = r0[x0], a1 = r1[x0];
            int32_t top = a0 + (((r0[x1] - a0) * wx) >> 16);
            int32_t bot = a1 + (((r1[x1] - a1) * wx) >> 16);
            drow[x] = (uint8_t)(top + (((bot - top) * wy) >> 16));
        }
    }
    free(x0a);
    free(wxa);
}

#endif /* NPU_DRIVER_H */
