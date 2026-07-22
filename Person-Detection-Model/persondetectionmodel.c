/* =============================================================================
 *  persondetectionmodel.c — PersonNet tiled person detector (archway_npu)
 *
 *  TILED MULTI-SCALE SCAN. The network is trained on INPUT_W x INPUT_H windows
 *  (v4 default: 181x181, persons 16..120 px tall) — squashing a whole
 *  megapixel frame into one pass makes persons 2..5 px and invisible (that is
 *  exactly what the v1 program did). This scans a pyramid of tiles across the
 *  image, runs every tile on the NPU, and merges detections with global NMS
 *  in image coordinates. A letterbox "fit" pass covers persons larger than any
 *  tile. No top-1 fallback: no persons -> no boxes.
 *
 *  DIMENSION-GENERIC: every size comes from model_arch.h (INPUT_W/H, GRID,
 *  TILE_STRIDE, HEAD_CH). The same source compiles for the 128 or the 181
 *  export — just point -I at the right export dir. The default --scales below
 *  MUST match the trainer's --tile-scales (it picks CONF_THRESH under them).
 *
 *  Build (board, VEGA AT1051 rv32ima, Linux, FPGA NPU):
 *      riscv32-linux-gcc -O3 -flto -funroll-loops -fno-math-errno -static \
 *          -I export_detector -DUSE_FPGA -o PM persondetectionmodel.c -lm
 *  Pure-software bit-exact reference (any host, no NPU): drop -DUSE_FPGA.
 *  Per-layer HW self-test build: add -DNPU_SELFTEST=1 (single-pass mode).
 *  Fast JPEG decode (optional, big win on the 40MHz core): cross-compile
 *  libjpeg-turbo and add -DUSE_LIBJPEG -I<jpeg>/include <jpeg>/lib/libjpeg.a
 *
 *  Usage (run as root on the board — /dev/mem):
 *      ./PM <image.jpg|input_q.bin> <weights.bin> [output.txt] [options]
 *        --conf X       confidence threshold   (default CONF_THRESH)
 *        --stride N     tile stride in px      (default TILE_STRIDE)
 *        --scales a,b,c pyramid scales rel. to the decoded image.  Default is
 *                       the trainer-exported TILE_SCALES from model_arch.h, so
 *                       the on-board pyramid always matches the protocol
 *                       CONF_THRESH was chosen under (1.2 UPSAMPLES so tiny
 *                       aerial persons reach the 16px band; each scale adds
 *                       tiles!).  A merged detection then needs MIN_VOTES
 *                       overlapping tiles (or one >= HIGH_CONF) to survive.
 *        --max-tiles N  tile budget, coarse scales first (default 512)
 *        --denom N      libjpeg-only: decode at 1/N resolution (1,2,4,8).
 *                       The pyramid then runs on the SMALLER image — the
 *                       cheap way to scan huge photos.
 *        --single       legacy one-pass mode (center-crop square squash);
 *                       forced for .bin inputs and NPU_SELFTEST builds
 *        --top1         emit the top-1 cell when nothing passes --conf
 *                       (single mode only; default OFF)
 *
 *  output.txt:
 *      # transform none                       (tiled mode: coords are
 *      <cx> <cy> <w> <h> <conf>                normalized to the FULL image)
 *      — or in --single mode the v1 "# transform crop ..." contract.
 * ============================================================================ */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "model_arch.h"
#include "npu_driver.h"

#ifndef TILE_STRIDE            /* older model_arch.h exports */
#define TILE_STRIDE (INPUT_W * 2 / 3)   /* ~34% overlap, matches trainer */
#endif
/* v6 tiled cross-scale AGREEMENT gate (single source of truth: model_arch.h).
 * A real person is caught by several overlapping tiles/scales; an INT8 phantom
 * fires in one isolated tile.  A merged cluster survives only if it collected
 * >= MIN_VOTES tiles OR one tile cleared HIGH_CONF.  Older headers lack these
 * -> fall back to plain NMS (MIN_VOTES 1).  MUST mirror train_persondetection-
 * model.py: tiled_cluster_vote(). */
#ifndef MIN_VOTES
#define MIN_VOTES 1            /* 1 = plain NMS (no agreement gate) */
#endif
#ifndef HIGH_CONF
#define HIGH_CONF 1.0f
#endif
/* collect tiles down to VOTE_FLOOR so weak supporting tiles can still vote;
 * CONF_THRESH is applied AFTER the agreement vote (mirrors the trainer). */
#ifndef VOTE_FLOOR
#define VOTE_FLOOR 0.2f
#endif

#ifdef USE_LIBJPEG
#include <jpeglib.h>
#include <setjmp.h>
#else
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#include "stb_image.h"
#endif

#define MAX_DETS   512
#define MAX_SCALES 6

typedef struct { float conf, cx, cy, w, h; } det_t;

static det_t g_dets[MAX_DETS];
static int g_ndets = 0;

/* timing accumulators (ms) */
static double tm_decode, tm_prep, tm_acts, tm_weights, tm_compute, tm_read;
static int g_tiles = 0;

static double wall_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int ends_with(const char *s, const char *suf)
{
    size_t ls = strlen(s), lf = strlen(suf);
    return ls >= lf && strcmp(s + ls - lf, suf) == 0;
}

static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

static float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static float iou(const det_t *a, const det_t *b)
{
    float ax0 = a->cx - a->w / 2, ax1 = a->cx + a->w / 2;
    float ay0 = a->cy - a->h / 2, ay1 = a->cy + a->h / 2;
    float bx0 = b->cx - b->w / 2, bx1 = b->cx + b->w / 2;
    float by0 = b->cy - b->h / 2, by1 = b->cy + b->h / 2;
    float iw = (ax1 < bx1 ? ax1 : bx1) - (ax0 > bx0 ? ax0 : bx0);
    float ih = (ay1 < by1 ? ay1 : by1) - (ay0 > by0 ? ay0 : by0);
    if (iw <= 0 || ih <= 0) return 0.0f;
    float inter = iw * ih;
    float ua = a->w * a->h + b->w * b->h - inter;
    return ua > 0 ? inter / ua : 0.0f;
}

static int8_t *load_weights(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz != (long)WEIGHTS_TOTAL_BYTES) {
        fprintf(stderr, "weights.bin is %ld bytes, expected %d — wrong "
                "model export?\n", sz, WEIGHTS_TOTAL_BYTES);
        fclose(f);
        return NULL;
    }
    int8_t *w = malloc(sz);
    if (fread(w, 1, sz, f) != (size_t)sz) {
        fprintf(stderr, "short read on %s\n", path);
        free(w); fclose(f); return NULL;
    }
    fclose(f);
    return w;
}

/* ---------------------------------------------------------------------------
 *  JPEG -> 8-bit grayscale.
 * ------------------------------------------------------------------------- */
#ifdef USE_LIBJPEG
struct jerr_jmp { struct jpeg_error_mgr mgr; jmp_buf jb; };
static void jerr_exit(j_common_ptr ci)
{
    struct jerr_jmp *e = (struct jerr_jmp *)ci->err;
    (*ci->err->output_message)(ci);
    longjmp(e->jb, 1);
}

static uint8_t *decode_gray(const char *path, int *W, int *H, int denom)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    struct jpeg_decompress_struct ci;
    struct jerr_jmp jerr;
    ci.err = jpeg_std_error(&jerr.mgr);
    jerr.mgr.error_exit = jerr_exit;
    uint8_t *img = NULL;
    if (setjmp(jerr.jb)) {
        jpeg_destroy_decompress(&ci);
        fclose(f);
        free(img);
        return NULL;
    }
    jpeg_create_decompress(&ci);
    jpeg_stdio_src(&ci, f);
    jpeg_read_header(&ci, TRUE);
    ci.out_color_space = JCS_GRAYSCALE;   /* skips chroma entirely */
    ci.scale_num = 1;
    ci.scale_denom = denom;               /* 1/2/4/8: scaled IDCT, cheap */
    ci.dct_method = JDCT_IFAST;
    ci.do_fancy_upsampling = FALSE;
    jpeg_start_decompress(&ci);
    *W = ci.output_width;
    *H = ci.output_height;
    img = malloc((size_t)*W * *H);
    while (ci.output_scanline < ci.output_height) {
        JSAMPROW row = img + (size_t)ci.output_scanline * *W;
        jpeg_read_scanlines(&ci, &row, 1);
    }
    jpeg_finish_decompress(&ci);
    jpeg_destroy_decompress(&ci);
    fclose(f);
    return img;
}
#else
static uint8_t *decode_gray(const char *path, int *W, int *H, int denom)
{
    if (denom != 1)
        fprintf(stderr, "note: --denom needs -DUSE_LIBJPEG, ignoring\n");
    int comp;
    uint8_t *g = stbi_load(path, W, H, &comp, 1);  /* stb converts to luma */
    if (!g)
        fprintf(stderr, "decode %s failed: %s\n", path,
                stbi_failure_reason());
    return g;
}
#endif

/* ---------------------------------------------------------------------------
 *  One INPUT_W x INPUT_H quantized tile through the model.
 *  q_in: INPUT_BYTES values 0..127. out: OUTPUT_BYTES int8 head.
 * ------------------------------------------------------------------------- */
#ifdef USE_FPGA
static npu_wprep_t g_wprep[NUM_LAYERS];

static int run_tile(const uint8_t *q_in, int8_t *out_head)
{
    double t0 = wall_ms();
    npu_write_acts(q_in, INPUT_BYTES, npu_cur_bank);
    double t1 = wall_ms();
    tm_acts += t1 - t0;
    for (int l = 0; l < NUM_LAYERS; ++l) {
        const layer_config_t *L = &model_layers[l];
        double ta = wall_ms();
        npu_load_weights_prepared(&g_wprep[l]);
        double tb = wall_ms();
        tm_weights += tb - ta;
        npu_w32(R_CIN, L->in_ch);   npu_w32(R_COUT, L->out_ch);
        npu_w32(R_HIN, L->in_h);    npu_w32(R_WIN, L->in_w);
        npu_w32(R_HOUT, L->out_h);  npu_w32(R_WOUT, L->out_w);
        npu_w32(R_K, L->k);         npu_w32(R_STRIDE, L->stride);
        npu_w32(R_PAD, L->pad);
        npu_w32(R_QSCALE, L->qscale); npu_w32(R_QSHIFT, L->qshift);
        npu_w32(R_RELU, L->relu);
        npu_w32(R_START, 0);
        npu_w32(R_START, 1);
        if (npu_wait_done(3000) != 0) {
            fprintf(stderr, "layer %d timeout\n", l + 1);
            return -1;
        }
        npu_w32(R_START, 0);
        npu_cur_bank ^= 1;
        tm_compute += wall_ms() - tb;
    }
    double t2 = wall_ms();
    if (out_head)
        npu_read_acts((uint8_t *)out_head, OUTPUT_BYTES, npu_cur_bank);
    tm_read += wall_ms() - t2;
    ++g_tiles;
    return 0;
}
#else
/* pure-software path: bit-exact CPU reference, ping-pong like the HW */
static const int8_t *g_weights_bin;

static int run_tile(const uint8_t *q_in, int8_t *out_head)
{
    static uint8_t ping[32768], pong[32768];
    memcpy(ping, q_in, INPUT_BYTES);
    uint8_t *a = ping, *b = pong;
    double t0 = wall_ms();
    for (int l = 0; l < NUM_LAYERS; ++l) {
        npu_ref_layer(a, &model_layers[l], g_weights_bin, b);
        uint8_t *t = a; a = b; b = t;
    }
    memcpy(out_head, a, OUTPUT_BYTES);
    tm_compute += wall_ms() - t0;
    ++g_tiles;
    return 0;
}
#endif

/* ---------------------------------------------------------------------------
 *  Tiled scan. obj_thr_i8 = smallest head int8 that can pass conf_thresh
 *  (logit-domain compare: no expf per cell). On the FPGA build the head
 *  is read sparsely: 256 obj bytes, then 4 more bytes per surviving cell.
 * ------------------------------------------------------------------------- */
static int g_obj_thr_i8;
static float g_conf_thresh;

static void decode_cell(int cell, const int8_t *t5, float xn, float yn,
                        float wn, float hn)
{
    if (g_ndets >= MAX_DETS) return;
    int row = cell / GRID, col = cell % GRID;
    det_t d;
    d.conf = sigmoid_f((float)t5[0] * LAST_ACC_SCALE);
    if (d.conf < g_conf_thresh) return;
    float cx = (col + sigmoid_f((float)t5[1] * LAST_ACC_SCALE)) / GRID;
    float cy = (row + sigmoid_f((float)t5[2] * LAST_ACC_SCALE)) / GRID;
    float w = ANCHOR_W * expf(clampf((float)t5[3] * LAST_ACC_SCALE,
                                     -4.f, 4.f));
    float h = ANCHOR_H * expf(clampf((float)t5[4] * LAST_ACC_SCALE,
                                     -4.f, 4.f));
    d.cx = xn + cx * wn;
    d.cy = yn + cy * hn;
    d.w = w * wn;
    d.h = h * hn;
    g_dets[g_ndets++] = d;
}

/* scan one already-scaled gray image; tile coords map into the ORIGINAL
 * normalized frame via (xn0 + tilex/sw * wsc, ...) with wsc = 1 for the
 * pyramid scales and the letterbox transform for the fit pass */
static int scan_scaled(const uint8_t *img, int sw, int sh, int stride,
                       float xn0, float yn0, float wsc, float hsc,
                       int max_tiles)
{
    static uint8_t q[INPUT_BYTES];
#ifndef USE_FPGA
    static int8_t head[OUTPUT_BYTES];
#endif
    int nx = 0, ny = 0;
    int xs[256], ys[256];   /* upsampled large frames need >127 positions/axis */
    for (int x = 0; x + INPUT_W < sw && nx < 255; x += stride) xs[nx++] = x;
    xs[nx++] = sw - INPUT_W;
    for (int y = 0; y + INPUT_H < sh && ny < 255; y += stride) ys[ny++] = y;
    ys[ny++] = sh - INPUT_H;
    for (int yi = 0; yi < ny; ++yi)
        for (int xi = 0; xi < nx; ++xi) {
            if (g_tiles >= max_tiles) return 1;   /* budget hit */
            int x0 = xs[xi], y0 = ys[yi];
            for (int r = 0; r < INPUT_H; ++r) {
                const uint8_t *s = img + (size_t)(y0 + r) * sw + x0;
                uint8_t *d = q + r * INPUT_W;
                for (int c = 0; c < INPUT_W; ++c) d[c] = s[c] >> 1;
            }
            float txn = xn0 + (float)x0 / sw * wsc;
            float tyn = yn0 + (float)y0 / sh * hsc;
            float twn = (float)INPUT_W / sw * wsc;
            float thn = (float)INPUT_H / sh * hsc;
#ifdef USE_FPGA
            double t0 = wall_ms();
            /* run all layers, then sparse head readout */
            npu_write_acts(q, INPUT_BYTES, npu_cur_bank);
            tm_acts += wall_ms() - t0;
            int fail = 0;
            for (int l = 0; l < NUM_LAYERS && !fail; ++l) {
                const layer_config_t *L = &model_layers[l];
                double ta = wall_ms();
                npu_load_weights_prepared(&g_wprep[l]);
                double tb = wall_ms();
                tm_weights += tb - ta;
                npu_w32(R_CIN, L->in_ch);   npu_w32(R_COUT, L->out_ch);
                npu_w32(R_HIN, L->in_h);    npu_w32(R_WIN, L->in_w);
                npu_w32(R_HOUT, L->out_h);  npu_w32(R_WOUT, L->out_w);
                npu_w32(R_K, L->k);         npu_w32(R_STRIDE, L->stride);
                npu_w32(R_PAD, L->pad);
                npu_w32(R_QSCALE, L->qscale); npu_w32(R_QSHIFT, L->qshift);
                npu_w32(R_RELU, L->relu);
                npu_w32(R_START, 0);
                npu_w32(R_START, 1);
                if (npu_wait_done(3000) != 0) fail = 1;
                npu_w32(R_START, 0);
                npu_cur_bank ^= 1;
                tm_compute += wall_ms() - tb;
            }
            if (fail) return -1;
            ++g_tiles;
            double t2 = wall_ms();
            npu_prime_acts(npu_cur_bank);
            for (int cell = 0; cell < GRID * GRID; ++cell) {
                int8_t obj = (int8_t)npu_read_act_byte(
                                 (uint32_t)cell * HEAD_CH, npu_cur_bank);
                if ((int)obj < g_obj_thr_i8)
                    continue;
                int8_t t5[5];
                t5[0] = obj;
                for (int k = 1; k < 5; ++k)
                    t5[k] = (int8_t)npu_read_act_byte(
                                (uint32_t)cell * HEAD_CH + k, npu_cur_bank);
                decode_cell(cell, t5, txn, tyn, twn, thn);
            }
            tm_read += wall_ms() - t2;
#else
            if (run_tile(q, head) != 0) return -1;
            for (int cell = 0; cell < GRID * GRID; ++cell) {
                const int8_t *t5 = head + cell * HEAD_CH;
                if ((int)t5[0] < g_obj_thr_i8) continue;
                decode_cell(cell, t5, txn, tyn, twn, thn);
            }
#endif
        }
    return 0;
}

/* integer bilinear downscale of the whole frame (16.16 fixed point,
 * from npu_driver.h) — used to build the pyramid levels */
static uint8_t *scale_image(const uint8_t *src, int W, int H, float s,
                            int *sw, int *sh)
{
    *sw = (int)(W * s + 0.5f);
    *sh = (int)(H * s + 0.5f);
    uint8_t *img = malloc((size_t)*sw * *sh);
    img_resize_gray(src, W, H, img, *sw, *sh);
    return img;
}

int main(int argc, char **argv)
{
    const char *inpath = NULL, *wpath = NULL, *outpath = "output.txt";
    int single = 0, top1 = 0, npos = 0, denom = 1;
    int stride = TILE_STRIDE, max_tiles = 512;
    float conf = CONF_THRESH;
    float scales[MAX_SCALES];
    int nscales = 0;
    const char *scales_arg = NULL;   /* NULL => use the header TILE_SCALES */

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--single"))    { single = 1; continue; }
        if (!strcmp(argv[i], "--top1"))      { top1 = 1; continue; }
        if (!strcmp(argv[i], "--no-top1"))   { top1 = 0; continue; }
        if (!strcmp(argv[i], "--conf") && i + 1 < argc)
            { conf = atof(argv[++i]); continue; }
        if (!strcmp(argv[i], "--stride") && i + 1 < argc)
            { stride = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--max-tiles") && i + 1 < argc)
            { max_tiles = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--denom") && i + 1 < argc)
            { denom = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--scales") && i + 1 < argc)
            { scales_arg = argv[++i]; continue; }
        if (npos == 0)      inpath = argv[i];
        else if (npos == 1) wpath = argv[i];
        else if (npos == 2) outpath = argv[i];
        ++npos;
    }
    if (!inpath || !wpath) {
        fprintf(stderr, "usage: %s <image.jpg|input_q.bin> <weights.bin> "
                "[output.txt]\n  [--conf X] [--stride N] [--scales a,b,c] "
                "[--max-tiles N] [--denom N] [--single] [--top1]\n",
                argv[0]);
        return 1;
    }
    {   /* scale list: --scales overrides; else the trainer-exported default
         * (TILE_SCALES) so the on-board pyramid always matches the protocol
         * CONF_THRESH was chosen under — no hand-syncing this list any more */
        if (scales_arg) {
            char buf[128];
            strncpy(buf, scales_arg, sizeof buf - 1);
            buf[sizeof buf - 1] = 0;
            for (char *tok = strtok(buf, ","); tok && nscales < MAX_SCALES;
                 tok = strtok(NULL, ","))
                scales[nscales++] = (float)atof(tok);
        } else {
#ifdef TILE_SCALES
            float ts[] = TILE_SCALES;
            int nts = NUM_TILE_SCALES;
            for (int i = 0; i < nts && nscales < MAX_SCALES; ++i)
                scales[nscales++] = ts[i];
#else                                   /* older header: v4 default */
            scales[nscales++] = 1.2f;
            scales[nscales++] = 0.5f;
            scales[nscales++] = 0.2f;
#endif
        }
    }
#if defined(NPU_SELFTEST) && NPU_SELFTEST
    single = 1;                         /* selftest is a one-pass protocol */
#endif
    if (ends_with(inpath, ".bin"))
        single = 1;

    int8_t *weights = load_weights(wpath);
    if (!weights) return 1;
    g_conf_thresh = conf;
    /* logit-domain integer gate: sigmoid(i*S) >= conf  <=>
     * i >= logit(conf)/S. floor+1 keeps it exact for the >= compare. */
    {
        float lg = logf(conf / (1.0f - conf)) / LAST_ACC_SCALE;
        g_obj_thr_i8 = (int)floorf(lg);
        if (g_obj_thr_i8 < -127) g_obj_thr_i8 = -127;
        if (g_obj_thr_i8 > 127)  g_obj_thr_i8 = 127;
    }

#ifdef USE_FPGA
    if (npu_map() != 0 || npu_probe_bank() != 0) return 1;
    npu_prepare_weights(model_layers, NUM_LAYERS, weights, g_wprep);
#else
    g_weights_bin = weights;
#endif

    /* ======================= SINGLE-PASS MODE ========================== */
    if (single) {
        uint8_t q[INPUT_BYTES];
        char transform[160];
        if (ends_with(inpath, ".bin")) {
            FILE *f = fopen(inpath, "rb");
            if (!f) { perror(inpath); return 1; }
            if (fread(q, 1, INPUT_BYTES, f) != INPUT_BYTES) {
                fprintf(stderr, "%s: need exactly %d bytes\n", inpath,
                        INPUT_BYTES);
                return 1;
            }
            fclose(f);
            strcpy(transform, "none");
        } else {
            double td = wall_ms();
            int W, H;
            uint8_t *gray = decode_gray(inpath, &W, &H, denom);
            if (!gray) return 1;
            tm_decode = wall_ms() - td;
            int side = W < H ? W : H;
            int x0 = (W - side) / 2, y0 = (H - side) / 2;
            uint8_t *sq = malloc((size_t)side * side);
            for (int y = 0; y < side; ++y)
                memcpy(sq + (size_t)y * side,
                       gray + (size_t)(y0 + y) * W + x0, side);
            free(gray);
            uint8_t *rs = malloc((size_t)INPUT_W * INPUT_H);
            img_resize_gray(sq, side, side, rs, INPUT_W, INPUT_H);
            free(sq);
            for (int i = 0; i < INPUT_BYTES; ++i) q[i] = rs[i] >> 1;
            free(rs);
            sprintf(transform, "crop %d %d %d of %dx%d", x0, y0, side, W, H);
        }
        static int8_t out[OUTPUT_BYTES];
        double t0 = wall_ms();
#if defined(USE_FPGA) && defined(NPU_SELFTEST) && NPU_SELFTEST
        {
            uint8_t *ref_a = malloc(32768), *ref_b = malloc(32768);
            uint8_t *hw_tmp = malloc(32768);
            memcpy(ref_a, q, INPUT_BYTES);
            npu_write_acts(q, INPUT_BYTES, npu_cur_bank);
            for (int l = 0; l < NUM_LAYERS; ++l) {
                if (npu_exec_layer_prepared(&model_layers[l],
                                            &g_wprep[l]) != 0)
                    return 1;
                const layer_config_t *L = &model_layers[l];
                uint32_t nb = (uint32_t)L->out_h * L->out_w * L->out_ch;
                npu_ref_layer(ref_a, L, weights, ref_b);
                npu_read_acts(hw_tmp, nb, npu_cur_bank);
                uint32_t bad = 0;
                for (uint32_t i = 0; i < nb; ++i)
                    if (hw_tmp[i] != ref_b[i]) {
                        if (bad < 8)
                            fprintf(stderr, "[SELFTEST] L%d byte %u: hw=%d "
                                    "ref=%d\n", l + 1, i, (int8_t)hw_tmp[i],
                                    (int8_t)ref_b[i]);
                        ++bad;
                    }
                printf("[SELFTEST] layer %d: %s (%u/%u mismatches)\n",
                       l + 1, bad ? "FAIL" : "ok", bad, nb);
                uint8_t *t = ref_a; ref_a = ref_b; ref_b = t;
            }
            npu_read_acts((uint8_t *)out, OUTPUT_BYTES, npu_cur_bank);
        }
#else
        if (run_tile(q, out) != 0) return 1;
#endif
        double t1 = wall_ms();

        det_t best = {-1, 0, 0, 0, 0};
        for (int cell = 0; cell < GRID * GRID; ++cell) {
            const int8_t *t5 = out + cell * HEAD_CH;
            float c0 = sigmoid_f((float)t5[0] * LAST_ACC_SCALE);
            if (c0 > best.conf) {
                int row = cell / GRID, col = cell % GRID;
                best.conf = c0;
                best.cx = (col + sigmoid_f((float)t5[1] * LAST_ACC_SCALE))
                          / GRID;
                best.cy = (row + sigmoid_f((float)t5[2] * LAST_ACC_SCALE))
                          / GRID;
                best.w = ANCHOR_W * expf(clampf((float)t5[3]
                                                * LAST_ACC_SCALE, -4, 4));
                best.h = ANCHOR_H * expf(clampf((float)t5[4]
                                                * LAST_ACC_SCALE, -4, 4));
            }
            if ((int)t5[0] >= g_obj_thr_i8)
                decode_cell(cell, t5, 0.0f, 0.0f, 1.0f, 1.0f);
        }
        if (g_ndets == 0 && top1 && best.conf >= 0) {
            g_dets[g_ndets++] = best;
            printf("no detection >= %.2f — emitting top-1 fallback "
                   "(conf %.3f)\n", (double)conf, best.conf);
        }
        /* NMS (insertion sort by conf, then greedy) */
        for (int i = 1; i < g_ndets; ++i) {
            det_t d = g_dets[i];
            int j = i - 1;
            while (j >= 0 && g_dets[j].conf < d.conf) {
                g_dets[j + 1] = g_dets[j]; --j;
            }
            g_dets[j + 1] = d;
        }
        det_t kept1[MAX_DETS];
        int nk1 = 0;
        for (int i = 0; i < g_ndets; ++i) {
            int ok = 1;
            for (int j = 0; j < nk1 && ok; ++j)
                if (iou(&g_dets[i], &kept1[j]) > NMS_IOU) ok = 0;
            if (ok) kept1[nk1++] = g_dets[i];
        }
        FILE *f = fopen(outpath, "w");
        if (!f) { perror(outpath); return 1; }
        fprintf(f, "# transform %s\n", transform);
        for (int i = 0; i < nk1; ++i)
            fprintf(f, "%.6f %.6f %.6f %.6f %.4f\n", kept1[i].cx,
                    kept1[i].cy, kept1[i].w, kept1[i].h, kept1[i].conf);
        fclose(f);
        printf("%d detection(s) -> %s  (npu %.1f ms, decode %.1f ms)\n",
               nk1, outpath, t1 - t0, tm_decode);
        for (int i = 0; i < nk1; ++i)
            printf("  person cx=%.3f cy=%.3f w=%.3f h=%.3f conf=%.3f\n",
                   kept1[i].cx, kept1[i].cy, kept1[i].w, kept1[i].h,
                   kept1[i].conf);
        return 0;
    }

    /* ======================= TILED SCAN MODE =========================== */
    double td = wall_ms();
    int W, H;
    uint8_t *gray = decode_gray(inpath, &W, &H, denom);
    if (!gray) return 1;
    tm_decode = wall_ms() - td;
    printf("decoded %dx%d gray (%.0f ms)\n", W, H, tm_decode);
    printf("tiled scan: %d scale(s)", nscales);
    for (int si = 0; si < nscales; ++si) printf(" %.3g", scales[si]);
    printf(" | stride %d  conf %.3f  agree>=%d or conf>=%.2f\n",
           stride, conf, MIN_VOTES, (double)HIGH_CONF);

    /* Collect tiles at VOTE_FLOOR (weak tiles can still cast a vote); the real
     * CONF_THRESH is applied AFTER the agreement vote — exactly the order the
     * trainer's tiled_image_dets + tiled sweep use, so deployed precision/
     * recall track the sweep CONF_THRESH was chosen under. */
    float final_conf = conf;
    float vote_floor = (VOTE_FLOOR < conf) ? VOTE_FLOOR : conf;
    g_conf_thresh = vote_floor;
    {
        float lg = logf(vote_floor / (1.0f - vote_floor)) / LAST_ACC_SCALE;
        g_obj_thr_i8 = (int)floorf(lg);
        if (g_obj_thr_i8 < -127) g_obj_thr_i8 = -127;
        if (g_obj_thr_i8 > 127)  g_obj_thr_i8 = 127;
    }

    /* fit pass first: letterbox the whole frame into one tile (large
     * persons), pad with mid-gray 114 like the training letterbox */
    {
        double tp = wall_ms();
        int side = W > H ? W : H;
        uint8_t *fit = malloc(INPUT_BYTES);
        int fw = W * INPUT_W / side, fh = H * INPUT_H / side;
        uint8_t *fs = malloc((size_t)fw * fh);
        img_resize_gray(gray, W, H, fs, fw, fh);
        memset(fit, 114, INPUT_BYTES);
        int ox = (INPUT_W - fw) / 2, oy = (INPUT_H - fh) / 2;
        for (int y = 0; y < fh; ++y)
            memcpy(fit + (size_t)(oy + y) * INPUT_W + ox,
                   fs + (size_t)y * fw, fw);
        free(fs);
        for (int i = 0; i < INPUT_BYTES; ++i) fit[i] >>= 1;
        tm_prep += wall_ms() - tp;
        /* the letterbox tile maps to [-oxn, 1+oxn] of the image */
        float wsc = (float)INPUT_W / fw;   /* >= 1 */
        float hsc = (float)INPUT_H / fh;
        float xn0 = -((float)ox / fw);
        float yn0 = -((float)oy / fh);
        static int8_t head[OUTPUT_BYTES];
        if (run_tile(fit, head) == 0) {
            for (int cell = 0; cell < GRID * GRID; ++cell) {
                const int8_t *t5 = head + cell * HEAD_CH;
                if ((int)t5[0] >= g_obj_thr_i8)
                    decode_cell(cell, t5, xn0, yn0, wsc, hsc);
            }
        }
        free(fit);
    }

    /* pyramid, coarse scales first (ascending scale value = ascending tile
     * count): a tile-budget cut then costs the fine level, not the cheap
     * wide views */
    for (int i = 0; i < nscales; ++i)
        for (int j = i + 1; j < nscales; ++j)
            if (scales[j] < scales[i]) {
                float t = scales[i]; scales[i] = scales[j]; scales[j] = t;
            }
    for (int si = 0; si < nscales; ++si) {
        float s = scales[si];
        if (s <= 0.0f || s > 1.001f) continue;
        int sw, sh;
        uint8_t *img;
        double tp = wall_ms();
        if (s >= 0.999f) {
            img = gray;
            sw = W; sh = H;
        } else {
            img = scale_image(gray, W, H, s, &sw, &sh);
        }
        tm_prep += wall_ms() - tp;
        if (sw >= INPUT_W && sh >= INPUT_H) {
            int rc = scan_scaled(img, sw, sh, stride, 0.0f, 0.0f,
                                 1.0f, 1.0f, max_tiles);
            if (rc < 0) return 1;
            if (rc == 1)
                printf("tile budget %d hit at scale %.2f — raise "
                       "--max-tiles or --stride for full coverage\n",
                       max_tiles, s);
        }
        if (img != gray) free(img);
    }
    free(gray);

    /* global NMS in image coords */
    double tmerge0 = wall_ms();
    for (int i = 1; i < g_ndets; ++i) {
        det_t d = g_dets[i];
        int j = i - 1;
        while (j >= 0 && g_dets[j].conf < d.conf) {
            g_dets[j + 1] = g_dets[j]; --j;
        }
        g_dets[j + 1] = d;
    }
    /* agreement-vote cluster (mirrors tiled_cluster_vote): each still-unused
     * (highest-conf) det seeds a cluster and absorbs its IoU>=NMS_IOU overlaps;
     * the seed is kept only if the cluster reached MIN_VOTES tiles OR the seed
     * itself cleared HIGH_CONF.  This turns the ~189x tile amplification into a
     * false-positive FILTER instead of the precision killer. */
    det_t kept[MAX_DETS];
    int nk = 0;
    static uint8_t vused[MAX_DETS];
    memset(vused, 0, sizeof vused);
    for (int i = 0; i < g_ndets; ++i) {
        /* g_dets is sorted by conf DESC. A seed is kept only if its own conf
         * >= final_conf (see the gate below), and every det at index > i has
         * conf <= g_dets[i].conf, so once we fall below final_conf nothing
         * further can ever be kept -> stop. This is OUTPUT-IDENTICAL, not an
         * approximation: the weak sub-threshold dets are NOT discarded, they
         * are still scanned as VOTES by the inner loops of the higher-conf
         * seeds above (which ran first); we only skip pointlessly seeding a
         * cluster from them. On dark/noisy frames the raw buffer saturates at
         * MAX_DETS with hundreds of sub-0.67 noise dets, and seeding each of
         * them ran a full O(n) soft-float iou() scan for nothing -- that was
         * the ~8 s "other". */
        if (g_dets[i].conf < final_conf) break;
        if (vused[i]) continue;
        vused[i] = 1;
        int votes = 1;
        for (int j = i + 1; j < g_ndets; ++j)
            if (!vused[j] && iou(&g_dets[i], &g_dets[j]) >= NMS_IOU) {
                vused[j] = 1;
                ++votes;
            }
        /* survive the vote, THEN pass CONF_THRESH (vote-then-threshold) */
        if ((votes >= MIN_VOTES || g_dets[i].conf >= HIGH_CONF)
                && g_dets[i].conf >= final_conf)
            kept[nk++] = g_dets[i];
    }
    double tm_merge = wall_ms() - tmerge0;   /* sort + agreement vote (the O(n^2)
                                              * soft-float NMS that floods on dark
                                              * frames) — now surfaced in [TIME] */

    FILE *f = fopen(outpath, "w");
    if (!f) { perror(outpath); return 1; }
    fprintf(f, "# transform none\n");
    for (int i = 0; i < nk; ++i)
        fprintf(f, "%.6f %.6f %.6f %.6f %.4f\n",
                clampf(kept[i].cx, 0, 1), clampf(kept[i].cy, 0, 1),
                kept[i].w, kept[i].h, kept[i].conf);
    fclose(f);

    double npu_total = tm_acts + tm_weights + tm_compute + tm_read;
    printf("%d person(s) -> %s\n", nk, outpath);
    for (int i = 0; i < nk; ++i)
        printf("  person cx=%.3f cy=%.3f w=%.3f h=%.3f conf=%.3f\n",
               kept[i].cx, kept[i].cy, kept[i].w, kept[i].h, kept[i].conf);
    printf("[TIME] decode %.0f  pyramid %.0f  npu %.0f  merge %.0f ms over "
           "%d tiles (%.1f ms/tile: acts %.1f, weights %.1f, compute %.1f, "
           "read %.1f)\n",
           tm_decode, tm_prep, npu_total, tm_merge, g_tiles,
           g_tiles ? npu_total / g_tiles : 0.0,
           g_tiles ? tm_acts / g_tiles : 0.0,
           g_tiles ? tm_weights / g_tiles : 0.0,
           g_tiles ? tm_compute / g_tiles : 0.0,
           g_tiles ? tm_read / g_tiles : 0.0);
    free(weights);
    return 0;
}
