// n2_model.c —— nano2 推理核：int4 权重 + int8 激活 + GQA + 滑窗(环形)KV + 全局首 token
//
// 与 PC 侧 sim_mcu.py 逐项对齐：
//   - 权重 int4（每输出行一个 f32 scale），int32 累加后一次乘回 sx*scale
//   - 激活逐 token int8 动态量化，away-from-zero 舍入
//   - K/V 以 int8 + 逐位置 scale 存环形窗口 N2_WIN，另存 position 0 的全局槽
//   - GELU 用 161 点 LUT + 线性插值
//   - LayerNorm / softmax 全程 f32

#include <math.h>
#include <string.h>

#include "stm32f10x.h"
#include "n2_weights.h"
#include "n2_model.h"

typedef struct {
    const uint8_t *q;   // int4 打包，行主序 [rows][cols/2]
    const float   *s;   // 每行一个 scale
    uint16_t rows, cols;
} n2_mat_t;

#define MAT(l, nm, R, C) { n2_l##l##_##nm##_q, n2_l##l##_##nm##_s, (R), (C) }

static const n2_mat_t M_TOK = { n2_tok_emb_q, n2_tok_emb_s, N2_VOCAB, N2_D };
static const n2_mat_t M_WQ[N2_LAYERS] = {
    MAT(0, wq, N2_NH * N2_HD, N2_D), MAT(1, wq, N2_NH * N2_HD, N2_D), MAT(2, wq, N2_NH * N2_HD, N2_D)
};
static const n2_mat_t M_WK[N2_LAYERS] = {
    MAT(0, wk, N2_KVD, N2_D), MAT(1, wk, N2_KVD, N2_D), MAT(2, wk, N2_KVD, N2_D)
};
static const n2_mat_t M_WV[N2_LAYERS] = {
    MAT(0, wv, N2_KVD, N2_D), MAT(1, wv, N2_KVD, N2_D), MAT(2, wv, N2_KVD, N2_D)
};
static const n2_mat_t M_WO[N2_LAYERS] = {
    MAT(0, wo, N2_D, N2_NH * N2_HD), MAT(1, wo, N2_D, N2_NH * N2_HD), MAT(2, wo, N2_D, N2_NH * N2_HD)
};
static const n2_mat_t M_W1[N2_LAYERS] = {
    MAT(0, w1, N2_FFN, N2_D), MAT(1, w1, N2_FFN, N2_D), MAT(2, w1, N2_FFN, N2_D)
};
static const n2_mat_t M_W2[N2_LAYERS] = {
    MAT(0, w2, N2_D, N2_FFN), MAT(1, w2, N2_D, N2_FFN), MAT(2, w2, N2_D, N2_FFN)
};

#define LN1W(l) n2_l##l##_ln1_w
#define LN1B(l) n2_l##l##_ln1_b
#define LN2W(l) n2_l##l##_ln2_w
#define LN2B(l) n2_l##l##_ln2_b
#define BQ(l)   n2_l##l##_bq
#define BK(l)   n2_l##l##_bk
#define BV(l)   n2_l##l##_bv
#define BO(l)   n2_l##l##_bo
#define B1(l)   n2_l##l##_b1
#define B2(l)   n2_l##l##_b2

static const float *const LN1W_T[N2_LAYERS] = { LN1W(0), LN1W(1), LN1W(2) };
static const float *const LN1B_T[N2_LAYERS] = { LN1B(0), LN1B(1), LN1B(2) };
static const float *const LN2W_T[N2_LAYERS] = { LN2W(0), LN2W(1), LN2W(2) };
static const float *const LN2B_T[N2_LAYERS] = { LN2B(0), LN2B(1), LN2B(2) };
static const int16_t *const BQ_T[N2_LAYERS] = { BQ(0), BQ(1), BQ(2) };
static const float BQ_S[N2_LAYERS] = { n2_l0_bq_s, n2_l1_bq_s, n2_l2_bq_s };
static const int16_t *const BK_T[N2_LAYERS] = { BK(0), BK(1), BK(2) };
static const float BK_S[N2_LAYERS] = { n2_l0_bk_s, n2_l1_bk_s, n2_l2_bk_s };
static const int16_t *const BV_T[N2_LAYERS] = { BV(0), BV(1), BV(2) };
static const float BV_S[N2_LAYERS] = { n2_l0_bv_s, n2_l1_bv_s, n2_l2_bv_s };
static const int16_t *const BO_T[N2_LAYERS] = { BO(0), BO(1), BO(2) };
static const float BO_S[N2_LAYERS] = { n2_l0_bo_s, n2_l1_bo_s, n2_l2_bo_s };
static const int16_t *const B1_T[N2_LAYERS] = { B1(0), B1(1), B1(2) };
static const float B1_S[N2_LAYERS] = { n2_l0_b1_s, n2_l1_b1_s, n2_l2_b1_s };
static const int16_t *const B2_T[N2_LAYERS] = { B2(0), B2(1), B2(2) };
static const float B2_S[N2_LAYERS] = { n2_l0_b2_s, n2_l1_b2_s, n2_l2_b2_s };

// ---------------- 工作缓冲（全部静态，无 malloc） ----------------
static float  s_x[N2_D];
static float  s_row[N2_FFN];
static float  s_ff[N2_FFN];
static float  s_proj[N2_D];
static float  s_q[N2_D];
static float  s_k[N2_KVD];
static float  s_v[N2_KVD];
static float  s_out[N2_D];
static float  s_sc[N2_WIN + 1];
static int8_t s_act[N2_FFN];
static int8_t s_aq[N2_D];
static int8_t s_qq[N2_D];
static int8_t s_ak[N2_KVD];
static int8_t s_av[N2_KVD];
static int8_t s_ao[N2_D];

// KV 环形窗口
static int8_t s_kc[N2_LAYERS][N2_WIN][N2_KVD];
static int8_t s_vc[N2_LAYERS][N2_WIN][N2_KVD];
static float  s_kcs[N2_LAYERS][N2_WIN];
static float  s_vcs[N2_LAYERS][N2_WIN];
static int8_t s_kg[N2_LAYERS][N2_KVD];
static int8_t s_vg[N2_LAYERS][N2_KVD];
static float  s_kgs[N2_LAYERS];
static float  s_vgs[N2_LAYERS];

static int s_t;          // 当前位置（0 起）
static int s_argmax;
static float s_logits[N2_VOCAB];

// ---------------- 基础算子 ----------------
static void ln_row(const float *x, const float *w, const float *b, float *y)
{
    float mean = 0.0f, var = 0.0f, rstd, d;
    int i;
    for (i = 0; i < N2_D; i++) mean += x[i];
    mean /= (float)N2_D;
    for (i = 0; i < N2_D; i++) { d = x[i] - mean; var += d * d; }
    var /= (float)N2_D;
    rstd = 1.0f / sqrtf(var + 1e-5f);
    for (i = 0; i < N2_D; i++) y[i] = (x[i] - mean) * rstd * w[i] + b[i];
}

static float quant8(const float *x, int n, int8_t *q)
{
    float amax = 0.0f, inv, v;
    int32_t iv;
    int i;
    for (i = 0; i < n; i++) { v = fabsf(x[i]); if (v > amax) amax = v; }
    if (amax < 1e-12f) amax = 1e-12f;
    inv = 127.0f / amax;
    for (i = 0; i < n; i++) {
        v = x[i] * inv;
        iv = (int32_t)(fabsf(v) + 0.5f);
        if (iv > 127) iv = 127;
        q[i] = (int8_t)(v < 0.0f ? -iv : iv);
    }
    return amax / 127.0f;
}

// out[j] = (sum_c xq[c] * wq[j][c]) * sx * w->s[j] + bias[j]
static void gemm4(const int8_t *xq, float sx, const n2_mat_t *w,
                  const int16_t *bias, float bias_scale, float *out)
{
    int j, c;
    const uint16_t half = (uint16_t)(w->cols >> 1);
    for (j = 0; j < (int)w->rows; j++) {
        const uint8_t *row = w->q + (uint32_t)j * half;
        int32_t acc = 0;
        for (c = 0; c < (int)w->cols; c += 2) {
            uint8_t b = row[c >> 1];
            int8_t lo = (int8_t)((b & 0x0F) << 4) >> 4;
            int8_t hi = (int8_t)(b & 0xF0) >> 4;
            acc += (int32_t)xq[c] * (int32_t)lo;
            acc += (int32_t)xq[c + 1] * (int32_t)hi;
        }
        out[j] = (float)acc * sx * w->s[j] + (bias ? (float)bias[j] * bias_scale : 0.0f);
    }
}

static float gelu_one(float x)
{
    const float hi = N2_LUT_LO + (float)(N2_LUT_N - 1) * N2_LUT_STEP;
    if (x <= N2_LUT_LO) return n2_gelu_lut[0];
    if (x >= hi) return n2_gelu_lut[N2_LUT_N - 1] + (x - hi);
    {
        float t = (x - N2_LUT_LO) * (1.0f / N2_LUT_STEP);
        int i = (int)t;
        float f = t - (float)i;
        return n2_gelu_lut[i] + (n2_gelu_lut[i + 1] - n2_gelu_lut[i]) * f;
    }
}

// ---------------- 单步前向 ----------------
static void step(uint8_t byte)
{
    const int pos = (s_t < N2_SEQ) ? s_t : (N2_SEQ - 1);
    const uint8_t *er = M_TOK.q + (uint32_t)byte * (N2_D >> 1);
    const float es = M_TOK.s[byte];
    const int8_t *pr = n2_pos_q + (uint32_t)pos * N2_D;
    const float ps = n2_pos_s[pos];
    const float att = 1.0f / sqrtf((float)N2_HD);
    int c, L, h, d, i;

    for (c = 0; c < N2_D; c += 2) {
        uint8_t b = er[c >> 1];
        int8_t lo = (int8_t)((b & 0x0F) << 4) >> 4;
        int8_t hi = (int8_t)(b & 0xF0) >> 4;
        s_x[c]     = (float)lo * es + (float)pr[c]     * ps;
        s_x[c + 1] = (float)hi * es + (float)pr[c + 1] * ps;
    }

    for (L = 0; L < N2_LAYERS; L++) {
        float sx, sq, so, s2, sa, inv;
        int slot, start, use_global, idx, n;

        // ---- 注意力 ----
        ln_row(s_x, LN1W_T[L], LN1B_T[L], s_row);
        sx = quant8(s_row, N2_D, s_aq);
        gemm4(s_aq, sx, &M_WQ[L], BQ_T[L], BQ_S[L], s_q);
        gemm4(s_aq, sx, &M_WK[L], BK_T[L], BK_S[L], s_k);
        gemm4(s_aq, sx, &M_WV[L], BV_T[L], BV_S[L], s_v);
        sq = quant8(s_q, N2_D, s_qq);
        {
            const float sk = quant8(s_k, N2_KVD, s_ak);
            const float sv = quant8(s_v, N2_KVD, s_av);

            slot = s_t % N2_WIN;
            memcpy(s_kc[L][slot], s_ak, N2_KVD);
            memcpy(s_vc[L][slot], s_av, N2_KVD);
            s_kcs[L][slot] = sk;
            s_vcs[L][slot] = sv;
            if (s_t == 0) {
                memcpy(s_kg[L], s_ak, N2_KVD);
                memcpy(s_vg[L], s_av, N2_KVD);
                s_kgs[L] = sk;
                s_vgs[L] = sv;
            }
        }

        start = (s_t >= N2_WIN) ? (s_t - N2_WIN + 1) : 0;
        use_global = (s_t >= N2_WIN);
        for (c = 0; c < N2_D; c++) s_out[c] = 0.0f;

        for (h = 0; h < N2_NH; h++) {
            const int g = h / (N2_NH / N2_G);
            const int8_t *qh = s_qq + h * N2_HD;
            float mx = -1e30f, sum = 0.0f;

            idx = 0;
            for (i = start; i <= s_t; i++, idx++) {
                const int8_t *kr = s_kc[L][i % N2_WIN] + g * N2_HD;
                int32_t acc = 0;
                for (d = 0; d < N2_HD; d++) acc += (int32_t)qh[d] * (int32_t)kr[d];
                s_sc[idx] = (float)acc * sq * s_kcs[L][i % N2_WIN] * att;
                if (s_sc[idx] > mx) mx = s_sc[idx];
            }
            if (use_global) {
                const int8_t *kr = s_kg[L] + g * N2_HD;
                int32_t acc = 0;
                for (d = 0; d < N2_HD; d++) acc += (int32_t)qh[d] * (int32_t)kr[d];
                s_sc[idx] = (float)acc * sq * s_kgs[L] * att;
                if (s_sc[idx] > mx) mx = s_sc[idx];
                idx++;
            }
            n = idx;
            for (idx = 0; idx < n; idx++) {
                s_sc[idx] = expf(s_sc[idx] - mx);
                sum += s_sc[idx];
            }
            inv = 1.0f / sum;
            idx = 0;
            for (i = start; i <= s_t; i++, idx++) {
                float wv = s_sc[idx] * inv * s_vcs[L][i % N2_WIN];
                const int8_t *vr = s_vc[L][i % N2_WIN] + g * N2_HD;
                for (d = 0; d < N2_HD; d++) s_out[h * N2_HD + d] += wv * (float)vr[d];
            }
            if (use_global) {
                float wv = s_sc[idx] * inv * s_vgs[L];
                const int8_t *vr = s_vg[L] + g * N2_HD;
                for (d = 0; d < N2_HD; d++) s_out[h * N2_HD + d] += wv * (float)vr[d];
            }
        }

        so = quant8(s_out, N2_D, s_ao);
        gemm4(s_ao, so, &M_WO[L], BO_T[L], BO_S[L], s_proj);
        for (c = 0; c < N2_D; c++) s_x[c] += s_proj[c];

        // ---- FFN ----
        ln_row(s_x, LN2W_T[L], LN2B_T[L], s_row);
        s2 = quant8(s_row, N2_D, s_aq);
        gemm4(s_aq, s2, &M_W1[L], B1_T[L], B1_S[L], s_ff);
        for (i = 0; i < N2_FFN; i++) s_ff[i] = gelu_one(s_ff[i]);
        sa = quant8(s_ff, N2_FFN, s_act);
        gemm4(s_act, sa, &M_W2[L], B2_T[L], B2_S[L], s_proj);
        for (c = 0; c < N2_D; c++) s_x[c] += s_proj[c];
    }

    // ---- 末层归一化 + 权重绑定的 lm_head（int32 累加，按行 scale 还原） ----
    {
        int best = 0;
        float bestv = -1e30f;
        float sx;
        ln_row(s_x, n2_ln_f_w, n2_ln_f_b, s_row);
        sx = quant8(s_row, N2_D, s_aq);
        for (i = 0; i < N2_VOCAB; i++) {
            const uint8_t *row = M_TOK.q + (uint32_t)i * (N2_D >> 1);
            int32_t acc = 0;
            for (c = 0; c < N2_D; c += 2) {
                uint8_t b = row[c >> 1];
                int8_t lo = (int8_t)((b & 0x0F) << 4) >> 4;
                int8_t hi = (int8_t)(b & 0xF0) >> 4;
                acc += (int32_t)s_aq[c] * (int32_t)lo;
                acc += (int32_t)s_aq[c + 1] * (int32_t)hi;
            }
            s_logits[i] = (float)acc * sx * M_TOK.s[i];
            if (s_logits[i] > bestv) { bestv = s_logits[i]; best = i; }
        }
        s_argmax = best;
    }
    s_t++;
}

// ---------------- 对外接口 ----------------
void N2_Reset(void)
{
    s_t = 0;
    s_argmax = 0;
}

int N2_DebugTop(uint8_t *ids, int n)
{
    float ceil_ = 1e30f;
    int k = 0, j;
    for (; k < n; k++) {
        float bv = -1e30f;
        int bi = -1;
        for (j = 0; j < N2_VOCAB; j++) {
            if (s_logits[j] < ceil_ && s_logits[j] > bv) { bv = s_logits[j]; bi = j; }
        }
        if (bi < 0) break;
        ids[k] = (uint8_t)bi;
        ceil_ = bv;
    }
    return k;
}

int N2_Generate(const uint8_t *prompt, int plen, uint8_t *out, int max_out,
                void (*on_byte)(uint8_t), void (*on_tick)(void))
{
    int i, k = 0, last = N2_EOS;

    N2_Reset();
    if (plen > N2_SEQ) { prompt += (plen - N2_SEQ); plen = N2_SEQ; }
    for (i = 0; i < plen; i++) {
        if (on_tick) on_tick();
        step(prompt[i]);
        last = s_argmax;
    }
    while (k < max_out) {
        if (last == N2_EOS) break;
        out[k++] = (uint8_t)last;
        if (on_byte) on_byte((uint8_t)last);
        if (on_tick) on_tick();
        step((uint8_t)last);
        last = s_argmax;
    }
    return k;
}
