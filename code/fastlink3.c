/* fastlink.c — env3 信道链 C 加速核（2026-10-25 死线令：C+ctypes+OpenMP）。
 * 覆盖 _interp（双线性+斜率）→ _link（SINR/速率链）→ _refresh_link
 * （softmax 关联）→ _aggregates（ρ/R_avail/u/T）全链，单次 ctypes 调用。
 * 数学逐行照抄 env3.py（数值等价：全程 double，金样 allclose 验证）。
 * 并行：OpenMP 按 M（用户）维——列独立；attn 的 max/sum 归约两段。
 * 升级版（2026-10-27 go/c++ 升级令）：全指针 restrict + interp/SINR/softmax
 * omp simd 显式向量化 + guided 调度；数值语义与原版逐行等价（金样对拍）。
 * 编译：gcc -O3 -march=native -funroll-loops -fopenmp -shared -fPIC  *       fastlink2.c -o fastlink2.so -lm
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LOG2 0.6931471805599453

/*
 * fastlink3.c —— 动态信道解析径 C 核（2026-10-27 用户裁定 A + 加速令）
 * 与 fastlink2.c 同源：下游段（link/agg/denom/bsh/R/attn/softmax/assoc/
 * R_avail/rho/u/T）逐行相同；interp 段替换为解析增益段——
 *   G_db(i,u) = C0[u] - 10*gamma*log10(d_iu + 50)
 *   斜率      = -10*gamma*dx/(ln10*D^2)，D=d+50（对机位置解析导数）
 * 用户位置实时参与（移动人群物理贯通）。编译为独立 fastlink3.so。
 */
#include <math.h>
#include <omp.h>

#define LOG2 0.6931471805599453

/* 解析增益（按机独立扫 M——与 interp_one 同位替换） */
static inline void pl_one(
    double px, double py, const double *restrict user_pos,
    const double *restrict c0, double gamma, int M,
    const double *restrict sh_row,
    double *restrict gain, double *restrict slx, double *restrict sly)
{
    const double LN10 = 2.302585092994046;
#pragma omp simd
    for (int m = 0; m < M; m++) {
        double dx = px - user_pos[2 * m];
        double dy = py - user_pos[2 * m + 1];
        double D = sqrt(dx * dx + dy * dy) + 50.0;
        gain[m] = c0[m] - 10.0 * gamma * log10(D);
        double k = -10.0 * gamma / (LN10 * D * D);
        slx[m] = k * dx;
        sly[m] = k * dy;
    }
    if (sh_row) {                      /* 链路级随机阴影（逐链路 dB） */
#pragma omp simd
        for (int m = 0; m < M; m++) gain[m] += sh_row[m];
    }
}

void fast_link_dyn(
    const double *restrict pos, const double *restrict b,
    const double *restrict pi,
    const double *restrict p, const double *restrict dem,
    double *restrict assoc,
    const double *restrict user_pos, const double *restrict c0,
    const double *restrict shadow,
    double gamma, int M, int N,
    double n0_lin, double p_ref_lin, double alpha, double kappa,
    double tau_u, int bw_assoc_equal,
    double *restrict G_db, double *restrict sx, double *restrict sy,
    double *restrict g_out, double *restrict P_out,
    double *restrict sig, double *restrict denom, double *restrict sinr,
    double *restrict r, double *restrict bsh,
    double *restrict R, double *restrict agg, double *restrict attn,
    double *restrict x_star,
    double *restrict R_avail, double *restrict rho, double *restrict u,
    double *restrict T,
    const double *restrict R_avail_prev,
    const double *restrict rho_prev)
{
    /* ---- 解析增益（按机并行：每机独立扫 M；用户位置实时参与） ---- */
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++) {
        pl_one(pos[2 * i], pos[2 * i + 1], user_pos, c0, gamma, M,
               shadow ? shadow + (long)i * M : 0,
               G_db + (long)i * M, sx + (long)i * M, sy + (long)i * M);
    }
    /* ---- link ---- */
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++) {
        double Pi = p[i] * p_ref_lin;
        P_out[i] = Pi;
        const double *Gi = G_db + (long)i * M;
        double *gi = g_out + (long)i * M;
        double *si = sig + (long)i * M;
#pragma omp simd
        for (int m = 0; m < M; m++) {
            gi[m] = exp(Gi[m] * 0.23025850929940458);  /* 10^(x/10) 快路 */
            si[m] = Pi * gi[m];                        /* sig=P*g */
        }
    }
    /* agg[m]=Σ_i assoc[i,m]*sig[i,m]（按 M 并行跨 N 规约） */
#pragma omp parallel for schedule(guided)
    for (int m = 0; m < M; m++) {
        double s = 0.0;
        for (int i = 0; i < N; i++)
            s += assoc[(long)i * M + m] * sig[(long)i * M + m];
        agg[m] = s;
    }
    /* denom = n0 + agg - assoc*sig；sinr/r 同循环 */
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++)
#pragma omp simd
        for (int m = 0; m < M; m++) {
            long k = (long)i * M + m;
            double d_ = n0_lin + agg[m] - assoc[k] * sig[k];
            denom[k] = d_;
            double sn = sig[k] / (d_ > 1e-30 ? d_ : 1e-30);
            sinr[k] = sn;
            r[k] = log1p(sn) / LOG2;
        }
    /* bsh/R（按机：wgt 行和后归一） */
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++) {
        const double *ai = assoc + (long)i * M;
        double *bi = bsh + (long)i * M;
        double *Ri = R + (long)i * M;
        double wsum = 0.0;
        for (int m = 0; m < M; m++) {
            bi[m] = bw_assoc_equal ? ai[m] : ai[m] * dem[m];
            wsum += bi[m];
        }
        double inv = b[i] / (wsum > 1e-9 ? wsum : 1e-9);
        for (int m = 0; m < M; m++) {
            bi[m] *= inv;
            Ri[m] = bi[m] * r[(long)i * M + m];
        }
    }
    /* ---- refresh：attn → softmax(x_star) → EMA ---- */
    /* attn 列 max（reduction） */
    double *amax = (double *)malloc(sizeof(double) * M);
#pragma omp parallel for schedule(guided)
    for (int m = 0; m < M; m++) amax[m] = -1e300;
#pragma omp parallel for collapse(2) schedule(guided)
    for (int i = 0; i < N; i++)
        for (int m = 0; m < M; m++) {
            long k = (long)i * M + m;
            double v = alpha * r[k]
                + kappa * (R_avail_prev[i] / (rho_prev[i] + 1e-3))
                + pi[i];
            attn[k] = v;
        }
#pragma omp parallel for schedule(guided)
    for (int m = 0; m < M; m++) {
        double mx = -1e300;
        for (int i = 0; i < N; i++) {
            double v = attn[(long)i * M + m];
            if (v > mx) mx = v;
        }
        amax[m] = mx;
    }
    /* exp+列归一 */
    double *xsum = (double *)malloc(sizeof(double) * M);
#pragma omp parallel for schedule(guided)
    for (int m = 0; m < M; m++) {
        double s = 0.0;
#pragma omp simd reduction(+ : s)
        for (int i = 0; i < N; i++) {
            long k = (long)i * M + m;
            double e = exp(attn[k] - amax[m]);
            x_star[k] = e;
            s += e;
        }
        xsum[m] = s;
    }
    double a = 1.0 / tau_u;
#pragma omp parallel for collapse(2) schedule(guided)
    for (int i = 0; i < N; i++)
        for (int m = 0; m < M; m++) {
            long k = (long)i * M + m;
            assoc[k] = (1.0 - a) * assoc[k] + a * (x_star[k] / xsum[m]);
        }
    /* ---- aggregates（按机规约） ---- */
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++) {
        const double *ai = assoc + (long)i * M;
        const double *Ri = R + (long)i * M;
        double ra = 0.0, rh = 0.0, tt = 0.0;
        for (int m = 0; m < M; m++) {
            ra += ai[m] * Ri[m];
            rh += ai[m] * dem[m];
            double dmin = dem[m] < Ri[m] ? dem[m] : Ri[m];
            tt += ai[m] * dmin;
        }
        R_avail[i] = ra;
        rho[i] = rh;
        u[i] = rh / (ra > 1e-9 ? ra : 1e-9);
        T[i] = tt;
    }
    free(amax);
    free(xsum);
}
