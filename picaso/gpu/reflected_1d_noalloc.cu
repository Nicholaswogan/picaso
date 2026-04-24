#include <cuda_runtime.h>
#include <math_functions.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#define SQ3 1.7320508075688772
#define PI 3.14159265358979323846264338327950288419716939937510

__device__ __forceinline__ int global_wno_idx() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int layer_w_idx(int layer, int w, int nwno) {
    return layer * nwno + w;
}

__device__ inline void solve_tridiagonal_inplace(int n, double *a, double *b, double *c, double *d) {
    d[0] = d[0] / b[0];
    c[0] = c[0] / b[0];

    for (int i = 1; i < n - 1; ++i) {
        double denom = b[i] - a[i] * c[i - 1];
        double inv_denom = 1.0 / denom;
        c[i] = c[i] * inv_denom;
        d[i] = (d[i] - a[i] * d[i - 1]) * inv_denom;
    }

    if (n > 1) {
        double denom = b[n - 1] - a[n - 1] * c[n - 2];
        d[n - 1] = (d[n - 1] - a[n - 1] * d[n - 2]) / denom;

        for (int i = n - 2; i >= 0; --i) {
            d[i] = d[i] - c[i] * d[i + 1];
        }
    }
}

__device__ inline void compute_toon_coefficients(
    int layer,
    int w,
    int nwno,
    const double *w0_dev,
    const double *ftau_cld_dev,
    const double *cosb_dev,
    int toon_coefficients,
    double *g1,
    double *g2)
{
    const int idx = layer_w_idx(layer, w, nwno);
    const double w0 = w0_dev[idx];
    const double ftau = ftau_cld_dev[idx];
    const double cb = cosb_dev[idx];

    if (toon_coefficients == 1) {
        *g1 = (7.0 - w0 * (4.0 + 3.0 * ftau * cb)) / 4.0;
        *g2 = -(1.0 - w0 * (4.0 - 3.0 * ftau * cb)) / 4.0;
    } else {
        *g1 = 0.5 * SQ3 * (2.0 - w0 * (1.0 + ftau * cb));
        *g2 = 0.5 * SQ3 * w0 * (1.0 - ftau * cb);
    }
}

__device__ inline void compute_phase_terms(
    int layer,
    int w,
    int nwno,
    const double *w0_dev,
    const double *cosb_dev,
    const double *ftau_cld_dev,
    const double *ftau_ray_dev,
    const double *gcos2_dev,
    const double *cosb_og_dev,
    const double *w0_og_dev,
    const double *tau_og_dev,
    const double *dtau_og_dev,
    const double *tau_dev,
    const double *dtau_dev,
    const double *lambda_row,
    const double *gama_row,
    const double *F0PI_dev,
    double u0,
    double u1,
    double cos_theta,
    int single_phase,
    int multi_phase,
    double frac_a,
    double frac_b,
    double frac_c,
    double constant_back,
    double constant_forward,
    int toon_coefficients,
    double *g3,
    double *a_minus,
    double *a_plus,
    double *c_minus_up,
    double *c_plus_up,
    double *c_minus_down,
    double *c_plus_down,
    double *exptrm,
    double *exptrm_positive,
    double *exptrm_minus)
{
    const int idx = layer_w_idx(layer, w, nwno);
    const double lambda = lambda_row[layer];
    const double gama = gama_row[layer];
    const double f0pi = F0PI_dev[w];
    const double inv_u0 = 1.0 / u0;

    double g1;
    double g2;
    compute_toon_coefficients(
        layer,
        w,
        nwno,
        w0_dev,
        ftau_cld_dev,
        cosb_dev,
        toon_coefficients,
        &g1,
        &g2);

    if (toon_coefficients == 1) {
        *g3 = (2.0 - 3.0 * ftau_cld_dev[idx] * cosb_dev[idx] * u0) / 4.0;
    } else {
        *g3 = 0.5 * (1.0 - SQ3 * ftau_cld_dev[idx] * cosb_dev[idx] * u0);
    }

    const double g4 = 1.0 - *g3;
    const double denom = lambda * lambda - inv_u0 * inv_u0;
    const double w0 = w0_dev[idx];

    *a_minus = f0pi * w0 * (g4 * (g1 + inv_u0) + g2 * (*g3)) / denom;
    *a_plus = f0pi * w0 * ((*g3) * (g1 - inv_u0) + g2 * g4) / denom;

    const double tau_here = tau_dev[idx];
    const double tau_next = tau_dev[layer_w_idx(layer + 1, w, nwno)];
    const double exp_top = exp(-tau_here * inv_u0);
    const double exp_bottom = exp(-tau_next * inv_u0);

    *c_minus_up = (*a_minus) * exp_top;
    *c_plus_up = (*a_plus) * exp_top;
    *c_minus_down = (*a_minus) * exp_bottom;
    *c_plus_down = (*a_plus) * exp_bottom;

    double exptrm_val = lambda * dtau_dev[idx];
    if (exptrm_val > 35.0) {
        exptrm_val = 35.0;
    }
    *exptrm = exptrm_val;
    *exptrm_positive = exp(exptrm_val);
    *exptrm_minus = 1.0 / (*exptrm_positive);

    (void)ftau_ray_dev;
    (void)gcos2_dev;
    (void)cosb_og_dev;
    (void)w0_og_dev;
    (void)tau_og_dev;
    (void)dtau_og_dev;
    (void)multi_phase;
    (void)frac_a;
    (void)frac_b;
    (void)frac_c;
    (void)constant_back;
    (void)constant_forward;
    (void)u1;
    (void)cos_theta;
}

__device__ inline double compute_single_phase_source(
    int layer,
    int w,
    int nwno,
    const double *ftau_cld_dev,
    const double *ftau_ray_dev,
    const double *gcos2_dev,
    const double *cosb_og_dev,
    const double *w0_og_dev,
    double cos_theta,
    int single_phase,
    double frac_a,
    double frac_b,
    double frac_c,
    double constant_back,
    double constant_forward)
{
    const int idx = layer_w_idx(layer, w, nwno);
    const double ftau_cld = ftau_cld_dev[idx];
    const double ftau_ray = ftau_ray_dev[idx];
    const double cb = cosb_og_dev[idx];

    double g_forward = 0.0;
    double g_back = 0.0;
    double f = 0.0;

    if (single_phase != 1) {
        g_forward = constant_forward * cb;
        g_back = constant_back * cb;
        f = frac_a + frac_b * pow(g_back, frac_c);
    }

    if (single_phase == 0) {
        const double hg_forward = (1.0 - g_forward * g_forward) /
            sqrt((1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta) *
                 (1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta) *
                 (1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta));
        const double hg_backward = (1.0 - g_back * g_back) /
            sqrt((1.0 + g_back * g_back + 2.0 * g_back * cos_theta) *
                 (1.0 + g_back * g_back + 2.0 * g_back * cos_theta) *
                 (1.0 + g_back * g_back + 2.0 * g_back * cos_theta));
        return f * hg_forward + (1.0 - f) * hg_backward + gcos2_dev[idx];
    } else if (single_phase == 1) {
        return (1.0 - cb * cb) /
            sqrt((1.0 + cb * cb + 2.0 * cb * cos_theta) *
                 (1.0 + cb * cb + 2.0 * cb * cos_theta) *
                 (1.0 + cb * cb + 2.0 * cb * cos_theta));
    } else if (single_phase == 2) {
        const double hg_forward = (1.0 - g_forward * g_forward) /
            sqrt((1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta) *
                 (1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta) *
                 (1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta));
        const double hg_backward = (1.0 - g_back * g_back) /
            sqrt((1.0 + g_back * g_back + 2.0 * g_back * cos_theta) *
                 (1.0 + g_back * g_back + 2.0 * g_back * cos_theta) *
                 (1.0 + g_back * g_back + 2.0 * g_back * cos_theta));
        return f * hg_forward + (1.0 - f) * hg_backward;
    } else {
        const double hg_forward = (1.0 - g_forward * g_forward) /
            sqrt((1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta) *
                 (1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta) *
                 (1.0 + g_forward * g_forward + 2.0 * g_forward * cos_theta));
        const double hg_backward = (1.0 - g_back * g_back) /
            sqrt((1.0 + g_back * g_back + 2.0 * g_back * cos_theta) *
                 (1.0 + g_back * g_back + 2.0 * g_back * cos_theta) *
                 (1.0 + g_back * g_back + 2.0 * g_back * cos_theta));
        return ftau_cld * (f * hg_forward + (1.0 - f) * hg_backward) +
            ftau_ray * (0.75 * (1.0 + cos_theta * cos_theta));
    }

    (void)w0_og_dev;
}

extern "C" __global__ void reflected_prepare_constants_kernel(
    const double *w0_dev,
    const double *ftau_cld_dev,
    const double *cosb_dev,
    int nlayer,
    int nwno,
    int toon_coefficients,
    double *lambda_dev,
    double *gama_dev)
{
    const int w = global_wno_idx();
    if (w >= nwno) {
        return;
    }

    for (int layer = 0; layer < nlayer; ++layer) {
        const int idx = layer_w_idx(layer, w, nwno);
        double g1;
        double g2;
        compute_toon_coefficients(
            layer,
            w,
            nwno,
            w0_dev,
            ftau_cld_dev,
            cosb_dev,
            toon_coefficients,
            &g1,
            &g2);

        const double lam = sqrt(fmax(g1 * g1 - g2 * g2, 0.0));
        lambda_dev[idx] = lam;
        gama_dev[idx] = (g1 - lam) / g2;
    }
}

extern "C" __global__ void reflected_solve_kernel(
    int nlevel,
    int nlayer,
    int nwno,
    int nang,
    const double *wno_dev,
    const double *dtau_dev,
    const double *tau_dev,
    const double *w0_dev,
    const double *cosb_dev,
    const double *gcos2_dev,
    const double *ftau_cld_dev,
    const double *ftau_ray_dev,
    const double *dtau_og_dev,
    const double *tau_og_dev,
    const double *w0_og_dev,
    const double *cosb_og_dev,
    const double *surf_reflect_dev,
    const double *ubar0_dev,
    const double *ubar1_dev,
    double cos_theta,
    const double *F0PI_dev,
    int single_phase,
    int multi_phase,
    double frac_a,
    double frac_b,
    double frac_c,
    double constant_back,
    double constant_forward,
    int get_toa_intensity,
    int get_lvl_flux,
    int toon_coefficients,
    double b_top,
    const double *lambda_dev,
    const double *gama_dev,
    double *A_dev,
    double *B_dev,
    double *C_dev,
    double *D_dev,
    double *xint_at_top_dev,
    double *flux_minus_all_dev,
    double *flux_plus_all_dev,
    double *flux_minus_midpt_all_dev,
    double *flux_plus_midpt_all_dev)
{
    const int w = global_wno_idx();
    if (w >= nwno) {
        return;
    }

    const int tri_size = 2 * nlayer;
    double *A = A_dev + w * tri_size;
    double *B = B_dev + w * tri_size;
    double *C = C_dev + w * tri_size;
    double *D = D_dev + w * tri_size;
    const double f0pi_w = F0PI_dev[w];
    const double surf_reflect_w = surf_reflect_dev[w];
    const double inv_pi = 1.0 / PI;

    for (int ang = 0; ang < nang; ++ang) {
        const double u0 = ubar0_dev[ang];
        const double u1 = ubar1_dev[ang];
        const double inv_u0 = 1.0 / u0;
        const double inv_u1 = 1.0 / u1;
        const double inv_u0u1 = inv_u0 * inv_u1;
        const double sum_u = u0 + u1;
        const double inv_sum_u = 1.0 / sum_u;
        const double u0_over_sum = u0 * inv_sum_u;
        const double u0_scale = u0 * f0pi_w;
        double c_minus_down_last = 0.0;
        double c_plus_down_last = 0.0;

        if (get_lvl_flux) {
            for (int lvl = 0; lvl < nlevel; ++lvl) {
                const int out_base = (ang * nlevel + lvl) * nwno + w;
                flux_minus_all_dev[out_base] = 0.0;
                flux_plus_all_dev[out_base] = 0.0;
                flux_minus_midpt_all_dev[out_base] = 0.0;
                flux_plus_midpt_all_dev[out_base] = 0.0;
            }
        }

        if (get_toa_intensity) {
            xint_at_top_dev[ang * nwno + w] = 0.0;
        }

        // Build the tridiagonal system for this angle.
        for (int layer = 0; layer < nlayer; ++layer) {
            double g3;
            double a_minus;
            double a_plus;
            double c_minus_up;
            double c_plus_up;
            double c_minus_down;
            double c_plus_down;
            double exptrm;
            double exptrm_positive;
            double exptrm_minus;

            compute_phase_terms(
                layer,
                w,
                nwno,
                w0_dev,
                cosb_dev,
                ftau_cld_dev,
                ftau_ray_dev,
                gcos2_dev,
                cosb_og_dev,
                w0_og_dev,
                tau_og_dev,
                dtau_og_dev,
                tau_dev,
                dtau_dev,
                lambda_dev + w * nlayer,
                gama_dev + w * nlayer,
                F0PI_dev,
                u0,
                u1,
                cos_theta,
                single_phase,
                multi_phase,
                frac_a,
                frac_b,
                frac_c,
                constant_back,
                constant_forward,
                toon_coefficients,
                &g3,
                &a_minus,
                &a_plus,
                &c_minus_up,
                &c_plus_up,
                &c_minus_down,
                &c_plus_down,
                &exptrm,
                &exptrm_positive,
                &exptrm_minus);

            const double lambda_here = lambda_dev[w * nlayer + layer];
            const double gama_here = gama_dev[w * nlayer + layer];
            const int idx = layer_w_idx(layer, w, nwno);

            if (layer == 0) {
                A[0] = 0.0;
                B[0] = gama_here + 1.0;
                C[0] = gama_here - 1.0;
                D[0] = b_top - c_minus_up;
            }

            if (layer < nlayer - 1) {
                const int next_idx = layer_w_idx(layer + 1, w, nwno);
                const double lambda_next = lambda_dev[w * nlayer + layer + 1];
                const double gama_next = gama_dev[w * nlayer + layer + 1];

                double g3_next;
                double a_minus_next;
                double a_plus_next;
                double c_minus_up_next;
                double c_plus_up_next;
                double c_minus_down_next;
                double c_plus_down_next;
                double exptrm_next;
                double exptrm_positive_next;
                double exptrm_minus_next;
                compute_phase_terms(
                    layer + 1,
                    w,
                    nwno,
                    w0_dev,
                    cosb_dev,
                    ftau_cld_dev,
                    ftau_ray_dev,
                    gcos2_dev,
                    cosb_og_dev,
                    w0_og_dev,
                    tau_og_dev,
                    dtau_og_dev,
                    tau_dev,
                    dtau_dev,
                    lambda_dev + w * nlayer,
                    gama_dev + w * nlayer,
                    F0PI_dev,
                    u0,
                    u1,
                    cos_theta,
                    single_phase,
                    multi_phase,
                    frac_a,
                    frac_b,
                    frac_c,
                    constant_back,
                    constant_forward,
                    toon_coefficients,
                    &g3_next,
                    &a_minus_next,
                    &a_plus_next,
                    &c_minus_up_next,
                    &c_plus_up_next,
                    &c_minus_down_next,
                    &c_plus_down_next,
                    &exptrm_next,
                    &exptrm_positive_next,
                    &exptrm_minus_next);

                const double e1 = exptrm_positive + gama_here * exptrm_minus;
                const double e2 = exptrm_positive - gama_here * exptrm_minus;
                const double e3 = gama_here * exptrm_positive + exptrm_minus;
                const double e4 = gama_here * exptrm_positive - exptrm_minus;

                const int row1 = 2 * layer + 1;
                A[row1] = (e1 + e3) * (gama_next - 1.0);
                B[row1] = (e2 + e4) * (gama_next - 1.0);
                C[row1] = 2.0 * (1.0 - gama_next * gama_next);
                D[row1] = (gama_next - 1.0) * (c_plus_up_next - c_plus_down) +
                          (1.0 - gama_next) * (c_minus_down - c_minus_up_next);

                const int row2 = 2 * layer + 2;
                A[row2] = 2.0 * (1.0 - gama_here * gama_here);
                B[row2] = (e1 - e3) * (gama_next + 1.0);
                C[row2] = (e1 + e3) * (gama_next - 1.0);
                D[row2] = e3 * (c_plus_up_next - c_plus_down) + e1 * (c_minus_down - c_minus_up_next);

                (void)lambda_next;
                (void)g3_next;
                (void)a_minus_next;
                (void)a_plus_next;
                (void)c_plus_down_next;
                (void)exptrm_next;
                (void)exptrm_positive_next;
                (void)exptrm_minus_next;
            }

            if (layer == nlayer - 1) {
                const double e1 = exptrm_positive + gama_here * exptrm_minus;
                const double e2 = exptrm_positive - gama_here * exptrm_minus;
                const double e3 = gama_here * exptrm_positive + exptrm_minus;
                const double e4 = gama_here * exptrm_positive - exptrm_minus;
                const double b_surface = surf_reflect_w * u0 * f0pi_w * exp(-tau_dev[layer_w_idx(nlevel - 1, w, nwno)] * inv_u0);
                c_minus_down_last = c_minus_down;
                c_plus_down_last = c_plus_down;

                A[tri_size - 1] = e1 - surf_reflect_w * e3;
                B[tri_size - 1] = e2 - surf_reflect_w * e4;
                C[tri_size - 1] = 0.0;
                D[tri_size - 1] = b_surface - c_plus_down + surf_reflect_w * c_minus_down;
            }
        }

        solve_tridiagonal_inplace(tri_size, A, B, C, D);

        const int last = nlayer - 1;
        const int last_base = 2 * last;
        const double last_pos = D[last_base] + D[last_base + 1];
        const double last_neg = D[last_base] - D[last_base + 1];
        const double lambda_last = lambda_dev[w * nlayer + last];
        const double gama_last = gama_dev[w * nlayer + last];
        const double exptrm_last = lambda_last * dtau_dev[layer_w_idx(last, w, nwno)];
        const double exptrm_last_clip = exptrm_last > 35.0 ? 35.0 : exptrm_last;
        const double exptrm_positive_last = exp(exptrm_last_clip);
        const double exptrm_minus_last = 1.0 / exptrm_positive_last;

        if (get_toa_intensity) {
            double flux_zero = last_pos * exptrm_positive_last +
                gama_last * last_neg * exptrm_minus_last +
                c_plus_down_last;
            double xint = flux_zero * inv_pi;

            for (int layer = last; layer >= 0; --layer) {
                const int idx = layer_w_idx(layer, w, nwno);
                const double lambda_here = lambda_dev[w * nlayer + layer];
                const double gama_here = gama_dev[w * nlayer + layer];
                const double dtau_here = dtau_dev[idx];
                const double tau_here = tau_dev[idx];
                const double tau_og_here = tau_og_dev[idx];
                const double dtau_og_here = dtau_og_dev[idx];
                const double w0_og_here = w0_og_dev[idx];
                const double c_og = cosb_og_dev[idx];
                const double w0_here = w0_dev[idx];

                double g3;
                double a_minus;
                double a_plus;
                double c_minus_up;
                double c_plus_up;
                double c_minus_down;
                double c_plus_down;
                double exptrm;
                double exptrm_positive;
                double exptrm_minus;
                compute_phase_terms(
                    layer,
                    w,
                    nwno,
                    w0_dev,
                    cosb_dev,
                    ftau_cld_dev,
                    ftau_ray_dev,
                    gcos2_dev,
                    cosb_og_dev,
                    w0_og_dev,
                    tau_og_dev,
                    dtau_og_dev,
                    tau_dev,
                    dtau_dev,
                    lambda_dev + w * nlayer,
                    gama_dev + w * nlayer,
                    F0PI_dev,
                    u0,
                    u1,
                    cos_theta,
                    single_phase,
                    multi_phase,
                    frac_a,
                    frac_b,
                    frac_c,
                    constant_back,
                    constant_forward,
                    toon_coefficients,
                    &g3,
                    &a_minus,
                    &a_plus,
                    &c_minus_up,
                    &c_plus_up,
                    &c_minus_down,
                    &c_plus_down,
                    &exptrm,
                    &exptrm_positive,
                    &exptrm_minus);

                const int base = 2 * layer;
                const double positive = D[base] + D[base + 1];
                const double negative = D[base] - D[base + 1];
                const double ubar2 = 0.767;
                const double phase_term = 3.0 * ubar2 * ubar2 * u1 * u1 - 1.0;
                double multi_plus = 0.0;
                double multi_minus = 0.0;

                if (multi_phase == 0) {
                    multi_plus = 1.0 + 1.5 * ftau_cld_dev[idx] * cosb_dev[idx] * u1 + gcos2_dev[idx] * phase_term / 2.0;
                    multi_minus = 1.0 - 1.5 * ftau_cld_dev[idx] * cosb_dev[idx] * u1 + gcos2_dev[idx] * phase_term / 2.0;
                } else {
                    multi_plus = 1.0 + 1.5 * ftau_cld_dev[idx] * cosb_dev[idx] * u1;
                    multi_minus = 1.0 - 1.5 * ftau_cld_dev[idx] * cosb_dev[idx] * u1;
                }

                const double geom_A = (multi_plus * c_plus_up + multi_minus * c_minus_up) * w0_here * 0.5 / PI;
                const double geom_G = positive * (multi_plus + gama_here * multi_minus) * w0_here * 0.5 / PI;
                const double geom_H = negative * (gama_here * multi_plus + multi_minus) * w0_here * 0.5 / PI;
                const double direct_term = (w0_og_here * f0pi_w / (4.0 * PI)) *
                    compute_single_phase_source(
                        layer,
                        w,
                        nwno,
                        ftau_cld_dev,
                        ftau_ray_dev,
                        gcos2_dev,
                        cosb_og_dev,
                        w0_og_dev,
                        cos_theta,
                        single_phase,
                        frac_a,
                        frac_b,
                        frac_c,
                        constant_back,
                        constant_forward) *
                    exp(-tau_og_here * inv_u0) *
                    (1.0 - exp(-dtau_og_here * sum_u * inv_u0u1)) *
                    u0_over_sum;
                const double source_term = geom_A * (1.0 - exp(-dtau_here * sum_u * inv_u0u1)) * u0_over_sum +
                    geom_G * (exp(exptrm - dtau_here * inv_u1) - 1.0) / (lambda_here * u1 - 1.0) +
                    geom_H * (1.0 - exp(-(exptrm + dtau_here * inv_u1))) / (lambda_here * u1 + 1.0);

                xint = xint * exp(-dtau_here * inv_u1) + direct_term + source_term;

                if (get_lvl_flux) {
                    const int out_base = (ang * nlevel + layer) * nwno + w;
                    flux_minus_all_dev[out_base] = positive * gama_here + negative + c_minus_up +
                        u0_scale * exp(-tau_here * inv_u0);
                    flux_plus_all_dev[out_base] = positive + gama_here * negative + c_plus_up;

                    const double exptrm_mid = exp(0.5 * exptrm);
                    const double exptrm_mid_minus = 1.0 / exptrm_mid;
                    const double taumid = tau_here + 0.5 * dtau_here;
                    const double c_plus_mid = a_plus * exp(-taumid * inv_u0);
                    const double c_minus_mid = a_minus * exp(-taumid * inv_u0);
                    flux_minus_midpt_all_dev[out_base] = gama_here * positive * exptrm_mid +
                        negative * exptrm_mid_minus +
                        c_minus_mid +
                        u0_scale * exp(-taumid * inv_u0);
                    flux_plus_midpt_all_dev[out_base] = positive * exptrm_mid +
                        gama_here * negative * exptrm_mid_minus +
                        c_plus_mid;
                }
            }

            if (get_lvl_flux) {
                const int out_base = (ang * nlevel + last) * nwno + w;
                flux_minus_all_dev[out_base] = gama_last * last_pos * exptrm_positive_last +
                    last_neg * exptrm_minus_last +
                    c_minus_down_last;
                flux_plus_all_dev[out_base] = last_pos * exptrm_positive_last +
                    gama_last * last_neg * exptrm_minus_last +
                    c_plus_down_last;
                flux_minus_midpt_all_dev[out_base] = 0.0;
                flux_plus_midpt_all_dev[out_base] = 0.0;
            }

            xint_at_top_dev[ang * nwno + w] = xint;
        }

        (void)wno_dev;
    }
}
