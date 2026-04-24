from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import numba as nb
import pytest

# Get the root of the repo and prepend to path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

cp = pytest.importorskip("cupy")

from picaso import fluxes_noalloc
from picaso.fluxes_reflected_gpu import ReflectedLightGPUContext


def _make_reflected_case(nlevel=60, nwno=100_000, numg=10, numt=1):
    nlayer = nlevel - 1

    wno = np.linspace(1.0, 5_000.0, nwno, dtype=np.float64)
    w_scale = np.linspace(0.0, 1.0, nwno, dtype=np.float64)[None, :]
    layer_scale = np.linspace(0.2, 0.8, nlayer, dtype=np.float64)[:, None]

    dtau = 0.010 + 0.004 * layer_scale + 0.0002 * w_scale
    tau = np.vstack([np.zeros((1, nwno), dtype=np.float64), np.cumsum(dtau, axis=0)])

    w0 = 0.55 + 0.08 * layer_scale + 0.01 * w_scale
    cosb = 0.12 + 0.04 * layer_scale + 0.003 * w_scale
    gcos2 = 0.02 + 0.01 * layer_scale + 0.001 * w_scale
    ftau_cld = 0.35 + 0.05 * layer_scale + 0.002 * w_scale
    ftau_ray = 0.15 + 0.02 * layer_scale + 0.001 * w_scale

    dtau_og = 0.80 * dtau
    tau_og = np.vstack([np.zeros((1, nwno), dtype=np.float64), np.cumsum(dtau_og, axis=0)])
    w0_og = 0.95 * w0
    cosb_og = 0.90 * cosb

    surf_reflect = np.full(nwno, 0.25, dtype=np.float64)
    ubar0 = np.linspace(0.18, 0.72, numg, dtype=np.float64).reshape(numg, 1)
    ubar1 = np.linspace(0.28, 0.82, numg, dtype=np.float64).reshape(numg, 1)
    F0PI = np.linspace(1.0, 1.2, nwno, dtype=np.float64)

    return dict(
        nlevel=nlevel,
        nwno=nwno,
        numg=numg,
        numt=numt,
        wno=wno,
        dtau=dtau,
        tau=tau,
        w0=w0,
        cosb=cosb,
        gcos2=gcos2,
        ftau_cld=ftau_cld,
        ftau_ray=ftau_ray,
        dtau_og=dtau_og,
        tau_og=tau_og,
        w0_og=w0_og,
        cosb_og=cosb_og,
        surf_reflect=surf_reflect,
        ubar0=ubar0,
        ubar1=ubar1,
        cos_theta=0.37,
        F0PI=F0PI,
        single_phase=3,
        multi_phase=0,
        frac_a=0.17,
        frac_b=0.27,
        frac_c=1.3,
        constant_back=0.29,
        constant_forward=0.39,
        b_top=0.07,
    )


def _call_cpu_reflected(case):
    return fluxes_noalloc.get_reflected_1d(
        case["nlevel"],
        case["wno"].copy(),
        case["nwno"],
        case["numg"],
        case["numt"],
        case["dtau"].copy(),
        case["tau"].copy(),
        case["w0"].copy(),
        case["cosb"].copy(),
        case["gcos2"].copy(),
        case["ftau_cld"].copy(),
        case["ftau_ray"].copy(),
        case["dtau_og"].copy(),
        case["tau_og"].copy(),
        case["w0_og"].copy(),
        case["cosb_og"].copy(),
        case["surf_reflect"].copy(),
        case["ubar0"].copy(),
        case["ubar1"].copy(),
        case["cos_theta"],
        case["F0PI"].copy(),
        case["single_phase"],
        case["multi_phase"],
        case["frac_a"],
        case["frac_b"],
        case["frac_c"],
        case["constant_back"],
        case["constant_forward"],
        1,
        0,
        0,
        case["b_top"],
    )


def _set_new_gpu_inputs(ctx, case):
    ctx.set_inputs(
        case["wno"],
        case["dtau"],
        case["tau"],
        case["w0"],
        case["cosb"],
        case["gcos2"],
        case["ftau_cld"],
        case["ftau_ray"],
        case["dtau_og"],
        case["tau_og"],
        case["w0_og"],
        case["cosb_og"],
        case["surf_reflect"],
        case["ubar0"],
        case["ubar1"],
        case["cos_theta"],
        case["F0PI"],
        case["single_phase"],
        case["multi_phase"],
        case["frac_a"],
        case["frac_b"],
        case["frac_c"],
        case["constant_back"],
        case["constant_forward"],
        0,
        case["b_top"],
    )


def _benchmark_gpu_against_cpu(case, dtype, rtol, atol):
    nb.set_num_threads(16)

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA device not available.")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    case = _make_reflected_case()

    # Warm up the CPU JIT with a tiny problem so the timed run is about solver
    # execution, not compilation.
    warm_case = _make_reflected_case(nlevel=4, nwno=8, numg=1, numt=1)
    _call_cpu_reflected(warm_case)

    cpu_t0 = time.perf_counter()
    cpu_xint, _ = _call_cpu_reflected(case)
    cpu_time = time.perf_counter() - cpu_t0

    gpu_ctx = ReflectedLightGPUContext(
        case["nlevel"],
        case["nwno"],
        case["numg"],
        case["numt"],
        get_lvl_flux=0,
        get_toa_intensity=1,
        dtype=dtype,
    )
    gpu_setup_t0 = time.perf_counter()
    _set_new_gpu_inputs(gpu_ctx, case)
    cp.cuda.Stream.null.synchronize()
    gpu_setup_time = time.perf_counter() - gpu_setup_t0

    # Warm up the GPU path once, then time the steady-state kernel execution.
    gpu_ctx.run(return_host=True)
    cp.cuda.Stream.null.synchronize()

    gpu_t0 = time.perf_counter()
    _set_new_gpu_inputs(gpu_ctx, case)
    gpu_xint, _ = gpu_ctx.run(return_host=True)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - gpu_t0
    gpu_total_time = gpu_time

    speedup = cpu_time / gpu_total_time if gpu_total_time > 0 else float("inf")
    max_abs = np.max(np.abs(cpu_xint - gpu_xint))
    max_rel = np.max(np.abs(cpu_xint - gpu_xint) / np.maximum(np.abs(cpu_xint), 1e-15))

    print(f"CPU reflected solve: {cpu_time:.3f}s")
    print(f"GPU reflected setup: {gpu_setup_time:.3f}s")
    print(f"GPU reflected end-to-end: {gpu_total_time:.3f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Max abs diff: {max_abs:.6e}")
    print(f"Max rel diff: {max_rel:.6e}")

    np.testing.assert_allclose(cpu_xint, gpu_xint, rtol=rtol, atol=atol)


def test_reflected_gpu_matches_cpu_and_reports_runtime():
    case = _make_reflected_case()
    _benchmark_gpu_against_cpu(case, np.float64, rtol=1e-6, atol=1e-8)


def test_reflected_gpu_float32_matches_cpu_and_reports_runtime():
    case = _make_reflected_case()
    _benchmark_gpu_against_cpu(case, np.float32, rtol=1e-4, atol=1e-5)
