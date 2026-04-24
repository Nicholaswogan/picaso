from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional GPU dependency
    cp = None


__all__ = ["ReflectedLightGPUContext", "get_reflected_1d"]


def _require_cupy():
    if cp is None:
        raise RuntimeError("CuPy is required for the reflected-light GPU solver.")


def _as_device_array(arr, name, ndim=None):
    _require_cupy()
    out = cp.ascontiguousarray(cp.asarray(arr, dtype=cp.float64))
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {out.ndim}.")
    return out


class ReflectedLightGPUContext:
    """Persistent GPU reflected-light solver state.

    The context mirrors the no-allocation CPU solver structure:
    it owns reusable work buffers, stores device inputs, and executes the
    reflected-light solve without rebuilding temporary arrays on every call.
    """

    def __init__(
        self,
        nlevel: int,
        nwno: int,
        numg: int,
        numt: int,
        get_lvl_flux: int = 0,
        get_toa_intensity: int = 1,
        source_path: str | None = None,
        compile_options: tuple[str, ...] | None = None,
    ):
        _require_cupy()

        self.nlevel = int(nlevel)
        self.nlayer = self.nlevel - 1
        self.nwno = int(nwno)
        self.numg = int(numg)
        self.numt = int(numt)
        self.nang = self.numg * self.numt
        self.get_lvl_flux = int(get_lvl_flux)
        self.get_toa_intensity = int(get_toa_intensity)
        self._source_path = (
            Path(source_path)
            if source_path is not None
            else Path(__file__).resolve().with_name("gpu") / "reflected_1d_noalloc.cu"
        )
        self._compile_options = compile_options or ("--std=c++14",)
        self._module = None
        self._prepare_kernel = None
        self._solve_kernel = None

        self._allocate_results()
        self._allocate_workspace()
        self._compile()

        self._inputs_ready = False

    def _compile(self):
        code = self._source_path.read_text()
        self._module = cp.RawModule(code=code, options=self._compile_options)
        self._prepare_kernel = self._module.get_function("reflected_prepare_constants_kernel")
        self._solve_kernel = self._module.get_function("reflected_solve_kernel")

    def _allocate_results(self):
        if self.get_toa_intensity:
            self.xint_at_top = cp.empty((self.nang, self.nwno), dtype=cp.float64)
        else:
            self.xint_at_top = cp.empty((0, 0), dtype=cp.float64)

        if self.get_lvl_flux:
            self.flux_minus_all = cp.empty((self.nang, self.nlevel, self.nwno), dtype=cp.float64)
            self.flux_plus_all = cp.empty((self.nang, self.nlevel, self.nwno), dtype=cp.float64)
            self.flux_minus_midpt_all = cp.empty((self.nang, self.nlevel, self.nwno), dtype=cp.float64)
            self.flux_plus_midpt_all = cp.empty((self.nang, self.nlevel, self.nwno), dtype=cp.float64)
        else:
            self.flux_minus_all = cp.empty((0, 0, 0), dtype=cp.float64)
            self.flux_plus_all = cp.empty((0, 0, 0), dtype=cp.float64)
            self.flux_minus_midpt_all = cp.empty((0, 0, 0), dtype=cp.float64)
            self.flux_plus_midpt_all = cp.empty((0, 0, 0), dtype=cp.float64)

    def _allocate_workspace(self):
        self.lambda_dev = cp.empty((self.nwno, self.nlayer), dtype=cp.float64)
        self.gama_dev = cp.empty((self.nwno, self.nlayer), dtype=cp.float64)
        self.A_dev = cp.empty((self.nwno, 2 * self.nlayer), dtype=cp.float64)
        self.B_dev = cp.empty((self.nwno, 2 * self.nlayer), dtype=cp.float64)
        self.C_dev = cp.empty((self.nwno, 2 * self.nlayer), dtype=cp.float64)
        self.D_dev = cp.empty((self.nwno, 2 * self.nlayer), dtype=cp.float64)

    def _solve_block_size(self):
        if self.nang < 1:
            raise ValueError("numg * numt must be at least 1.")
        if self.nang > 1024:
            raise ValueError(
                f"GPU reflected solver currently supports at most 1024 angles per call, got {self.nang}."
            )
        return (self.nang,)

    def ensure(self, nlevel, nwno, numg, numt, get_lvl_flux, get_toa_intensity):
        changed = (
            int(nlevel) != self.nlevel
            or int(nwno) != self.nwno
            or int(numg) != self.numg
            or int(numt) != self.numt
            or int(get_lvl_flux) != self.get_lvl_flux
            or int(get_toa_intensity) != self.get_toa_intensity
        )
        if not changed:
            return

        self.nlevel = int(nlevel)
        self.nlayer = self.nlevel - 1
        self.nwno = int(nwno)
        self.numg = int(numg)
        self.numt = int(numt)
        self.nang = self.numg * self.numt
        self.get_lvl_flux = int(get_lvl_flux)
        self.get_toa_intensity = int(get_toa_intensity)
        self._allocate_results()
        self._allocate_workspace()

    def set_inputs(
        self,
        wno,
        dtau,
        tau,
        w0,
        cosb,
        gcos2,
        ftau_cld,
        ftau_ray,
        dtau_og,
        tau_og,
        w0_og,
        cosb_og,
        surf_reflect,
        ubar0,
        ubar1,
        cos_theta,
        F0PI,
        single_phase,
        multi_phase,
        frac_a,
        frac_b,
        frac_c,
        constant_back,
        constant_forward,
        toon_coefficients,
        b_top,
    ):
        self.wno = _as_device_array(wno, "wno", ndim=1)
        self.dtau = _as_device_array(dtau, "dtau", ndim=2)
        self.tau = _as_device_array(tau, "tau", ndim=2)
        self.w0 = _as_device_array(w0, "w0", ndim=2)
        self.cosb = _as_device_array(cosb, "cosb", ndim=2)
        self.gcos2 = _as_device_array(gcos2, "gcos2", ndim=2)
        self.ftau_cld = _as_device_array(ftau_cld, "ftau_cld", ndim=2)
        self.ftau_ray = _as_device_array(ftau_ray, "ftau_ray", ndim=2)
        self.dtau_og = _as_device_array(dtau_og, "dtau_og", ndim=2)
        self.tau_og = _as_device_array(tau_og, "tau_og", ndim=2)
        self.w0_og = _as_device_array(w0_og, "w0_og", ndim=2)
        self.cosb_og = _as_device_array(cosb_og, "cosb_og", ndim=2)
        self.surf_reflect = _as_device_array(surf_reflect, "surf_reflect", ndim=1)
        self.ubar0 = _as_device_array(ubar0, "ubar0").reshape(-1)
        self.ubar1 = _as_device_array(ubar1, "ubar1").reshape(-1)
        self.F0PI = _as_device_array(F0PI, "F0PI", ndim=1)

        if self.wno.size != self.nwno:
            raise ValueError(f"wno length mismatch: expected {self.nwno}, got {self.wno.size}.")
        if self.ubar0.size != self.nang or self.ubar1.size != self.nang:
            raise ValueError(f"ubar arrays must flatten to {self.nang} elements.")

        self.cos_theta = float(cos_theta)
        self.single_phase = int(single_phase)
        self.multi_phase = int(multi_phase)
        self.frac_a = float(frac_a)
        self.frac_b = float(frac_b)
        self.frac_c = float(frac_c)
        self.constant_back = float(constant_back)
        self.constant_forward = float(constant_forward)
        self.toon_coefficients = int(toon_coefficients)
        self.b_top = float(b_top)
        self._inputs_ready = True

    def run(self, return_host: bool = False):
        if not self._inputs_ready:
            raise RuntimeError("set_inputs() must be called before run().")

        prepare_block = (256,)
        prepare_grid = ((self.nwno + prepare_block[0] - 1) // prepare_block[0],)
        solve_block = self._solve_block_size()
        solve_grid = (self.nwno,)

        self._prepare_kernel(
            prepare_grid,
            prepare_block,
            (
                self.w0,
                self.ftau_cld,
                self.cosb,
                np.int32(self.nlayer),
                np.int32(self.nwno),
                np.int32(self.toon_coefficients),
                self.lambda_dev,
                self.gama_dev,
            ),
        )

        self._solve_kernel(
            solve_grid,
            solve_block,
            (
                np.int32(self.nlevel),
                np.int32(self.nlayer),
                np.int32(self.nwno),
                np.int32(self.nang),
                self.wno,
                self.dtau,
                self.tau,
                self.w0,
                self.cosb,
                self.gcos2,
                self.ftau_cld,
                self.ftau_ray,
                self.dtau_og,
                self.tau_og,
                self.w0_og,
                self.cosb_og,
                self.surf_reflect,
                self.ubar0,
                self.ubar1,
                self.cos_theta,
                self.F0PI,
                np.int32(self.single_phase),
                np.int32(self.multi_phase),
                float(self.frac_a),
                float(self.frac_b),
                float(self.frac_c),
                float(self.constant_back),
                float(self.constant_forward),
                np.int32(self.get_toa_intensity),
                np.int32(self.get_lvl_flux),
                np.int32(self.toon_coefficients),
                float(self.b_top),
                self.lambda_dev,
                self.gama_dev,
                self.A_dev,
                self.B_dev,
                self.C_dev,
                self.D_dev,
                self.xint_at_top,
                self.flux_minus_all,
                self.flux_plus_all,
                self.flux_minus_midpt_all,
                self.flux_plus_midpt_all,
            ),
        )

        if return_host:
            if self.get_toa_intensity:
                xint = cp.asnumpy(self.xint_at_top).reshape(self.numg, self.numt, self.nwno)
            else:
                xint = cp.asnumpy(self.xint_at_top)

            if self.get_lvl_flux:
                fluxes = (
                    cp.asnumpy(self.flux_minus_all).reshape(self.numg, self.numt, self.nlevel, self.nwno),
                    cp.asnumpy(self.flux_plus_all).reshape(self.numg, self.numt, self.nlevel, self.nwno),
                    cp.asnumpy(self.flux_minus_midpt_all).reshape(self.numg, self.numt, self.nlevel, self.nwno),
                    cp.asnumpy(self.flux_plus_midpt_all).reshape(self.numg, self.numt, self.nlevel, self.nwno),
                )
            else:
                fluxes = (
                    cp.asnumpy(self.flux_minus_all),
                    cp.asnumpy(self.flux_plus_all),
                    cp.asnumpy(self.flux_minus_midpt_all),
                    cp.asnumpy(self.flux_plus_midpt_all),
                )
            return xint, fluxes

        if self.get_toa_intensity:
            xint = self.xint_at_top.reshape(self.numg, self.numt, self.nwno)
        else:
            xint = self.xint_at_top

        if self.get_lvl_flux:
            fluxes = (
                self.flux_minus_all.reshape(self.numg, self.numt, self.nlevel, self.nwno),
                self.flux_plus_all.reshape(self.numg, self.numt, self.nlevel, self.nwno),
                self.flux_minus_midpt_all.reshape(self.numg, self.numt, self.nlevel, self.nwno),
                self.flux_plus_midpt_all.reshape(self.numg, self.numt, self.nlevel, self.nwno),
            )
        else:
            fluxes = (
                self.flux_minus_all,
                self.flux_plus_all,
                self.flux_minus_midpt_all,
                self.flux_plus_midpt_all,
            )

        return xint, fluxes

    def free(self):
        for name in (
            "wno",
            "dtau",
            "tau",
            "w0",
            "cosb",
            "gcos2",
            "ftau_cld",
            "ftau_ray",
            "dtau_og",
            "tau_og",
            "w0_og",
            "cosb_og",
            "surf_reflect",
            "ubar0",
            "ubar1",
            "F0PI",
            "lambda_dev",
            "gama_dev",
            "A_dev",
            "B_dev",
            "C_dev",
            "D_dev",
            "xint_at_top",
            "flux_minus_all",
            "flux_plus_all",
            "flux_minus_midpt_all",
            "flux_plus_midpt_all",
        ):
            if hasattr(self, name):
                setattr(self, name, None)


def get_reflected_1d(
    nlevel,
    wno,
    nwno,
    numg,
    numt,
    dtau,
    tau,
    w0,
    cosb,
    gcos2,
    ftau_cld,
    ftau_ray,
    dtau_og,
    tau_og,
    w0_og,
    cosb_og,
    surf_reflect,
    ubar0,
    ubar1,
    cos_theta,
    F0PI,
    single_phase,
    multi_phase,
    frac_a,
    frac_b,
    frac_c,
    constant_back,
    constant_forward,
    get_toa_intensity=1,
    get_lvl_flux=0,
    toon_coefficients=0,
    b_top=0.0,
    return_host=False,
):
    """Convenience wrapper mirroring the CPU no-alloc reflected API."""

    ctx = ReflectedLightGPUContext(
        nlevel,
        nwno,
        numg,
        numt,
        get_lvl_flux,
        get_toa_intensity,
    )
    ctx.set_inputs(
        wno,
        dtau,
        tau,
        w0,
        cosb,
        gcos2,
        ftau_cld,
        ftau_ray,
        dtau_og,
        tau_og,
        w0_og,
        cosb_og,
        surf_reflect,
        ubar0,
        ubar1,
        cos_theta,
        F0PI,
        single_phase,
        multi_phase,
        frac_a,
        frac_b,
        frac_c,
        constant_back,
        constant_forward,
        toon_coefficients,
        b_top,
    )
    return ctx.run(return_host=return_host)
