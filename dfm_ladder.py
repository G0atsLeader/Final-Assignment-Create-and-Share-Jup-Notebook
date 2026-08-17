"""
DFM ladder estimation spine (Stage 1).

Custom single-factor state-space model with:
  * GT loading fixed to 1.0  (factor scale = "GT-equivalent std units"; sign +GT)
  * factor dynamics: local_level | llt_damped | ar1 | ar2
  * idiosyncratic:   white       | ar1
  * native Kalman missing-data handling (NaNs allowed in endog)

State vector = [ factor block ... , idiosyncratic block ... ].
Observation i:  y_it = lambda_i * (loaded factor component) + e_it
  - white idiosyncratic: e_it ~ N(0, sigma2_i)   -> lives in obs_cov H
  - ar1   idiosyncratic: e_it = rho_i e_{i,t-1} + u_it, u ~ N(0,sigma2_ui)
                                                   -> own state, H = 0
"""
import numpy as np
import warnings
from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tools.sm_exceptions import ConvergenceWarning


FACTOR_KINDS = {"local_level": 1, "llt_damped": 2, "ar1": 1, "ar2": 2}


class LadderDFM(MLEModel):
    def __init__(self, endog, factor="ar1", idio="white"):
        endog = np.asarray(endog, float)
        if endog.ndim == 1:
            endog = endog[:, None]
        self.k_series = endog.shape[1]
        self.factor = factor
        self.idio = idio
        self.kf = FACTOR_KINDS[factor]                       # factor state dim
        self.ki = self.k_series if idio == "ar1" else 0      # idio state dim
        k_states = self.kf + self.ki
        k_posdef = self.kf + self.ki                         # every state has an innovation
        super().__init__(endog, k_states=k_states, k_posdef=k_posdef,
                         initialization="approximate_diffuse")
        self.ssm.filter_univariate = True                    # robust with H possibly singular

        # --- fixed parts of the design (loadings filled in update) ---
        # factor is loaded through its FIRST state component (index 0)
        # idiosyncratic states (if any) start at index kf, each loads 1 on its series
        if self.idio == "ar1":
            for i in range(self.k_series):
                self["design", i, self.kf + i] = 1.0
        # selection = identity (each state has its own innovation)
        self["selection"] = np.eye(k_states)[:, :k_posdef]

    # ---------------------------------------------------------------- params
    @property
    def param_names(self):
        names = [f"lambda_{i+1}" for i in range(1, self.k_series)]   # lambda_1 fixed = 1
        if self.factor == "local_level":
            names += ["sigma2_level"]
        elif self.factor == "llt_damped":
            names += ["phi_trend", "sigma2_level", "sigma2_slope"]
        elif self.factor == "ar1":
            names += ["phi_f", "sigma2_f"]
        elif self.factor == "ar2":
            names += ["phi_f1", "phi_f2", "sigma2_f"]
        if self.idio == "white":
            names += [f"sigma2_eps_{i+1}" for i in range(self.k_series)]
        else:  # ar1
            for i in range(self.k_series):
                names += [f"rho_{i+1}", f"sigma2_u_{i+1}"]
        return names

    @property
    def start_params(self):
        v = np.nanvar(self.endog, axis=0).ravel()
        p = list(np.ones(self.k_series - 1))                 # loadings
        if self.factor == "local_level":
            p += [0.3 * v[0]]
        elif self.factor == "llt_damped":
            p += [0.9, 0.2 * v[0], 0.01 * v[0]]
        elif self.factor == "ar1":
            p += [0.7, 0.3 * v[0]]
        elif self.factor == "ar2":
            p += [0.5, 0.1, 0.3 * v[0]]
        if self.idio == "white":
            p += list(0.5 * v)
        else:
            for i in range(self.k_series):
                p += [0.3, 0.5 * v[i]]
        return np.array(p, float)

    # variances must be > 0; AR coeffs kept in (-1,1) via tanh
    def transform_params(self, u):
        u = np.array(u, float); c = u.copy()
        nm = self.param_names
        for j, name in enumerate(nm):
            if name.startswith("sigma2"):
                c[j] = u[j] ** 2
            elif name.startswith(("phi", "rho")):
                c[j] = np.tanh(u[j])
        return c

    def untransform_params(self, c):
        c = np.array(c, float); u = c.copy()
        nm = self.param_names
        for j, name in enumerate(nm):
            if name.startswith("sigma2"):
                u[j] = np.sqrt(np.abs(c[j]))
            elif name.startswith(("phi", "rho")):
                u[j] = np.arctanh(np.clip(c[j], -0.999, 0.999))
        return u

    # ---------------------------------------------------------------- update
    def update(self, params, **kw):
        params = super().update(params, **kw)
        nm = self.param_names
        g = {name: params[i] for i, name in enumerate(nm)}
        kf, ki, kser = self.kf, self.ki, self.k_series

        # loadings on factor's first component
        self["design", 0, 0] = 1.0
        for i in range(1, kser):
            self["design", i, 0] = g[f"lambda_{i+1}"]

        # transition + state cov for the factor block
        # dtype follows params so complex-step differentiation (used for SEs) is preserved
        T = np.zeros((self.k_states, self.k_states), dtype=params.dtype)
        Q = np.zeros((self.k_states, self.k_states), dtype=params.dtype)
        if self.factor == "local_level":
            T[0, 0] = 1.0
            Q[0, 0] = g["sigma2_level"]
        elif self.factor == "llt_damped":
            T[0, 0] = 1.0; T[0, 1] = 1.0; T[1, 1] = g["phi_trend"]
            Q[0, 0] = g["sigma2_level"]; Q[1, 1] = g["sigma2_slope"]
        elif self.factor == "ar1":
            T[0, 0] = g["phi_f"]
            Q[0, 0] = g["sigma2_f"]
        elif self.factor == "ar2":
            T[0, 0] = g["phi_f1"]; T[0, 1] = g["phi_f2"]; T[1, 0] = 1.0
            Q[0, 0] = g["sigma2_f"]

        # idiosyncratic block
        if self.idio == "white":
            H = np.diag([g[f"sigma2_eps_{i+1}"] for i in range(kser)])
            self["obs_cov"] = H
        else:  # ar1 idiosyncratic states
            for i in range(kser):
                T[kf + i, kf + i] = g[f"rho_{i+1}"]
                Q[kf + i, kf + i] = g[f"sigma2_u_{i+1}"]
            self["obs_cov"] = np.zeros((kser, kser), dtype=params.dtype)

        self["transition"] = T
        self["state_cov"] = Q


# ==================================================================== fitting
def fit_multistart(endog, factor, idio, n_starts=3, seed=0, maxiter=1000):
    """Fit from several dispersed starts; return (best_result, agreement_report)."""
    rng = np.random.default_rng(seed)
    mod = LadderDFM(endog, factor=factor, idio=idio)
    base = mod.start_params
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for s in range(n_starts):
            sp = base if s == 0 else base * np.exp(rng.normal(0, 0.5, size=base.shape))
            try:
                r = mod.fit(sp, disp=False, method="lbfgs", maxiter=maxiter)
                r = mod.fit(r.params, disp=False, method="nm", maxiter=2 * maxiter)
                if np.isfinite(r.llf):
                    results.append(r)
            except Exception:
                pass
    if not results:
        raise RuntimeError(f"all starts failed for factor={factor}, idio={idio}")
    best = max(results, key=lambda r: r.llf)
    llfs = np.array([r.llf for r in results])
    agree = int(np.sum(np.abs(llfs - best.llf) < 1e-2))
    report = {"n_starts": len(results), "n_agree": agree,
              "llf_spread": float(llfs.max() - llfs.min()),
              "converged": bool(best.mle_retvals.get("converged", False))}
    best._agree = report
    return best, report


# ============================================================= smoke test
if __name__ == "__main__":
    import pandas as pd
    rng = np.random.default_rng(1)
    T = 112
    # planted AR(1) factor + AR(1) idiosyncratic
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = 0.85 * f[t-1] + rng.normal(0, 0.5)
    def series(lam, rho, s):
        e = np.zeros(T)
        for t in range(1, T):
            e[t] = rho * e[t-1] + rng.normal(0, s)
        return lam * f + e
    GT   = series(1.0, 0.3, 0.4)
    shar = series(0.8, 0.4, 0.5)
    tone = series(0.2, 0.5, 0.9)          # deliberately weak
    Z = np.c_[GT, shar, tone]
    Z = (Z - Z.mean(0)) / Z.std(0)        # standardize (levels)

    specs = {
        "M0": (Z[:, :1], "local_level", "white"),
        "M1": (Z[:, :1], "llt_damped",  "white"),
        "M2": (Z[:, :2], "ar1",         "white"),
        "M3": (Z[:, :2], "ar1",         "ar1"),
        "M4": (Z[:, :3], "ar1",         "ar1"),
        "M5": (Z[:, :3], "ar2",         "ar1"),
        "M6": (np.diff(Z, axis=0)[:, :3], "ar1", "ar1"),   # first differences
    }
    print(f"{'spec':4} {'conv':5} {'agree':6} {'llf':>9} {'aic':>9} {'seOK':5}  key params")
    for name, (e, fac, idi) in specs.items():
        res, rep = fit_multistart(e, fac, idi, n_starts=3)
        nm = res.model.param_names
        se_ok = bool(np.all(np.isfinite(res.bse)))
        show = {k: round(float(res.params[nm.index(k)]), 3)
                for k in nm if k.startswith("lambda") or k in ("phi_f", "phi_trend")}
        print(f"{name:4} {str(rep['converged']):5} "
              f"{rep['n_agree']}/{rep['n_starts']:<3}  {res.llf:9.2f} {res.aic:9.2f} {str(se_ok):5}  {show}")
