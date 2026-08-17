"""
DFM Triage — do tone and MediaCloud share carry common-component information?

Screening stage. Four fits (T1-T4), single AR(1) factor, all three series,
GT loading fixed to 1.0. Reports both loadings jointly (estimate/SE/CI),
LR tests (tone=0, media=0, joint=0) built over the SAME 3-series data vector,
the covariance of the two loading estimates, Ljung-Box per series, and a
multi-start convergence check.

Depends on dfm_ladder.LadderDFM / fit_multistart.
Series order everywhere: [GT, media, tone]  ->  lambda_2 = media, lambda_3 = tone.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from dfm_ladder import LadderDFM, fit_multistart

COL = dict(gt="gtrends", count="mediacloud_AI_specific",
           share="mediacloud_AI_specific_share", tone="avg_tone")
FLOOR_END = "2025-04-01"


# ----------------------------------------------------------------- data prep
def build_Z(W, media_col, diff=False, subsample_from=None):
    """log(GT+1), log(media), tone levels -> optional subsample -> optional
    first difference -> standardise. Returns a demeaned/unit-var 3-col frame
    [GT, media, tone]."""
    df = pd.DataFrame(index=W.index)
    df["GT"]    = np.log(W[COL["gt"]] + 1.0)
    df["media"] = np.log(W[media_col].replace(0, np.nan))     # log(share) or log(count)
    df["tone"]  = W[COL["tone"]].astype(float)
    if subsample_from is not None:
        df = df.loc[df.index >= pd.Timestamp(subsample_from)]
    if diff:
        df = df.diff()
    df = df.dropna(how="all")
    return (df - df.mean()) / df.std()


# ------------------------------------------------------------- restricted fit
def fit_restricted(endog, idio, fix, maxiter=1500):
    """MLE with one or more loadings fixed (for the LR tests)."""
    mod = LadderDFM(endog, factor="ar1", idio=idio)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = mod.fit_constrained(fix, method="lbfgs", disp=False, maxiter=maxiter)
    return res


def lr_test(llf_full, llf_restricted, df):
    stat = 2.0 * (llf_full - llf_restricted)
    return float(stat), float(stats.chi2.sf(max(stat, 0.0), df))


# ------------------------------------------------------------------ one spec
def run_spec(spec_id, W, media_col, idio, diff=False, subsample_from=None):
    Z = build_Z(W, media_col, diff=diff, subsample_from=subsample_from)
    endog = Z.values
    full, rep = fit_multistart(endog, factor="ar1", idio=idio, n_starts=5)
    nm = full.model.param_names
    i_media, i_tone = nm.index("lambda_2"), nm.index("lambda_3")

    p, se = full.params, full.bse
    lam_media, lam_tone = float(p[i_media]), float(p[i_tone])
    se_media, se_tone = float(se[i_media]), float(se[i_tone])

    # joint covariance / correlation of the two loading estimates
    cov = np.asarray(full.cov_params())
    c_mt = cov[i_media, i_tone]
    denom = np.sqrt(cov[i_media, i_media] * cov[i_tone, i_tone])
    load_corr = float(c_mt / denom) if denom > 0 else np.nan

    # LR tests (all restricted models use the SAME 3-series data)
    r_tone  = fit_restricted(endog, idio, {"lambda_3": 0.0})
    r_media = fit_restricted(endog, idio, {"lambda_2": 0.0})
    r_both  = fit_restricted(endog, idio, {"lambda_2": 0.0, "lambda_3": 0.0})
    lr_tone  = lr_test(full.llf, r_tone.llf, 1)
    lr_media = lr_test(full.llf, r_media.llf, 1)
    lr_joint = lr_test(full.llf, r_both.llf, 2)

    # Ljung-Box on standardised one-step residuals, per series
    resid = pd.DataFrame(full.standardized_forecasts_error.T,
                         columns=["GT", "media", "tone"]).dropna()
    lb = {}
    for s in ["GT", "media", "tone"]:
        tab = acorr_ljungbox(resid[s], lags=[4, 8, 12], return_df=True)
        for L in (4, 8, 12):
            lb[f"lb_p_{s}_{L}"] = float(tab.loc[L, "lb_pvalue"])

    z = 1.959963985
    row = {
        "spec": spec_id, "media_col": media_col, "idio": idio,
        "diff": diff, "subsample_from": subsample_from or "",
        "n": int(np.isfinite(endog).any(axis=1).sum()),
        "converged": rep["converged"], "n_agree": f"{rep['n_agree']}/{rep['n_starts']}",
        "llf": float(full.llf), "aic": float(full.aic), "bic": float(full.bic),
        "lambda_media": lam_media, "se_media": se_media,
        "media_ci_lo": lam_media - z*se_media, "media_ci_hi": lam_media + z*se_media,
        "lambda_tone": lam_tone, "se_tone": se_tone,
        "tone_ci_lo": lam_tone - z*se_tone, "tone_ci_hi": lam_tone + z*se_tone,
        "loading_corr": load_corr,
        "LR_tone_stat": lr_tone[0], "LR_tone_p": lr_tone[1],
        "LR_media_stat": lr_media[0], "LR_media_p": lr_media[1],
        "LR_joint_stat": lr_joint[0], "LR_joint_p": lr_joint[1],
        # sign check: tone ~ -0.675 with GT in levels -> expect lambda_tone < 0
        "tone_sign_ok": bool(lam_tone < 0) if not diff else "n/a(diff)",
        **lb,
    }
    return row, full


# --------------------------------------------------------------- bug check 1.1
def assert_configs_differ(W):
    Zc = build_Z(W, COL["count"])["media"]
    Zs = build_Z(W, COL["share"])["media"]
    m = Zc.notna() & Zs.notna()
    identical = np.allclose(Zc[m], Zs[m])
    corr = float(np.corrcoef(Zc[m], Zs[m])[0, 1])
    assert not identical, "BUG (1.1): count and share configs receive identical data!"
    return {"identical": identical, "corr_count_share_after_transform": corr}


# --------------------------------------------------------------- denominator
def plot_denominator(W, fname="fig_mediacloud_denominator.png"):
    """Total weekly volume = AI count / AI share (share = count/total)."""
    share = W[COL["share"]].replace(0, np.nan)
    total = W[COL["count"]] / share
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(W.index, total, color="#37474F", lw=1.3)
    ax.plot(W.index, total.rolling(8, min_periods=1).mean(), color="#E64A19",
            lw=1.0, ls="--", label="8-wk mean")
    ax.set_title("MediaCloud denominator — total weekly article volume")
    ax.set_ylabel("total articles / week"); ax.set_xlabel("week")
    ax.grid(alpha=.3); ax.legend()
    plt.tight_layout(); plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
    return total


# ------------------------------------------------------------------- verdict
def _survives(res, series, specs=("T2", "T3", "T4")):
    """A loading survives triage if, across the robustness specs, LR p<0.05 AND
    the 95% CI excludes 0. T3 (differences) is the decisive co-trending gate."""
    lo, hi, pcol = f"{series}_ci_lo", f"{series}_ci_hi", f"LR_{series}_p"
    sub = res[res.spec.isin(specs)]
    ok = (sub[pcol] < 0.05) & ((sub[lo] > 0) | (sub[hi] < 0))
    t3 = res[res.spec == "T3"]
    t3_ok = bool(((t3[pcol] < 0.05) & ((t3[lo] > 0) | (t3[hi] < 0))).all())
    return bool(ok.all()), t3_ok


def generate_notes(res, bug11, fname="NOTES.md"):
    tone_ok, tone_t3 = _survives(res, "tone")
    media_ok, media_t3 = _survives(res, "media")
    L = []
    L.append("# DFM Triage — findings\n")
    L.append("## §1 Bug checks\n")
    L.append(f"- **§1.1 count vs share receive different data:** "
             f"`identical={bug11['identical']}` (assertion passed → not a bug). "
             f"Post-transform corr(count, share) = "
             f"{bug11['corr_count_share_after_transform']:.3f}. "
             f"{'High collinearity explains near-identical prior outputs.' if bug11['corr_count_share_after_transform']>0.9 else 'The two series are meaningfully different inputs.'}")
    L.append("- **§1.2 LR test construction:** restricted models fix `lambda_*=0` on the "
             "**same 3-series data** as the unrestricted fit (via `fit_constrained`); "
             "likelihoods are comparable. Valid.\n")
    L.append("## §3 T3 (first-difference) co-trending verdict\n")
    for s in ("tone", "media"):
        t3 = res[res.spec == "T3"].iloc[0]
        surv = t3[f"LR_{s}_p"] < 0.05 and (t3[f"{s}_ci_lo"] > 0 or t3[f"{s}_ci_hi"] < 0)
        L.append(f"- **{s}:** T3 λ={t3[f'lambda_{s}']:.3f} "
                 f"CI[{t3[f'{s}_ci_lo']:.3f}, {t3[f'{s}_ci_hi']:.3f}], LR p={t3[f'LR_{s}_p']:.3g} → "
                 f"{'survives differencing (association not purely trend)' if surv else 'does NOT survive differencing (association lives at the trend)'}")
    L.append("")
    L.append("## Decisions\n")
    L.append(f"- **Tone:** {'**survives triage**' if tone_ok else '**does not survive triage**'} "
             f"(LR + CI across T2/T3/T4; T3 gate {'passed' if tone_t3 else 'failed'}). "
             f"{'Carries common-component information.' if tone_ok else 'Cannot be modelled as a contemporaneous indicator; drop from the measurement-equation arm.'}")
    L.append(f"- **Media (share):** {'**survives triage**' if media_ok else '**does not survive triage**'} "
             f"(T3 gate {'passed' if media_t3 else 'failed'}). "
             "Note: media is a β driver regardless; a tight near-zero loading is positive "
             "evidence for a single-indicator observation structure, not a null.")
    L.append("\n*A DFM loading is undirected — surviving triage establishes only that there is "
             "common-component information to model, not where the series enters the mechanistic model.*\n")
    with open(fname, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return "\n".join(L)


# ------------------------------------------------------------------- driver
def run_triage(W, write=True):
    bug11 = assert_configs_differ(W)
    specs = [
        ("T1", COL["share"], "white", False, None),
        ("T2", COL["share"], "ar1",   False, None),
        ("T3", COL["share"], "ar1",   True,  None),
        ("T4", COL["share"], "ar1",   False, FLOOR_END),
        ("T2_raw", COL["count"], "ar1", False, None),   # §2 raw swap (confirmation)
    ]
    rows, fits = [], {}
    for sid, mcol, idio, diff, sub in specs:
        row, fit = run_spec(sid, W, mcol, idio, diff=diff, subsample_from=sub)
        rows.append(row); fits[sid] = fit
    res = pd.DataFrame(rows)
    if write:
        plot_denominator(W)
        res.to_csv("triage_results.csv", index=False)
        generate_notes(res, bug11)
    return res, fits, bug11


# --------------------------------------------------------------- smoke test
if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n = 112
    idx = pd.date_range("2024-06-02", periods=n, freq="W")
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = 0.9 * f[t-1] + rng.normal(0, 0.4)          # smooth factor
    def ar_noise(rho, s):
        e = np.zeros(n)
        for t in range(1, n):
            e[t] = rho * e[t-1] + rng.normal(0, s)
        return e
    total = 400 + np.linspace(0, 300, n) + rng.normal(0, 20, n)   # growing denominator
    ai_share = np.clip(0.02 + 0.01*(f - f.min())/(np.ptp(f)+1e-9) + ar_noise(0.3, 0.002), 1e-4, None)
    ai_count = np.round(ai_share * total)
    gt = np.clip(np.round(np.expm1(2 + 1.0*f + ar_noise(0.3, 0.3))), 0, 100)
    tone = 0.8 - 0.2*(f - f.mean()) + ar_noise(0.5, 0.05)         # tone NEG related to factor
    W = pd.DataFrame({COL["gt"]: gt, COL["count"]: ai_count,
                      COL["share"]: ai_share, COL["tone"]: tone}, index=idx)

    res, fits, bug11 = run_triage(W, write=False)
    print("bug-check 1.1:", bug11)
    cols = ["spec", "converged", "n_agree", "lambda_media", "lambda_tone",
            "loading_corr", "LR_tone_p", "LR_media_p", "LR_joint_p", "tone_sign_ok"]
    print(res[cols].round(3).to_string(index=False))
    print("\nLjung-Box tone p (lags 4/8/12):",
          res.loc[res.spec=="T1", ["lb_p_tone_4","lb_p_tone_8","lb_p_tone_12"]].round(3).values)
