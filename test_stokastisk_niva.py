"""
Test: Behövs stokastisk nivå (random walk)?
Jämför för både envariat och multivariat modell:
  M1: σ²_η fri (stokastisk nivå)
  M2: σ²_η = 0 (konstant nivå)
Alla modeller kör nseason=0 för rent test.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from projekt_nivaer import (
    load_base_station, load_candidate_stations,
    dataframe_multi, GroundwaterSSM, GroundwaterSSM_Multi, MLEModel,
    basror_22W102, basror_17XX01U, basror_G1101,
)


class GroundwaterSSM_FixedLevel(MLEModel):
    """Envariat modell med deterministisk (konstant) nivå: σ²_η = 0."""

    def __init__(self, endog, **kwargs):
        super().__init__(endog, k_states=1, k_posdef=1, **kwargs)
        self._param_names = ["sigma2_eps_ref", "beta_ref"]
        self.ssm.initialize_diffuse()

    @property
    def param_names(self):
        return self._param_names

    @property
    def start_params(self):
        return np.array([0.1, 1.0])

    def transform_params(self, unc):
        p = unc.copy()
        p[0] = np.exp(unc[0])  # σ²_ε,ref > 0
        return p

    def untransform_params(self, con):
        p = con.copy()
        p[0] = np.log(con[0])
        return p

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        s2_ref, beta_ref = params
        self.ssm["transition"] = np.array([[1.0]])
        self.ssm["design"] = np.array([[1.0], [beta_ref]])
        self.ssm["obs_cov"] = np.diag([0.0, s2_ref])
        self.ssm["state_cov"] = np.array([[0.0]])  # σ²_η = 0 → konstant nivå
        self.ssm["selection"] = np.array([[1.0]])


STATIONS = [
    (basror_22W102, "22W102"),
    (basror_17XX01U, "17XX01U"),
    (basror_G1101, "G1101"),
]

results = []

for station_path, station_id in STATIONS:
    print("\n" + "=" * 60)
    print(f"STATION: {station_id}")
    print("=" * 60)

    base = load_base_station(station_path)
    span_days = (base.index.max() - base.index.min()).days
    n_obs = base.notna().sum()
    avg_gap = span_days / max(n_obs - 1, 1)
    freq = "MS" if avg_gap > 21 else "7D"

    refs = load_candidate_stations(base, base_station_id=station_id)
    df_top1 = dataframe_multi(base, refs, freq=freq, max_refs=1)
    ref_id = [c for c in df_top1.columns if c != "base"][0]
    df = df_top1.rename(columns={ref_id: "ref"})[["base", "ref"]]
    endog = df[["base", "ref"]].values.astype(float)

    # M1: Stokastisk nivå (σ²_η fri) — nseason=0 för rent test

    m1 = GroundwaterSSM(endog)
    r1 = m1.fit(method="lbfgs", maxiter=2000, disp=False)

    # M2: Deterministisk nivå (σ²_η = 0)
    m2 = GroundwaterSSM_FixedLevel(endog)
    r2 = m2.fit(method="lbfgs", maxiter=2000, disp=False)

    print(f"\n  M1 (stokastisk): sigma2_eta={r1.params[0]:.6e}, LL={r1.llf:.4f}, AIC={r1.aic:.4f}, params={len(r1.params)}")
    print(f"  M2 (konstant):   sigma2_eta=0 (last),          LL={r2.llf:.4f}, AIC={r2.aic:.4f}, params={len(r2.params)}")
    print(f"  -> DeltaLL={r1.llf - r2.llf:.4f}, DeltaAIC={r1.aic - r2.aic:.4f}")

    results.append({
        "station": station_id, "ref": ref_id,
        "sigma2_eta": r1.params[0],
        "LL_stokastisk": r1.llf, "LL_konstant": r2.llf,
        "AIC_stokastisk": r1.aic, "AIC_konstant": r2.aic,
    })

print("\n" + "=" * 60)
print("SAMMANFATTNING - Envariat modell")
print("=" * 60)
print(f"\n{'Station':<12} {'Ref':<12} {'sigma2_eta':<14} {'DeltaLL':<10} {'DeltaAIC':<10} {'Slutsats'}")
print("-" * 75)
for r in results:
    dll = r["LL_stokastisk"] - r["LL_konstant"]
    daic = r["AIC_stokastisk"] - r["AIC_konstant"]
    if daic < -2:
        slutsats = "Stokastisk niva behovs"
    elif daic > 2:
        slutsats = "Konstant niva racker"
    else:
        slutsats = "Likvardig"
    print(f"{r['station']:<12} {r['ref']:<12} {r['sigma2_eta']:<14.6e} {dll:<10.4f} {daic:<10.4f} {slutsats}")


# ══════════════════════════════════════════════════════════════════
#  DEL 2: MULTIVARIAT MODELL — stokastisk vs konstant nivå
# ══════════════════════════════════════════════════════════════════

class GroundwaterSSM_Multi_FixedLevel(MLEModel):
    """Multivariat modell med konstant nivå: σ²_η = 0, nseason=0."""

    def __init__(self, endog, ref_ids=None, **kwargs):
        self.ref_ids = ref_ids if ref_ids is not None else []
        self.n_refs = len(self.ref_ids)
        self.nseason = 0

        super().__init__(endog, k_states=1, k_posdef=1, **kwargs)

        # Parametrar per ref: sigma2_eps, beta, alpha (ingen sigma2_eta_level)
        self._param_names = []
        for rid in self.ref_ids:
            self._param_names += [f"sigma2_eps_{rid}", f"beta_{rid}", f"alpha_{rid}"]

        self.ssm.initialize_diffuse()

    @property
    def param_names(self):
        return self._param_names

    @property
    def start_params(self):
        y1 = self.endog[:, 0]
        y1_valid = y1[~np.isnan(y1)]
        level = np.full(len(y1), np.nanmean(y1_valid) if len(y1_valid) > 0 else 0.0)
        params = []
        for i in range(self.n_refs):
            y_ref = self.endog[:, 1 + i]
            mask = ~(np.isnan(y_ref) | np.isnan(level))
            X = np.column_stack([np.ones(mask.sum()), level[mask]])
            ols = np.linalg.lstsq(X, y_ref[mask], rcond=None)
            alpha_i = float(ols[0][0])
            beta_i = float(ols[0][1])
            s2_eps_i = float(np.var(y_ref[mask] - X @ ols[0]))
            params += [max(s2_eps_i, 1e-6), beta_i, alpha_i]
        return np.array(params)

    def transform_params(self, unc):
        p = unc.copy()
        for i in range(self.n_refs):
            p[3 * i] = np.exp(unc[3 * i])  # sigma2_eps > 0
        return p

    def untransform_params(self, con):
        p = con.copy()
        for i in range(self.n_refs):
            p[3 * i] = np.log(con[3 * i])
        return p

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        n_endog = 1 + self.n_refs

        ref_params = [
            (params[3 * i], params[3 * i + 1], params[3 * i + 2])
            for i in range(self.n_refs)
        ]

        # T: konstant nivå (identity men σ²_η = 0)
        self.ssm["transition"] = np.array([[1.0]])

        # Z: design
        Z = np.zeros((n_endog, 1))
        Z[0, 0] = 1.0  # base ← nivå
        for i, (_, beta_i, _) in enumerate(ref_params):
            Z[1 + i, 0] = beta_i
        self.ssm["design"] = Z

        # d: intercept
        d = np.zeros((n_endog, 1))
        for i, (_, _, alpha_i) in enumerate(ref_params):
            d[1 + i, 0] = alpha_i
        self.ssm["obs_intercept"] = d

        # H: obs cov
        obs_cov = np.zeros((n_endog, n_endog))
        for i, (s2_eps_i, _, _) in enumerate(ref_params):
            obs_cov[1 + i, 1 + i] = s2_eps_i
        self.ssm["obs_cov"] = obs_cov

        # Q: state cov = 0 (konstant nivå)
        self.ssm["state_cov"] = np.array([[0.0]])
        self.ssm["selection"] = np.array([[1.0]])


results_multi = []

print("\n\n" + "╔" + "═" * 60 + "╗")
print("║  DEL 2: MULTIVARIAT MODELL — stokastisk vs konstant nivå  ║")
print("╚" + "═" * 60 + "╝")

for station_path, station_id in STATIONS:
    print("\n" + "=" * 60)
    print(f"STATION: {station_id}")
    print("=" * 60)

    base = load_base_station(station_path)
    span_days = (base.index.max() - base.index.min()).days
    n_obs = base.notna().sum()
    avg_gap = span_days / max(n_obs - 1, 1)
    freq = "MS" if avg_gap > 21 else "7D"

    refs = load_candidate_stations(base, base_station_id=station_id)
    df_multi = dataframe_multi(base, refs, freq=freq, max_refs=4)
    ref_ids = [c for c in df_multi.columns if c != "base"]
    endog = df_multi[["base"] + ref_ids].values.astype(float)

    print(f"  Referensrör: {ref_ids}")

    # M1: Stokastisk nivå (σ²_η fri), nseason=0
    m1 = GroundwaterSSM_Multi(endog, nseason=0, ref_ids=ref_ids)
    r1 = m1.fit(method="powell", maxiter=5000, disp=False, cov_type="none")

    # M2: Konstant nivå (σ²_η = 0), nseason=0
    m2 = GroundwaterSSM_Multi_FixedLevel(endog, ref_ids=ref_ids)
    r2 = m2.fit(method="powell", maxiter=5000, disp=False, cov_type="none")

    print(f"\n  M1 (stokastisk): sigma2_eta={r1.params[0]:.6e}, LL={r1.llf:.4f}, AIC={r1.aic:.4f}, params={len(r1.params)}")
    print(f"  M2 (konstant):   sigma2_eta=0 (låst),          LL={r2.llf:.4f}, AIC={r2.aic:.4f}, params={len(r2.params)}")
    print(f"  -> DeltaLL={r1.llf - r2.llf:.4f}, DeltaAIC={r1.aic - r2.aic:.4f}")

    results_multi.append({
        "station": station_id, "refs": ref_ids,
        "sigma2_eta": r1.params[0],
        "LL_stokastisk": r1.llf, "LL_konstant": r2.llf,
        "AIC_stokastisk": r1.aic, "AIC_konstant": r2.aic,
        "n_params_m1": len(r1.params), "n_params_m2": len(r2.params),
    })

print("\n" + "=" * 60)
print("SAMMANFATTNING - Multivariat modell")
print("=" * 60)
print(f"\n{'Station':<12} {'#refs':<6} {'sigma2_eta':<14} {'DeltaLL':<10} {'DeltaAIC':<10} {'k_M1':<5} {'k_M2':<5} {'Slutsats'}")
print("-" * 85)
for r in results_multi:
    dll = r["LL_stokastisk"] - r["LL_konstant"]
    daic = r["AIC_stokastisk"] - r["AIC_konstant"]
    if daic < -2:
        slutsats = "Stokastisk niva behovs"
    elif daic > 2:
        slutsats = "Konstant niva racker"
    else:
        slutsats = "Likvardig"
    print(f"{r['station']:<12} {len(r['refs']):<6} {r['sigma2_eta']:<14.6e} {dll:<10.4f} {daic:<10.4f} {r['n_params_m1']:<5} {r['n_params_m2']:<5} {slutsats}")
