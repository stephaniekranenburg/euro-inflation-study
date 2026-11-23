"""
Does Joining the Euro Reduce Inflation?
Replication code: downloads the data, builds the panel, estimates every model
and writes the LaTeX tables and figures used in paper/paper.tex.

Run:  python analysis.py
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from linearmodels.panel import PanelOLS, PooledOLS

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT_T = ROOT / "paper" / "tables"
OUT_F = ROOT / "paper" / "figures"
for p in (RAW, OUT_T, OUT_F):
    p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1. Sample definition
# --------------------------------------------------------------------------- #
# Year in which conversion rates were irrevocably fixed (start of membership).
ADOPTION = {
    "AUT": 1999, "BEL": 1999, "DEU": 1999, "ESP": 1999, "FIN": 1999, "FRA": 1999,
    "IRL": 1999, "ITA": 1999, "LUX": 1999, "NLD": 1999, "PRT": 1999,
    "GRC": 2001, "SVN": 2007, "CYP": 2008, "MLT": 2008, "SVK": 2009,
    "EST": 2011, "LVA": 2014, "LTU": 2015, "HRV": 2023,
}
# Current or former EU members that kept their own currency during 1995-2023.
CONTROLS = ["SWE", "DNK", "GBR", "POL", "CZE", "HUN", "ROU", "BGR"]
TRANSITION = ["SVN", "SVK", "EST", "LVA", "LTU", "HRV", "POL", "CZE", "HUN", "ROU", "BGR"]
START, END = 1995, 2023

SOURCES = {  # World Bank WDI series, mirrored as open CSVs by github.com/datasets
    "inflation-consumer.csv": "https://raw.githubusercontent.com/datasets/inflation/main/data/inflation-consumer.csv",
    "gdp.csv": "https://raw.githubusercontent.com/datasets/gdp/main/data/gdp.csv",
    "population.csv": "https://raw.githubusercontent.com/datasets/population/main/data/population.csv",
}


def download_raw() -> None:
    import urllib.request
    for name, url in SOURCES.items():
        if not (RAW / name).exists():
            print("downloading", name)
            urllib.request.urlretrieve(url, RAW / name)


def build_panel() -> pd.DataFrame:
    download_raw()
    inf = pd.read_csv(RAW / "inflation-consumer.csv").rename(
        columns={"Country Code": "iso", "Year": "year", "Inflation": "inflation", "Country": "country"})
    gdp = pd.read_csv(RAW / "gdp.csv").rename(columns={"Country Code": "iso", "Year": "year", "Value": "gdp"})
    pop = pd.read_csv(RAW / "population.csv").rename(columns={"Country Code": "iso", "Year": "year", "Value": "pop"})
    df = (inf.merge(gdp[["iso", "year", "gdp"]], on=["iso", "year"], how="left")
             .merge(pop[["iso", "year", "pop"]], on=["iso", "year"], how="left"))
    df = df[df.iso.isin(list(ADOPTION) + CONTROLS) & df.year.between(START, END)].copy()

    df["adopt_year"] = df.iso.map(ADOPTION)
    df["treated"] = df.adopt_year.notna().astype(int)
    df["euro"] = ((df.year >= df.adopt_year) & df.treated.astype(bool)).astype(int)
    df["rel_time"] = df.year - df.adopt_year
    df["transition"] = df.iso.isin(TRANSITION).astype(int)
    df["gdppc"] = df.gdp / df["pop"]

    lo, hi = df.inflation.quantile([0.025, 0.975])
    df["infl_w"] = df.inflation.clip(lo, hi)
    df["dist_target"] = (df.infl_w - 2).abs()
    df.attrs["winsor"] = (float(lo), float(hi))
    return df.sort_values(["iso", "year"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Estimators
# --------------------------------------------------------------------------- #
def fit(df, y, entity=True, time=True, extra=()):
    d = df.set_index(["iso", "year"])
    X = d[["euro", *extra]].astype(float)
    if not entity and not time:
        X = X.assign(const=1.0)
        return PooledOLS(d[y], X).fit(cov_type="clustered", cluster_entity=True)
    return PanelOLS(d[y], X, entity_effects=entity, time_effects=time).fit(
        cov_type="clustered", cluster_entity=True)


def with_trends(df):
    d = df.copy()
    t = d.year - d.year.min()
    cols = []
    for c in sorted(d.iso.unique())[1:]:  # one trend dropped: collinear with year FE
        d[f"tr_{c}"] = (d.iso == c) * t
        cols.append(f"tr_{c}")
    return d, cols


# --------------------------------------------------------------------------- #
# 3. Table helpers
# --------------------------------------------------------------------------- #
def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def reg_table(models, labels, rows, path, caption, label, note):
    k = len(models)
    size = r"\small" if k <= 4 else r"\footnotesize\setlength{\tabcolsep}{4pt}"
    L = [r"\begin{table}[htbp]\centering" + size, rf"\caption{{{caption}}}\label{{{label}}}",
         r"\begin{tabular}{l" + "c" * k + "}", r"\toprule",
         " & " + " & ".join(f"({i + 1})" for i in range(k)) + r" \\",
         " & " + " & ".join(labels) + r" \\", r"\midrule"]
    b = [m.params["euro"] for m in models]
    se = [m.std_errors["euro"] for m in models]
    p = [m.pvalues["euro"] for m in models]
    L.append(r"$\text{Euro}_{it}$ & " + " & ".join(f"${x:.3f}^{{{stars(q)}}}$" for x, q in zip(b, p)) + r" \\")
    L.append(" & " + " & ".join(f"({s:.3f})" for s in se) + r" \\")
    L.append(r"$p$-value & " + " & ".join(f"[{q:.3f}]" for q in p) + r" \\[2pt]")
    L.append(r"\midrule")
    for name, vals in rows:
        L.append(name + " & " + " & ".join(vals) + r" \\")
    L.append("Countries & " + " & ".join(str(int(m.entity_info["total"])) for m in models) + r" \\")
    L.append("Observations & " + " & ".join(str(int(m.nobs)) for m in models) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{0.95\linewidth}\vspace{4pt}\footnotesize\textit{Notes:} " + note + r"\end{minipage}",
          r"\end{table}"]
    path.write_text("\n".join(L))


def r2w(m):
    try:
        return f"{m.rsquared_within:.3f}"
    except Exception:
        return "--"


def main():
    df = build_panel()
    lo, hi = df.attrs["winsor"]
    facts = {"winsor": [round(lo, 2), round(hi, 2)], "n_obs": len(df), "n_countries": int(df.iso.nunique())}

    # ---- Descriptive statistics ------------------------------------------ #
    groups = {
        "Adopters, before": df[(df.treated == 1) & (df.euro == 0)],
        "Adopters, after": df[df.euro == 1],
        "Non-euro EU": df[df.treated == 0],
    }
    R = [r"\begin{table}[htbp]\centering\small",
         r"\caption{Descriptive statistics of annual CPI inflation (\%), " + f"{START}--{END}" + r"}\label{tab:desc}",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"Group & Mean & Median & Std.\ dev. & Min & Max \\", r"\midrule"]
    for g, d in groups.items():
        R.append(f"{g} & {d.inflation.mean():.2f} & {d.inflation.median():.2f} & {d.inflation.std():.2f} "
                 f"& {d.inflation.min():.2f} & {d.inflation.max():.2f}" + r" \\")
    R.append(r"\midrule")
    R.append(f"Full sample & {df.inflation.mean():.2f} & {df.inflation.median():.2f} & {df.inflation.std():.2f} "
             f"& {df.inflation.min():.2f} & {df.inflation.max():.2f}" + r" \\")
    R += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{0.9\linewidth}\vspace{4pt}\footnotesize\textit{Notes:} Unwinsorized values. "
          + ", ".join(f"{g}: $n={len(d)}$" for g, d in groups.items())
          + r". ``Before'' and ``after'' are defined relative to each adopter's own adoption year. "
            r"Source: World Bank, WDI (FP.CPI.TOTL.ZG).\end{minipage}", r"\end{table}"]
    (OUT_T / "descriptives.tex").write_text("\n".join(R))

    # ---- Country table (appendix) ----------------------------------------- #
    rows = []
    for iso, d in df.groupby("iso"):
        name = d.country.iloc[0]
        ay = ADOPTION.get(iso)
        pre = d[d.euro == 0].inflation.mean()
        post = d[d.euro == 1].inflation.mean() if ay else np.nan
        rows.append((0 if ay else 1, ay or 9999, name, ay, pre, post))
    rows.sort()
    C = [r"\begin{table}[htbp]\centering\small", r"\caption{Mean CPI inflation by country (\%)}\label{tab:countries}",
         r"\begin{tabular}{llccc}", r"\toprule",
         r"Country & Group & Adoption & Before / whole period & After \\", r"\midrule"]
    for ctrl, _, name, ay, pre, post in rows:
        name = name.replace("&", r"\&")
        C.append(f"{name} & {'Control' if ctrl else 'Adopter'} & {ay if ay else '--'} & {pre:.2f} & "
                 f"{'--' if np.isnan(post) else f'{post:.2f}'}" + r" \\")
    C += [r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{0.85\linewidth}\vspace{4pt}\footnotesize\textit{Notes:} For control countries the "
          r"fourth column is the 1995--2023 mean. Unwinsorized values.\end{minipage}", r"\end{table}"]
    (OUT_T / "countries.tex").write_text("\n".join(C))

    # ---- Main results: from pooled OLS to TWFE ---------------------------- #
    m_pool = fit(df, "infl_w", entity=False, time=False)
    m_fe = fit(df, "infl_w", entity=True, time=False)
    m_twfe = fit(df, "infl_w")
    d_tr, tr_cols = with_trends(df)
    m_trend = fit(d_tr, "infl_w", extra=tr_cols)
    reg_table(
        [m_pool, m_fe, m_twfe, m_trend], ["Pooled OLS", "Country FE", "Two-way FE", "TWFE + trends"],
        [("Country fixed effects", ["No", "Yes", "Yes", "Yes"]),
         ("Year fixed effects", ["No", "No", "Yes", "Yes"]),
         ("Country-specific trends", ["No", "No", "No", "Yes"])],
        OUT_T / "main.tex", "Euro adoption and inflation: baseline estimates", "tab:main",
        f"Dependent variable: annual CPI inflation in percentage points, winsorized at the 2.5th and 97.5th "
        f"percentiles ({lo:.1f}\\% and {hi:.1f}\\%). Standard errors clustered by country in parentheses, "
        f"$p$-values in brackets. Column (1) includes a constant. $^{{*}}p<0.10$, $^{{**}}p<0.05$, $^{{***}}p<0.01$.")
    facts["main"] = {n: [round(float(m.params["euro"]), 3), round(float(m.std_errors["euro"]), 3),
                         round(float(m.pvalues["euro"]), 3)]
                     for n, m in zip(["pooled", "fe", "twfe", "trends"], [m_pool, m_fe, m_twfe, m_trend])}

    # ---- Robustness -------------------------------------------------------- #
    later = df[(df.adopt_year != 1999) | (df.treated == 0)]
    rob = {
        "Unwinsorized": fit(df.assign(raw=df.inflation), "raw"),
        "Until 2019": fit(df[df.year <= 2019], "infl_w"),
        "From 2000": fit(df[df.year >= 2000], "infl_w"),
        "Excl.\\ 1999": fit(later, "infl_w"),
        "Transition": fit(df[df.transition == 1], "infl_w"),
        "Non-trans.": fit(df[df.transition == 0], "infl_w"),
        "$|\\pi-2|$": fit(df, "dist_target"),
    }
    reg_table(
        list(rob.values()), list(rob.keys()), [], OUT_T / "robustness.tex",
        "Robustness checks (two-way fixed effects)", "tab:rob",
        "All columns include country and year fixed effects. (1) raw inflation; (2) excludes the pandemic and "
        "energy-shock years; (3) starts after the 1990s disinflation; (4) drops the eleven founding members; "
        "(5)--(6) split the sample into Central and Eastern European transition economies and the rest; (7) uses "
        "the absolute distance between inflation and 2\\% as dependent variable. Standard errors clustered by "
        "country. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.")
    facts["rob"] = {k: [round(float(m.params["euro"]), 3), round(float(m.pvalues["euro"]), 3)] for k, m in rob.items()}

    # ---- Cross-section: initial inflation vs change after adoption -------- #
    cs = []
    for iso, d in df[df.treated == 1].groupby("iso"):
        pre, post = d[d.euro == 0].inflation, d[d.euro == 1].inflation
        if len(pre) >= 2 and len(post) >= 1:
            cs.append({"iso": iso, "pre": d[d.euro == 0].infl_w.mean(), "post": d[d.euro == 1].infl_w.mean()})
    cs = pd.DataFrame(cs)
    cs["change"] = cs.post - cs.pre
    X = np.column_stack([np.ones(len(cs)), cs.pre.values])
    y = cs.change.values
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ (X.T * e**2) @ X @ XtXi * n / (n - k)  # HC1
    se = np.sqrt(np.diag(V))
    tval = beta[1] / se[1]
    pval = 2 * stats.t.sf(abs(tval), n - k)
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    facts["cross"] = {"a": round(float(beta[0]), 3), "b": round(float(beta[1]), 3), "se_b": round(float(se[1]), 3),
                      "t": round(float(tval), 2), "p": round(float(pval), 4), "r2": round(float(r2), 3), "n": int(n)}

    # Placebo: controls with a pseudo-adoption year equal to the median later-adoption year.
    pseudo = int(np.median([y for y in ADOPTION.values() if y > 1999]))
    pc = []
    for iso, d in df[df.treated == 0].groupby("iso"):
        pc.append({"iso": iso, "pre": d[d.year < pseudo].infl_w.mean(), "post": d[d.year >= pseudo].infl_w.mean()})
    pc = pd.DataFrame(pc); pc["change"] = pc.post - pc.pre
    allc = pd.concat([cs.assign(D=1), pc.assign(D=0)], ignore_index=True)
    Z = np.column_stack([np.ones(len(allc)), allc.pre, allc.D, allc.D * allc.pre])
    yz = allc.change.values
    bz = np.linalg.solve(Z.T @ Z, Z.T @ yz)
    ez = yz - Z @ bz
    nz, kz = Z.shape
    Zi = np.linalg.inv(Z.T @ Z)
    Vz = Zi @ (Z.T * ez**2) @ Z @ Zi * nz / (nz - kz)
    sez = np.sqrt(np.diag(Vz))
    pz = 2 * stats.t.sf(np.abs(bz / sez), nz - kz)
    facts["placebo"] = {"pseudo_year": pseudo, "n": int(nz),
                        "coef": [round(float(x), 3) for x in bz], "se": [round(float(x), 3) for x in sez],
                        "p": [round(float(x), 3) for x in pz]}
    names = ["Constant", r"$\bar{\pi}_i^{\,pre}$", r"$D_i$ (adopter)", r"$D_i \times \bar{\pi}_i^{\,pre}$"]
    T = [r"\begin{table}[htbp]\centering\small",
         r"\caption{Initial inflation and the change in inflation after adoption}\label{tab:cross}",
         r"\begin{tabular}{lcc}", r"\toprule", r" & (1) Adopters only & (2) With placebo controls \\", r"\midrule"]
    vals1 = {0: (beta[0], se[0], 2 * stats.t.sf(abs(beta[0] / se[0]), n - k)), 1: (beta[1], se[1], pval)}
    for j, nm in enumerate(names):
        c1 = f"${vals1[j][0]:.3f}^{{{stars(vals1[j][2])}}}$" if j in vals1 else ""
        s1 = f"({vals1[j][1]:.3f})" if j in vals1 else ""
        T.append(f"{nm} & {c1} & ${bz[j]:.3f}^{{{stars(pz[j])}}}$" + r" \\")
        T.append(f" & {s1} & ({sez[j]:.3f})" + r" \\")
    T += [r"\midrule", f"$R^2$ & {r2:.3f} & {1 - (ez @ ez) / ((yz - yz.mean()) @ (yz - yz.mean())):.3f}" + r" \\",
          f"Countries & {n} & {nz}" + r" \\", r"\bottomrule", r"\end{tabular}",
          r"\begin{minipage}{0.9\linewidth}\vspace{4pt}\footnotesize\textit{Notes:} Dependent variable: "
          r"$\Delta\bar{\pi}_i = \bar{\pi}_i^{\,post} - \bar{\pi}_i^{\,pre}$ (winsorized inflation). In column (2) the eight "
          f"control countries are assigned a pseudo-adoption year of {pseudo}. Heteroskedasticity-robust (HC1) "
          r"standard errors in parentheses. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.\end{minipage}", r"\end{table}"]
    (OUT_T / "cross.tex").write_text("\n".join(T))

    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.scatter(cs.pre, cs.change, color="#1f3b73", s=28, zorder=3)
    for _, r in cs.iterrows():
        ax.annotate(r.iso, (r.pre, r.change), xytext=(4, 3), textcoords="offset points", fontsize=8, color="#444")
    xs = np.linspace(cs.pre.min() - 0.3, cs.pre.max() + 0.3, 50)
    ax.plot(xs, beta[0] + beta[1] * xs, color="#b5452c", lw=1.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Mean inflation before adoption (%, winsorized)")
    ax.set_ylabel("Change in mean inflation after adoption (p.p.)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT_F / "cross_section.pdf"); fig.savefig(OUT_F / "cross_section.png", dpi=200)
    plt.close(fig)

    # ---- Descriptive pattern around adoption (later adopters) ------------- #
    ctrl_mean = df[df.treated == 0].groupby("year").infl_w.mean()
    la = df[(df.treated == 1) & (df.adopt_year > 1999)].copy()
    la["gap"] = la.infl_w - la.year.map(ctrl_mean)
    la = la[la.rel_time.between(-6, 8)]
    prof = la.groupby("rel_time").agg(gap=("gap", "mean"), level=("infl_w", "mean"), n=("gap", "size"))
    prof.to_csv(OUT_T / "around_adoption.csv")
    facts["around"] = {"gap_m5_m3": round(float(prof.loc[-5:-3, "gap"].mean()), 2),
                       "gap_m2_m1": round(float(prof.loc[-2:-1, "gap"].mean()), 2),
                       "gap_0_8": round(float(prof.loc[0:8, "gap"].mean()), 2),
                       "level_m2_m1": round(float(prof.loc[-2:-1, "level"].mean()), 2),
                       "level_m5_m3": round(float(prof.loc[-5:-3, "level"].mean()), 2),
                       "level_0_8": round(float(prof.loc[0:8, "level"].mean()), 2)}
    fig, ax = plt.subplots(figsize=(6.8, 3.7))
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(-0.5, color="grey", lw=0.8, ls="--")
    ax.bar(prof.index, prof.gap, color=["#9a9a9a" if k < 0 else "#1f3b73" for k in prof.index], width=0.7)
    ax.set_xlabel("Years relative to euro adoption")
    ax.set_ylabel("Inflation gap vs. non-euro EU (p.p.)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT_F / "around_adoption.pdf"); fig.savefig(OUT_F / "around_adoption.png", dpi=200)
    plt.close(fig)

    # ---- Raw trends -------------------------------------------------------- #
    g = df.assign(group=np.select([df.adopt_year == 1999, df.treated == 1],
                                  ["Founding members (1999)", "Later adopters (2001-2023)"], "Non-euro EU")
                  ).groupby(["group", "year"]).inflation.median().unstack(0)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for col, c, ls in zip(g.columns, ["#1f3b73", "#9a9a9a", "#b5452c"], ["-", "--", "-"]):
        ax.plot(g.index, g[col], color=c, ls=ls, lw=1.8, label=col)
    ax.axvline(1999, color="grey", lw=0.7, ls=":")
    ax.set_ylabel("Median CPI inflation (%)"); ax.set_ylim(-1, 16)
    ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT_F / "trends.pdf"); fig.savefig(OUT_F / "trends.png", dpi=200)
    plt.close(fig)

    (OUT_T / "facts.json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))
    print(prof.round(2).to_string())
    df.to_csv(ROOT / "data" / "panel.csv", index=False)


if __name__ == "__main__":
    main()
