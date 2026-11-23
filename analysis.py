"""
Does Joining the Euro Reduce Inflation?
Replication code: builds the panel, estimates all models and writes the
LaTeX tables and figures used in paper/paper.tex.

Run:  python analysis.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

ROOT = Path(__file__).resolve().parent
RAW, OUT_T, OUT_F = ROOT / "data" / "raw", ROOT / "paper" / "tables", ROOT / "paper" / "figures"
OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261123)

# --------------------------------------------------------------------------- #
# 1. Sample definition
# --------------------------------------------------------------------------- #
# Year in which the euro became the country's currency (irrevocably fixed rates).
ADOPTION = {
    "AUT": 1999, "BEL": 1999, "DEU": 1999, "ESP": 1999, "FIN": 1999, "FRA": 1999,
    "IRL": 1999, "ITA": 1999, "LUX": 1999, "NLD": 1999, "PRT": 1999,
    "GRC": 2001, "SVN": 2007, "CYP": 2008, "MLT": 2008, "SVK": 2009,
    "EST": 2011, "LVA": 2014, "LTU": 2015, "HRV": 2023,
}
# EU members (current or former) that did not use the euro during the sample.
CONTROLS = ["SWE", "DNK", "GBR", "POL", "CZE", "HUN", "ROU", "BGR"]
TRANSITION = ["SVN", "SVK", "EST", "LVA", "LTU", "HRV", "POL", "CZE", "HUN", "ROU", "BGR"]
START, END = 1995, 2023


SOURCES = {  # World Bank WDI series, mirrored as open CSVs by github.com/datasets
    "inflation-consumer.csv": "https://raw.githubusercontent.com/datasets/inflation/main/data/inflation-consumer.csv",
    "gdp.csv": "https://raw.githubusercontent.com/datasets/gdp/main/data/gdp.csv",
    "population.csv": "https://raw.githubusercontent.com/datasets/population/main/data/population.csv",
}


def download_raw() -> None:
    """Download the raw World Bank series once (kept in data/raw/)."""
    import urllib.request
    RAW.mkdir(parents=True, exist_ok=True)
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
    df["log_gdppc"] = np.log(df.gdp / df["pop"])

    lo, hi = df.inflation.quantile([0.025, 0.975])
    df["infl_w"] = df.inflation.clip(lo, hi)              # main outcome (winsorized)
    df["dist_target"] = (df.infl_w - 2).abs()             # distance from 2% price-stability goal
    df.attrs["winsor"] = (lo, hi)
    return df.sort_values(["iso", "year"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Estimators
# --------------------------------------------------------------------------- #
def twfe(df, y, x=("euro",)):
    d = df.set_index(["iso", "year"])
    res = PanelOLS(d[y], d[list(x)], entity_effects=True, time_effects=True).fit(
        cov_type="clustered", cluster_entity=True)
    return res


def twfe_trends(df, y):
    """TWFE plus country-specific linear trends (one country's trend dropped: collinear with year FE)."""
    d = df.copy()
    t = d.year - d.year.min()
    cols = ["euro"]
    for c in sorted(d.iso.unique())[1:]:
        d[f"tr_{c}"] = (d.iso == c) * t
        cols.append(f"tr_{c}")
    return twfe(d, y, cols)


def wild_cluster_bootstrap_p(df, y, reps=999):
    """Wild cluster bootstrap-t (Rademacher, null imposed); Cameron, Gelbach & Miller (2008)."""
    d = df.dropna(subset=[y]).copy()
    # Two-way demeaning is equivalent to TWFE in a (near-)balanced panel; we use
    # explicit dummies instead to stay exact in the unbalanced case.
    X_fe = pd.get_dummies(d[["iso", "year"]].astype(str), drop_first=True).astype(float)
    X_fe.insert(0, "const", 1.0)
    X = pd.concat([d[["euro"]].astype(float), X_fe], axis=1).to_numpy()
    Xr = X_fe.to_numpy()
    yv = d[y].to_numpy()
    groups = d.iso.to_numpy()
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}

    def t_stat(yy):
        XtX_inv = np.linalg.pinv(X.T @ X)
        b = XtX_inv @ X.T @ yy
        e = yy - X @ b
        meat = sum(np.outer(X[i].T @ e[i], X[i].T @ e[i]) for i in idx.values())
        G = len(uniq)
        V = XtX_inv @ meat @ XtX_inv * G / (G - 1)
        return b[0] / np.sqrt(V[0, 0])

    t_obs = t_stat(yv)
    br = np.linalg.lstsq(Xr, yv, rcond=None)[0]
    fit_r, e_r = Xr @ br, yv - Xr @ br
    t_boot = np.empty(reps)
    for r in range(reps):
        w = RNG.choice([-1.0, 1.0], size=len(uniq))
        wmap = dict(zip(uniq, w))
        t_boot[r] = t_stat(fit_r + e_r * np.array([wmap[g] for g in groups]))
    return float(np.mean(np.abs(t_boot) >= abs(t_obs)))


def event_study(df, y, window=(-6, 8)):
    """Leads/lags around adoption; never-treated countries as controls; t = -1 omitted; endpoints binned."""
    d = df.copy()
    lo, hi = window
    rel = d.rel_time.clip(lo, hi)
    cols = []
    for k in range(lo, hi + 1):
        if k == -1:
            continue
        c = f"e_{k}".replace("-", "m")
        d[c] = ((rel == k) & (d.treated == 1)).astype(float)
        cols.append((k, c))
    res = twfe(d, y, [c for _, c in cols])
    ci = res.conf_int()
    out = pd.DataFrame({
        "k": [k for k, _ in cols],
        "coef": [res.params[c] for _, c in cols],
        "lo": [ci.loc[c, "lower"] for _, c in cols],
        "hi": [ci.loc[c, "upper"] for _, c in cols],
    })
    out = pd.concat([out, pd.DataFrame({"k": [-1], "coef": [0.0], "lo": [0.0], "hi": [0.0]})])
    return out.sort_values("k").reset_index(drop=True), res


# --------------------------------------------------------------------------- #
# 3. Output helpers
# --------------------------------------------------------------------------- #
def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def regression_table(models, labels, path, caption, label, note):
    size = r"\small" if len(models) <= 4 else r"\footnotesize\setlength{\tabcolsep}{4pt}"
    lines = [r"\begin{table}[t]\centering" + size, rf"\caption{{{caption}}}\label{{{label}}}",
             r"\begin{tabular}{l" + "c" * len(models) + "}", r"\toprule",
             " & " + " & ".join(f"({i + 1})" for i in range(len(models))) + r" \\",
             " & " + " & ".join(labels) + r" \\", r"\midrule"]
    b = [m["res"].params["euro"] for m in models]
    se = [m["res"].std_errors["euro"] for m in models]
    p = [m["res"].pvalues["euro"] for m in models]
    lines.append(r"Euro adoption & " + " & ".join(f"${x:.3f}^{{{stars(pp)}}}$" for x, pp in zip(b, p)) + r" \\")
    lines.append(" & " + " & ".join(f"({s:.3f})" for s in se) + r" \\[2pt]")
    for key, name in [("boot_p", "Wild bootstrap $p$-value"), ("mean_y", "Mean of dep.\\ variable")]:
        if all(m.get(key) is None for m in models):
            continue
        lines.append(name + " & " + " & ".join(
            (f"{m[key]:.3f}" if key == "boot_p" else f"{m[key]:.2f}") if m.get(key) is not None else "--" for m in models) + r" \\")
    lines += [r"\midrule",
              "Country FE & " + " & ".join(["Yes"] * len(models)) + r" \\",
              "Year FE & " + " & ".join(["Yes"] * len(models)) + r" \\",
              "Countries & " + " & ".join(str(int(m["res"].entity_info["total"])) for m in models) + r" \\",
              "Observations & " + " & ".join(str(int(m["res"].nobs)) for m in models) + r" \\",
              r"\bottomrule", r"\end{tabular}",
              r"\begin{minipage}{0.94\linewidth}\vspace{4pt}\footnotesize\textit{Notes:} " + note + r"\end{minipage}",
              r"\end{table}"]
    path.write_text("\n".join(lines))


def main():
    df = build_panel()
    lo, hi = df.attrs["winsor"]
    print(f"Panel: {df.iso.nunique()} countries, {len(df)} obs, {START}-{END}; winsor [{lo:.2f}, {hi:.2f}]")

    # ---- Descriptives ----------------------------------------------------- #
    pre = df[(df.treated == 1) & (df.euro == 0)]
    post = df[df.euro == 1]
    ctrl = df[df.treated == 0]
    desc = pd.DataFrame({
        "Euro adopters, before": [pre.inflation.mean(), pre.inflation.median(), pre.inflation.std(), len(pre)],
        "Euro adopters, after": [post.inflation.mean(), post.inflation.median(), post.inflation.std(), len(post)],
        "Non-euro EU (controls)": [ctrl.inflation.mean(), ctrl.inflation.median(), ctrl.inflation.std(), len(ctrl)],
    }, index=["Mean", "Median", "Std.\\ dev.", "Observations"])
    rows = [r"\begin{table}[t]\centering\small",
            r"\caption{Descriptive statistics: annual CPI inflation (\%), " + f"{START}--{END}" + r"}\label{tab:desc}",
            r"\begin{tabular}{lccc}", r"\toprule", " & " + " & ".join(desc.columns) + r" \\", r"\midrule"]
    for name, r in desc.iterrows():
        rows.append(name + " & " + " & ".join(f"{v:.0f}" if name == "Observations" else f"{v:.2f}" for v in r) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}",
             r"\begin{minipage}{0.86\linewidth}\vspace{4pt}\footnotesize\textit{Notes:} Unwinsorized values. "
             r"``Before'' and ``after'' refer to each adopter's own adoption year. Source: World Bank, WDI (FP.CPI.TOTL.ZG).\end{minipage}",
             r"\end{table}"]
    (OUT_T / "descriptives.tex").write_text("\n".join(rows))

    # ---- Main results ----------------------------------------------------- #
    def model(data, y, boot=True):
        return {"res": twfe(data, y), "boot_p": wild_cluster_bootstrap_p(data, y) if boot else None,
                "mean_y": data[y].mean()}

    main_models = [
        model(df, "infl_w"),
        model(df[df.year <= 2019], "infl_w"),
        model(df, "dist_target"),
        model(df[df.year <= 2019], "dist_target"),
    ]
    regression_table(
        main_models, ["Inflation", "Inflation", "$|\\pi-2|$", "$|\\pi-2|$"], OUT_T / "main.tex",
        "Effect of euro adoption on inflation: two-way fixed effects", "tab:main",
        f"Dependent variable in (1)--(2) is annual CPI inflation in percentage points, winsorized at the 2.5th and 97.5th "
        f"percentiles ({lo:.1f}\\% and {hi:.1f}\\%); in (3)--(4) the absolute distance between inflation and 2\\%. "
        f"Columns (2) and (4) end the sample in 2019, before the pandemic and energy shock. Standard errors clustered "
        f"by country in parentheses. The wild cluster bootstrap $p$-value uses 999 Rademacher draws with the null imposed. "
        f"$^{{*}}p<0.10$, $^{{**}}p<0.05$, $^{{***}}p<0.01$.")

    # ---- Robustness ------------------------------------------------------- #
    later = df[(df.adopt_year != 1999) | (df.treated == 0)]
    rob = [
        {"res": twfe(df.assign(infl_raw=df.inflation), "infl_raw"), "mean_y": df.inflation.mean()},
        {"res": twfe(df[df.year >= 2000], "infl_w"), "mean_y": df[df.year >= 2000].infl_w.mean()},
        {"res": twfe(later, "infl_w"), "mean_y": later.infl_w.mean()},
        {"res": twfe(df[df.transition == 1], "infl_w"), "mean_y": df[df.transition == 1].infl_w.mean()},
        {"res": twfe(df[df.transition == 0], "infl_w"), "mean_y": df[df.transition == 0].infl_w.mean()},
        {"res": twfe_trends(df, "infl_w"), "mean_y": df.infl_w.mean()},
    ]
    regression_table(
        rob, ["Unwinsorized", "From 2000", "Excl.\\ 1999", "Transition", "Non-trans.", "Country trends"],
        OUT_T / "robustness.tex", "Robustness checks (dependent variable: CPI inflation)", "tab:rob",
        "(1) uses raw inflation; (2) starts the sample in 2000; (3) drops the eleven 1999 founding members, so identification "
        "comes only from later adopters; (4) and (5) split the sample into Central and Eastern European transition economies "
        "and the rest; (6) adds country-specific linear time trends. Standard errors clustered by country. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.")

    # ---- Event study ------------------------------------------------------ #
    es, es_res = event_study(df[df.adopt_year.fillna(0) != 1999], "infl_w")  # later adopters: both pre and post observed
    pre_cols = [c for c in es_res.params.index if c.startswith("e_m")]
    wald = es_res.wald_test(formula=" = ".join(pre_cols[:1]) + " = 0" if len(pre_cols) == 1 else
                            ", ".join(f"{c} = 0" for c in pre_cols))
    es.to_csv(OUT_T / "event_study.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(-0.5, color="grey", lw=0.8, ls="--")
    ax.fill_between(es.k, es.lo, es.hi, color="#1f3b73", alpha=0.15, lw=0)
    ax.plot(es.k, es.coef, "o-", color="#1f3b73", ms=4)
    ax.set_xlabel("Years relative to euro adoption (endpoints binned)")
    ax.set_ylabel("Effect on inflation (p.p.)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT_F / "event_study.pdf"); fig.savefig(OUT_F / "event_study.png", dpi=200)
    plt.close(fig)

    # ---- Raw trends figure ------------------------------------------------ #
    g = df.assign(group=np.select([df.adopt_year == 1999, df.treated == 1], ["Founding members (1999)", "Later adopters (2001-2023)"],
                                  "Non-euro EU")).groupby(["group", "year"]).inflation.median().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for col, c, ls in zip(g.columns, ["#1f3b73", "#9a9a9a", "#b5452c"], ["-", "--", "-"]):
        ax.plot(g.index, g[col], color=c, ls=ls, lw=1.8, label=col)
    ax.axvline(1999, color="grey", lw=0.7, ls=":")
    ax.set_ylabel("Median CPI inflation (%)"); ax.set_ylim(-1, 16)
    ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT_F / "trends.pdf"); fig.savefig(OUT_F / "trends.png", dpi=200)
    plt.close(fig)

    # ---- Numbers quoted in the text ---------------------------------------- #
    m0 = main_models[0]["res"]
    post_avg = es[es.k >= 0].coef.mean()
    pre_avg = es[(es.k < -1)].coef.mean()
    facts = {
        "n_countries": df.iso.nunique(), "n_obs": len(df), "n_treated": int(df[df.treated == 1].iso.nunique()),
        "n_controls": int(df[df.treated == 0].iso.nunique()),
        "beta": m0.params["euro"], "se": m0.std_errors["euro"], "p": m0.pvalues["euro"],
        "boot_p": main_models[0]["boot_p"],
        "beta_2019": main_models[1]["res"].params["euro"], "p_2019": main_models[1]["res"].pvalues["euro"],
        "beta_dist": main_models[2]["res"].params["euro"], "p_dist": main_models[2]["res"].pvalues["euro"],
        "boot_p_dist": main_models[2]["boot_p"],
        "es_pre_avg": pre_avg, "es_post_avg": post_avg, "pretrend_p": float(wald.pval),
        "rob": {n: (r["res"].params["euro"], r["res"].pvalues["euro"]) for n, r in
                zip(["raw", "from2000", "excl1999", "transition", "nontransition", "trends"], rob)},
        "winsor": (lo, hi),
    }
    pd.Series({k: v for k, v in facts.items()}).to_json(OUT_T / "facts.json", indent=1)
    for k, v in facts.items():
        print(f"{k}: {v}")
    print("\nEvent study:\n", es.round(2).to_string(index=False))
    df.to_csv(ROOT / "data" / "panel.csv", index=False)


if __name__ == "__main__":
    main()
