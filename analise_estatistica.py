"""
Análise estatística dos resultados do experimento_eamc_v3.py (não recalcula modelos; só lê os CSVs).

Uso:  python analise_estatistica.py cron_2a cron_2b cron_lee        (prefixos dos arquivos)
Lê:   {prefixo}_trajetorias.csv e {prefixo}_paradas.csv
Gera: {prefixo}_estatistica.txt  e  {prefixo}_compacto.csv (acurácia por sujeito nos orçamentos)

Análises declaradas (todas com IC95 por t pareado sobre sujeitos; Holm dentro de cada família):
  F1  transferência − intra nos orçamentos de 20, 40 e 80 trials          (efeito positivo)
  F2  equivalência (TOST, margem ±5 pp; IC90 também mostrado contra ±2 pp):
        transf com 40 trials  vs  intra com a calibração completa
  F3  critérios de parada vs orçamento fixo de 40 trials (Δ acurácia e Δ trials) + ρ de Spearman
      com o oráculo (IC95 por bootstrap)                                   (resultado negativo)
  F4  poucos canais: transf completa com 8 e 3 canais vs a montagem maior (TOST ±5 pp)
  F5  limite estatístico: ruído do sinal de parada (cv) vs o ganho real por passo de calibração
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

ORCAMENTOS = [20, 40, 80]
MARGEM, MARGEM_ESTRITA = 0.05, 0.02
EPS, PAC = 0.02, 3
CRITERIOS = ["preq", "cv", "extrap"]


def holm(ps):
    ps = np.asarray(ps, float)
    ordem = np.argsort(ps)
    aj = np.empty_like(ps)
    acum = 0.0
    for r, i in enumerate(ordem):
        acum = max(acum, min(1.0, (len(ps) - r) * ps[i]))
        aj[i] = acum
    return aj


def ic(d, nivel=0.95):
    d = np.asarray(d, float)
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    t = stats.t.ppf(0.5 + nivel / 2, len(d) - 1)
    return m, m - t * se, m + t * se


def wil(d):
    d = np.asarray(d, float)
    return 1.0 if np.allclose(d, 0) else stats.wilcoxon(d).pvalue


def tost(d, margem):
    d = np.asarray(d, float)
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    if se == 0:
        return 0.0 if abs(m) < margem else 1.0
    p_inf = 1 - stats.t.cdf((m + margem) / se, n - 1)   # H0: m <= -margem
    p_sup = stats.t.cdf((m - margem) / se, n - 1)       # H0: m >= +margem
    return max(p_inf, p_sup)


def rho_boot(x, y, B=2000, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    r = stats.spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        i = rng.integers(len(x), size=len(x))
        if len(np.unique(x[i])) > 1 and len(np.unique(y[i])) > 1:
            bs.append(stats.spearmanr(x[i], y[i]).correlation)
    return r, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def pp(v):
    return f"{100 * v:+5.1f}"


def acc_no_orcamento(traj, n_alvo):
    """Acurácia por sujeito no checkpoint mais próximo de n_alvo (média sobre permutações)."""
    g = traj.groupby(["config", "cond", "sujeito", "n"]).acc_E.mean().reset_index()
    g["dist"] = (g.n - n_alvo).abs()
    g = g.sort_values(["dist", "n"]).groupby(["config", "cond", "sujeito"]).head(1)
    return g.set_index(["config", "cond", "sujeito"])[["acc_E", "n"]]


def acc_completa(traj):
    g = traj.groupby(["config", "cond", "sujeito", "n"]).acc_E.mean().reset_index()
    return g.sort_values("n").groupby(["config", "cond", "sujeito"]).acc_E.last()


def analisar(pref):
    traj = pd.read_csv(f"{pref}_trajetorias.csv")
    par = pd.read_csv(f"{pref}_paradas.csv")
    configs = list(dict.fromkeys(traj.config))
    nsuj = traj.sujeito.nunique()
    out = [f"######## {pref} — {nsuj} sujeitos | IC95 = t pareado sobre sujeitos | valores em pontos percentuais (pp)\n"]

    # compacto: acurácia por sujeito nos orçamentos (para comparar ordens de trials depois)
    comp = []
    for nb in ORCAMENTOS:
        a = acc_no_orcamento(traj, nb).reset_index()
        a["orcamento"] = nb
        comp.append(a)
    full = acc_completa(traj).reset_index()
    full["orcamento"], full["n"] = "completa", np.nan
    comp.append(full)
    pd.concat(comp).to_csv(f"{pref}_compacto.csv", index=False)
    full = acc_completa(traj)

    # F1
    out.append("F1  Transferência − intra, mesmo nº de trials do usuário novo (Holm sobre a família)")
    linhas, ps = [], []
    for cfg in configs:
        for nb in ORCAMENTOS:
            a = acc_no_orcamento(traj, nb)
            d = (a.loc[(cfg, "transf")].acc_E - a.loc[(cfg, "intra")].acc_E).values
            m, lo, hi = ic(d)
            linhas.append((cfg, nb, m, lo, hi, int((d > 0).sum()), len(d)))
            ps.append(wil(d))
    for (cfg, nb, m, lo, hi, k, n), p, pa in zip(linhas, ps, holm(ps)):
        out.append(f"  {cfg:<10} {nb:>3} trials: Δ {pp(m)} [{pp(lo)}, {pp(hi)}]  transf melhor em {k}/{n}  "
                   f"p {p:.4f}  p_Holm {pa:.4f}")

    # F2
    out.append(f"\nF2  Equivalência: transf com 40 trials vs intra com calibração completa "
               f"(TOST ±{100 * MARGEM:.0f} pp; IC90 vs ±{100 * MARGEM_ESTRITA:.0f} pp)")
    a40 = acc_no_orcamento(traj, 40)
    for cfg in configs:
        d = (a40.loc[(cfg, "transf")].acc_E - full.loc[(cfg, "intra")]).values
        m, lo, hi = ic(d)
        _, lo90, hi90 = ic(d, 0.90)
        p = tost(d, MARGEM)
        estrito = "dentro" if (lo90 > -MARGEM_ESTRITA and hi90 < MARGEM_ESTRITA) else "fora"
        out.append(f"  {cfg:<10} Δ {pp(m)} IC95 [{pp(lo)}, {pp(hi)}] IC90 [{pp(lo90)}, {pp(hi90)}]  "
                   f"p_TOST(±5) {p:.4f} → {'EQUIVALENTE' if p < 0.05 else 'inconclusivo'}  | IC90 {estrito} de ±2 pp")

    # F3
    out.append("\nF3  Critérios de parada vs orçamento fixo de 40 trials (mesmas condições; Holm sobre a família)")
    princ = par[((par.criterio == "extrap") & np.isclose(par.eps, EPS)) |
                (par.criterio.isin(["preq", "cv"]) & np.isclose(par.eps, EPS) & np.isclose(par.pac, PAC)) |
                (par.criterio == "fixo_40")]
    ps_ = princ.groupby(["config", "cond", "criterio", "sujeito"])[["n_parada", "acc_parada", "n_oraculo"]].mean()
    linhas, ps = [], []
    for cfg in configs:
        for cond in ["intra", "transf", "adapt"]:
            fx = ps_.loc[(cfg, cond, "fixo_40")]
            for crit in CRITERIOS:
                c = ps_.loc[(cfg, cond, crit)]
                d = (c.acc_parada - fx.acc_parada).values
                dn = (c.n_parada - fx.n_parada).values
                m, lo, hi = ic(d)
                r, rlo, rhi = rho_boot(c.n_parada.values, c.n_oraculo.values)
                linhas.append((cfg, cond, crit, m, lo, hi, dn.mean(), r, rlo, rhi))
                ps.append(wil(d))
    for (cfg, cond, crit, m, lo, hi, dn, r, rlo, rhi), p, pa in zip(linhas, ps, holm(ps)):
        rt = "  ρ oráculo    n/d (oráculo constante)" if np.isnan(r) else f"  ρ oráculo {r:+.2f} [{rlo:+.2f}, {rhi:+.2f}]"
        out.append(f"  {cfg:<10} {cond:<6} {crit:<6} Δacc {pp(m)} [{pp(lo)}, {pp(hi)}]  Δtrials {dn:+5.1f}"
                   f"{rt}  p {p:.3f}  p_Holm {pa:.3f}")
    rhos = np.array([l[7] for l in linhas], float)
    rhos = rhos[np.isfinite(rhos)]
    if len(rhos):
        out.append(f"  → ρ com o oráculo: mediana {np.median(rhos):+.2f}; "
                   f"{int((np.abs(rhos) < 0.3).sum())}/{len(rhos)} combinações com |ρ| < 0,3")

    # F4
    if len(configs) > 1:
        out.append(f"\nF4  Poucos canais: transf com calibração completa vs {configs[0]} (TOST ±5 pp)")
        for cfg in configs[1:]:
            for cond in ["transf", "intra"]:
                d = (full.loc[(cfg, cond)] - full.loc[(configs[0], cond)]).values
                m, lo, hi = ic(d)
                p = tost(d, MARGEM)
                out.append(f"  {cfg:<10} {cond:<6} Δ {pp(m)} [{pp(lo)}, {pp(hi)}]  p_Wilcoxon {wil(d):.4f}  "
                           f"p_TOST(±5) {p:.4f} → {'EQUIVALENTE' if p < 0.05 else 'inconclusivo'}")

    # F5
    out.append("\nF5  Limite estatístico do sinal de parada (por sujeito, entre 32 e 48 trials)")
    for cfg in configs:
        for cond in ["intra", "transf"]:
            t = traj[(traj.config == cfg) & (traj.cond == cond)].sort_values(["sujeito", "perm", "n"])
            t = t.assign(dcv=t.groupby(["sujeito", "perm"]).cv.diff(),
                         dacc=t.groupby(["sujeito", "perm"]).acc_E.diff())
            j = t[(t.n >= 32) & (t.n <= 48)].dropna(subset=["dcv", "dacc"])
            ruido = j.groupby("sujeito").dcv.std().mean()          # oscilação do cv a cada +4 trials
            ganho = j.groupby("sujeito").dacc.mean().mean()        # ganho real médio a cada +4 trials
            p = full.loc[(cfg, cond)].mean()
            se40 = np.sqrt(p * (1 - p) / 40)
            out.append(f"  {cfg:<10} {cond:<6} ganho real/passo {pp(ganho)} pp | oscilação do cv/passo ±{100 * ruido:4.1f} pp"
                       f" | razão sinal/ruído {abs(ganho) / ruido:5.2f} | EP binomial com 40 trials ±{100 * se40:.1f} pp")
    out.append(f"  → um critério que precisa detectar ganhos de ~{100 * EPS:.0f} pp está olhando um sinal "
               f"dominado por ruído.")

    texto = "\n".join(out)
    print(texto + "\n")
    with open(f"{pref}_estatistica.txt", "w", encoding="utf-8") as f:
        f.write(texto + "\n")


if __name__ == "__main__":
    for pref in sys.argv[1:] or ["cron_2a", "cron_2b", "cron_lee"]:
        analisar(pref)
