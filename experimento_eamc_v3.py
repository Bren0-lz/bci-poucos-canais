"""
Experimento EAMC — v3: duas hipóteses novas + validação em dataset não visto

Pergunta: um critério de parada adaptativa encurta a calibração de uma BCI de imagética motora
com poucos canais, sem perda relevante de acurácia?

O que a v2 mostrou (BCI IV 2a): critérios de PLATÔ (preq/cv) param cedo demais sem transferência
e não superam um tempo fixo de 5 min com transferência. A v3 testa duas hipóteses derivadas disso:

  H1 — critério por EXTRAPOLAÇÃO da curva de aprendizado ("extrap"):
       ajusta acc(n) = a - b·n^(-1/2) à validação cruzada acumulada e para quando o ganho PREVISTO
       até o fim da calibração for < eps. Usa a tendência inteira, não as últimas oscilações.
  H2 — modelo ADAPTATIVO ("adapt"): a cada checkpoint escolhe, pela validação cruzada nos dados da
       calibração, entre o modelo com transferência e o modelo só do usuário. Hipótese: fica perto da
       melhor das duas curvas em qualquer tempo de calibração.

Protocolo anti-autoengano:
  - Parâmetros de H1/H2 fixados ANTES de rodar no dataset de teste (abaixo, seção PARÂMETROS).
  - DATASET="2a": desenvolvimento (já visto).  DATASET="2b": TESTE (BCI IV 2b, 9 sujeitos novos,
    3 canais nativos C3/Cz/C4 — mesma montagem do cenário de baixo custo). Rode o 2b uma única vez.

Colab:
  from google.colab import files; files.upload()      # este arquivo
  !pip install -q moabb pyriemann
  !python experimento_eamc_v3.py 2a
  !python experimento_eamc_v3.py 2b
Saídas (prefixo 2a_ ou 2b_): trajetorias.csv, paradas.csv, resumo_criterios.csv,
  numeros_para_o_resumo.txt, curvas_aprendizado.png/.pdf, parada_intra.png/.pdf, parada_adapt.png/.pdf
"""
import itertools
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import binom, spearmanr, wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ PARÂMETROS (fixados a priori)
_POS = [a for a in sys.argv[1:] if not a.startswith("--")]
_OPC = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
MODO = _POS[0] if _POS else "2a"                         # 2a | 2b | lee | lee-baixar
DATASET = "lee" if MODO.startswith("lee") else MODO
LEE_CACHE = _POS[1] if len(_POS) > 1 else "/content/drive/MyDrive/eamc_lee2019"
# Ordem de chegada dos trials de calibração do usuário novo:
#   aleatoria — sorteio de toda a sessão (versão original; mantida para comparação)
#   cron      — ordem real de gravação, uma vez por sujeito (resultado principal)
#   bloco     — bloco contínuo começando em ponto sorteado (protocolo de Wu, 2025; robustez)
ORDEM = _OPC.get("ordem", "aleatoria")
assert ORDEM in ("aleatoria", "cron", "bloco"), f"--ordem inválida: {ORDEM}"
PREFIXO = DATASET if ORDEM == "aleatoria" else f"{ORDEM}_{DATASET}"   # resultados antigos não são sobrescritos
FMIN, FMAX = 8, 30
TMIN, TMAX = 0.5, 2.5
SEG_POR_TRIAL = {"2a": 7.5, "2b": 7.5, "lee": 10.0}[DATASET]   # ~duração de um trial no protocolo

OITO = ["FC3", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4"]
CONFIGS_POR_DATASET = {
    "2a": {"22 canais": None, "8 canais": OITO, "3 canais": ["C3", "Cz", "C4"]},
    "2b": {"3 canais": None},  # o 2b só tem C3, Cz, C4
    # Lee2019: 62 canais seria lento demais com 54 sujeitos; usa os 20 canais motores
    "lee": {"20 canais": ["FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz",
                          "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"],
            "8 canais": OITO, "3 canais": ["C3", "Cz", "C4"]},
}
CONFIGS = CONFIGS_POR_DATASET[DATASET]
CONDICOES = ["intra", "transf"]          # "adapt" é derivada destas duas

N0, PASSO, JANELA_PREQ = 12, 4, 20
N_PERM = 1 if ORDEM == "cron" else (3 if DATASET == "lee" else 5)   # cron: só existe uma ordem real
MIN_POR_CLASSE = 2            # mínimo de trials por classe para o primeiro modelo (ordem real pode desbalancear o início)
EPS_PRINCIPAL, PAC_PRINCIPAL = 0.02, 3
GRADE_EPS, GRADE_PAC = [0.01, 0.02, 0.03], [3, 5]
FIXOS = [40, 80]
TOL_ORACULO = 0.02
EXP_CURVA = 0.5               # H1: expoente da lei de potência acc(n) = a - b·n^(-0.5)
MIN_PONTOS_EXTRAP = 4         # H1: checkpoints mínimos para ajustar a curva
CRITERIOS = ["preq", "cv", "extrap"]


# ------------------------------------------------------------------ dados
def carregar():
    import mne
    mne.set_log_level("ERROR")
    from moabb.datasets import BNCI2014_001, BNCI2014_004
    from moabb.paradigms import LeftRightImagery

    ds = {"2a": BNCI2014_001, "2b": BNCI2014_004}[DATASET]()
    par = LeftRightImagery(fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX)
    dados, nomes = {}, None
    for s in ds.subject_list:
        ep, y, meta = par.get_data(ds, subjects=[s], return_epochs=True)
        X = ep.get_data() * 1e6
        nomes = list(ep.ch_names)
        sessoes = sorted(meta.session.unique())
        assert len(sessoes) >= 2, f"sujeito {s}: sessões {sessoes}"
        yb = (np.asarray(y) == "right_hand").astype(int)
        # calibração = 1ª sessão; avaliação = última sessão (outro dia)
        dados[s] = {}
        for nome, sid in [("T", sessoes[0]), ("E", sessoes[-1])]:
            idx = np.where((meta.session == sid).values)[0]
            runs = meta.run.values[idx].astype(str)
            idx = idx[np.lexsort((idx, runs))]                # ordem cronológica: run, depois posição no run
            dados[s][nome] = (X[idx], yb[idx])
            if nome == "T":
                print(f"   sujeito {s}: runs da calibração em ordem {list(dict.fromkeys(runs[np.lexsort((np.arange(len(runs)), runs))]))}")
        print(f"sujeito {s}: sessões {sessoes} | T={len(dados[s]['T'][1])} E={len(dados[s]['E'][1])} | canais {nomes}")
    return dados, nomes


def chave(cfg):
    return cfg.replace(" ", "_")


def _apagar_brutos(s):
    import glob
    import os
    import mne
    raizes = {os.path.expanduser("~/mne_data"), mne.get_config("MNE_DATA") or ""}
    for raiz in filter(None, raizes):
        for f in glob.glob(f"{raiz}/**/*subj{s:02d}_EEG_MI*", recursive=True):
            os.remove(f)


def baixar_lee(cache=LEE_CACHE):
    """Baixa o Lee2019 um sujeito por vez, salva só as covariâncias (MB) e apaga o bruto (GB).
    Retomável: sujeitos já salvos em `cache` são pulados."""
    import os
    import time
    import mne
    mne.set_log_level("ERROR")
    from moabb.datasets import Lee2019_MI
    from moabb.paradigms import LeftRightImagery
    from pyriemann.estimation import Covariances

    os.makedirs(cache, exist_ok=True)
    ds = Lee2019_MI()                                         # fase de treino (offline), sessões 1 e 2
    par = LeftRightImagery(fmin=FMIN, fmax=FMAX, tmin=TMIN, tmax=TMAX, resample=250)
    est = Covariances(estimator="oas")
    for s in ds.subject_list:
        arq = f"{cache}/sujeito_{s:02d}.npz"
        if os.path.exists(arq):
            continue
        t = time.time()
        try:
            ep, y, meta = par.get_data(ds, subjects=[s], return_epochs=True)
        except Exception as erro:                            # um sujeito com problema não derruba o resto
            print(f"sujeito {s}: ERRO {erro!r} — pulando")
            continue
        X, nomes = ep.get_data() * 1e6, list(ep.ch_names)
        sessoes = sorted(meta.session.unique())
        yb = (np.asarray(y) == "right_hand").astype(int)
        saida = {}
        for nome, sid in [("T", sessoes[0]), ("E", sessoes[-1])]:
            m = (meta.session == sid).values
            saida[f"y_{nome}"] = yb[m]
            for cfg, canais in CONFIGS.items():
                idx = [nomes.index(c) for c in canais]
                saida[f"C_{nome}_{chave(cfg)}"] = est.fit_transform(X[m][:, idx, :])
        np.savez_compressed(arq, **saida)
        _apagar_brutos(s)
        print(f"sujeito {s}: sessões {sessoes}, T={len(saida['y_T'])} E={len(saida['y_E'])} trials "
              f"— {time.time() - t:.0f} s")


def carregar_cache(cache=LEE_CACHE):
    import glob
    arqs = sorted(glob.glob(f"{cache}/sujeito_*.npz"))
    assert arqs, f"nenhum sujeito em {cache} — rode antes: python experimento_eamc_v3.py lee-baixar"
    covs = {cfg: {} for cfg in CONFIGS}
    for a in arqs:
        s = int(a.split("_")[-1].split(".")[0])
        z = np.load(a)
        for cfg in CONFIGS:
            covs[cfg][s] = {ses: (z[f"C_{ses}_{chave(cfg)}"], z[f"y_{ses}"]) for ses in "TE"}
    print(f"{len(arqs)} sujeitos carregados do cache")
    return covs


def preparar(dados, nomes):
    from pyriemann.estimation import Covariances
    est = Covariances(estimator="oas")
    covs = {}
    for cfg, canais in CONFIGS.items():
        if canais is None:
            idx = list(range(len(nomes)))
        else:
            faltam = [c for c in canais if c not in nomes]
            assert not faltam, f"canais ausentes: {faltam} (disponíveis: {nomes})"
            idx = [nomes.index(c) for c in canais]
        covs[cfg] = {s: {ses: (est.fit_transform(dados[s][ses][0][:, idx, :]), dados[s][ses][1]) for ses in "TE"}
                     for s in dados}
    return covs


# ------------------------------------------------------------------ EA + CSP + LDA
def raiz_inv(R):
    w, V = np.linalg.eigh(R)
    return (V * w ** -0.5) @ V.T


def alinhar(C, R):
    A = raiz_inv(R)
    return A @ C @ A


def atributos(C, W):
    v = np.einsum("ij,nij->nj", W, np.matmul(C, W))          # diag(Wᵀ C W) via BLAS
    return np.log(v / v.sum(1, keepdims=True))


def treinar(C, y):
    S0, S1 = C[y == 0].mean(0), C[y == 1].mean(0)
    _, V = eigh(S0, S0 + S1)
    n_ch = C.shape[1]
    k = min(6, n_ch - n_ch % 2) // 2
    W = np.concatenate([V[:, :k], V[:, -k:]], axis=1)
    return W, LDA(solver="lsqr", shrinkage="auto").fit(atributos(C, W), y)


def prever(m, C):
    return m[1].predict(atributos(C, m[0]))


def conjunto_treino(cond, fonte, Ccal, ycal):
    if cond == "intra":
        return Ccal, ycal
    Cs, ys, n_por_fonte = fonte
    rep = max(1, round(n_por_fonte / len(ycal)))
    return np.concatenate([Cs, np.repeat(Ccal, rep, 0)]), np.concatenate([ys, np.repeat(ycal, rep)])


def validacao_cruzada(cond, fonte, C, y):
    k = min(5, np.bincount(y, minlength=2).min())
    ac = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(C, y):
        m = treinar(*conjunto_treino(cond, fonte, C[tr], y[tr]))
        ac.append((prever(m, C[te]) == y[te]).mean())
    return float(np.mean(ac))


# ------------------------------------------------------------------ simulação
def sortear_ordem(y, rng):
    while True:
        o = rng.permutation(len(y))
        if np.bincount(y[o[:N0]], minlength=2).min() >= 3:
            return o


def gerar_ordem(y, rng):
    n = len(y)
    if ORDEM == "aleatoria":
        return sortear_ordem(y, rng)
    if ORDEM == "cron":
        return np.arange(n)
    inicio = int(rng.integers(n))                          # bloco: contínuo a partir de um ponto sorteado
    return np.roll(np.arange(n), -inicio)


def inicio_valido(y, ordem):
    """Primeiro checkpoint da grade (N0, N0+PASSO, ...) com pelo menos MIN_POR_CLASSE trials de cada classe."""
    for n in range(N0, len(ordem) + 1, PASSO):
        if np.bincount(y[ordem[:n]], minlength=2).min() >= MIN_POR_CLASSE:
            return n
    raise ValueError("calibração sem trials suficientes de uma das classes")


def trajetoria(cond, CT, yT, CE, yE, fonte, ordem):
    pontos = list(range(inicio_valido(yT, ordem), len(ordem) + 1, PASSO))
    if pontos[-1] != len(ordem):
        pontos.append(len(ordem))
    online, linhas = [], []
    for i, n in enumerate(pontos):
        idx = ordem[:n]
        Ccal, ycal = CT[idx], yT[idx]
        R = None
        if cond == "transf":
            R = Ccal.mean(0)
            Ccal, CE_ = alinhar(Ccal, R), alinhar(CE, R)
        else:
            CE_ = CE
        m = treinar(*conjunto_treino(cond, fonte, Ccal, ycal))
        linhas.append(dict(n=n, acc_E=(prever(m, CE_) == yE).mean(),
                           cv=validacao_cruzada(cond, fonte, Ccal, ycal),
                           preq=np.mean(online[-JANELA_PREQ:]) if len(online) >= JANELA_PREQ // 2 else np.nan))
        if i + 1 < len(pontos):
            prox = ordem[n:pontos[i + 1]]
            Cp = alinhar(CT[prox], R) if R is not None else CT[prox]
            online.extend((prever(m, Cp) == yT[prox]).astype(int))
    return linhas


def rodar(covs, pasta_progresso=None):
    """Salva cada (montagem, sujeito) em disco: se o Colab cair, a próxima execução retoma."""
    import os
    if pasta_progresso is None:   # no Lee, guarda o progresso no Drive para sobreviver a quedas do Colab
        sufixo = "" if ORDEM == "aleatoria" else f"_{ORDEM}"
        pasta_progresso = f"{LEE_CACHE}/progresso{sufixo}" if DATASET == "lee" else f"{PREFIXO}_progresso"
    os.makedirs(pasta_progresso, exist_ok=True)
    partes = []
    for cfg in CONFIGS:
        D = covs[cfg]
        sujeitos = sorted(D)
        alinhadas = {s: {ses: alinhar(D[s][ses][0], D[s][ses][0].mean(0)) for ses in "TE"} for s in sujeitos}
        for alvo in sujeitos:
            arq = f"{pasta_progresso}/{chave(cfg)}_s{alvo:02d}.csv"
            if os.path.exists(arq):
                partes.append(pd.read_csv(arq))
                continue
            CT, yT = D[alvo]["T"]
            CE, yE = D[alvo]["E"]
            fontes = [s for s in sujeitos if s != alvo]
            Cs = np.concatenate([alinhadas[s][ses] for s in fontes for ses in "TE"])
            ys = np.concatenate([D[s][ses][1] for s in fontes for ses in "TE"])
            fonte = (Cs, ys, len(ys) // len(fontes))
            linhas = []
            for perm in range(N_PERM):
                ordem = gerar_ordem(yT, np.random.default_rng(1000 * alvo + perm))
                for cond in CONDICOES:
                    for l in trajetoria(cond, CT, yT, CE, yE, fonte, ordem):
                        l.update(config=cfg, sujeito=alvo, cond=cond, perm=perm)
                        linhas.append(l)
            parte = pd.DataFrame(linhas)
            parte.to_csv(arq, index=False)
            partes.append(parte)
            print(f"[{DATASET} | {cfg}] sujeito {alvo} ok")
    return pd.concat(partes, ignore_index=True)


def adicionar_adapt(traj):
    """H2: em cada checkpoint, usa o modelo (intra ou transf) com maior validação cruzada na calibração."""
    chaves = ["config", "sujeito", "perm", "n"]
    w = traj.pivot_table(index=chaves, columns="cond", values=["acc_E", "cv", "preq"]).reset_index()
    usa_transf = (w[("cv", "transf")] >= w[("cv", "intra")]).values
    pega = lambda col: np.where(usa_transf, w[(col, "transf")], w[(col, "intra")])
    ad = pd.DataFrame({k: w[(k, "")] for k in chaves})
    ad["acc_E"], ad["preq"] = pega("acc_E"), pega("preq")
    ad["cv"] = np.maximum(w[("cv", "transf")], w[("cv", "intra")])
    ad["cond"], ad["usa_transf"] = "adapt", usa_transf.astype(int)
    return pd.concat([traj, ad], ignore_index=True)


# ------------------------------------------------------------------ critérios de parada
def parada_plato(serie, eps, pac):
    for i in range(pac, len(serie)):
        if np.isnan(serie[i - pac]):
            continue
        if np.nanmax(serie[i - pac + 1: i + 1]) - np.nanmax(serie[: i - pac + 1]) < eps:
            return i
    return len(serie) - 1


def parada_extrap(ns, serie, n_full, eps):
    """H1: ganho previsto até o fim = b·(n^-c − N^-c), com a, b ajustados por mínimos quadrados."""
    x_all = ns.astype(float) ** -EXP_CURVA
    for i in range(MIN_PONTOS_EXTRAP - 1, len(serie)):
        x, y = x_all[: i + 1], serie[: i + 1]
        ok = np.isfinite(y)
        if ok.sum() < MIN_PONTOS_EXTRAP:
            continue
        inclinacao = np.polyfit(x[ok], y[ok], 1)[0]           # acc = a + inclinacao·x  (b = −inclinacao)
        ganho = -inclinacao * (x_all[i] - n_full ** -EXP_CURVA)
        if ganho < eps:
            return i
    return len(serie) - 1


def aplicar_criterios(traj):
    linhas = []
    for (cfg, s, cond, perm), g in traj.groupby(["config", "sujeito", "cond", "perm"]):
        g = g.sort_values("n")
        n, acc = g.n.values, g.acc_E.values
        base = dict(config=cfg, sujeito=s, cond=cond, perm=perm, n_full=n[-1], acc_full=acc[-1],
                    n_oraculo=n[np.argmax(acc >= acc[-1] - TOL_ORACULO)])
        for crit in CRITERIOS:
            for eps in GRADE_EPS:
                for pac in (GRADE_PAC if crit != "extrap" else [np.nan]):
                    if crit == "extrap":
                        i = parada_extrap(n, g.cv.values, n[-1], eps)
                    else:
                        i = parada_plato(g[crit].values, eps, int(pac))
                    linhas.append({**base, "criterio": crit, "eps": eps, "pac": pac,
                                   "n_parada": n[i], "acc_parada": acc[i]})
        for nf in FIXOS:
            i = min(int(np.searchsorted(n, nf)), len(n) - 1)
            linhas.append({**base, "criterio": f"fixo_{nf}", "eps": np.nan, "pac": np.nan,
                           "n_parada": n[i], "acc_parada": acc[i]})
    return pd.DataFrame(linhas)


def eh_principal(df):
    return ((df.criterio.str.startswith("fixo")) |
            ((df.criterio == "extrap") & (df.eps == EPS_PRINCIPAL)) |
            (df.criterio.isin(["preq", "cv"]) & (df.eps == EPS_PRINCIPAL) & (df.pac == PAC_PRINCIPAL)))


# ------------------------------------------------------------------ análise
def por_sujeito(par):
    chaves = ["config", "cond", "criterio", "eps", "pac", "sujeito"]
    ps = par.groupby(chaves, dropna=False)[["n_parada", "acc_parada", "acc_full", "n_oraculo", "n_full"]] \
        .mean().reset_index()
    ps["delta"] = ps.acc_parada - ps.acc_full
    ps["min_parada"] = ps.n_parada * SEG_POR_TRIAL / 60
    ps["economia"] = 1 - ps.n_parada / ps.n_full
    return ps


def resumir(ps):
    def rho(g):
        if g.n_parada.nunique() < 2 or g.n_oraculo.nunique() < 2:
            return np.nan
        return spearmanr(g.n_parada, g.n_oraculo).correlation

    def p_wil(g):
        return wilcoxon(g.acc_parada, g.acc_full).pvalue if (g.delta != 0).any() else 1.0

    linhas = []
    for chave, g in ps.groupby(["config", "cond", "criterio", "eps", "pac"], dropna=False):
        linhas.append(dict(zip(["config", "cond", "criterio", "eps", "pac"], chave),
                           min_medio=g.min_parada.mean(), min_min=g.min_parada.min(), min_max=g.min_parada.max(),
                           economia_pct=100 * g.economia.mean(), acc_parada=g.acc_parada.mean(),
                           acc_full=g.acc_full.mean(), delta_medio=g.delta.mean(), delta_pior=g.delta.min(),
                           n_suj_perda_5pp=int((g.delta < -0.05).sum()), spearman_oraculo=rho(g),
                           p_wilcoxon=p_wil(g)))
    r = pd.DataFrame(linhas)
    r.to_csv(f"{PREFIXO}_resumo_criterios.csv", index=False)
    return r


def so_n_comuns(ms):
    """Mantém só os checkpoints presentes em TODOS os sujeitos (evita médias com sujeitos faltando
    no início — ordem real — ou no fim — sessões de calibração mais longas, como o sujeito 8 do 2b)."""
    nsuj = ms.groupby(["config", "cond"]).sujeito.transform("nunique")
    cont = ms.groupby(["config", "cond", "n"]).sujeito.transform("nunique")
    return ms[cont == nsuj]


def texto_resumo(traj, r, n_teste):
    limiar = binom.ppf(0.95, n_teste, 0.5) / n_teste
    out = [f"DATASET {DATASET} | ordem dos trials de calibração: {ORDEM} | sujeitos: {traj.sujeito.nunique()} | trials de teste: {n_teste} | "
           f"limiar de acaso (p<0,05): {limiar:.3f}",
           f"Critérios principais a priori: preq/cv eps={EPS_PRINCIPAL} pac={PAC_PRINCIPAL}; extrap eps={EPS_PRINCIPAL}",
           f"Wilcoxon pareado por sujeito (n={traj.sujeito.nunique()})\n"]

    # curvas: acurácia média por tempo de calibração (para comparar intra, transf e adapt)
    ms_todos = traj.groupby(["config", "cond", "sujeito", "n"]).acc_E.mean().reset_index()
    ms = so_n_comuns(ms_todos)
    for cfg in CONFIGS:
        out.append(f"##### {cfg} — acurácia média na sessão de avaliação por tempo de calibração")
        for alvo_min in [1.5, 2.5, 5, 10]:
            n_alvo = int(round(alvo_min * 60 / SEG_POR_TRIAL))
            lin = []
            for cond in ["intra", "transf", "adapt"]:
                s = ms[(ms.config == cfg) & (ms.cond == cond)]
                nn = s.n.unique()
                n_usado = nn[np.argmin(abs(nn - n_alvo))]
                lin.append(f"{cond} {s[s.n == n_usado].acc_E.mean():.3f}")
            out.append(f"  ~{alvo_min:>4} min: " + " | ".join(lin))
        lin = []
        for cond in ["intra", "transf", "adapt"]:
            s = ms_todos[(ms_todos.config == cfg) & (ms_todos.cond == cond)]
            lin.append(f"{cond} {s.sort_values('n').groupby('sujeito').acc_E.last().mean():.3f}")   # último ponto de cada sujeito
        out.append("  completa: " + " | ".join(lin))
        frac = traj[(traj.config == cfg) & (traj.cond == "adapt")].groupby("n").usa_transf.mean()
        trocas = frac[frac < 0.5]
        out.append(f"  adapt: fração usando transferência no início {frac.iloc[0]:.2f}, no fim {frac.iloc[-1]:.2f}"
                   + (f"; maioria passa a usar só o usuário a partir de ~{trocas.index[0] * SEG_POR_TRIAL / 60:.1f} min"
                      if len(trocas) else "; maioria usa transferência até o fim"))
        out.append("")

    rp = r[eh_principal(r)]
    for cfg in CONFIGS:
        for cond in ["intra", "transf", "adapt"]:
            sub = rp[(rp.config == cfg) & (rp.cond == cond)]
            out.append(f"=== {cfg} | {cond} | acurácia com calibração completa: {sub.acc_full.iloc[0]:.3f}")
            ordem = [f"fixo_{nf}" for nf in FIXOS] + CRITERIOS
            for crit in ordem:
                l = sub[sub.criterio == crit].iloc[0]
                out.append(f"  {crit:<8} ~{l.min_medio:4.1f} min ({l.min_min:.1f}–{l.min_max:.1f}) | "
                           f"economia {l.economia_pct:3.0f}% | acc {l.acc_parada:.3f} | Δ {l.delta_medio:+.3f} | "
                           f"pior {l.delta_pior:+.3f} | perda>5pp: {l.n_suj_perda_5pp} | "
                           f"ρ oráculo {l.spearman_oraculo:+.2f} | p {l.p_wilcoxon:.4f}")
            out.append("")
    texto = "\n".join(out)
    print("\n" + texto)
    with open(f"{PREFIXO}_numeros_para_o_resumo.txt", "w", encoding="utf-8") as f:
        f.write(texto)
    return limiar


# ------------------------------------------------------------------ figuras
ESTILO = {"intra": ("Só o usuário (intra)", "tab:blue"),
          "transf": ("Transferência + EA", "tab:red"),
          "adapt": ("Adaptativo (H2)", "tab:green")}


def plotar_curvas(traj, limiar):
    ms = so_n_comuns(traj.groupby(["config", "cond", "sujeito", "n"]).acc_E.mean().reset_index())
    ag = ms.groupby(["config", "cond", "n"]).acc_E.agg(["mean", "std"]).reset_index()
    fig, axs = plt.subplots(1, len(CONFIGS), figsize=(3.8 * len(CONFIGS), 3.6), sharey=True, squeeze=False)
    for ax, cfg in zip(axs[0], CONFIGS):
        for cond, (rot, cor) in ESTILO.items():
            s = ag[(ag.config == cfg) & (ag.cond == cond)]
            x = s.n * SEG_POR_TRIAL / 60
            ax.plot(x, s["mean"], color=cor, label=rot, lw=2 if cond == "adapt" else 1.3)
            if cond != "adapt":
                ax.fill_between(x, s["mean"] - s["std"], s["mean"] + s["std"], color=cor, alpha=0.10)
        ax.axhline(limiar, ls=":", color="black", lw=0.8, label="Limiar de acaso (p<0,05)")
        ax.set_title(f"{DATASET} — {cfg}" + ("" if ORDEM == "aleatoria" else f" [{ORDEM}]"))
        ax.set_xlabel("Minutos de calibração")
        ax.grid(alpha=0.3)
    axs[0][0].set_ylabel("Acurácia na sessão de avaliação")
    axs[0][0].set_ylim(0.45, 1.0)
    axs[0][-1].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{PREFIXO}_curvas_aprendizado.png", dpi=300)
    fig.savefig(f"{PREFIXO}_curvas_aprendizado.pdf")


def plotar_paradas(ps, cond):
    marc = {"preq": ("o", "tab:purple"), "cv": ("s", "tab:orange"), "extrap": ("D", "tab:green")}
    pp = ps[eh_principal(ps)]
    fig, axs = plt.subplots(1, len(CONFIGS), figsize=(3.8 * len(CONFIGS), 3.6), sharey=True, squeeze=False)
    for ax, cfg in zip(axs[0], CONFIGS):
        g = pp[(pp.config == cfg) & (pp.cond == cond)]
        ref = g[g.criterio == "extrap"].sort_values("sujeito")
        x = np.arange(len(ref))
        ax.scatter(x, ref.n_oraculo * SEG_POR_TRIAL / 60, marker="x", color="black", label="Oráculo (gabarito)")
        for j, (crit, (mk, cor)) in enumerate(marc.items()):
            s = g[g.criterio == crit].sort_values("sujeito")
            ax.scatter(x + (j - 1) * 0.18, s.min_parada, marker=mk, color=cor, s=22, label=f"Parada: {crit}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s}" for s in ref.sujeito])
        ax.set_title(f"{DATASET} — {cfg} ({cond})")
        ax.grid(alpha=0.3)
    axs[0][0].set_ylabel("Minutos de calibração")
    axs[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{PREFIXO}_parada_{cond}.png", dpi=300)
    fig.savefig(f"{PREFIXO}_parada_{cond}.pdf")


# ------------------------------------------------------------------ main
def main(dados=None, nomes=None, covs=None):
    if covs is None:
        if DATASET == "lee":
            covs = carregar_cache()
        else:
            if dados is None:
                dados, nomes = carregar()
            covs = preparar(dados, nomes)
    traj = adicionar_adapt(rodar(covs))
    traj.to_csv(f"{PREFIXO}_trajetorias.csv", index=False)
    par = aplicar_criterios(traj)
    par.to_csv(f"{PREFIXO}_paradas.csv", index=False)
    ps = por_sujeito(par)
    r = resumir(ps)
    primeiro = next(iter(covs.values()))
    n_teste = int(np.median([len(v["E"][1]) for v in primeiro.values()]))
    limiar = texto_resumo(traj, r, n_teste)
    plotar_curvas(traj, limiar)
    plotar_paradas(ps, "intra")
    plotar_paradas(ps, "adapt")
    print(f"Pronto. Arquivos com prefixo {PREFIXO}_")


if __name__ == "__main__":
    if MODO == "lee-baixar":
        baixar_lee()
    else:
        main()
