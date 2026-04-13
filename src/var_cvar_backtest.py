import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock
from scipy.stats import norm, t, binom, chi2
from arch import arch_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 결과 저장할 폴더 생성
os.makedirs("results_png", exist_ok=True)
os.makedirs("results_txt", exist_ok=True)

# 분석 대상 ticker 목록
tickers_yf = ['ACWI', 'ACWX', 'EFA', 'VWO', 'BNDX', 'EMB', 'IEF', 'TLT', 'USO', 'VNQ']
tickers_kr = ['069500', '229200', '114260', '148070']

# 날짜 및 공통 파라미터 설정
start_date = "2015-01-01"
end_date   = "2022-12-31"
window     = 250
alpha_var  = 0.01
alpha_cvar = 0.025


def _krx_date(date_str):
    return date_str.replace("-", "")


def fetch_krx_close(ticker, start_date, end_date):
    start = _krx_date(start_date)
    end = _krx_date(end_date)
    df = stock.get_etf_ohlcv_by_date(start, end, ticker)
    if df.empty:
        df = stock.get_market_ohlcv_by_date(start, end, ticker)
    if df.empty:
        raise ValueError(f"No KRX price data returned for {ticker} from {start_date} to {end_date}")

    for close_col in ["종가", "Close", "close", "Adj Close"]:
        if close_col in df.columns:
            return df[[close_col]].rename(columns={close_col: "Close"}).dropna()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        fallback_col = numeric_cols[-1]
        return df[[fallback_col]].rename(columns={fallback_col: "Close"}).dropna()

    raise KeyError(f"Could not find a close-price column for {ticker}. Columns: {list(df.columns)}")


# ✅ VaR 및 CVaR 백테스트 결과 저장을 위한 초기화
summary_var_results = {model: [] for model in ["VaR_Hist", "VaR_Param_n", "VaR_Param_t", "VaR_GARCH_t", "VaR_GARCH_n"]}
summary_var_results.update({model + "_LR": [] for model in ["VaR_Hist", "VaR_Param_n", "VaR_Param_t", "VaR_GARCH_t", "VaR_GARCH_n"]})

summary_cvar_results = {model: {"down": [], "up": []} for model in ["CVaR_Hist", "CVaR_Param_n", "CVaR_Param_t", "CVaR_GARCH_t", "CVaR_GARCH_n"]}

for ticker in tickers_yf + tickers_kr:
    print(f"\n=== {ticker} 분석 시작 ===")
    if ticker in tickers_yf:
        raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
        df = raw[["Close"]].dropna()
    else:
        df = fetch_krx_close(ticker, start_date, end_date)
    df.rename(columns={"Close": "Close"}, inplace=True)
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)

    n = len(df)
    dates = df.index.to_numpy()
    returns = df["Return"].to_numpy()

    # 결과 초기화
    var_hist = np.full(n, np.nan)
    cvar_hist = np.full(n, np.nan)
    var_param = np.full(n, np.nan)
    cvar_param = np.full(n, np.nan)
    var_param_t_all = np.full(n, np.nan)
    cvar_param_t_all = np.full(n, np.nan)
    var_garch = np.full(n, np.nan)
    cvar_garch = np.full(n, np.nan)
    var_garch_norm_all = np.full(n, np.nan)
    cvar_garch_norm_all = np.full(n, np.nan)

    for i in range(window, n):
        window_returns = returns[i-window:i]
        mu = window_returns.mean()
        sigma = window_returns.std(ddof=1)

        var_hist[i] = np.quantile(window_returns, alpha_var)
        cvar_hist[i] = window_returns[window_returns <= var_hist[i]].mean()

        z_var = norm.ppf(alpha_var)
        var_param[i] = mu + sigma * z_var
        cvar_param[i] = mu - sigma * (norm.pdf(z_var) / alpha_cvar)

        t_df = 5
        t_quant = t.ppf(alpha_var, df=t_df)
        var_param_t_all[i] = mu + sigma * t_quant
        t_pdf = t.pdf(t_quant, df=t_df)
        cvar_param_t_all[i] = mu - sigma * ((t_df + t_quant**2) / (t_df - 1)) * (t_pdf / alpha_cvar)

        am_norm = arch_model(window_returns * 100, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res_norm = am_norm.fit(disp="off")
        sigma2_fcast_n = res_norm.forecast(horizon=1, reindex=False).variance.values[-1, 0] / 10000
        mu_gn = res_norm.params["mu"]
        sigma_gn = np.sqrt(sigma2_fcast_n)
        z_g = norm.ppf(alpha_var)
        var_garch_norm_all[i] = mu_gn / 100 + sigma_gn * z_g
        cvar_garch_norm_all[i] = mu_gn / 100 - sigma_gn * (norm.pdf(z_g) / alpha_cvar)

        am = arch_model(window_returns * 100, vol="Garch", p=1, q=1, dist="t", rescale=False)
        res = am.fit(disp="off")
        sigma2_fcast = res.forecast(horizon=1, reindex=False).variance.values[-1, 0] / 10000
        nu = res.params["nu"]
        mu_g = res.params["mu"]
        sigma_g = np.sqrt(sigma2_fcast)
        t_quant = t.ppf(alpha_var, df=nu)
        var_garch[i] = mu_g / 100 + sigma_g * t_quant
        pdf_t = t.pdf(t_quant, df=nu)
        cvar_garch[i] = mu_g / 100 - sigma_g * ((nu + t_quant**2)/(nu - 1)) * (pdf_t / alpha_cvar)

    result = pd.DataFrame({
        "Close": df["Close"].to_numpy().ravel(),
        "Return": df["Return"].to_numpy().ravel(),
        "VaR_Hist": var_hist.ravel(),
        "CVaR_Hist": cvar_hist.ravel(),
        "VaR_Param_n": var_param.ravel(),
        "CVaR_Param_n": cvar_param.ravel(),
        "VaR_Param_t": var_param_t_all.ravel(),
        "CVaR_Param_t": cvar_param_t_all.ravel(),
        "VaR_GARCH_t": var_garch.ravel(),
        "CVaR_GARCH_t": cvar_garch.ravel(),
        "VaR_GARCH_n": var_garch_norm_all.ravel(),
        "CVaR_GARCH_n": cvar_garch_norm_all.ravel()
    }, index=dates)

    bt = result.dropna().copy()
    R   = bt["Return"].values
    T   = len(R)
    VH, CH = bt["VaR_Hist"].values, bt["CVaR_Hist"].values
    VP, CP = bt["VaR_Param_n"].values, bt["CVaR_Param_n"].values
    VPT, CPT = bt["VaR_Param_t"].values, bt["CVaR_Param_t"].values
    VG, CG = bt["VaR_GARCH_t"].values, bt["CVaR_GARCH_t"].values
    VGN, CGN = bt["VaR_GARCH_n"].values, bt["CVaR_GARCH_n"].values

    def var_backtest(returns, var_series, alpha):
        violations = (returns < var_series).astype(int)
        x = violations.sum()
        N = len(violations)
        green_cut = binom.ppf(0.95, N, alpha)
        red_cut   = binom.ppf(0.9999, N, alpha)
        if x <= green_cut: zone = "GREEN"
        elif x <= red_cut: zone = "YELLOW"
        else: zone = "RED"
        N00 = N01 = N10 = N11 = 0
        for t in range(1, N):
            prev, curr = violations[t-1], violations[t]
            if prev == 0 and curr == 0: N00 += 1
            elif prev == 0 and curr == 1: N01 += 1
            elif prev == 1 and curr == 0: N10 += 1
            else: N11 += 1
        p01 = N01 / (N00 + N01) if (N00 + N01) > 0 else 0
        p11 = N11 / (N10 + N11) if (N10 + N11) > 0 else 0
        pUC = (N01 + N11) / N
        eps = 1e-10
        safe_log = lambda x: np.log(x + eps)
        L0 = ((1-pUC)**N00)*(pUC**N01)*((1-pUC)**N10)*(pUC**N11)
        L1 = ((1-p01)**N00)*(p01**N01)*((1-p11)**N10)*(p11**N11)
        LR_CC = -2 * (safe_log(L0) - safe_log(L1))
        p_value_cc = 1 - chi2.cdf(LR_CC, df=1)
        return {"violations": int(x), "zone": zone, "LR_CC": LR_CC, "p_value_CC": p_value_cc}

    def cvar_dual_test(returns, var_series, cvar_series, alpha):
        def test(side):
            if side == 'down':
                v = (returns < var_series).astype(int)
            else:
                v = (returns > var_series).astype(int)
            W = v * (returns - cvar_series) / alpha
            W_clean = W[~np.isnan(W) & ~np.isinf(W)]
            if len(W_clean) == 0:
                return {"Z": np.nan, "p_value": np.nan, "pass": False}
            Z = W_clean.mean() / (W_clean.std(ddof=1) / np.sqrt(len(W_clean)))
            return {"Z": Z, "p_value": 1 - norm.cdf(abs(Z)), "pass": abs(Z) <= 1.96}
        return {"down": test("down"), "up": test("up")}

    bt_results = {}
    for name, var_s in [("VaR_Hist", VH), ("VaR_Param_n", VP), ("VaR_Param_t", VPT), ("VaR_GARCH_t", VG), ("VaR_GARCH_n", VGN)]:
        bt_results[name] = var_backtest(R, var_s, alpha_var)

    for name, (var_s, cvar_s) in [("CVaR_Hist", (VH, CH)), ("CVaR_Param_n", (VP, CP)), ("CVaR_Param_t", (VPT, CPT)), ("CVaR_GARCH_t", (VG, CG)), ("CVaR_GARCH_n", (VGN, CGN))]:
        bt_results[name] = cvar_dual_test(R, var_s, cvar_s, alpha_cvar)

    print(f"\n===== VaR Backtest Results for {ticker} =====")
    for model in ["VaR_Hist", "VaR_Param_n", "VaR_Param_t", "VaR_GARCH_t", "VaR_GARCH_n"]:
        r = bt_results[model]
        print(f"\n[{model}]\n  Violations: {r['violations']} out of {T} (α={alpha_var})")
        print(f"  Zone: {r['zone']}")
        print(f"  LR_CC: {r['LR_CC']:.4f}, p: {r['p_value_CC']:.4f}")

    print(f"\n===== CVaR Backtest Results for {ticker} =====")
    print("\nModel            | Downside Z  | p-val | Pass | Upside Z   | p-val | Pass")
    print("----------------|-------------|-------|------|------------|-------|------")
    for model in ["CVaR_Hist", "CVaR_Param_n", "CVaR_Param_t", "CVaR_GARCH_t", "CVaR_GARCH_n"]:
        r = bt_results[model]
        d, u = r['down'], r['up']
        print(f"{model:<16} | {d['Z']:+.4f}     | {d['p_value']:.4f} | {str(d['pass']):<4} | " f"{u['Z']:+.4f}    | {u['p_value']:.4f} | {str(u['pass']):<4}")

    with open(f"results_txt/{ticker}_backtest.txt", "w", encoding="utf-8") as f:
        f.write("===== VaR Backtest Results =====\n")
        for model in ["VaR_Hist", "VaR_Param_n", "VaR_Param_t", "VaR_GARCH_t", "VaR_GARCH_n"]:
            r = bt_results[model]
            f.write(f"\n[{model}]\n  Violations: {r['violations']} out of {T} (α={alpha_var})\n")
            f.write(f"  Zone: {r['zone']}\n")
            f.write(f"  LR_CC: {r['LR_CC']:.4f}, p: {r['p_value_CC']:.4f}\n")
        f.write("\n===== CVaR Backtest Results =====\n")
        f.write("\nModel            | Downside Z  | p-val | Pass | Upside Z   | p-val | Pass\n")
        f.write("----------------|-------------|-------|------|------------|-------|------\n")
        for model in ["CVaR_Hist", "CVaR_Param_n", "CVaR_Param_t", "CVaR_GARCH_t", "CVaR_GARCH_n"]:
            r = bt_results[model]
            d, u = r['down'], r['up']
            f.write(f"{model:<16} | {d['Z']:+.4f}     | {d['p_value']:.4f} | {str(d['pass']):<4} | " f"{u['Z']:+.4f}    | {u['p_value']:.4f} | {str(u['pass']):<4}\n")
    # ✅ 모델별로 결과 누적
    for model in ["VaR_Hist", "VaR_Param_n", "VaR_Param_t", "VaR_GARCH_t", "VaR_GARCH_n"]:
        summary_var_results[model].append(bt_results[model]["violations"])
        summary_var_results[model + "_LR"].append(bt_results[model]["LR_CC"])

    for model in ["CVaR_Hist", "CVaR_Param_n", "CVaR_Param_t", "CVaR_GARCH_t", "CVaR_GARCH_n"]:
        summary_cvar_results[model]["down"].append(bt_results[model]["down"]["pass"])
        summary_cvar_results[model]["up"].append(bt_results[model]["up"]["pass"])



# ===== 최종 요약 Table 3 생성 =====
import pandas as pd
output_dir = "results_txt"
os.makedirs(output_dir, exist_ok=True)

critical_val = 3.841  # chi2(1) at 95%
summary_var_rows = []
for model in ["VaR_Hist", "VaR_Param_n", "VaR_Param_t", "VaR_GARCH_t", "VaR_GARCH_n"]:
    vios = np.array(summary_var_results[model])
    lrs = np.array(summary_var_results[model + "_LR"])
    AE = np.mean(lrs - critical_val)
    zones = []
    for v in vios:
        if v <= binom.ppf(0.95, T, alpha_var):
            zones.append("GREEN")
        elif v <= binom.ppf(0.9999, T, alpha_var):
            zones.append("YELLOW")
        else:
            zones.append("RED")
    g = zones.count("GREEN")
    y = zones.count("YELLOW")
    r = zones.count("RED")
    summary_var_rows.append([model, np.mean(vios), AE, g, y, r])

df_var = pd.DataFrame(summary_var_rows, columns=["Model", "Mean_Violations", "AE_LR_CC", "GREEN", "YELLOW", "RED"])
df_var.to_csv(os.path.join(output_dir, "table3_var_summary.csv"), index=False)

summary_cvar_rows = []
for model in summary_cvar_results:
    down = summary_cvar_results[model]["down"]
    up = summary_cvar_results[model]["up"]
    summary_cvar_rows.append([model, sum(down), sum(up)])

df_cvar = pd.DataFrame(summary_cvar_rows, columns=["Model", "Pass_Down", "Pass_Up"])
df_cvar.to_csv(os.path.join(output_dir, "table3_cvar_summary.csv"), index=False)
