# CVaR downside, CVaR upside 횟수 출력하는 코드 추가
import os
import numpy as np
import pandas as pd
import yfinance as yf
from pykrx import stock
from scipy.stats import t

# 폴더 생성
os.makedirs("results_csv", exist_ok=True)

# 사용자 설정
tickers_yf = ['ACWI', 'ACWX', 'EFA', 'VWO', 'BNDX', 'EMB', 'IEF', 'TLT', 'USO', 'VNQ']       # Yahoo Finance 티커 (필요시 수정)
tickers_kr = ['069500', '229200', '114260', '148070']  # KRX 티커 (필요시 수정)

start_date = "2019-02-14"  # 원하는 기간 시작일 (YYYY-MM-DD), rolling window 때문에 약 1년 전부터 데이터를 다운받아야 함
end_date   = "2020-04-07"  # 원하는 기간 종료일 (YYYY-MM-DD)

window   = 250    # 계산에 사용할 이동윈도우 길이
alpha    = 0.01   # VaR 신뢰수준
t_df     = 5      # Student-t 자유도 (필요시 수정)

def _krx_date(date_str):
    return date_str.replace("-", "")

def fetch_price_data(ticker, start_date, end_date):
    """
    Yahoo Finance 또는 KRX에서 종가 데이터를 가져와서 DataFrame으로 반환.
    """
    if ticker in tickers_yf:
        raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
        df = raw[["Close"]].dropna()
    else:
        start = _krx_date(start_date)
        end = _krx_date(end_date)
        df = stock.get_etf_ohlcv_by_date(start, end, ticker)
        if df.empty:
            df = stock.get_market_ohlcv_by_date(start, end, ticker)
        if df.empty:
            raise ValueError(f"No KRX price data returned for {ticker} from {start_date} to {end_date}")

        for close_col in ["종가", "Close", "close", "Adj Close"]:
            if close_col in df.columns:
                df = df[[close_col]].rename(columns={close_col: "Close"}).dropna()
                break
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise KeyError(f"Could not find a close-price column for {ticker}. Columns: {list(df.columns)}")
            fallback_col = numeric_cols[-1]
            df = df[[fallback_col]].rename(columns={fallback_col: "Close"}).dropna()
    return df

def calculate_return(df):
    """
    로그수익률(Return)을 계산하고 DataFrame에 추가.
    """
    df = df.copy()
    df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    return df

def compute_param_t_var(df, window, alpha, t_df):
    """
    Student-t 기반 Parametric VaR를 계산하여 시계열로 반환.
    """
    n = len(df)
    var_t = np.full(n, np.nan)
    returns = df["Return"].to_numpy()
    
    for i in range(window, n):
        window_rt = returns[i-window:i]
        mu = window_rt.mean()
        sigma = window_rt.std(ddof=1)
        
        t_quant = t.ppf(alpha, df=t_df)
        var_t[i] = mu + sigma * t_quant
    
    result = pd.DataFrame({
        "Close": df["Close"].to_numpy().ravel(),
        "Return": returns,
        "VaR_Param_t": var_t
    }, index=df.index)
    
    return result

# 메인 처리: 각 티커별로 계산하고 CSV 저장
for ticker in tickers_yf + tickers_kr:
    print(f"=== {ticker} VaR (Parametric t) 계산 중 ===")
    
    # 데이터 불러오기 및 수익률 계산
    price_df = fetch_price_data(ticker, start_date, end_date)
    data_df = calculate_return(price_df)
    
    # Student-t Parametric VaR 계산
    var_df = compute_param_t_var(data_df, window, alpha, t_df)
    
    # 결과 필터: 사용자가 지정한 기간 내에서 NaN이 아닌 VaR 값만 추출
    var_filtered = var_df.loc[start_date:end_date, ["VaR_Param_t"]].dropna()
    
    # CSV로 저장
    output_path = f"results_csv/{ticker}_param_t_var_{start_date}_to_{end_date}.csv"
    var_filtered.to_csv(output_path, index_label="Date")
    
    print(f"{ticker} 결과가 '{output_path}'에 저장되었습니다.")

print("모든 티커 처리 완료.")

