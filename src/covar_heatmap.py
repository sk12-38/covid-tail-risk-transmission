import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from arch import arch_model
from scipy import stats
from datetime import datetime
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

def get_data(tickers, start_date, end_date):
    """
    Yahoo Finance에서 데이터를 다운로드하고 로그 수익률을 계산합니다.
    """
    print("데이터 다운로드 중...")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    returns = np.log(data / data.shift(1)).dropna()
    return returns

def fit_garch(returns, p=1, q=1):
    """
    GARCH(p,q) 모델을 적합하고 조건부 평균과 변동성을 반환합니다.
    """
    try:
        model = arch_model(returns, vol='GARCH', p=p, q=q, mean='Zero', dist='normal')
        results = model.fit(disp='off', show_warning=False, options={'maxiter': 1000})
        return results
    except Exception as e:
        print(f"GARCH 모델 적합 중 오류 발생: {e}")
        return None

def compute_var(garch_results, alpha=0.05):
    """
    GARCH 모델 결과를 기반으로 VaR를 계산합니다.
    """
    z_alpha = stats.norm.ppf(alpha)
    conditional_vol = garch_results.conditional_volatility
    var = z_alpha * conditional_vol
    return var

def compute_covar(returns_y, returns_x, var_x, alpha=0.05):
    """
    CoVaR를 계산합니다.
    """
    # OLS 회귀를 통한 beta 추정
    X = sm.add_constant(returns_x)
    model = sm.OLS(returns_y, X).fit()
    alpha_hat, beta = model.params
    
    # Y의 GARCH 모델 적합
    garch_y = fit_garch(returns_y)
    sigma_y = garch_y.conditional_volatility
    
    # CoVaR 계산
    z_alpha = stats.norm.ppf(alpha)
    covar = alpha_hat + beta * var_x + z_alpha * sigma_y
    
    return covar

def create_covar_matrix(returns, alpha=0.05):
    """
    모든 자산 쌍에 대한 CoVaR 행렬을 생성합니다.
    """
    n_assets = len(returns.columns)
    covar_matrix = np.zeros((n_assets, n_assets))
    covar_matrix[:] = np.nan
    
    for i, asset_y in enumerate(returns.columns):
        for j, asset_x in enumerate(returns.columns):
            if i != j:
                try:
                    # X의 GARCH 모델 적합 및 VaR 계산
                    garch_x = fit_garch(returns[asset_x])
                    if garch_x is not None:
                        var_x = compute_var(garch_x, alpha)
                        
                        # CoVaR 계산
                        covar = compute_covar(returns[asset_y], returns[asset_x], var_x, alpha)
                        covar_matrix[i, j] = covar.mean()
                except Exception as e:
                    print(f"CoVaR 계산 중 오류 발생 ({asset_y}, {asset_x}): {e}")
                    continue
    
    return pd.DataFrame(covar_matrix, 
                       index=returns.columns, 
                       columns=returns.columns)

def plot_heatmap(covar_matrix, title, save_path=None, vmin=None, vmax=None):
    """
    CoVaR 행렬의 히트맵을 생성합니다.
    
    Parameters:
    -----------
    covar_matrix : pandas.DataFrame
        CoVaR 행렬
    title : str
        히트맵 제목
    save_path : str, optional
        저장할 파일 경로
    vmin : float, optional
        색상 스케일의 최소값
    vmax : float, optional
        색상 스케일의 최대값
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(covar_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                vmin=vmin,
                vmax=vmax)
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_results_to_txt(returns, var_results, covar_matrix, alpha, period_name, filename):
    """
    VaR와 CoVaR 결과를 txt 파일로 저장합니다.
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        수익률 데이터
    var_results : dict
        각 자산의 VaR 결과
    covar_matrix : pandas.DataFrame
        CoVaR 행렬
    alpha : float
        유의수준
    period_name : str
        분석 기간 이름
    filename : str
        저장할 파일 이름
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"=== {period_name} 분석 결과 ===\n")
        f.write(f"유의수준: {alpha}\n\n")
        
        # VaR 결과 저장
        f.write("=== 개별 자산 VaR ===\n")
        for asset, var in var_results.items():
            f.write(f"{asset}: {var:.4f}\n")
        
        # CoVaR 행렬은 CSV 파일로 저장
        csv_filename = filename.replace('.txt', '_matrix.csv')
        
        # CSV 파일에 설명 추가
        with open(csv_filename, 'w', encoding='utf-8') as csv_file:
            csv_file.write("# CoVaR 행렬 설명:\n")
            csv_file.write("# - 행(row): 영향을 받는 자산\n")
            csv_file.write("# - 열(column): 영향을 주는 자산\n")
            csv_file.write("# - 값: 열의 자산이 위기 상황일 때 행의 자산의 CoVaR 값\n\n")
        
        # 기존 설명 위에 데이터 추가
        covar_matrix.to_csv(csv_filename, mode='a', encoding='utf-8')
        f.write(f"\nCoVaR 행렬은 {csv_filename} 파일에 저장되었습니다.\n")

def main():
    # 설정
    tickers = ['ACWI', 'ACWX', '069500.KS', '229200.KS', 'EFA', 'VWO',  # 주식
               'BNDX', 'EMB', 'IEF', 'TLT', '114260.KS', '148070.KS',    # 채권
               'USO', 'GLD',                                              # 원자재
               'VNQ']                                                     # 부동산
    start_date = '2005-01-01'
    end_date = '2023-12-31'
    alpha = 0.05
    
    # 데이터 수집
    returns = get_data(tickers, start_date, end_date)
    
    # 전체 기간 분석
    # 개별 VaR 계산
    var_results = {}
    for asset in returns.columns:
        garch_results = fit_garch(returns[asset])
        if garch_results is not None:
            var = compute_var(garch_results, alpha)
            var_results[asset] = var.mean()
    
    # 전체 기간 CoVaR 계산
    covar_matrix = create_covar_matrix(returns, alpha)
    
    # 전체 기간의 최대/최소값 계산
    vmin = covar_matrix.min().min()
    vmax = covar_matrix.max().max()
    
    # 결과 저장
    save_results_to_txt(returns, var_results, covar_matrix, alpha, 
                       "전체 기간 (2005-2023)", "results_full_period.txt")
    
    # 히트맵 시각화
    plot_heatmap(covar_matrix, 
                'CoVaR Matrix (2005-2023)',
                'covar_heatmap_full.png',
                vmin=vmin,
                vmax=vmax)
    
    # 위기 기간 분석
    crisis_periods = {
        'Global Finance Crises': ('2008-09-01', '2009-03-31'),
        'COVID-19': ('2020-02-01', '2020-04-30')
    }
    
    for crisis_name, (start, end) in crisis_periods.items():
        crisis_returns = returns[start:end]
        if len(crisis_returns) > 30:  # 최소 30일 이상의 데이터가 있는지 확인
            # 위기 기간 VaR 계산
            crisis_var_results = {}
            for asset in crisis_returns.columns:
                garch_results = fit_garch(crisis_returns[asset])
                if garch_results is not None:
                    var = compute_var(garch_results, alpha)
                    crisis_var_results[asset] = var.mean()
            
            # 위기 기간 CoVaR 계산
            crisis_covar = create_covar_matrix(crisis_returns, alpha)
            
            # 결과 저장
            save_results_to_txt(crisis_returns, crisis_var_results, crisis_covar, alpha,
                              crisis_name, f"results_{crisis_name.lower().replace(' ', '_')}.txt")
            
            # 히트맵 시각화
            plot_heatmap(crisis_covar,
                        f'CoVaR Matrix - {crisis_name}',
                        f'covar_heatmap_{crisis_name.lower()}.png',
                        vmin=vmin,
                        vmax=vmax)
        else:
            print(f"{crisis_name} 기간의 데이터가 충분하지 않습니다. (데이터 수: {len(crisis_returns)})")

if __name__ == "__main__":
    main()
