# COVID-19 Tail Risk Transmission Across Asset Classes

## 한국어

이 저장소는 COVID-19 위기 전후 글로벌 자산군 간 tail risk 전이를 VaR, CVaR, GARCH, CoVaR 계열 위험 측도로 분석한 금융계량 프로젝트입니다. 글로벌 ETF와 국내 상장 ETF를 함께 사용해 주식, 채권, 원자재, 부동산 자산군의 위험 전이와 VaR/CVaR 모형의 backtest 성능을 비교합니다.

### 연구 질문

COVID-19 기간 동안 자산군 간 tail risk 전이는 어떻게 변화했는가? 여러 자산군에서 VaR/CVaR 모형 중 어떤 모형이 backtesting 관점에서 더 좋은 성능을 보였는가?

### 자산군

- 글로벌 주식: `ACWI`, `ACWX`
- 선진국 및 신흥국 주식: `EFA`, `VWO`
- 한국 주식: `069500` KODEX 200, `229200` KODEX KOSDAQ 150
- 글로벌 및 신흥국 채권: `BNDX`, `EMB`
- 미국 국채: `IEF`, `TLT`
- 한국 국채: `114260` KOSEF 3년 국고채, `148070` KOSEF 10년 국고채
- 원자재: `USO`, `GLD`
- 부동산: `VNQ`

### 방법론

- ETF 가격 데이터에서 일간 log return 생성
- GARCH 기반 VaR 추정
- 자산군 간 tail-risk spillover 분석을 위한 CoVaR matrix 구성
- 전체 기간과 COVID-19 위기 기간 비교
- VaR violation count와 conditional coverage 통계량 기반 backtest
- CVaR downside/upside pass count 기반 backtest

### 주요 파일

- `src/covar_heatmap.py`: GARCH 기반 VaR와 CoVaR matrix를 만들고 heatmap을 저장합니다.
- `src/var_cvar_backtest.py`: ETF universe에 대해 VaR/CVaR를 계산하고 backtest summary table을 생성합니다.
- `src/parametric_var_export.py`: 일부 ticker와 기간에 대해 Student-t parametric VaR series를 저장합니다.
- `paper/tail_risk_transmission_covid19.pdf`: 최종 보고서
- `results/figures/`: 대표 그림
- `results/tables/`: 최종 요약 표와 CoVaR matrix

### 실행 방법

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/covar_heatmap.py
python src/var_cvar_backtest.py
```

## English

This repository studies tail-risk spillovers across global asset classes during the COVID-19 crisis using VaR, CVaR, GARCH, and CoVaR-style risk measures. The analysis covers equity, bond, commodity, and real estate ETFs, including both global ETFs and Korean listed ETFs.

### Research Question

How did cross-asset tail-risk transmission change during COVID-19, and which VaR/CVaR models performed better in backtesting across multiple asset classes?

### Asset Universe

- Global equity: `ACWI`, `ACWX`
- Developed and emerging equity: `EFA`, `VWO`
- Korean equity: `069500` KODEX 200, `229200` KODEX KOSDAQ 150
- Global and emerging bonds: `BNDX`, `EMB`
- US Treasuries: `IEF`, `TLT`
- Korean bonds: `114260` KOSEF 3Y Treasury Bond, `148070` KOSEF 10Y Treasury Bond
- Commodities: `USO`, `GLD`
- Real estate: `VNQ`

### Methodology

- Daily log-return construction from ETF price data
- GARCH-based VaR estimation
- CoVaR matrix construction for cross-asset tail-risk spillover analysis
- Full-period and COVID-19 crisis-period comparison
- VaR backtesting with violation counts and conditional coverage statistics
- CVaR backtesting using downside and upside pass counts

### Main Files

- `src/covar_heatmap.py`: Builds GARCH-based VaR and CoVaR matrices and exports heatmaps.
- `src/var_cvar_backtest.py`: Computes VaR/CVaR estimates and creates backtest summary tables.
- `src/parametric_var_export.py`: Exports Student-t parametric VaR series for selected tickers and periods.
- `paper/tail_risk_transmission_covid19.pdf`: Final report
- `results/figures/`: Representative figures
- `results/tables/`: Final summary tables and CoVaR matrices

### How to Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/covar_heatmap.py
python src/var_cvar_backtest.py
```
