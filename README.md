# COVID-19 Tail Risk Transmission Across Asset Classes

This project studies tail-risk spillovers across global asset classes during the COVID-19 crisis using VaR, CVaR, GARCH, and CoVaR-style risk measures.

The analysis covers equity, bond, commodity, and real estate ETFs, including both global ETFs and Korean listed ETFs. The project compares risk model performance through VaR/CVaR backtests and visualizes cross-asset tail-risk transmission with CoVaR heatmaps.

## Research Question

How did cross-asset tail-risk transmission change during COVID-19, and which VaR/CVaR models performed better in backtesting across multiple asset classes?

## Asset Universe

- Global equity: `ACWI`, `ACWX`
- Developed and emerging equity: `EFA`, `VWO`
- Korean equity: `069500` KODEX 200, `229200` KODEX KOSDAQ 150
- Global and emerging bonds: `BNDX`, `EMB`
- US Treasuries: `IEF`, `TLT`
- Korean bonds: `114260` KOSEF 3Y Treasury Bond, `148070` KOSEF 10Y Treasury Bond
- Commodities: `USO`, `GLD`
- Real estate: `VNQ`

## Methodology

- Daily log-return construction from ETF price data
- GARCH-based VaR estimation
- CoVaR matrix construction for cross-asset tail-risk spillover analysis
- Full-period and COVID-19 crisis-period comparison
- VaR backtesting with traffic-light style violation counts and Christoffersen-style conditional coverage statistics
- CVaR backtesting using downside and upside pass counts
- Summary tables for model comparison across tickers

## Key Files

- `src/covar_heatmap.py`: builds GARCH-based VaR and CoVaR matrices, then exports heatmaps and matrix outputs.
- `src/var_cvar_backtest.py`: computes historical, parametric, and GARCH VaR/CVaR estimates across the ETF universe and creates summary backtest tables.
- `src/parametric_var_export.py`: exports Student-t parametric VaR series for selected tickers and periods.
- `paper/tail_risk_transmission_covid19.pdf`: final report.
- `results/figures/`: selected representative figures.
- `results/tables/`: selected final summary tables and CoVaR matrices.

## Repository Structure

```text
.
|-- data/
|   `-- README.md
|-- paper/
|   `-- tail_risk_transmission_covid19.pdf
|-- results/
|   |-- figures/
|   |   |-- covar_heatmap_covid19.png
|   |   |-- covar_heatmap_full.png
|   |   `-- model_comparison.png
|   `-- tables/
|       |-- results_covid19_matrix.csv
|       |-- results_full_period_matrix.csv
|       |-- table3_cvar_summary.csv
|       `-- table3_var_summary.csv
|-- src/
|   |-- covar_heatmap.py
|   |-- parametric_var_export.py
|   `-- var_cvar_backtest.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Data Availability

Raw price data is excluded from this repository. The scripts download public market data from Yahoo Finance and KRX via `yfinance` and `pykrx`, but downloaded CSV files and intermediate outputs are intentionally not tracked.

## How to Run

Create a Python environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the CoVaR heatmap analysis:

```powershell
python src/covar_heatmap.py
```

Run the VaR/CVaR backtesting pipeline:

```powershell
python src/var_cvar_backtest.py
```

Run the Student-t parametric VaR export helper:

```powershell
python src/parametric_var_export.py
```

## Notes

This repository is a curated portfolio version of a financial econometrics team project. Intermediate experiments, duplicate scripts, raw data files, generated per-ticker outputs, and submission archives were excluded to keep the repository focused on the final research workflow.
