
# XAUUSD Strategy Repo

This repository implements an institutional-style XAU/USD strategy with:

- Daily signal generation and email report at 09:30 UAE (05:30 UTC) using GitHub Actions.
- A 15-year daily backtest pipeline.

## Structure

- `.github/workflows/xauusd-daily-report.yml` – daily signal + email workflow.
- `.github/workflows/xauusd-backtest-15y.yml` – weekly 15-year backtest + email.
- `scripts/xauusd_strategy.py` – data fetching, fundamental/technical score, backtest.
- `requirements.txt` – Python dependencies.
- `output/` – generated reports and tables (ignored by git).

## Data sources

- XAU/USD daily prices via Yahoo Finance (`XAUUSD=X`).
- DXY (U.S. Dollar Index) via Yahoo Finance (`DX-Y.NYB`).
- 10Y real yield via FRED API (series `DFII10`).

## GitHub Actions scheduling

GitHub scheduled workflows run in UTC only. The daily report is configured at 05:30 UTC, which corresponds to 09:30 in the UAE (UTC+4).

## Setup

1. Create a new GitHub repository and push this bundle.
2. In repository **Settings → Secrets and variables → Actions → New repository secret**, define:
   - `SMTP_SERVER_ADDRESS` (e.g., `smtp.gmail.com`)
   - `SMTP_SERVER_PORT` (e.g., `465`)
   - `SMTP_USERNAME`
   - `SMTP_PASSWORD` (App Password for your SMTP provider)
   - `MAIL_FROM` (e.g., `xauusd.bot@gmail.com`)
   - `FRED_API_KEY` (your API key from St. Louis Fed FRED for real-yield data)
3. Enable Actions for the repository.
4. Optionally dispatch the workflows manually to test them.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run daily report locally
python scripts/xauusd_strategy.py   --mode daily-report   --lookback-years 15   --out-html output/xauusd_daily_report.html   --out-csv output/xauusd_daily_table.csv

# Run backtest locally
python scripts/xauusd_strategy.py   --mode backtest   --lookback-years 15   --out-html output/xauusd_backtest_15y_report.html   --out-csv output/xauusd_backtest_15y_table.csv
```

