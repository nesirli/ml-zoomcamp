# ML Zoomcamp

Hands-on machine learning reference notebooks and deployment examples following the [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) curriculum.

## Structure

| Module | Topic |
|---|---|
| `01_intro/` | NumPy, Linear Algebra, Pandas |
| `02_car-price/` | Regression — Car Price Prediction |
| `03_telco-churn/` | Classification — Telco Churn |
| `04_evalutation/` | Model Evaluation Metrics |
| `05_deployment/` | Model Deployment (FastAPI, Docker, AWS Lambda) |
| `06_credit-risk/` | Credit Risk Scoring |

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

Open any notebook with Jupyter:

```bash
uv run jupyter notebook
```

## Deployment module

The `05_deployment/` folder is a self-contained FastAPI service with its own `pyproject.toml`. Run it with:

```bash
cd 05_deployment
uv run uvicorn app:app --reload
```
