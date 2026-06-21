# Streamlit Dashboard

This dashboard is the visual interface for the Battery RUL system. It does not load the model directly. Instead, it calls the FastAPI backend, which serves the selected GLU model.

## Architecture

```text
Streamlit Dashboard -> FastAPI Backend -> Saved GLU Model -> Predicted RUL
```

## Install

```powershell
python -m pip install -r requirements-api.txt
python -m pip install -r requirements-dashboard.txt
```

## Run

Start FastAPI first:

```powershell
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Then start Streamlit in a second terminal:

```powershell
python -m streamlit run dashboard/app.py
```

## What The Dashboard Shows

- The selected GLU model metrics: MAE, RMSE, R2, and test window count.
- A live prediction section that calls FastAPI.
- Saved test predictions from battery B6.
- Trend plots for true RUL, predicted RUL, and prediction error.
- The generated experiment figures from the GLU results folder.

## Future XAI Integration

When the XAI part is ready, the dashboard can add a button that calls a new FastAPI endpoint such as `POST /explain`. That endpoint can return feature importance values for the selected window, and Streamlit can display them as a bar chart.
