# Battery Health Monitor Dashboard

This dashboard is the client-facing visual interface for the Battery RUL system. It does not load the model directly. Instead, it calls the FastAPI backend, which serves the selected GLU model.

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

- A simple battery health status: Healthy, Warning, or Critical.
- Estimated remaining useful life in cycles.
- A recommended action for the user.
- An estimated end-of-life cycle.
- A battery RUL trend chart.
- Optional technical sections for model metrics, raw validation values, and experiment figures.

## Data Source Modes

The dashboard has two prediction modes:

- `Official test battery`: uses the saved final test windows and predictions from the selected GLU experiment.
- `All processed batteries`: uses `data/processed/processed_data.csv`, aggregates it by `battery_id + cycle`, creates a 10-cycle window, and sends the window to FastAPI using `POST /predict`.
- `Upload client CSV`: lets a user upload a preprocessed CSV, creates a 10-cycle window, and sends it to FastAPI using `POST /predict`.

The processed CSV mode can show B5, B6, B7, and B18 because those batteries exist in `processed_data.csv`. Remember that B5 and B7 were training batteries, B18 was validation, and B6 was the official test battery.

## Alert Logic

The API returns one of three alert levels:

- `healthy`: predicted RUL is above 50 cycles.
- `warning`: predicted RUL is 50 cycles or below.
- `critical`: predicted RUL is 20 cycles or below.

The dashboard translates these into client-friendly messages and maintenance recommendations.

## Upload CSV Format

The uploaded CSV must already be preprocessed/scaled in the same style as the training data. It must include these columns:

```text
battery_id, cycle, chI, chV, chT, disI, disV, BCt, SOH
```

The `RUL` column is optional. If it exists, the dashboard can show validation values such as true RUL and prediction error. If it does not exist, the dashboard will still show the health alert and estimated remaining cycles.

A sample file is included here:

```text
dashboard/sample_client_battery_upload.csv
```

## Future XAI Integration

When the XAI part is ready, the dashboard can add a button that calls a new FastAPI endpoint such as `POST /explain`. That endpoint can return feature importance values for the selected window, and Streamlit can display them as a bar chart.
