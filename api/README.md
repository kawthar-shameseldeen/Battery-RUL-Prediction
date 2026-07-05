# Battery RUL FastAPI Backend

This API serves the selected GLU model, `Two GLU Blocks + Pooling`, so the dashboard or any external system can request Remaining Useful Life predictions.

## How It Works

The API loads the saved Keras model once when the server starts. A request sends one battery window with shape `(10, 7)`, where:

- `10` = ten battery cycles in the window
- `7` = the selected input features
- feature order = `chI`, `chV`, `chT`, `disI`, `disV`, `BCt`, `SOH`

The API returns the predicted RUL and a simple status:

- `healthy` when predicted RUL is above 50
- `warning` when predicted RUL is 50 or below
- `critical` when predicted RUL is 20 or below

## Install

```powershell
python -m pip install -r requirements-api.txt
```

## Run

```powershell
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Useful Endpoints

- `GET /health`: checks that the API is running and shows the model input shape.
- `POST /predict`: predicts RUL from a custom `(10, 7)` window.
- `GET /sample/test/0`: returns one saved test window from the processed dataset.
- `GET /predict-sample/test/0`: predicts RUL for one saved test window.

## Dashboard Connection

Later, Streamlit can call `POST /predict` when the user uploads or selects a battery window. XAI can also be added later as a separate endpoint, for example `POST /explain`, using the same input window.
