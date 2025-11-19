import os
import joblib
import yfinance as yf
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

app = FastAPI()

# Paths
BASE_DIR = r"C:\Users\Administrator\PycharmProjects\work\sr_model"
MODEL_DIR = os.path.join(BASE_DIR, "fusion_model")
MODEL_PATH = os.path.join(MODEL_DIR, "fusion_meta_v1_20251108_021325.pkl")
SCALER_PATH_1 = os.path.join(MODEL_DIR, "fusion_meta_v1_20251108_021325_scaler.pkl")

model = joblib.load(MODEL_PATH)
SCALER_PATH = os.path.join(MODEL_DIR, "fusion_meta_v1_20251108_021325_scaler.pkl")
scaler = joblib.load(SCALER_PATH)

# Load HTML
with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
    HTML_PAGE = f.read()


# Feature prediction
@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        features = data["features"]

        required = ["open","high","low","close","volume","rsi","vwap","vol_avg","vwap_zone_diff"]
        for f in required:
            if f not in features:
                return {"error": f"Missing feature {f}"}

        open_p = float(features["open"])
        high_p = float(features["high"])
        low_p = float(features["low"])
        close_p = float(features["close"])
        volume = float(features["volume"])
        rsi = float(features["rsi"])
        vwap = float(features["vwap"])
        vol_avg = float(features["vol_avg"])
        vwap_zone_diff = float(features["vwap_zone_diff"])

        # Auto features
        zone_price = (high_p + low_p + close_p) / 3
        distance_from_zone = close_p - zone_price
        vwap_distance = close_p - vwap
        rsi_overbought = 1 if rsi > 70 else 0
        rsi_oversold = 1 if rsi < 30 else 0

        # FINAL features array (14 features)
        final = [
            open_p, high_p, low_p, close_p, volume, rsi, vwap,
            zone_price, distance_from_zone, vwap_distance,
            rsi_overbought, rsi_oversold,
            vol_avg, vwap_zone_diff
        ]

        X_scaled = scaler.transform([final])
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0]

        return {
            "prediction": "BUY" if pred == 1 else "SELL",
            "confidence": float(max(prob)),
            "auto_features": {
                "zone_price": zone_price,
                "distance_from_zone": distance_from_zone,
                "vwap_distance": vwap_distance,
                "rsi_overbought": rsi_overbought,
                "rsi_oversold": rsi_oversold
            }
        }

    except Exception as e:
        return {"error": str(e)}

# Batch predict
@app.post("/batch_predict")
async def batch_predict(request: Request):
    try:
        data = await request.json()
        rows = data.get("rows", [])
        if not rows: return {"error":"No rows provided"}

        X = []
        for r in rows:
            open_p = float(r["open"])
            high_p = float(r["high"])
            low_p = float(r["low"])
            close_p = float(r["close"])
            volume = float(r["volume"])
            rsi = float(r["rsi"])
            vwap = float(r["vwap"])

            zone_price = (high_p+low_p+close_p)/3
            distance_from_zone = close_p - zone_price
            vwap_distance = close_p - vwap
            rsi_overbought = 1 if rsi>70 else 0
            rsi_oversold = 1 if rsi<30 else 0

            X.append([open_p, high_p, low_p, close_p, volume, rsi, vwap,
                      zone_price, distance_from_zone, vwap_distance,
                      rsi_overbought, rsi_oversold])

        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)

        results = []
        for p, pr in zip(preds, probs):
            results.append({"prediction":"BUY" if p==1 else "SELL","confidence":float(max(pr))})

        return {"results":results}
    except Exception as e:
        return {"error":str(e)}

# Live predict (unchanged)
@app.get("/live_predict/{symbol}")
def live_predict(symbol: str):
    df = yf.download(symbol, period="5d", interval="1m")
    if df.empty: return {"error":"invalid ticker"}
    latest = df.iloc[-1]
    features = [float(latest["Close"])]+[0]*13
    scaled = scaler.transform([features])
    pred = model.predict(scaled)[0]
    return {"market":symbol,"current_price":float(features[0]),"prediction":"BUY" if pred==1 else "SELL"}

# Dashboard
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return HTMLResponse(content=HTML_PAGE)
