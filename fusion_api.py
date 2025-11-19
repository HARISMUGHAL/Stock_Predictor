import os
import joblib
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict
import numpy as np

app = FastAPI()

# --------------------------------------------------------
#  ABSOLUTE PATH FIX
# --------------------------------------------------------
BASE_DIR = r"C:\Users\Administrator\PycharmProjects\work\sr_model"
MODEL_DIR = os.path.join(BASE_DIR, "fusion_model")

MODEL_PATH = os.path.join(MODEL_DIR, "fusion_meta_v1_20251108_021325.pkl")
SCALER_PATH_1 = os.path.join(MODEL_DIR, "fusion_meta_v1_20251108_021325_scaler.pkl")
SCALER_PATH_2 = os.path.join(MODEL_DIR, "fusion_scaler_v1_20251108_021325.pkl")

# Load model
model = joblib.load(MODEL_PATH)

# Load scaler
if os.path.exists(SCALER_PATH_1):
    scaler = joblib.load(SCALER_PATH_1)
else:
    scaler = joblib.load(SCALER_PATH_2)

# Model expects 14 features
MODEL_FEATURES = [f"Column_{i}" for i in range(14)]

# --------------------------------------------------------
#  Pydantic models for API
# --------------------------------------------------------
class PredictRequest(BaseModel):
    features: Dict[str, float]


class BatchPredictRequest(BaseModel):
    rows: List[Dict[str, float]]


# --------------------------------------------------------
#  Helper: fix features to 14
# --------------------------------------------------------
def fix_features(input_dict):
    arr = [0] * 14
    i = 0
    for _, v in input_dict.items():
        if i < 14:
            arr[i] = v
            i += 1
    return arr


# --------------------------------------------------------
#  Live feature generator
# --------------------------------------------------------
def create_live_features(symbol):
    df = yf.download(symbol, period="5d", interval="1m")
    if df.empty:
        return None, "❌ Invalid ticker or data not found"

    df["sma_5"] = df["Close"].rolling(5).mean().fillna(df["Close"])
    df["ema_5"] = df["Close"].ewm(span=5).mean()
    df["volatility"] = df["Close"].pct_change().rolling(5).std().fillna(0)
    df["price_change"] = df["Close"].pct_change().fillna(0)

    latest = df.iloc[-1]

    features = [
        float(latest["Close"]),
        float(latest["sma_5"]),
        float(latest["ema_5"]),
        float(latest["volatility"]),
        float(latest["price_change"]),
        float(latest["Open"]),
        float(latest["High"]),
        float(latest["Low"]),
        float(latest["Volume"]),
        0, 0, 0, 0, 0  # dummy fillers
    ]
    return features, None


# --------------------------------------------------------
#  API: Single predict
# --------------------------------------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        # Required inputs from user: close, open, high, low, volume
        close = req.features.get("Close", 0)
        open_ = req.features.get("Open", 0)
        high = req.features.get("High", 0)
        low = req.features.get("Low", 0)
        volume = req.features.get("Volume", 0)

        # Calculated features
        sma_5 = req.features.get("SMA_5", close)       # user can input or use close as default
        ema_5 = req.features.get("EMA_5", close)
        volatility = req.features.get("Volatility", 0)
        price_change = req.features.get("Price_Change", 0)
        vwap = req.features.get("VWAP", close)          # default close if not provided

        zone_price = (high + low + close) / 3
        distance_from_zone = close - zone_price
        vwap_distance = close - vwap
        rsi = req.features.get("RSI", 50)
        rsi_overbought = 1 if rsi > 70 else 0
        rsi_oversold = 1 if rsi < 30 else 0

        # Build feature array in order expected by model
        features = [
            close, open_, high, low, volume,       # Columns 0-4: show names in dashboard
            sma_5, ema_5, volatility, price_change,  # Columns 5-8
            zone_price, distance_from_zone, vwap_distance,  # Columns 9-11
            rsi_overbought, rsi_oversold            # Columns 12-13
        ]

        scaled = scaler.transform([features])
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0].max()
        decision = "BUY" if pred == 1 else "SELL"
        return {"market": "Custom", "prediction": decision, "confidence": round(float(prob), 4)}
    except Exception as e:
        return {"error": str(e)}
# --------------------------------------------------------
#  API: Batch predict
# --------------------------------------------------------
@app.post("/batch_predict")
def batch_predict(req: BatchPredictRequest):
    output = []
    try:
        rows = [fix_features(r) for r in req.rows]
        scaled = scaler.transform(rows)
        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)
        for i in range(len(rows)):
            decision = "BUY" if preds[i] == 1 else "SELL"
            conf = float(probs[i].max())
            output.append({"prediction": decision, "confidence": round(conf, 4)})
        return output
    except Exception as e:
        return {"error": str(e)}


# --------------------------------------------------------
#  API: Live predict
# --------------------------------------------------------
@app.get("/live_predict/{symbol}")
def live_predict(symbol: str):
    features, err = create_live_features(symbol)
    if err:
        return {"error": err}
    scaled = scaler.transform([features])
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0].max()
    decision = "BUY" if pred == 1 else "SELL"
    return {
        "market": symbol.upper(),
        "current_price": float(features[0]),
        "prediction": decision,
        "confidence": round(float(prob), 4)
    }


# --------------------------------------------------------
#  HTML Dashboard
# --------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fusion Meta v1 – Stock Predictor Dashboard</title>
        <style>
            body {
                background-color: #1f1f2e;
                color: #e0e0e0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                text-align: center;
                padding: 20px;
            }
            h1 {
                font-size: 36px;
                color: #00bfff;
                margin-bottom: 20px;
            }
            input, button {
                font-size: 16px;
                padding: 10px 15px;
                margin: 5px;
                border-radius: 6px;
                border: none;
            }
            input { width: 250px; }
            button { cursor: pointer; transition: 0.3s; }
            .btn-live { background-color: #007acc; color: #fff; }
            .btn-features { background-color: #28a745; color: #fff; }
            .btn-batch { background-color: #ff7f50; color: #fff; }
            button:hover { opacity: 0.85; }
            #result, #batch-result { 
                margin-top: 20px; 
                padding: 20px; 
                background-color: #2b2b3d; 
                border-radius: 10px; 
                width: 70%; 
                margin-left: auto; margin-right: auto; 
                font-size: 18px; 
                word-wrap: break-word;
            }
            #batch-upload { margin-top: 15px; display:none; }
            .feature-input { width: 80px; margin:3px; }
        </style>
    </head>
    <body>
        <h1>Fusion Meta v1 – Stock Predictor Dashboard</h1>

        <!-- Live Predict -->
        <div>
            <input id="symbol" placeholder="Enter Stock Symbol (AAPL, TSLA)" />
            <button class="btn-live" onclick="livePredict()">Live Predict</button>
        </div>

        <!-- Predict by Features -->
        <div id="feature-div" style="margin-top:20px;">
            <h3>Predict by Features</h3>
            <div id="feature-inputs">
            </div>
            <button class="btn-features" onclick="predictByFeatures()">Predict Features</button>
        </div>

        <!-- Batch Predict -->
        <div id="batch-div" style="margin-top:20px;">
            <button class="btn-batch" onclick="showBatchUpload()">Batch Predict (CSV)</button>
            <div id="batch-upload">
                <input type="file" id="csvFile" accept=".csv"/>
                <button onclick="uploadCSV()">Upload & Predict</button>
            </div>
        </div>

        <div id="result">Prediction result will appear here.</div>
        <div id="batch-result"></div>

        <div id="refresh-note" style="margin-top:15px; color:#aaa;">Dashboard auto-refreshes every 20 seconds ⭐</div>

        <script>
            // Create 14 feature inputs
            const featureDiv = document.getElementById("feature-inputs");
            for(let i=0;i<14;i++){
                const input = document.createElement("input");
                input.className = "feature-input";
                input.id = "feat_"+i;
                input.placeholder = "Col_"+i;
                featureDiv.appendChild(input);
            }

            // Auto-refresh live prediction every 20s
            setInterval(() => {
                const symbol = document.getElementById("symbol").value;
                if(symbol.trim() !== "") livePredict();
            }, 20000);

            async function livePredict() {
                const symbol = document.getElementById("symbol").value.toUpperCase();
                if(!symbol) return;

                const res = await fetch(`/live_predict/${symbol}`);
                const data = await res.json();
                if(data.error){
                    document.getElementById("result").innerText = data.error;
                } else {
                    document.getElementById("result").innerHTML = `
                        <b>Symbol:</b> ${data.market} <br>
                        <b>Price:</b> $${data.current_price.toFixed(2)} <br>
                        <b>Prediction:</b> ${data.prediction} <br>
                        <b>Confidence:</b> ${data.confidence}
                    `;
                }
            }

            function predictByFeatures() {
                const features = {};
                for(let i=0;i<14;i++){
                    const val = parseFloat(document.getElementById("feat_"+i).value) || 0;
                    features["Column_"+i] = val;
                }
                fetch('/predict',{
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({features})
                })
                .then(res=>res.json())
                .then(data=>{
                    if(data.error){
                        document.getElementById("result").innerText = data.error;
                    } else {
                        document.getElementById("result").innerHTML = `
                            <b>Prediction:</b> ${data.prediction} <br>
                            <b>Confidence:</b> ${data.confidence}
                        `;
                    }
                });
            }

            function showBatchUpload(){
                document.getElementById("batch-upload").style.display = "block";
            }

            function uploadCSV(){
                const fileInput = document.getElementById("csvFile");
                if(!fileInput.files.length) return alert("Please select a CSV file.");

                const reader = new FileReader();
                reader.onload = async function(e){
                    const text = e.target.result;
                    const rows = text.split("\\n").filter(r=>r.trim()!=="").map(r=>{
                        const vals = r.split(",").map(Number);
                        return Object.fromEntries(vals.map((v,i)=>["Column_"+i,v]));
                    });

                    const res = await fetch('/batch_predict',{
                        method:'POST',
                        headers:{'Content-Type':'application/json'},
                        body:JSON.stringify({rows})
                    });
                    const data = await res.json();
                    document.getElementById("batch-result").innerHTML = "<pre>"+JSON.stringify(data,null,2)+"</pre>";
                };
                reader.readAsText(fileInput.files[0]);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
