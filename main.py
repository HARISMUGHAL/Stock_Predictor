import os
import logging
import traceback
import time
import requests
import joblib
import yfinance as yf
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# ---------------- Railway-compatible relative paths ----------------
MODEL_DIR = "fusion_model"
MODEL_PATH = "fusion_meta_v1_20251108_021325.pkl"
SCALER_PATH = "fusion_meta_v1_20251108_021325_scaler.pkl"
HTML_PATH = "index.html"

# ---------------- Load model, scaler, HTML ----------------
logger.info("=" * 50)
logger.info("Starting application initialization...")
logger.info(f"MODEL_PATH: {MODEL_PATH}")
logger.info(f"SCALER_PATH: {SCALER_PATH}")
logger.info(f"HTML_PATH: {HTML_PATH}")

try:
    logger.info("Loading model from: " + MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    logger.info("✓ Model loaded successfully")
except Exception as e:
    logger.error(f"✗ ERROR loading model: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

try:
    logger.info("Loading scaler from: " + SCALER_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("✓ Scaler loaded successfully")
except Exception as e:
    logger.error(f"✗ ERROR loading scaler: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

try:
    logger.info("Loading HTML from: " + HTML_PATH)
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        HTML_PAGE = f.read()
    logger.info(f"✓ HTML loaded successfully ({len(HTML_PAGE)} characters)")
except Exception as e:
    logger.error(f"✗ ERROR loading HTML: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

logger.info("=" * 50)
logger.info("Application initialization complete!")
logger.info("=" * 50)

# Feature prediction
@app.post("/predict")
async def predict(request: Request):
    logger.info("=" * 50)
    logger.info("POST /predict - Request received")
    try:
        logger.info("Step 1: Parsing JSON request body...")
        data = await request.json()
        logger.info(f"✓ JSON parsed successfully. Keys: {list(data.keys())}")
        
        logger.info("Step 2: Extracting features from request...")
        features = data.get("features", {})
        if not features:
            logger.error("✗ ERROR: 'features' key not found in request data")
            return {"error": "Missing 'features' key in request"}
        logger.info(f"✓ Features extracted. Keys: {list(features.keys())}")

        logger.info("Step 3: Validating required features...")
        required = ["high","low","close","volume","rsi","vwap","vol_avg"]
        for f in required:
            if f not in features:
                logger.error(f"✗ ERROR: Missing required feature '{f}'")
                logger.error(f"Available features: {list(features.keys())}")
                return {"error": f"Missing feature {f}"}
        logger.info("✓ All required features present")

        logger.info("Step 4: Converting features to float...")
        try:
            high_p = float(features["high"])
            low_p = float(features["low"])
            close_p = float(features["close"])
            volume = float(features["volume"])
            rsi = float(features["rsi"])
            vwap = float(features["vwap"])
            vol_avg = float(features["vol_avg"])
            logger.info(f"✓ Features converted: high={high_p}, low={low_p}, close={close_p}, volume={volume}, rsi={rsi}, vwap={vwap}, vol_avg={vol_avg}")
        except ValueError as ve:
            logger.error(f"✗ ERROR converting features to float: {str(ve)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"Invalid numeric value in features: {str(ve)}"}

        logger.info("Step 5: Calculating auto features...")
        zone_price = (high_p + low_p + close_p) / 3
        distance_from_zone = close_p - zone_price
        vwap_distance = close_p - vwap
        rsi_overbought = 1 if rsi > 70 else 0
        rsi_oversold = 1 if rsi < 30 else 0
        logger.info(f"✓ Auto features calculated: zone_price={zone_price}, distance_from_zone={distance_from_zone}, vwap_distance={vwap_distance}, rsi_overbought={rsi_overbought}, rsi_oversold={rsi_oversold}")

        logger.info("Step 6: Building feature array (14 features)...")
        final = [
            close_p,             # close
            zone_price,          # zone_price
            distance_from_zone,  # distance_from_zone
            volume,              # volume
            vol_avg,             # vol_avg
            0,                   # outcome_label (dummy)
            high_p,              # high
            low_p,               # low
            vwap,                # vwap
            rsi,                 # rsi
            vwap_distance,       # vwap_distance
            rsi_overbought,      # rsi_overbought
            rsi_oversold,        # rsi_oversold
            0                    # label (dummy)
        ]
        logger.info(f"✓ Feature array built: length={len(final)}")

        logger.info("Step 7: Scaling features...")
        try:
            X_scaled = scaler.transform([final])
            logger.info(f"✓ Features scaled successfully. Shape: {X_scaled.shape}")
        except Exception as se:
            logger.error(f"✗ ERROR in scaler.transform: {str(se)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"Scaling error: {str(se)}"}

        logger.info("Step 8: Making prediction...")
        try:
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0]
            confidence = float(max(prob))
            prediction_str = "BUY" if pred == 1 else "SELL"
            logger.info(f"✓ Prediction complete: {prediction_str} (confidence: {confidence:.4f})")
        except Exception as pe:
            logger.error(f"✗ ERROR in model prediction: {str(pe)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"Prediction error: {str(pe)}"}

        result = {
            "prediction": prediction_str,
            "confidence": confidence,
            "auto_features": {
                "zone_price": zone_price,
                "distance_from_zone": distance_from_zone,
                "vwap_distance": vwap_distance,
                "rsi_overbought": rsi_overbought,
                "rsi_oversold": rsi_oversold
            }
        }
        logger.info("=" * 50)
        logger.info("POST /predict - SUCCESS")
        return result

    except Exception as e:
        logger.error("=" * 50)
        logger.error(f"✗ POST /predict - EXCEPTION: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}


# Batch predict
@app.post("/batch_predict")
async def batch_predict(request: Request):
    logger.info("=" * 50)
    logger.info("POST /batch_predict - Request received")
    try:
        logger.info("Step 1: Parsing JSON request body...")
        import pandas as pd
        from io import StringIO

        data = await request.json()
        logger.info(f"✓ JSON parsed successfully. Keys: {list(data.keys())}")
        
        logger.info("Step 2: Extracting rows from request...")
        rows = data.get("rows", [])
        if not rows:
            logger.error("✗ ERROR: No rows provided in request")
            return {"error": "No rows provided"}
        logger.info(f"✓ Rows extracted: {len(rows)} rows")

        logger.info("Step 3: Converting rows to DataFrame...")
        try:
            df = pd.DataFrame(rows)
            logger.info(f"✓ DataFrame created. Shape: {df.shape}, Columns: {list(df.columns)}")
        except Exception as de:
            logger.error(f"✗ ERROR creating DataFrame: {str(de)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"DataFrame creation error: {str(de)}"}

        logger.info("Step 4: Ensuring required numeric columns exist...")
        numeric_cols = ["open","high","low","close","volume","rsi","vwap","vol_avg","zone_price","distance_from_zone","vwap_distance","rsi_overbought","rsi_oversold","vwap_zone_diff"]
        missing_cols = []
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
                missing_cols.append(col)
        if missing_cols:
            logger.info(f"✓ Added missing columns with zeros: {missing_cols}")
        else:
            logger.info("✓ All required columns present")

        logger.info("Step 5: Computing auto features...")
        try:
            df["zone_price"] = df["zone_price"].where(df["zone_price"]!=0, (df["high"] + df["low"] + df["close"])/3)
            df["distance_from_zone"] = df["distance_from_zone"].where(df["distance_from_zone"]!=0, df["close"] - df["zone_price"])
            df["vwap_distance"] = df["vwap_distance"].where(df["vwap_distance"]!=0, df["close"] - df["vwap"])
            df["rsi_overbought"] = df["rsi_overbought"].where(df["rsi_overbought"]!=0, df["rsi"].apply(lambda x: 1 if x>70 else 0))
            df["rsi_oversold"] = df["rsi_oversold"].where(df["rsi_oversold"]!=0, df["rsi"].apply(lambda x: 1 if x<30 else 0))
            logger.info("✓ Auto features computed")
        except Exception as fe:
            logger.error(f"✗ ERROR computing auto features: {str(fe)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"Feature computation error: {str(fe)}"}

        logger.info("Step 6: Building feature matrix...")
        try:
            X = df[numeric_cols].values
            logger.info(f"✓ Feature matrix built. Shape: {X.shape}")
        except Exception as me:
            logger.error(f"✗ ERROR building feature matrix: {str(me)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"Feature matrix error: {str(me)}"}

        logger.info("Step 7: Scaling features...")
        try:
            X_scaled = scaler.transform(X)
            logger.info(f"✓ Features scaled. Shape: {X_scaled.shape}")
        except Exception as se:
            logger.error(f"✗ ERROR in scaler.transform: {str(se)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"Scaling error: {str(se)}"}

        logger.info("Step 8: Making batch predictions...")
        try:
            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)
            logger.info(f"✓ Predictions complete. Count: {len(preds)}")
        except Exception as pe:
            logger.error(f"✗ ERROR in model prediction: {str(pe)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": f"Prediction error: {str(pe)}"}

        logger.info("Step 9: Formatting results...")
        results = []
        for i, (p, pr) in enumerate(zip(preds, probs)):
            results.append({"prediction": "BUY" if p==1 else "SELL", "confidence": float(max(pr))})
        logger.info(f"✓ Results formatted. Total results: {len(results)}")
        
        logger.info("=" * 50)
        logger.info("POST /batch_predict - SUCCESS")
        return {"results": results}

    except Exception as e:
        logger.error("=" * 50)
        logger.error(f"✗ POST /batch_predict - EXCEPTION: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}

# Live predict
@app.get("/live_predict/{symbol}")
def live_predict(symbol: str):
    logger.info("=" * 50)
    logger.info(f"GET /live_predict/{symbol} - Request received")
    original_symbol = symbol
    
    def make_prediction(close_price: float, method_name: str):
        """Helper function to make prediction from close price"""
        try:
            logger.info(f"  Building feature array from price {close_price}...")
            features = [close_price] + [0]*13
            logger.info(f"  ✓ Feature array built: length={len(features)}")
            
            logger.info("  Scaling features...")
            scaled = scaler.transform([features])
            logger.info(f"  ✓ Features scaled. Shape: {scaled.shape}")
            
            logger.info("  Making prediction...")
            pred = model.predict(scaled)[0]
            prediction_str = "BUY" if pred == 1 else "SELL"
            logger.info(f"  ✓ Prediction: {prediction_str}")
            
            return {
                "market": symbol,
                "current_price": close_price,
                "prediction": prediction_str
            }
        except Exception as pe:
            logger.error(f"  ✗ ERROR in prediction: {str(pe)}")
            logger.error(f"  Traceback: {traceback.format_exc()}")
            raise
    
    try:
        logger.info(f"Step 1: Processing symbol '{symbol}'...")
        symbol = symbol.upper().strip()
        logger.info(f"✓ Symbol normalized to: '{symbol}'")
        
        # Create custom session with proper headers to avoid blocking
        logger.info("Step 2: Creating custom session with headers...")
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        logger.info("✓ Custom session created")
        
        # Try to set yfinance to use our session globally
        try:
            import yfinance.utils as yf_utils
            if hasattr(yf_utils, '_get_session'):
                logger.info("  Configuring yfinance to use custom session...")
                # This is a workaround for yfinance session management
        except:
            pass
        
        # Method 1: Try using Ticker with custom session and info() for current price
        logger.info("Step 3: Method 1 - Trying yf.Ticker() with info()...")
        for attempt in range(3):
            try:
                logger.info(f"  Attempt {attempt + 1}/3: Creating Ticker object...")
                # Try with session if supported, otherwise without
                try:
                    ticker = yf.Ticker(symbol, session=session)
                except TypeError:
                    # Older yfinance versions don't support session parameter
                    logger.info(f"  Session parameter not supported, using default...")
                    ticker = yf.Ticker(symbol)
                logger.info(f"  ✓ Ticker object created")
                
                # Try to get current price from info first (fastest)
                logger.info(f"  Fetching ticker info...")
                try:
                    info = ticker.info
                    if info and isinstance(info, dict):
                        if 'regularMarketPrice' in info and info['regularMarketPrice']:
                            close_price = float(info['regularMarketPrice'])
                            logger.info(f"  ✓ Current price from info (regularMarketPrice): {close_price}")
                            result = make_prediction(close_price, "Ticker.info")
                            logger.info("=" * 50)
                            logger.info(f"GET /live_predict/{symbol} - SUCCESS (Method 1: Ticker.info)")
                            return result
                        elif 'currentPrice' in info and info['currentPrice']:
                            close_price = float(info['currentPrice'])
                            logger.info(f"  ✓ Current price from info (currentPrice): {close_price}")
                            result = make_prediction(close_price, "Ticker.info")
                            logger.info("=" * 50)
                            logger.info(f"GET /live_predict/{symbol} - SUCCESS (Method 1: Ticker.info)")
                            return result
                        elif 'previousClose' in info and info['previousClose']:
                            close_price = float(info['previousClose'])
                            logger.info(f"  ✓ Using previousClose from info: {close_price}")
                            result = make_prediction(close_price, "Ticker.info")
                            logger.info("=" * 50)
                            logger.info(f"GET /live_predict/{symbol} - SUCCESS (Method 1: Ticker.info)")
                            return result
                    else:
                        logger.warning(f"  ✗ Info returned invalid data: {type(info)}")
                except Exception as info_e:
                    logger.warning(f"  ✗ Info method failed: {str(info_e)}")
                    logger.debug(f"  Info error traceback: {traceback.format_exc()}")
                
                # Fallback to history
                logger.info(f"  Fetching history (period='5d', interval='1d')...")
                try:
                    hist = ticker.history(period="5d", interval="1d", timeout=15)
                except TypeError:
                    # Some versions don't support timeout
                    hist = ticker.history(period="5d", interval="1d")
                logger.info(f"  History fetched. Empty: {hist.empty}, Length: {len(hist)}")
                
                if not hist.empty and len(hist) > 0:
                    logger.info(f"  ✓ Data retrieved successfully via Ticker.history")
                    latest = hist.iloc[-1]
                    close_price = float(latest["Close"])
                    logger.info(f"  ✓ Close price extracted: {close_price}")
                    result = make_prediction(close_price, "Ticker.history")
                    logger.info("=" * 50)
                    logger.info(f"GET /live_predict/{symbol} - SUCCESS (Method 1: Ticker.history)")
                    return result
                else:
                    logger.warning(f"  ✗ Ticker history returned empty data")
                    if attempt < 2:
                        wait_time = (attempt + 1) * 1.0
                        logger.info(f"  Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
            except Exception as ticker_e:
                logger.warning(f"  ✗ Ticker attempt {attempt + 1} failed: {str(ticker_e)}")
                logger.warning(f"  Exception type: {type(ticker_e).__name__}")
                if attempt < 2:
                    wait_time = (attempt + 1) * 1.0
                    logger.info(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    logger.debug(f"  Ticker traceback: {traceback.format_exc()}")
        
        # Method 2: Try download with custom session and different periods
        logger.info("Step 4: Method 2 - Trying yf.download() with custom session...")
        periods_intervals = [
            ("1d", "1d"),
            ("5d", "1d"),
            ("1mo", "1d"),
            ("3mo", "1d"),  # Add longer period
        ]
        
        for idx, (period, interval) in enumerate(periods_intervals, 1):
            for attempt in range(3):  # Increase retries to 3
                try:
                    logger.info(f"  Attempt {idx}/{len(periods_intervals)}, retry {attempt + 1}/3: period='{period}', interval='{interval}'...")
                    # Try with session if supported
                    try:
                        df = yf.download(
                            symbol,
                            period=period,
                            interval=interval,
                            progress=False,
                            show_errors=False,
                            timeout=20,  # Increase timeout
                            session=session,
                            threads=False  # Disable threading for more reliable results
                        )
                    except TypeError:
                        # Older versions don't support session parameter
                        logger.info(f"    Session parameter not supported, trying without session...")
                        try:
                            df = yf.download(
                                symbol,
                                period=period,
                                interval=interval,
                                progress=False,
                                show_errors=False,
                                timeout=20,
                                threads=False
                            )
                        except TypeError:
                            # Some versions don't support threads parameter
                            df = yf.download(
                                symbol,
                                period=period,
                                interval=interval,
                                progress=False,
                                show_errors=False,
                                timeout=20
                            )
                    logger.info(f"  Download completed. Empty: {df.empty}, Length: {len(df)}, Shape: {df.shape}")
                    
                    if not df.empty and len(df) > 0:
                        logger.info(f"  ✓ Data retrieved successfully via download method")
                        latest = df.iloc[-1]
                        close_price = float(latest["Close"])
                        logger.info(f"  ✓ Close price extracted: {close_price}")
                        result = make_prediction(close_price, f"Download({period})")
                        logger.info("=" * 50)
                        logger.info(f"GET /live_predict/{symbol} - SUCCESS (Method 2: Download, period={period})")
                        return result
                    else:
                        logger.warning(f"  ✗ Download returned empty dataframe")
                        if attempt < 2:
                            wait_time = (attempt + 1) * 1.5
                            logger.info(f"  Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                except Exception as inner_e:
                    logger.warning(f"  ✗ Download attempt failed: {str(inner_e)}")
                    logger.warning(f"  Exception type: {type(inner_e).__name__}")
                    if attempt < 2:
                        wait_time = (attempt + 1) * 1.5
                        logger.info(f"  Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        logger.debug(f"  Download traceback: {traceback.format_exc()}")
        
        # Method 3: Try direct API call as last resort
        logger.info("Step 5: Method 3 - Trying direct Yahoo Finance API call...")
        try:
            import json
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
            logger.info(f"  Making direct API call to: {url}")
            response = session.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if 'chart' in data and 'result' in data['chart'] and len(data['chart']['result']) > 0:
                    result_data = data['chart']['result'][0]
                    if 'meta' in result_data and 'regularMarketPrice' in result_data['meta']:
                        close_price = float(result_data['meta']['regularMarketPrice'])
                        logger.info(f"  ✓ Current price from direct API: {close_price}")
                        result = make_prediction(close_price, "Direct API")
                        logger.info("=" * 50)
                        logger.info(f"GET /live_predict/{symbol} - SUCCESS (Method 3: Direct API)")
                        return result
                    elif 'indicators' in result_data and 'quote' in result_data['indicators']:
                        quotes = result_data['indicators']['quote'][0]
                        if 'close' in quotes and len(quotes['close']) > 0:
                            # Get last non-null close price
                            close_prices = [p for p in quotes['close'] if p is not None]
                            if close_prices:
                                close_price = float(close_prices[-1])
                                logger.info(f"  ✓ Current price from direct API (indicators): {close_price}")
                                result = make_prediction(close_price, "Direct API (indicators)")
                                logger.info("=" * 50)
                                logger.info(f"GET /live_predict/{symbol} - SUCCESS (Method 3: Direct API)")
                                return result
            else:
                logger.warning(f"  ✗ Direct API returned status code: {response.status_code}")
        except Exception as api_e:
            logger.warning(f"  ✗ Direct API method failed: {str(api_e)}")
            logger.debug(f"  API error traceback: {traceback.format_exc()}")
        
        # If all attempts failed
        logger.error("=" * 50)
        logger.error(f"✗ GET /live_predict/{symbol} - ALL METHODS FAILED")
        logger.error(f"  Symbol: {symbol}")
        logger.error(f"  Tried: Ticker.info (3x), Ticker.history (3x), Download (4 periods x 3 retries), Direct API")
        logger.error(f"  Final error: Unable to fetch data from Yahoo Finance")
        logger.error(f"  This could be due to:")
        logger.error(f"    - Symbol is invalid or delisted")
        logger.error(f"    - Yahoo Finance is blocking requests from this server")
        logger.error(f"    - Network connectivity issues")
        logger.error(f"    - Rate limiting from Yahoo Finance")
        return {
            "error": f"Unable to fetch data for {symbol}. The symbol may be invalid, delisted, or Yahoo Finance may be temporarily unavailable. Please try again later or use 'Predict by Features' instead."
        }
    except Exception as e:
        logger.error("=" * 50)
        logger.error(f"✗ GET /live_predict/{original_symbol} - EXCEPTION: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": f"Failed to fetch data for {symbol.upper()}: {str(e)}"}

# Dashboard
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    logger.info("GET / - Dashboard request received")
    try:
        logger.info(f"Returning HTML page ({len(HTML_PAGE)} characters)")
        return HTMLResponse(content=HTML_PAGE)
    except Exception as e:
        logger.error(f"✗ ERROR serving dashboard: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return HTMLResponse(content=f"<h1>Error loading dashboard</h1><p>{str(e)}</p>", status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)  # only for local testing

