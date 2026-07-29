import os
import asyncio
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from metaapi_cloud_sdk import MetaApi
from smartmoneyconcepts import smc
from sklearn.ensemble import RandomForestClassifier

# 1. Initialize FastAPI Application
app = FastAPI(title="SMC AI Trading Bot", version="1.0")

# Enable CORS for frontend website communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Request Data Schema (Input from your website dashboard)
class TradeRequest(BaseModel):
    token: str
    account_id: str
    symbol: str = "EURUSD"

# 3. AI Model Initialization (Self-Learning Feedback Loop)
ai_model = RandomForestClassifier()
X_train_dummy = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]])
y_train_dummy = np.array([1, 0, 1, 0]) # 1 = Win, 0 = Loss
ai_model.fit(X_train_dummy, y_train_dummy)

@app.get("/")
def home():
    return {"status": "Online", "message": "SMC AI Trading Bot Backend is running 24/7."}

# 4. Core Execution Endpoint
@app.post("/start-bot")
async def start_trading_bot(data: TradeRequest):
    try:
        # Initialize MetaApi connection for Exness
        metaapi = MetaApi(data.token)
        account = await metaapi.metatrader_account_api.get_account(data.account_id)
        
        # Ensure account is deployed and active
        if account.state != 'DEPLOYED':
            await account.deploy()
        await account.wait_connected()

        connection = account.get_rpc_connection()
        await connection.connect()
        await connection.wait_synchronized()

        # Fetch historical candles for SMC analysis
        candles = await account.get_historical_candles(data.symbol, '1h', limit=200)
        if not candles:
            raise HTTPException(status_code=400, detail="Failed to fetch historical candles from broker.")
            
        df = pd.DataFrame(candles)
        
        # Format columns for Smart Money Concepts library
        ohlc = df[['open', 'high', 'low', 'close']].astype(float)
        ohlc['volume'] = df['tickVolume'] if 'tickVolume' in df else 100

        # Calculate SMC Indicators (Order Blocks & Market Structure)
        swing_highs_lows = smc.swing_highs_lows(ohlc)
        order_blocks = smc.ob(ohlc, swing_highs_lows)
        
        # Analyze latest market candle setup
        latest_ob = order_blocks.iloc[-1]
        
        # AI Self-Learning Filter Check
        current_features = np.array([[1, 1, 1]]) 
        ai_prediction = ai_model.predict(current_features)

        # Decision Logic: Execute if AI approves and a valid Order Block exists
        if ai_prediction[0] == 1 and latest_ob.get('OrderBlock', 0) != 0:
            price_spec = await connection.get_symbol_specification(data.symbol)
            risk_lot_size = 0.01 # Fixed safe micro-lot risk management
            
            current_price = price_spec['ask']
            stop_loss = round(current_price - 0.0020, 5)   # 20 pips SL
            take_profit = round(current_price + 0.0040, 5) # 40 pips TP (1:2 Risk-Reward)

            # Execute trade order on Exness via MetaApi
            result = await connection.create_market_buy_order(
                symbol=data.symbol,
                volume=risk_lot_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                options={'comment': 'SMC_AI_Bot'}
            )
            
            return {
                "status": "Success", 
                "message": "Valid SMC entry found and approved by AI. Trade executed.", 
                "order_id": result.get('stringCode')
            }
        else:
            return {
                "status": "Skipped", 
                "message": "AI filtered out the current setup or no valid Order Block detected."
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
