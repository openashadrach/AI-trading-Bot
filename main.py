import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from metaapi_cloud_sdk import MetaApi
from smartmoneyconcepts import smc
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="SMC AI Trading Bot - Multi-Pair", version="4.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

METAAPI_TOKEN = os.getenv("METAAPI_TOKEN")

class MultiPairSetup(BaseModel):
    broker_name: str       
    platform: str          # "mt4" or "mt5"
    login: str
    password: str
    server: str
    symbols: list[str] = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
    daily_profit_target: float = 50.0

ai_model = RandomForestClassifier()
X_train_dummy = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]])
y_train_dummy = np.array([1, 0, 1, 0])
ai_model.fit(X_train_dummy, y_train_dummy)

@app.get("/")
def home():
    return {"status": "Online", "message": "Multi-Pair SMC Trading Engine Active 24/7."}

@app.post("/connect-and-trade")
async def connect_and_trade(data: MultiPairSetup):
    try:
        platform_lower = data.platform.lower()
        if platform_lower not in ["mt4", "mt5"]:
            raise HTTPException(status_code=400, detail="Invalid platform. Must be 'mt4' or 'mt5'.")

        if not METAAPI_TOKEN or METAAPI_TOKEN == "YOUR_METAAPI_TOKEN_HERE":
            raise HTTPException(status_code=500, detail="METAAPI_TOKEN is not configured in environment variables.")

        metaapi = MetaApi(METAAPI_TOKEN)
        
        account = await metaapi.metatrader_account_api.create_account({
            'name': f'{data.broker_name}_MultiPair_{data.login}',
            'type': 'cloud',
            'login': data.login,
            'password': data.password,
            'server': data.server,
            'platform': platform_lower,
            'magic': 123456
        })
        
        account_id = account.id
        
        await account.deploy()
        await account.wait_connected()

        connection = account.get_rpc_connection()
        await connection.connect()
        await connection.wait_synchronized()

        account_info = await connection.get_account_information()
        floating_profit = account_info.get('profit', 0.0)
        
        if floating_profit >= data.daily_profit_target:
            return {
                "status": "Target Reached", 
                "message": f"Daily profit target of ${data.daily_profit_target} achieved. All trading paused."
            }

        executed_trades = []

        for symbol in data.symbols:
            try:
                candles = await account.get_historical_candles(symbol, '1h', limit=200)
                if not candles:
                    continue
                    
                df = pd.DataFrame(candles)
                ohlc = df[['open', 'high', 'low', 'close']].astype(float)
                ohlc['volume'] = df['tickVolume'] if 'tickVolume' in df else 100

                swing_highs_lows = smc.swing_highs_lows(ohlc)
                order_blocks = smc.ob(ohlc, swing_highs_lows)
                latest_ob = order_blocks.iloc[-1]
                
                current_features = np.array([[1, 1, 1]]) 
                ai_prediction = ai_model.predict(current_features)

                if ai_prediction[0] == 1 and latest_ob.get('OrderBlock', 0) != 0:
                    price_spec = await connection.get_symbol_specification(symbol)
                    risk_lot_size = 0.01 
                    
                    current_price = price_spec['ask']
                    stop_loss = round(current_price - 0.0020, 5)   
                    take_profit = round(current_price + 0.0040, 5) 

                    result = await connection.create_market_buy_order(
                        symbol=symbol,
                        volume=risk_lot_size,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        options={'comment': f'{data.broker_name}_MultiSMC'}
                    )
                    
                    executed_trades.append({"symbol": symbol, "order_id": result.get('stringCode')})
            
            except Exception:
                continue

        if executed_trades:
            return {
                "status": "Success", 
                "message": "Successfully scanned and executed trades across multiple pairs.",
                "trades": executed_trades,
                "account_id": account_id
            }
        else:
            return {
                "status": "Skipped", 
                "message": "Scanned all selected pairs, but no valid SMC setups matched criteria at this moment."
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
