import os
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from metaapi_cloud_sdk import MetaApi


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="SMC AI Trading Bot - Multi-Pair",
    version="5.0",
    description="Multi-pair MetaTrader trading bot with market analysis."
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

METAAPI_TOKEN = os.getenv("METAAPI_TOKEN")


# ============================================================
# REQUEST MODEL
# ============================================================

class MultiPairSetup(BaseModel):
    broker_name: str = Field(..., min_length=1)
    platform: str = Field(..., min_length=2)
    login: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    server: str = Field(..., min_length=1)

    symbols: List[str] = Field(
        default_factory=lambda: [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "XAUUSD"
        ]
    )

    daily_profit_target: float = Field(
        default=50.0,
        gt=0
    )

    risk_percent: float = Field(
        default=1.0,
        gt=0,
        le=5
    )

    lot_size: float = Field(
        default=0.01,
        gt=0
    )

    dry_run: bool = False


# ============================================================
# HOME
# ============================================================

@app.get("/")
async def home():
    return {
        "status": "Online",
        "message": "Multi-Pair Trading Engine Active",
        "version": "5.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "metaapi_configured": bool(METAAPI_TOKEN),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ============================================================
# MARKET ANALYSIS
# ============================================================

def analyse_market(candles: list):
    """
    Simple market structure and momentum analysis.

    Returns:
    BUY
    SELL
    HOLD
    """

    if not candles or len(candles) < 50:
        return {
            "signal": "HOLD",
            "reason": "Not enough candle data"
        }

    rows = []

    for candle in candles:
        rows.append({
            "time": candle.get("time"),
            "open": float(candle.get("open", 0)),
            "high": float(candle.get("high", 0)),
            "low": float(candle.get("low", 0)),
            "close": float(candle.get("close", 0))
        })

    df = pd.DataFrame(rows)

    if len(df) < 50:
        return {
            "signal": "HOLD",
            "reason": "Insufficient valid candles"
        }

    # Moving averages
    df["ema_fast"] = df["close"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["ema_slow"] = df["close"].ewm(
        span=21,
        adjust=False
    ).mean()

    # Recent highs and lows
    recent = df.tail(20)

    current = df.iloc[-1]
    previous = df.iloc[-2]

    current_price = current["close"]

    highest_recent = recent["high"].max()
    lowest_recent = recent["low"].min()

    bullish_trend = current["ema_fast"] > current["ema_slow"]
    bearish_trend = current["ema_fast"] < current["ema_slow"]

    bullish_momentum = current["close"] > previous["close"]
    bearish_momentum = current["close"] < previous["close"]

    # --------------------------------------------------------
    # BUY SIGNAL
    # --------------------------------------------------------

    if bullish_trend and bullish_momentum:

        return {
            "signal": "BUY",
            "reason": (
                "Bullish trend confirmed by fast EMA above slow EMA "
                "and positive candle momentum"
            ),
            "price": current_price,
            "recent_high": highest_recent,
            "recent_low": lowest_recent
        }

    # --------------------------------------------------------
    # SELL SIGNAL
    # --------------------------------------------------------

    if bearish_trend and bearish_momentum:

        return {
            "signal": "SELL",
            "reason": (
                "Bearish trend confirmed by fast EMA below slow EMA "
                "and negative candle momentum"
            ),
            "price": current_price,
            "recent_high": highest_recent,
            "recent_low": lowest_recent
        }

    # --------------------------------------------------------
    # HOLD
    # --------------------------------------------------------

    return {
        "signal": "HOLD",
        "reason": "No clear market structure confirmation",
        "price": current_price,
        "recent_high": highest_recent,
        "recent_low": lowest_recent
    }


# ============================================================
# GET OR CREATE METAAPI ACCOUNT
# ============================================================

async def get_or_create_account(
    metaapi,
    broker_name,
    login,
    password,
    server,
    platform
):

    account_name = (
        f"{broker_name}_{platform}_{login}"
    )

    try:

        account = await metaapi.metatrader_account_api.create_account({
            "name": account_name,
            "type": "cloud",
            "login": str(login),
            "password": password,
            "server": server,
            "platform": platform,
            "magic": 123456
        })

        return account

    except Exception as create_error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to create MetaAPI account. "
                f"Check login, password, server and platform. "
                f"Error: {str(create_error)}"
            )
        )


# ============================================================
# CONNECT AND TRADE
# ============================================================

@app.post("/connect-and-trade")
async def connect_and_trade(data: MultiPairSetup):

    # --------------------------------------------------------
    # VALIDATE PLATFORM
    # --------------------------------------------------------

    platform = data.platform.lower().strip()

    if platform not in ["mt4", "mt5"]:
        raise HTTPException(
            status_code=400,
            detail="Platform must be mt4 or mt5"
        )

    # --------------------------------------------------------
    # VALIDATE METAAPI TOKEN
    # --------------------------------------------------------

    if not METAAPI_TOKEN:

        raise HTTPException(
            status_code=500,
            detail=(
                "METAAPI_TOKEN is not configured. "
                "Add it to Railway Environment Variables."
            )
        )

    # --------------------------------------------------------
    # CONNECT TO METAAPI
    # --------------------------------------------------------

    try:

        metaapi = MetaApi(METAAPI_TOKEN)

        account = await get_or_create_account(
            metaapi=metaapi,
            broker_name=data.broker_name,
            login=data.login,
            password=data.password,
            server=data.server,
            platform=platform
        )

        account_id = account.id

        # Deploy account
        await account.deploy()

        # Wait for broker connection
        await account.wait_connected()

        # Create RPC connection
        connection = account.get_rpc_connection()

        await connection.connect()

        await connection.wait_synchronized()

    except Exception as connection_error:

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to connect to MetaTrader account",
                "error": str(connection_error)
            }
        )

    # ========================================================
    # ACCOUNT INFORMATION
    # ========================================================

    try:

        account_information = (
            await connection.get_account_information()
        )

    except Exception as account_error:

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Connected but unable to retrieve account information",
                "error": str(account_error)
            }
        )

    # ========================================================
    # DAILY PROFIT CHECK
    # ========================================================

    floating_profit = float(
        account_information.get("profit", 0)
    )

    if floating_profit >= data.daily_profit_target:

        return {
            "status": "target_reached",
            "message": (
                f"Profit target of "
                f"${data.daily_profit_target} reached. "
                f"Trading is paused."
            ),
            "profit": floating_profit,
            "account_id": account_id
        }

    # ========================================================
    # RESULTS
    # ========================================================

    results = []

    executed_trades = []

    # ========================================================
    # SCAN SYMBOLS
    # ========================================================

    for symbol in data.symbols:

        symbol = symbol.strip()

        try:

            # ------------------------------------------------
            # GET HISTORICAL CANDLES
            # ------------------------------------------------

            start_time = (
                datetime.now(timezone.utc)
                - timedelta(hours=250)
            )

            candles = await connection.get_candles(
                symbol,
                "1h",
                start_time,
                200
            )

            if not candles:

                results.append({
                    "symbol": symbol,
                    "status": "skipped",
                    "reason": "No candle data returned"
                })

                continue

            # ------------------------------------------------
            # ANALYSE MARKET
            # ------------------------------------------------

            analysis = analyse_market(candles)

            signal = analysis["signal"]

            # ------------------------------------------------
            # NO TRADE
            # ------------------------------------------------

            if signal == "HOLD":

                results.append({
                    "symbol": symbol,
                    "status": "skipped",
                    "signal": signal,
                    "reason": analysis["reason"]
                })

                continue

            # ------------------------------------------------
            # GET LIVE PRICE
            # ------------------------------------------------

            price = await connection.get_symbol_price(
                symbol
            )

            ask = float(price.get("ask", 0))
            bid = float(price.get("bid", 0))

            if ask <= 0 or bid <= 0:

                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "reason": "Invalid live price returned"
                })

                continue

            # =================================================
            # BUY
            # =================================================

            if signal == "BUY":

                entry_price = ask

                # 0.5% stop loss
                stop_loss = (
                    entry_price * 0.995
                )

                # 1% take profit
                take_profit = (
                    entry_price * 1.01
                )

            # =================================================
            # SELL
            # =================================================

            elif signal == "SELL":

                entry_price = bid

                # 0.5% stop loss
                stop_loss = (
                    entry_price * 1.005
                )

                # 1% take profit
                take_profit = (
                    entry_price * 0.99
                )

            # ------------------------------------------------
            # DRY RUN
            # ------------------------------------------------

            if data.dry_run:

                results.append({
                    "symbol": symbol,
                    "status": "simulation",
                    "signal": signal,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "reason": analysis["reason"]
                })

                continue

            # =================================================
            # EXECUTE BUY
            # =================================================

            if signal == "BUY":

                order_result = (
                    await connection.create_market_buy_order(
                        symbol,
                        data.lot_size,
                        stop_loss,
                        take_profit,
                        {
                            "comment": "SMC_AI_BOT_BUY"
                        }
                    )
                )

            # =================================================
            # EXECUTE SELL
            # =================================================

            elif signal == "SELL":

                order_result = (
                    await connection.create_market_sell_order(
                        symbol,
                        data.lot_size,
                        stop_loss,
                        take_profit,
                        {
                            "comment": "SMC_AI_BOT_SELL"
                        }
                    )
                )

            trade = {
                "symbol": symbol,
                "signal": signal,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "order_result": order_result
            }

            executed_trades.append(trade)

            results.append({
                "symbol": symbol,
                "status": "trade_executed",
                "signal": signal,
                "reason": analysis["reason"]
            })

        except Exception as symbol_error:

            # IMPORTANT:
            # We now show the actual error instead of silently
            # hiding it.

            results.append({
                "symbol": symbol,
                "status": "error",
                "reason": str(symbol_error)
            })

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    if executed_trades:

        return {
            "status": "success",
            "message": (
                "Market scan completed and one or more "
                "trades were executed."
            ),
            "account_id": account_id,
            "account_profit": floating_profit,
            "trades": executed_trades,
            "results": results
        }

    return {
        "status": "completed",
        "message": (
            "Market scan completed. No trades were executed."
        ),
        "account_id": account_id,
        "account_profit": floating_profit,
        "results": results
    }
