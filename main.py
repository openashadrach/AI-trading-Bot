import os
from datetime import datetime, timezone, timedelta
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from metaapi_cloud_sdk import MetaApi


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="AI Trading Bot",
    version="5.1"
)

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
    broker_name: str
    platform: str
    login: str
    password: str
    server: str

    symbols: List[str] = Field(
        default_factory=lambda: ["EURUSD"]
    )

    daily_profit_target: float = 50.0

    lot_size: float = 0.01

    # True = analyse only, no real order
    # False = can send order to demo/live account
    dry_run: bool = True


# ============================================================
# HOME
# ============================================================

@app.get("/")
async def home():
    return {
        "status": "Online",
        "message": "AI Trading Engine Active"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "metaapi_token_configured": bool(METAAPI_TOKEN),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ============================================================
# MARKET ANALYSIS
# ============================================================

def analyse_market(candles):

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

    # Fast and slow trend averages
    df["ema_fast"] = df["close"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["ema_slow"] = df["close"].ewm(
        span=21,
        adjust=False
    ).mean()

    current = df.iloc[-1]
    previous = df.iloc[-2]

    bullish_trend = (
        current["ema_fast"] > current["ema_slow"]
    )

    bearish_trend = (
        current["ema_fast"] < current["ema_slow"]
    )

    bullish_momentum = (
        current["close"] > previous["close"]
    )

    bearish_momentum = (
        current["close"] < previous["close"]
    )

    # BUY
    if bullish_trend and bullish_momentum:

        return {
            "signal": "BUY",
            "reason": "Bullish trend and momentum detected"
        }

    # SELL
    if bearish_trend and bearish_momentum:

        return {
            "signal": "SELL",
            "reason": "Bearish trend and momentum detected"
        }

    # HOLD
    return {
        "signal": "HOLD",
        "reason": "No valid trading setup detected"
    }


# ============================================================
# FIND EXISTING METAAPI ACCOUNT OR CREATE ONE
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

        # ----------------------------------------------------
        # GET ALL EXISTING METAAPI ACCOUNTS
        # ----------------------------------------------------

        accounts = (
            await metaapi.metatrader_account_api.get_accounts()
        )

        # ----------------------------------------------------
        # LOOK FOR MATCHING ACCOUNT
        # ----------------------------------------------------

        for account in accounts:

            account_login = str(
                getattr(account, "login", "")
            )

            account_server = str(
                getattr(account, "server", "")
            )

            account_platform = str(
                getattr(account, "platform", "")
            ).lower()

            if (
                account_login == str(login)
                and account_server == str(server)
                and account_platform == str(platform).lower()
            ):

                print(
                    f"Using existing MetaApi account: "
                    f"{account.id}"
                )

                return account

        # ----------------------------------------------------
        # NO MATCH FOUND — CREATE ACCOUNT
        # ----------------------------------------------------

        print(
            "No existing MetaApi account found. "
            "Creating a new account."
        )

        account = (
            await metaapi.metatrader_account_api.create_account({
                "name": account_name,
                "type": "cloud",
                "login": str(login),
                "password": password,
                "server": server,
                "platform": platform,
                "magic": 123456
            })
        )

        return account

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Unable to find or create "
                    "MetaApi account"
                ),
                "error": str(error)
            }
        )


# ============================================================
# CONNECT AND TRADE
# ============================================================

@app.post("/connect-and-trade")
async def connect_and_trade(data: MultiPairSetup):

    # --------------------------------------------------------
    # VALIDATE TOKEN
    # --------------------------------------------------------

    if not METAAPI_TOKEN:

        raise HTTPException(
            status_code=500,
            detail="METAAPI_TOKEN is missing from Railway."
        )

    # --------------------------------------------------------
    # VALIDATE PLATFORM
    # --------------------------------------------------------

    platform = data.platform.lower().strip()

    if platform not in ["mt4", "mt5"]:

        raise HTTPException(
            status_code=400,
            detail="Platform must be mt4 or mt5."
        )

    try:

        # ----------------------------------------------------
        # CONNECT METAAPI
        # ----------------------------------------------------

        metaapi = MetaApi(METAAPI_TOKEN)

        # ----------------------------------------------------
        # FIND OR CREATE ACCOUNT
        # ----------------------------------------------------

        account = await get_or_create_account(
            metaapi=metaapi,
            broker_name=data.broker_name,
            login=data.login,
            password=data.password,
            server=data.server,
            platform=platform
        )

        account_id = account.id

        print(
            f"MetaApi account selected: {account_id}"
        )

        # ----------------------------------------------------
        # DEPLOY ONLY IF NECESSARY
        # ----------------------------------------------------

        account_state = str(
            getattr(account, "state", "")
        ).upper()

        if account_state != "DEPLOYED":

            print(
                "Deploying MetaApi account..."
            )

            await account.deploy()

        # ----------------------------------------------------
        # WAIT FOR CONNECTION
        # ----------------------------------------------------

        print(
            "Waiting for MetaTrader connection..."
        )

        await account.wait_connected()

        # ----------------------------------------------------
        # CREATE RPC CONNECTION
        # ----------------------------------------------------

        connection = account.get_rpc_connection()

        await connection.connect()

        await connection.wait_synchronized()

        print(
            "MetaTrader account synchronized."
        )

        # ----------------------------------------------------
        # GET ACCOUNT INFORMATION
        # ----------------------------------------------------

        account_information = (
            await connection.get_account_information()
        )

        account_profit = float(
            account_information.get("profit", 0)
        )

        # ----------------------------------------------------
        # DAILY PROFIT TARGET CHECK
        # ----------------------------------------------------

        if (
            account_profit >=
            data.daily_profit_target
        ):

            return {
                "status": "target_reached",
                "message": (
                    "Daily profit target reached. "
                    "No new trades opened."
                ),
                "account_id": account_id,
                "profit": account_profit
            }

        # ====================================================
        # SCAN SYMBOLS
        # ====================================================

        results = []

        executed_trades = []

        for symbol in data.symbols:

            symbol = symbol.strip()

            try:

                print(
                    f"Scanning symbol: {symbol}"
                )

                # ------------------------------------------------
                # GET CANDLES
                # ------------------------------------------------

                start_time = (
                    datetime.now(timezone.utc)
                    - timedelta(hours=250)
                )

                candles = (
                    await connection.get_candles(
                        symbol,
                        "1h",
                        start_time,
                        200
                    )
                )

                if not candles:

                    results.append({
                        "symbol": symbol,
                        "status": "error",
                        "reason": (
                            "No candle data returned."
                        )
                    })

                    continue

                # ------------------------------------------------
                # ANALYSE MARKET
                # ------------------------------------------------

                analysis = analyse_market(
                    candles
                )

                signal = analysis["signal"]

                # ------------------------------------------------
                # HOLD
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

                price = (
                    await connection.get_symbol_price(
                        symbol
                    )
                )

                ask = float(
                    price.get("ask", 0)
                )

                bid = float(
                    price.get("bid", 0)
                )

                if ask <= 0 or bid <= 0:

                    results.append({
                        "symbol": symbol,
                        "status": "error",
                        "reason": (
                            "Invalid market price returned."
                        )
                    })

                    continue

                # ------------------------------------------------
                # CALCULATE BUY LEVELS
                # ------------------------------------------------

                if signal == "BUY":

                    entry_price = ask

                    stop_loss = (
                        entry_price * 0.995
                    )

                    take_profit = (
                        entry_price * 1.01
                    )

                # ------------------------------------------------
                # CALCULATE SELL LEVELS
                # ------------------------------------------------

                elif signal == "SELL":

                    entry_price = bid

                    stop_loss = (
                        entry_price * 1.005
                    )

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
                                "comment": (
                                    "AI_TRADING_BOT_BUY"
                                )
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
                                "comment": (
                                    "AI_TRADING_BOT_SELL"
                                )
                            }
                        )
                    )

                # ------------------------------------------------
                # RECORD RESULT
                # ------------------------------------------------

                trade = {
                    "symbol": symbol,
                    "signal": signal,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "order_result": order_result
                }

                executed_trades.append(
                    trade
                )

                results.append({
                    "symbol": symbol,
                    "status": "trade_executed",
                    "signal": signal,
                    "reason": analysis["reason"]
                })

            # ====================================================
            # SHOW REAL ERRORS
            # ====================================================

            except Exception as symbol_error:

                print(
                    f"ERROR processing "
                    f"{symbol}: {str(symbol_error)}"
                )

                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "reason": str(symbol_error)
                })

        # ========================================================
        # FINAL RESPONSE
        # ========================================================

        return {
            "status": (
                "success"
                if executed_trades
                else "completed"
            ),
            "message": (
                "Trade scan completed."
                if executed_trades
                else "Market scan completed. "
                     "No trades were executed."
            ),
            "account_id": account_id,
            "account_profit": account_profit,
            "trades": executed_trades,
            "results": results
        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            f"CONNECTION ERROR: {str(error)}"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Trading engine failed."
                ),
                "error": str(error)
            }
        )
