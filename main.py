import os
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

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
    version="6.0"
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
# RUNTIME STORAGE
#
# This stores active bot configurations while Railway is running.
# If Railway restarts, the UI can reconnect the account again.
# ============================================================

BOT_CONFIGS: Dict[str, dict] = {}

BOT_TASKS: Dict[str, asyncio.Task] = {}

BOT_STATUS: Dict[str, dict] = {}

# Analytics is kept in memory while the Railway service is running.
# Closed-trade history is also requested from MetaApi when available.
ANALYTICS_STATE: Dict[str, dict] = {}


# ============================================================
# REQUEST MODELS
# ============================================================

class AccountConnectRequest(BaseModel):

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

    scan_interval_seconds: int = 60


class StartTradingRequest(BaseModel):

    symbols: Optional[List[str]] = None

    daily_profit_target: Optional[float] = None

    lot_size: Optional[float] = None

    scan_interval_seconds: Optional[int] = None


class LegacyMultiPairSetup(AccountConnectRequest):

    # False = real trading
    # True = analysis only
    dry_run: bool = True


# ============================================================
# METAAPI VALIDATION
# ============================================================

def validate_metaapi_token():

    if not METAAPI_TOKEN:

        raise HTTPException(
            status_code=500,
            detail="METAAPI_TOKEN is missing from Railway environment variables."
        )


# ============================================================
# HOME
# ============================================================

@app.get("/")
async def home():

    return {
        "status": "Online",
        "message": "AI Trading Engine Active",
        "version": "6.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "metaapi_token_configured": bool(METAAPI_TOKEN),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat()
    }


# ============================================================
# MARKET ANALYSIS
#
# EMA 9 + EMA 21
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
        current["ema_fast"]
        >
        current["ema_slow"]
    )

    bearish_trend = (
        current["ema_fast"]
        <
        current["ema_slow"]
    )

    bullish_momentum = (
        current["close"]
        >
        previous["close"]
    )

    bearish_momentum = (
        current["close"]
        <
        previous["close"]
    )

    if bullish_trend and bullish_momentum:

        return {
            "signal": "BUY",
            "reason": (
                "Bullish EMA trend and momentum detected"
            )
        }

    if bearish_trend and bearish_momentum:

        return {
            "signal": "SELL",
            "reason": (
                "Bearish EMA trend and momentum detected"
            )
        }

    return {
        "signal": "HOLD",
        "reason": "No valid trading setup detected"
    }


# ============================================================
# GET METAAPI CLIENT
# ============================================================

def get_metaapi():

    validate_metaapi_token()

    return MetaApi(METAAPI_TOKEN)


# ============================================================
# FIND METAAPI ACCOUNT BY ID
# ============================================================

async def get_account_by_id(
    metaapi,
    account_id: str
):

    accounts = (
        await metaapi
        .metatrader_account_api
        .get_accounts()
    )

    for account in accounts:

        if str(account.id) == str(account_id):

            return account

    raise HTTPException(
        status_code=404,
        detail=(
            f"MetaApi account {account_id} "
            "was not found."
        )
    )


# ============================================================
# FIND EXISTING ACCOUNT OR CREATE ONE
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

        accounts = (
            await metaapi
            .metatrader_account_api
            .get_accounts()
        )

        for account in accounts:

            account_login = str(
                getattr(
                    account,
                    "login",
                    ""
                )
            )

            account_server = str(
                getattr(
                    account,
                    "server",
                    ""
                )
            )

            account_platform = str(
                getattr(
                    account,
                    "platform",
                    ""
                )
            ).lower()

            if (
                account_login == str(login)
                and account_server == str(server)
                and account_platform
                == str(platform).lower()
            ):

                print(
                    "Using existing MetaApi account: "
                    f"{account.id}"
                )

                return account

        print(
            "No matching MetaApi account found. "
            "Creating a new account."
        )

        account = (
            await metaapi
            .metatrader_account_api
            .create_account({
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

    except HTTPException:

        raise

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
# DEPLOY AND CONNECT ACCOUNT
# ============================================================

async def connect_account(
    account
):

    account_state = str(
        getattr(
            account,
            "state",
            ""
        )
    ).upper()

    if account_state != "DEPLOYED":

        print(
            f"Deploying account {account.id}"
        )

        await account.deploy()

    print(
        f"Waiting for account "
        f"{account.id} to connect..."
    )

    await account.wait_connected()

    connection = (
        account.get_rpc_connection()
    )

    await connection.connect()

    await connection.wait_synchronized()

    return connection


# ============================================================
# FORMAT ACCOUNT INFORMATION
# ============================================================

def format_account_response(
    account,
    account_information,
    account_id
):

    return {
        "account_id": account_id,
        "name": getattr(
            account,
            "name",
            None
        ),
        "login": str(
            getattr(
                account,
                "login",
                ""
            )
        ),
        "server": getattr(
            account,
            "server",
            None
        ),
        "platform": getattr(
            account,
            "platform",
            None
        ),
        "state": getattr(
            account,
            "state",
            None
        ),
        "connection_status": "connected",
        "balance": account_information.get(
            "balance",
            0
        ),
        "equity": account_information.get(
            "equity",
            0
        ),
        "margin": account_information.get(
            "margin",
            0
        ),
        "free_margin": account_information.get(
            "freeMargin",
            0
        ),
        "profit": account_information.get(
            "profit",
            0
        ),
        "currency": account_information.get(
            "currency",
            None
        )
    }


# ============================================================
# CONNECT ACCOUNT
#
# This endpoint connects the account only.
# It DOES NOT start trading.
# ============================================================

@app.post("/accounts/connect")
async def connect_account_endpoint(
    data: AccountConnectRequest
):

    validate_metaapi_token()

    platform = (
        data.platform
        .lower()
        .strip()
    )

    if platform not in [
        "mt4",
        "mt5"
    ]:

        raise HTTPException(
            status_code=400,
            detail=(
                "Platform must be "
                "'mt4' or 'mt5'."
            )
        )

    try:

        metaapi = get_metaapi()

        account = (
            await get_or_create_account(
                metaapi=metaapi,
                broker_name=data.broker_name,
                login=data.login,
                password=data.password,
                server=data.server,
                platform=platform
            )
        )

        account_id = str(
            account.id
        )

        connection = (
            await connect_account(
                account
            )
        )

        account_information = (
            await connection
            .get_account_information()
        )

        BOT_CONFIGS[
            account_id
        ] = {
            "broker_name": data.broker_name,
            "platform": platform,
            "login": data.login,
            "server": data.server,
            "symbols": data.symbols,
            "daily_profit_target":
                data.daily_profit_target,
            "lot_size":
                data.lot_size,
            "scan_interval_seconds":
                data.scan_interval_seconds
        }

        if account_id not in BOT_STATUS:

            BOT_STATUS[
                account_id
            ] = {
                "trading_active": False,
                "last_scan": None,
                "last_error": None,
                "last_results": []
            }

        get_analytics_state(account_id)

        return {
            "status": "connected",
            "message": (
                "Account connected successfully. "
                "Trading has not started yet."
            ),
            "account": (
                format_account_response(
                    account,
                    account_information,
                    account_id
                )
            )
        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            f"CONNECT ERROR: {str(error)}"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Failed to connect account.",
                "error":
                    str(error)
            }
        )


# ============================================================
# GET ACCOUNT DETAILS
# ============================================================

@app.get("/accounts/{account_id}")
async def get_account_details(
    account_id: str
):

    try:

        metaapi = get_metaapi()

        account = (
            await get_account_by_id(
                metaapi,
                account_id
            )
        )

        connection = (
            await connect_account(
                account
            )
        )

        account_information = (
            await connection
            .get_account_information()
        )

        response = (
            format_account_response(
                account,
                account_information,
                account_id
            )
        )

        response["bot"] = (
            BOT_STATUS.get(
                account_id,
                {
                    "trading_active": False,
                    "last_scan": None,
                    "last_error": None,
                    "last_results": []
                }
            )
        )

        response["configuration"] = (
            BOT_CONFIGS.get(
                account_id,
                {}
            )
        )

        return response

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# GET OPEN POSITIONS
# ============================================================

@app.get(
    "/accounts/{account_id}/positions"
)
async def get_positions(
    account_id: str
):

    try:

        metaapi = get_metaapi()

        account = (
            await get_account_by_id(
                metaapi,
                account_id
            )
        )

        connection = (
            await connect_account(
                account
            )
        )

        positions = (
            await connection
            .get_positions()
        )

        return {
            "account_id": account_id,
            "count": len(positions),
            "positions": positions
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# CHECK IF SYMBOL ALREADY HAS A POSITION
#
# Prevents opening unlimited duplicate positions
# ============================================================

def has_open_position(
    positions,
    symbol
):

    for position in positions:

        position_symbol = position.get(
            "symbol",
            ""
        )

        if position_symbol == symbol:

            return True

    return False


# ============================================================
# SINGLE MARKET SCAN
#
# This is the core trading function.
# ============================================================

async def run_market_scan(
    account_id: str,
    dry_run: bool = False
):

    if account_id not in BOT_CONFIGS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Account configuration not found. "
                "Connect the account first."
            )
        )

    config = BOT_CONFIGS[
        account_id
    ]

    metaapi = get_metaapi()

    account = (
        await get_account_by_id(
            metaapi,
            account_id
        )
    )

    connection = (
        await connect_account(
            account
        )
    )

    account_information = (
        await connection
        .get_account_information()
    )

    account_profit = float(
        account_information.get(
            "profit",
            0
        )
    )

    daily_profit_target = float(
        config.get(
            "daily_profit_target",
            50
        )
    )

    if (
        account_profit
        >=
        daily_profit_target
    ):

        return {
            "status": "target_reached",
            "message": (
                "Daily profit target reached. "
                "No new trades opened."
            ),
            "account_id": account_id,
            "profit": account_profit,
            "results": []
        }

    results = []

    executed_trades = []

    positions = (
        await connection
        .get_positions()
    )

    symbols = config.get(
        "symbols",
        []
    )

    for symbol in symbols:

        symbol = (
            symbol
            .strip()
        )

        try:

            # --------------------------------------------
            # PREVENT DUPLICATE POSITION
            # --------------------------------------------

            if has_open_position(
                positions,
                symbol
            ):

                results.append({
                    "symbol": symbol,
                    "status": "skipped",
                    "reason": (
                        "An open position already "
                        "exists for this symbol."
                    )
                })

                continue

            print(
                f"Scanning symbol: "
                f"{symbol}"
            )

            # --------------------------------------------
            # GET CANDLES
            # --------------------------------------------

            start_time = (
                datetime.now(
                    timezone.utc
                )
                -
                timedelta(
                    hours=250
                )
            )

            candles = (
                await connection
                .get_candles(
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
                        "No candle data returned. "
                        "Check the broker symbol name."
                    )
                })

                continue

            # --------------------------------------------
            # ANALYSE MARKET
            # --------------------------------------------

            analysis = (
                analyse_market(
                    candles
                )
            )

            signal = (
                analysis["signal"]
            )

            # --------------------------------------------
            # HOLD
            # --------------------------------------------

            if signal == "HOLD":

                results.append({
                    "symbol": symbol,
                    "status": "skipped",
                    "signal": signal,
                    "reason":
                        analysis["reason"]
                })

                continue

            # --------------------------------------------
            # GET LIVE PRICE
            # --------------------------------------------

            price = (
                await connection
                .get_symbol_price(
                    symbol
                )
            )

            ask = float(
                price.get(
                    "ask",
                    0
                )
            )

            bid = float(
                price.get(
                    "bid",
                    0
                )
            )

            if ask <= 0 or bid <= 0:

                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "reason": (
                        "Invalid market price. "
                        "Check symbol availability."
                    )
                })

                continue

            # --------------------------------------------
            # BUY LEVELS
            # --------------------------------------------

            if signal == "BUY":

                entry_price = ask

                stop_loss = (
                    entry_price
                    *
                    0.995
                )

                take_profit = (
                    entry_price
                    *
                    1.01
                )

            # --------------------------------------------
            # SELL LEVELS
            # --------------------------------------------

            elif signal == "SELL":

                entry_price = bid

                stop_loss = (
                    entry_price
                    *
                    1.005
                )

                take_profit = (
                    entry_price
                    *
                    0.99
                )

            # --------------------------------------------
            # DRY RUN
            # --------------------------------------------

            if dry_run:

                results.append({
                    "symbol": symbol,
                    "status": "simulation",
                    "signal": signal,
                    "entry_price":
                        entry_price,
                    "stop_loss":
                        stop_loss,
                    "take_profit":
                        take_profit,
                    "reason":
                        analysis["reason"]
                })

                continue

            # --------------------------------------------
            # EXECUTE BUY
            # --------------------------------------------

            if signal == "BUY":

                order_result = (
                    await connection
                    .create_market_buy_order(
                        symbol,
                        float(
                            config[
                                "lot_size"
                            ]
                        ),
                        stop_loss,
                        take_profit,
                        {
                            "comment":
                                "AI_TRADING_BOT_BUY"
                        }
                    )
                )

            # --------------------------------------------
            # EXECUTE SELL
            # --------------------------------------------

            elif signal == "SELL":

                order_result = (
                    await connection
                    .create_market_sell_order(
                        symbol,
                        float(
                            config[
                                "lot_size"
                            ]
                        ),
                        stop_loss,
                        take_profit,
                        {
                            "comment":
                                "AI_TRADING_BOT_SELL"
                        }
                    )
                )

            trade = {
                "symbol": symbol,
                "signal": signal,
                "entry_price":
                    entry_price,
                "stop_loss":
                    stop_loss,
                "take_profit":
                    take_profit,
                "order_result":
                    order_result
            }

            executed_trades.append(
                trade
            )

            results.append({
                "symbol": symbol,
                "status":
                    "trade_executed",
                "signal": signal,
                "reason":
                    analysis["reason"]
            })

        except Exception as symbol_error:

            print(
                f"ERROR processing "
                f"{symbol}: "
                f"{str(symbol_error)}"
            )

            results.append({
                "symbol": symbol,
                "status": "error",
                "reason":
                    str(symbol_error)
            })

    return {
        "status": (
            "success"
            if executed_trades
            else "completed"
        ),
        "account_id":
            account_id,
        "account_profit":
            account_profit,
        "trades":
            executed_trades,
        "results":
            results
    }


# ============================================================
# BACKGROUND TRADING LOOP
# ============================================================

async def trading_loop(
    account_id: str
):

    print(
        f"Trading loop started "
        f"for {account_id}"
    )

    while True:

        try:

            status = (
                BOT_STATUS.get(
                    account_id,
                    {}
                )
            )

            if not status.get(
                "trading_active",
                False
            ):

                print(
                    f"Trading loop stopped "
                    f"for {account_id}"
                )

                break

            result = (
                await run_market_scan(
                    account_id,
                    dry_run=False
                )
            )

            BOT_STATUS[
                account_id
            ]["last_scan"] = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            BOT_STATUS[
                account_id
            ]["last_results"] = (
                result
            )

            record_scan_analytics(
                account_id,
                result
            )

            BOT_STATUS[
                account_id
            ]["last_error"] = None

        except asyncio.CancelledError:

            print(
                f"Trading task cancelled "
                f"for {account_id}"
            )

            break

        except Exception as error:

            print(
                f"TRADING LOOP ERROR "
                f"{account_id}: "
                f"{str(error)}"
            )

            if account_id in BOT_STATUS:

                BOT_STATUS[
                    account_id
                ]["last_error"] = (
                    str(error)
                )

        config = (
            BOT_CONFIGS.get(
                account_id,
                {}
            )
        )

        interval = int(
            config.get(
                "scan_interval_seconds",
                60
            )
        )

        interval = max(
            interval,
            15
        )

        await asyncio.sleep(
            interval
        )


# ============================================================
# START TRADING
# ============================================================

@app.post(
    "/accounts/{account_id}/start"
)
async def start_trading(
    account_id: str,
    data: StartTradingRequest
):

    if account_id not in BOT_CONFIGS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Account has not been configured "
                "by this API yet. "
                "Connect it first."
            )
        )

    config = BOT_CONFIGS[
        account_id
    ]

    if data.symbols is not None:

        config["symbols"] = (
            data.symbols
        )

    if (
        data.daily_profit_target
        is not None
    ):

        config[
            "daily_profit_target"
        ] = (
            data.daily_profit_target
        )

    if data.lot_size is not None:

        config[
            "lot_size"
        ] = (
            data.lot_size
        )

    if (
        data.scan_interval_seconds
        is not None
    ):

        config[
            "scan_interval_seconds"
        ] = (
            data
            .scan_interval_seconds
        )

    current_status = (
        BOT_STATUS.get(
            account_id,
            {}
        )
    )

    if current_status.get(
        "trading_active",
        False
    ):

        return {
            "status":
                "already_running",
            "account_id":
                account_id,
            "message":
                "Trading bot is already running."
        }

    BOT_STATUS[
        account_id
    ] = {
        "trading_active": True,
        "started_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "last_scan": None,
        "last_error": None,
        "last_results": []
    }

    task = (
        asyncio.create_task(
            trading_loop(
                account_id
            )
        )
    )

    BOT_TASKS[
        account_id
    ] = task

    return {
        "status": "started",
        "account_id":
            account_id,
        "message": (
            "Trading bot started successfully. "
            "The bot is now scanning the market "
            "and can open trades when a valid "
            "BUY or SELL signal is detected."
        ),
        "configuration":
            config
    }


# ============================================================
# STOP TRADING
# ============================================================

@app.post(
    "/accounts/{account_id}/stop"
)
async def stop_trading(
    account_id: str
):

    if account_id in BOT_STATUS:

        BOT_STATUS[
            account_id
        ]["trading_active"] = False

    task = BOT_TASKS.get(
        account_id
    )

    if task and not task.done():

        task.cancel()

    BOT_TASKS.pop(
        account_id,
        None
    )

    return {
        "status": "stopped",
        "account_id":
            account_id,
        "message": (
            "Trading bot stopped. "
            "The MetaTrader account remains "
            "connected."
        )
    }


# ============================================================
# GET BOT STATUS
# ============================================================

@app.get(
    "/accounts/{account_id}/bot-status"
)
async def get_bot_status(
    account_id: str
):

    return {
        "account_id":
            account_id,
        "bot":
            BOT_STATUS.get(
                account_id,
                {
                    "trading_active": False,
                    "message":
                        "Bot is not running."
                }
            ),
        "configuration":
            BOT_CONFIGS.get(
                account_id,
                {}
            )
    }


# ============================================================
# RUN ONE TEST SCAN
#
# This analyses the market but does NOT open real trades.
# Useful for testing Lovable.
# ============================================================

@app.post(
    "/accounts/{account_id}/test-scan"
)
async def test_scan(
    account_id: str
):

    result = (
        await run_market_scan(
            account_id,
            dry_run=True
        )
    )

    record_scan_analytics(
        account_id,
        result
    )

    return result




# ============================================================
# ANALYTICS HELPERS
# ============================================================

def get_analytics_state(account_id: str):
    if account_id not in ANALYTICS_STATE:
        ANALYTICS_STATE[account_id] = {
            "decision_log": [],
            "scan_history": []
        }
    return ANALYTICS_STATE[account_id]


def _to_float(value, default=0.0):
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _read_value(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def record_scan_analytics(account_id: str, result: dict):
    state = get_analytics_state(account_id)
    timestamp = datetime.now(timezone.utc).isoformat()

    scan_record = {
        "timestamp": timestamp,
        "status": result.get("status"),
        "account_profit": result.get("account_profit", 0),
        "trades_opened": len(result.get("trades", [])),
        "results": result.get("results", [])
    }
    state["scan_history"].append(scan_record)
    state["scan_history"] = state["scan_history"][-500:]

    for item in result.get("results", []):
        entry = {
            "timestamp": timestamp,
            "symbol": item.get("symbol"),
            "status": item.get("status"),
            "signal": item.get("signal", "HOLD"),
            "reason": item.get("reason"),
            "source": "trading_engine"
        }
        state["decision_log"].append(entry)

    state["decision_log"] = state["decision_log"][-1000:]


async def get_closed_deals_for_analytics(connection, days: int):
    """Fetch closed deal history from MetaApi when supported by the SDK."""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=max(1, min(days, 365)))

    try:
        deals = await connection.get_deals_by_time_range(
            start_time,
            end_time,
            0,
            1000
        )
        return deals or []
    except TypeError:
        try:
            deals = await connection.get_deals_by_time_range(
                start_time,
                end_time
            )
            return deals or []
        except Exception:
            return []
    except Exception:
        return []


def build_analytics_response(account_id: str, deals: list, days: int):
    normalized_deals = []

    for deal in deals or []:
        deal_type = str(_read_value(deal, "type", "")).upper()
        entry_type = str(_read_value(deal, "entryType", _read_value(deal, "entry_type", ""))).upper()
        profit = _to_float(_read_value(deal, "profit", 0))
        commission = _to_float(_read_value(deal, "commission", 0))
        swap = _to_float(_read_value(deal, "swap", 0))
        total_profit = profit + commission + swap
        time_value = _read_value(deal, "time", _read_value(deal, "brokerTime", None))

        # Keep balance/credit operations out of trade statistics.
        if "BALANCE" in deal_type or "CREDIT" in deal_type:
            continue

        # A deal with a symbol is considered trade-related. This keeps the
        # route tolerant of different MetaApi SDK object formats.
        symbol = _read_value(deal, "symbol", None)
        if not symbol:
            continue

        normalized_deals.append({
            "id": str(_read_value(deal, "id", _read_value(deal, "ticket", ""))),
            "symbol": symbol,
            "type": deal_type,
            "entry_type": entry_type,
            "profit": total_profit,
            "time": time_value
        })

    # Deduplicate by deal id when the API returns repeated records.
    unique = {}
    for deal in normalized_deals:
        unique[deal["id"] or f"{deal['symbol']}-{deal['time']}-{deal['profit']}"] = deal
    normalized_deals = list(unique.values())

    closed = [d for d in normalized_deals if d["profit"] != 0]
    wins = [d for d in closed if d["profit"] > 0]
    losses = [d for d in closed if d["profit"] < 0]

    trade_count = len(closed)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = round((win_count / trade_count) * 100, 2) if trade_count else 0.0

    gross_profit = sum(d["profit"] for d in wins)
    gross_loss = abs(sum(d["profit"] for d in losses))
    avg_win = gross_profit / win_count if win_count else 0.0
    avg_loss = gross_loss / loss_count if loss_count else 0.0
    avg_rr = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0

    daily = {}
    for deal in closed:
        time_value = str(deal.get("time") or "unknown")
        day = time_value[:10] if len(time_value) >= 10 else "unknown"
        if day not in daily:
            daily[day] = {"date": day, "wins": 0, "losses": 0, "profit": 0.0, "trades": 0}
        daily[day]["trades"] += 1
        daily[day]["profit"] += deal["profit"]
        if deal["profit"] > 0:
            daily[day]["wins"] += 1
        elif deal["profit"] < 0:
            daily[day]["losses"] += 1

    state = get_analytics_state(account_id)
    decision_log = state.get("decision_log", [])[-100:]

    return {
        "account_id": account_id,
        "period_days": days,
        "win_rate": win_rate,
        "trades": trade_count,
        "wins": win_count,
        "losses": loss_count,
        "avg_rr": avg_rr,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_profit": round(sum(d["profit"] for d in closed), 2),
        "daily": sorted(daily.values(), key=lambda item: item["date"]),
        "decision_log": decision_log,
        "scan_history": state.get("scan_history", [])[-100:],
        "message": (
            "Analytics generated from MetaApi deal history and the current bot runtime log. "
            "Decision history starts collecting after this service version is deployed."
        )
    }


# ============================================================
# ============================================================
# HOME DASHBOARD COMPATIBILITY ROUTES
# ============================================================

@app.get("/status")
async def get_status():

    if not BOT_CONFIGS:
        return {
            "status": "running",
            "trading_active": False,
            "connected": False,
            "account_id": None,
            "message": "Bot service is running. No account is currently connected."
        }

    account_id = next(iter(BOT_CONFIGS.keys()))

    bot_status = BOT_STATUS.get(
        account_id,
        {
            "trading_active": False,
            "last_scan": None,
            "last_error": None,
            "last_results": []
        }
    )

    return {
        "status": "running",
        "connected": True,
        "account_id": account_id,
        "trading_active": bot_status.get(
            "trading_active",
            False
        ),
        "last_scan": bot_status.get(
            "last_scan"
        ),
        "last_error": bot_status.get(
            "last_error"
        )
    }


@app.get("/account")
async def get_current_account():

    if not BOT_CONFIGS:
        return {
            "connected": False,
            "account_id": None,
            "balance": 0,
            "equity": 0,
            "profit": 0,
            "currency": None,
            "message": (
                "No account is configured in this Railway runtime."
            )
        }

    account_id = next(iter(BOT_CONFIGS.keys()))

    try:

        metaapi = get_metaapi()

        account = await get_account_by_id(
            metaapi,
            account_id
        )

        connection = await connect_account(
            account
        )

        account_information = (
            await connection.get_account_information()
        )

        return {
            "connected": True,
            "account_id": account_id,
            "balance": account_information.get(
                "balance",
                0
            ),
            "equity": account_information.get(
                "equity",
                0
            ),
            "profit": account_information.get(
                "profit",
                0
            ),
            "currency": account_information.get(
                "currency"
            )
        }

    except Exception as error:

        return {
            "connected": False,
            "account_id": account_id,
            "balance": 0,
            "equity": 0,
            "profit": 0,
            "currency": None,
            "message": (
                "Broker data is currently unavailable."
            ),
            "error": str(error)
        }


@app.get("/trades")
async def get_trades():

    if not BOT_CONFIGS:
        return {
            "account_id": None,
            "count": 0,
            "trades": []
        }

    account_id = next(iter(BOT_CONFIGS.keys()))

    bot_status = BOT_STATUS.get(
        account_id,
        {}
    )

    last_results = bot_status.get(
        "last_results",
        []
    )

    executed_trades = []

    for result in last_results:

        if result.get("status") in [
            "executed",
            "opened",
            "success"
        ]:

            executed_trades.append(
                result
            )

    return {
        "account_id": account_id,
        "count": len(executed_trades),
        "trades": executed_trades
    }
# ANALYTICS
#
# Supports both:
#   GET /analytics?account_id=<id>&days=30
#   GET /accounts/<account_id>/analytics?days=30
# ============================================================

@app.get("/analytics")
async def get_analytics(
    account_id: Optional[str] = None,
    days: int = 30
):
    days = max(1, min(days, 365))

    if account_id is None:
        if BOT_CONFIGS:
            account_id = next(iter(BOT_CONFIGS.keys()))
        else:
            return {
                "account_id": None,
                "period_days": days,
                "win_rate": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "avg_rr": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "net_profit": 0.0,
                "daily": [],
                "decision_log": [],
                "scan_history": [],
                "message": "No account is configured in this Railway runtime yet. Connect the account first."
            }

    try:
        metaapi = get_metaapi()
        account = await get_account_by_id(metaapi, account_id)
        connection = await connect_account(account)
        deals = await get_closed_deals_for_analytics(connection, days)
        return build_analytics_response(account_id, deals, days)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/accounts/{account_id}/analytics")
async def get_account_analytics(
    account_id: str,
    days: int = 30
):
    return await get_analytics(account_id=account_id, days=days)


# ============================================================
# LEGACY CONNECT AND TRADE ENDPOINT
#
# Keeps your existing Swagger/Lovable integration working.
# ============================================================

@app.post("/connect-and-trade")
async def connect_and_trade(
    data: LegacyMultiPairSetup
):

    connect_data = (
        AccountConnectRequest(
            broker_name=
                data.broker_name,
            platform=
                data.platform,
            login=
                data.login,
            password=
                data.password,
            server=
                data.server,
            symbols=
                data.symbols,
            daily_profit_target=
                data.daily_profit_target,
            lot_size=
                data.lot_size
        )
    )

    connect_result = (
        await connect_account_endpoint(
            connect_data
        )
    )

    account_id = (
        connect_result[
            "account"
        ][
            "account_id"
        ]
    )

    scan_result = (
        await run_market_scan(
            account_id,
            dry_run=data.dry_run
        )
    )

    return {
        "connection":
            connect_result,
        "scan":
            scan_result
    }
