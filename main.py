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

    return (
        await run_market_scan(
            account_id,
            dry_run=True
        )
    )


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
