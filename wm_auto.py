"""
wm_auto.py
Starter automation for W/M + Order Block strategy (XAUUSD)
SAFE MODE by default (no live orders unless you enable LIVE_TRADING).
Author: (you) — build, learn, iterate.
"""

import time
import math
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Optional: pip install MetaTrader5
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None
    print("MetaTrader5 package not found. Install with: pip install MetaTrader5")

# ---------- CONFIG ----------
SYMBOL = "XAUUSD"
TF = mt5.TIMEFRAME_M15 if mt5 else "M15"
LOOKBACK = 500                 # bars to pull
LOT_DEFAULT = 0.05
LIVE_TRADING = False           # Must set True to send orders
MAGIC = 444001

# Risk rules
RISK_PER_TRADE = 0.01          # 1% of balance
MAX_DAILY_LOSS_PCT = 0.08      # 8% daily drawdown limit

# Session filter: London + NewYork hours (UTC) typical windows
ALLOWED_SESSION_HOURS = [(7, 16), (12, 21)]  # examples; adjust to your broker timezone

# News avoidance window (minutes before/after) - set to 5 if you have calendar
NEWS_WINDOW_MIN = 5

# ---------- UTILITIES ----------
def load_config(fname="config.json"):
    with open(fname, "r") as f:
        return json.load(f)

def connect_mt5(cfg, max_retries=5, wait=5):
    if mt5 is None:
        raise RuntimeError("MetaTrader5 module not available.")
    terminal_path = cfg.get("path")
    login = cfg.get("login")
    password = cfg.get("password")
    server = cfg.get("server")

    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}: Connecting to MT5...")
        ok = mt5.initialize(path=terminal_path, login=login, password=password, server=server)
        if ok:
            print("✅ Connected to MetaTrader 5!")
            break
        else:
            err = mt5.last_error()
            print(f"⚠️ Connection failed: {err}, retrying in {wait}s...")
            time.sleep(wait)
    else:
        raise RuntimeError("❌ MT5 initialize failed after multiple attempts.")

    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"Failed to select symbol {SYMBOL}")

def get_ohlc(n=LOOKBACK):
    """Return DataFrame of OHLC for SYMBOL on TF with columns time, open, high, low, close, tick_volume."""
    if mt5 is None:
        raise RuntimeError("MT5 not installed in environment.")
    # assume already initialized in connect_mt5
    rates = mt5.copy_rates_from_pos(SYMBOL, TF, 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data retrieved for {SYMBOL} on {TF}, check symbol/timeframe availability")
    df = pd.DataFrame(rates)
    # Debug prints (can comment out later)
    print("DEBUG: DataFrame columns:", df.columns.tolist())
    print("DEBUG: DataFrame head:\n", df.head())
    if 'time' not in df.columns:
        raise KeyError("Column 'time' not found in DataFrame, check MT5 data structure")
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def calc_trend(df_h4):
    highs = df_h4['high'].iloc[-5:]
    lows = df_h4['low'].iloc[-5:]
    hh = highs.iloc[-1] > highs.iloc[-2]
    hl = lows.iloc[-1] > lows.iloc[-2]
    ll = lows.iloc[-1] < lows.iloc[-2]
    lh = highs.iloc[-1] < highs.iloc[-2]
    if hh and hl:
        return 'bull'
    if ll and lh:
        return 'bear'
    return 'sideways'

def detect_bos(df_1h):
    highs = df_1h['high']
    lows = df_1h['low']
    if len(highs) < 8:
        return None
    recent_high = highs.iloc[-3:-1].max()
    prev_high = highs.iloc[-6:-4].max()
    recent_low = lows.iloc[-3:-1].min()
    prev_low = lows.iloc[-6:-4].min()
    if recent_high > prev_high:
        return 'bull_bos'
    if recent_low < prev_low:
        return 'bear_bos'
    return None

def detect_wm(df_1h):
    closes = df_1h['close'].values
    if len(closes) < 12:
        return None
    last = df_1h.tail(12)
    lows = last['low'].values
    highs = last['high'].values
    if lows[-3] < lows[-4] and lows[-3] < lows[-2] and lows[-2] > lows[-1]:
        return 'W'
    if highs[-3] > highs[-4] and highs[-3] > highs[-2] and highs[-2] < highs[-1]:
        return 'M'
    return None

def detect_liquidity_sweep(df_m15):
    last3 = df_m15.tail(6)
    high_wick = last3['high'].max()
    low_wick = last3['low'].min()
    recent_high = df_m15['high'].iloc[-20:-5].max()
    recent_low = df_m15['low'].iloc[-20:-5].min()
    sweep_buy = low_wick < recent_low and last3['close'].iloc[-1] > last3['open'].iloc[-1]
    sweep_sell = high_wick > recent_high and last3['close'].iloc[-1] < last3['open'].iloc[-1]
    if sweep_buy:
        return 'buy_sweep', low_wick
    if sweep_sell:
        return 'sell_sweep', high_wick
    return None, None

def detect_order_block(df_1h):
    if len(df_1h) < 3:
        return None, None
    last = df_1h.iloc[-2]
    if last['close'] < last['open']:
        return 'buy', (last['low'], last['high'])
    if last['close'] > last['open']:
        return 'sell', (last['low'], last['high'])
    return None, None

def in_allowed_session(now=None):
    if now is None:
        now = datetime.utcnow()
    h = now.hour
    for start, end in ALLOWED_SESSION_HOURS:
        if start <= h < end:
            return True
    return False

def account_info():
    if mt5 is None:
        return {"balance":1000.0, "equity":1000.0}
    info = mt5.account_info()
    return {"balance": info.balance, "equity": info.equity}

# ---------- ORDER HANDLING (safe by default) ----------
def place_market_order(direction, lot, sl_price=None, tp_price=None):
    timestamp = datetime.utcnow().isoformat()
    print(f"[{timestamp}] ORDER REQUEST -> {direction} lot={lot} sl={sl_price} tp={tp_price}")
    if not LIVE_TRADING:
        return {"ret":"simulated", "direction":direction, "lot":lot}
    if mt5 is None:
        raise RuntimeError("MT5 not available for live orders.")
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if direction=='buy' else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if direction=='buy' else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": sl_price if sl_price else 0.0,
        "tp": tp_price if tp_price else 0.0,
        "deviation": 50,
        "magic": MAGIC,
        "comment": "wm_auto",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result

# ---------- RISK SIZING ----------
def lot_size_from_risk(balance, risk_pct, sl_pips):
    if mt5:
        info = mt5.symbol_info(SYMBOL)
        if info is None:
            return 0.01
        pip_value_1lot = 10.0
        risk_dollars = balance * risk_pct
        lot = max(0.01, (risk_dollars / (sl_pips * pip_value_1lot)))
        return round(lot, 2)
    else:
        return 0.01

# ---------- CORE LOGIC LOOP ----------
def run_once():
    df_m15 = get_ohlc(n=LOOKBACK)
    df_1h = df_m15.resample('1H').agg({'open':'first','high':'max','low':'min','close':'last','tick_volume':'sum'}).dropna()
    df_4h = df_m15.resample('4H').agg({'open':'first','high':'max','low':'min','close':'last','tick_volume':'sum'}).dropna()
    trend = calc_trend(df_4h)
    bos = detect_bos(df_1h)
    wm = detect_wm(df_1h)
    sweep_type, sweep_level = detect_liquidity_sweep(df_m15)
    ob_type, ob_zone = detect_order_block(df_1h)
    info = account_info()
    balance = info.get('balance', 1000.0)
    print(f"Trend={trend}, BOS={bos}, WM={wm}, Sweep={sweep_type}, OB={ob_type}, balance={balance}")

    if not in_allowed_session():
        print("Outside allowed session - skip")
        return

    if sweep_type == 'buy_sweep' and ob_type == 'buy' and (bos == 'bull_bos' or wm == 'W') and trend == 'bull':
        sl_price = ob_zone[0] - 0.5
        tp_price = df_m15['close'].iloc[-1] + (df_m15['close'].iloc[-1] - sl_price) * 2.5
        sl_pips = abs((df_m15['close'].iloc[-1] - sl_price) / (mt5.symbol_info(SYMBOL).point if mt5 else 0.01))
        lot = lot_size_from_risk(balance, RISK_PER_TRADE, sl_pips)
        print("Signal: BUY (sweep+OB+bos/w) lot:", lot)
        res = place_market_order('buy', lot, sl_price, tp_price)
        print("Order result:", res)
        return

    if sweep_type == 'sell_sweep' and ob_type == 'sell' and (bos == 'bear_bos' or wm == 'M') and trend == 'bear':
        sl_price = ob_zone[1] + 0.5
        tp_price = df_m15['close'].iloc[-1] - (sl_price - df_m15['close'].iloc[-1]) * 2.5
        sl_pips = abs((sl_price - df_m15['close'].iloc[-1]) / (mt5.symbol_info(SYMBOL).point if mt5 else 0.01))
        lot = lot_size_from_risk(balance, RISK_PER_TRADE, sl_pips)
        print("Signal: SELL (sweep+OB+bos/m) lot:", lot)
        res = place_market_order('sell', lot, sl_price, tp_price)
        print("Order result:", res)
        return

    print("No high-probability signal found this cycle.")

# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    print("WM Auto starter - SAFE MODE:", not LIVE_TRADING)
    if mt5 is None:
        print("MT5 not available, cannot proceed.")
    else:
        cfg = load_config("config.json")
        try:
            connect_mt5(cfg)
            print("Connected to MT5.")
            run_once()
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            try:
                mt5.shutdown()
            except Exception:
                pass
