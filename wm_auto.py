"""
wm_auto.py
Starter automation for W/M + Order Block strategy (XAUUSD)
SAFE MODE by default (no live orders unless you enable LIVE_TRADING).
Author: (you) — build, learn, iterate.
"""

import time
import math
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
def connect_mt5():
    if mt5 is None:
        raise RuntimeError("MetaTrader5 module not available.")
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize failed")
    # make sure symbol selected
    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"Failed to select symbol {SYMBOL}")

def get_ohlc(n=LOOKBACK):
    """Return DataFrame of OHLC for SYMBOL on TF with columns time, open, high, low, close, tick_volume."""
    if mt5 is None:
        raise RuntimeError("MT5 not installed in environment.")
    rates = mt5.copy_rates_from_pos(SYMBOL, TF, 0, n)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def calc_trend(df_h4):
    """Simple trend detection: Higher highs & higher lows = bullish; vice versa for bearish"""
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
    """Detect simple Break Of Structure (BOS) using recent swing highs/lows."""
    highs = df_1h['high']
    lows = df_1h['low']
    # recent swing high, low
    recent_high = highs.iloc[-3:-1].max()
    prev_high = highs.iloc[-6:-4].max()
    recent_low = lows.iloc[-3:-1].min()
    prev_low = lows.iloc[-6:-4].min()
    if recent_high > prev_high:
        return 'bull_bos'   # bullish breakout
    if recent_low < prev_low:
        return 'bear_bos'   # bearish breakout
    return None

def detect_wm(df_1h):
    """Naive W/M detection on 1H: returns 'W' or 'M' or None. You will refine this."""
    closes = df_1h['close'].values
    if len(closes) < 10:
        return None
    # find two peaks/troughs: very simple heuristic
    last = df_1h.tail(12)
    lows = last['low'].values
    highs = last['high'].values
    # detect W: middle trough lower than flanking troughs
    if lows[-3] < lows[-4] and lows[-3] < lows[-2] and lows[-2] > lows[-1]:
        return 'W'
    if highs[-3] > highs[-4] and highs[-3] > highs[-2] and highs[-2] < highs[-1]:
        return 'M'
    return None

def detect_liquidity_sweep(df_m15):
    """Naive liquidity sweep detection: a wick beyond recent highs/lows followed by rejection"""
    last3 = df_m15.tail(6)
    high_wick = last3['high'].max()
    low_wick = last3['low'].min()
    # check for recent wick that pulled price above recent high then closed back
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
    """Very naive OB detection: capture last bearish candle (for buy OB) or bullish candle (for sell OB). 
    Returns zone (low,high) and type."""
    last = df_1h.iloc[-2]   # use previous bar as potential OB
    if last['close'] < last['open']:
        # bearish candle — potential buy order block below
        return 'buy', (last['low'], last['high'])
    if last['close'] > last['open']:
        # bullish candle — potential sell order block above
        return 'sell', (last['low'], last['high'])
    return None, None

def in_allowed_session(now=None):
    """Check if current hour is within allowed session ranges (UTC assumed)."""
    if now is None:
        now = datetime.utcnow()
    h = now.hour
    for start, end in ALLOWED_SESSION_HOURS:
        if start <= h < end:
            return True
    return False

def account_info():
    if mt5 is None:
        return {"balance":1000.0}
    info = mt5.account_info()
    return {"balance": info.balance, "equity": info.equity}

# ---------- ORDER HANDLING (safe by default) ----------
def place_market_order(direction, lot, sl_price=None, tp_price=None):
    """Place an order or print simulated order depending on LIVE_TRADING."""
    timestamp = datetime.utcnow().isoformat()
    print(f"[{timestamp}] ORDER REQUEST -> {direction} lot={lot} sl={sl_price} tp={tp_price}")
    if not LIVE_TRADING:
        return {"ret":"simulated", "direction":direction, "lot":lot}
    # Live: prepare MT5 request
    if mt5 is None:
        raise RuntimeError("MT5 not available for live orders.")
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if direction=='buy' else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if direction=='buy' else mt5.ORDER_TYPE_SELL
    point = mt5.symbol_info(SYMBOL).point
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
    """Return lot size given balance, risk percent and stop loss in pips using simple formula:
       risk $ = balance * risk_pct. For gold, 1 lot pip value approx $1 per 0.01? We'll use symbol point/contract_size.
       This is approximate; for live use compute using symbol contract_size.
    """
    # use mt5 for precise contract size if available
    if mt5:
        info = mt5.symbol_info(SYMBOL)
        if info is None:
            return 0.01
        tick_value = info.trade_tick_value if info.trade_tick_value else 1.0
        point = info.point
        # risk $ per pip for 1 lot approx = contract_size * point * price? (approximation)
        # We'll use a conservative simple rule: scale lot so that sl_loss ≈ risk$
        risk_dollars = balance * risk_pct
        # estimate pip_value_per_lot (very rough): use 1 lot pip ~ $1 for 0.01 on gold -> scale accordingly
        # fallback approach: assume 1 lot -> $10 per pip
        pip_value_1lot = 10.0
        lot = max(0.01, (risk_dollars / (sl_pips * pip_value_1lot)))
        return round(lot, 2)
    else:
        # sandbox: use small default
        return 0.01

# ---------- CORE LOGIC LOOP ----------
def run_once():
    # pull data
    df_m15 = get_ohlc(n=LOOKBACK)
    # derive 1H and 4H by resampling
    df_1h = df_m15.resample('1H').agg({'open':'first','high':'max','low':'min','close':'last','tick_volume':'sum'}).dropna()
    df_4h = df_m15.resample('4H').agg({'open':'first','high':'max','low':'min','close':'last','tick_volume':'sum'}).dropna()
    # trend & structure
    trend = calc_trend(df_4h)
    bos = detect_bos(df_1h)
    wm = detect_wm(df_1h)
    sweep_type, sweep_level = detect_liquidity_sweep(df_m15)
    ob_type, ob_zone = detect_order_block(df_1h)
    info = account_info()
    balance = info['balance'] if 'balance' in info else info.get('equity', 1000.0)
    print(f"Trend={trend}, BOS={bos}, WM={wm}, Sweep={sweep_type}, OB={ob_type}, balance={balance}")

    # session & news filter
    if not in_allowed_session():
        print("Outside allowed session - skip")
        return

    # Example decision rule (conservative):
    # If we have a buy sweep AND OB buy zone AND BOS bullish OR W pattern in direction of trend -> place buy
    if sweep_type == 'buy_sweep' and ob_type == 'buy' and (bos == 'bull_bos' or wm == 'W') and trend == 'bull':
        # set SL below OB zone low, TP target a few R
        sl_price = ob_zone[0] - 0.5  # buffer in price units
        tp_price = df_m15['close'].iloc[-1] + (df_m15['close'].iloc[-1] - sl_price) * 2.5  # 2.5R target
        # estimate sl pips roughly
        sl_pips = abs((df_m15['close'].iloc[-1] - sl_price) / mt5.symbol_info(SYMBOL).point) if mt5 else 200
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
    if mt5:
        try:
            connect_mt5()
            print("Connected to MT5.")
        except Exception as e:
            print("MT5 connect error:", e)
    # run a single cycle for demo; wrap in loop for continuous running
    run_once()
