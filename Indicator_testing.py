"""
simple_indicator_framework.py

Goal: Make it EASY for anyone on the team to:
  1) Load a Databento .dbn file (file picker pops up).
  2) Compute a few indicators (CVD, ADX included).
  3) Compare indicators with the SAME simple metrics:
       - Spearman correlation vs forward returns
       - Directional hit rate (sign of indicator change vs sign of forward return)
  4) Add your own indicator in ~5 lines (see TEMPLATE at the bottom).

How to use:
  - Open this file in PyCharm and click "Run".
  - Pick a .dbn file when the file dialog opens.
  - Results print in the console; charts (optional) pop up.

Edit SETTINGS below if you want defaults (e.g., set DBN_PATH to a fixed file).
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Databento DBN
try:
    from databento import DBNStore
except Exception:
    DBNStore = None

# --------------------------
# SETTINGS (adjust if needed)
# --------------------------

DBN_PATH = None          # None -> show a file picker at runtime
SYMBOL = None            # e.g., "ES.U2025" (leave None to use everything)
BAR = "1min"             # For resampling trades -> OHLCV when needed
ADX_PERIOD = 14          # Default ADX lookback
HORIZONS = [1, 5, 20]    # Forward-return horizons (bars)
PLOTS = True             # Turn off if you don't want charts


# ==========================
# 1) Small utility helpers
# ==========================
def _file_picker() -> str | None:
    """Pick a .dbn file; fall back to console input if GUI isn't available."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            title="Select a Databento .dbn file",
            filetypes=[("Databento DBN", "*.dbn"), ("All files", "*.*")]
        )
        root.update(); root.destroy()
        return path or None
    except Exception:
        try:
            return input("Path to .dbn file (Enter to cancel): ").strip() or None
        except EOFError:
            return None


def _choose_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def load_dbn(path: str) -> pd.DataFrame:
    if DBNStore is None:
        raise ImportError("Please install Databento: pip install databento")
    store = DBNStore.from_file(path=path)
    df = store.to_df()
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Loaded DBN produced an empty DataFrame.")
    return df


def ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a datetime index. Accepts existing DatetimeIndex, epoch-like index, or timestamp columns."""
    df = df.copy()

    # Case A: already a DatetimeIndex (Databento often does this)
    if isinstance(df.index, pd.DatetimeIndex):
        # Normalize to tz-naive UTC if tz-aware (optional)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df.sort_index()

    # Case B: integer-like epoch index
    try:
        if np.issubdtype(df.index.dtype, np.integer):
            idx = pd.to_datetime(df.index, unit="ns", errors="coerce")
            if idx.notna().any():
                df.index = idx
                return df[~df.index.isna()].sort_index()
    except Exception:
        pass

    # Case C: look for a timestamp column (add a few common aliases)
    candidates = ("ts_event", "ts_recv", "ts", "time", "start_ts", "start", "date", "datetime")
    for tcol in candidates:
        if tcol in df.columns:
            s = df[tcol]
            # If dtype already datetime64, just parse; else try ns-epoch, then generic
            if np.issubdtype(s.dtype, np.datetime64):
                idx = pd.to_datetime(s, errors="coerce")
            else:
                idx = pd.to_datetime(s, unit="ns", errors="coerce")
                if idx.isna().mean() > 0.5:
                    idx = pd.to_datetime(s, errors="coerce")
            df.index = idx
            return df[~df.index.isna()].sort_index()

    # Helpful error with context
    cols_preview = list(df.columns)[:12]
    raise KeyError(
        f"No timestamp found. Tried {candidates}. "
        f"Index={type(df.index).__name__} (dtype={getattr(df.index, 'dtype', None)}), "
        f"Columns preview={cols_preview}"
    )

def filter_symbol(df: pd.DataFrame, symbol: str | None) -> pd.DataFrame:
    if symbol is None:
        return df
    for col in ("symbol", "instrument_id", "sym"):
        if col in df.columns:
            return df[df[col].astype(str) == str(symbol)]
    return df


def standardize_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure we have price/size; copy optional bid/ask/side if present."""
    df = df.copy()
    p = _choose_col(df, ("price", "px", "trade_px"))
    s = _choose_col(df, ("size", "sz", "qty", "size_lots"))
    if p is None or s is None:
        raise KeyError("Missing price/size columns in DBN trades.")
    df["price"] = df[p].astype(float)
    df["size"] = df[s].astype(float)
    b = _choose_col(df, ("bid_px", "best_bid_px"))
    a = _choose_col(df, ("ask_px", "best_ask_px"))
    if b: df["bid_px"] = df[b]
    if a: df["ask_px"] = df[a]
    side = _choose_col(df, ("side", "aggressor_side", "buyer_seller"))
    if side: df["side"] = df[side]
    return df


def resample_ohlcv(trades: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    """Build simple OHLCV from trades."""
    o = trades["price"].resample(rule).first()
    h = trades["price"].resample(rule).max()
    l = trades["price"].resample(rule).min()
    c = trades["price"].resample(rule).last()
    v = trades["size"].resample(rule).sum()
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna()


# ==========================
# 2) Included indicators
# ==========================
def ind_cvd(trades: pd.DataFrame) -> pd.Series:
    """
    CVD:
      1) If 'side' exists, map to +/-1 and use it.
      2) Else if bid/ask exist, classify trades by crossing.
      3) Else fallback to tick rule.
      Then CVD = cumsum(sign * size)
    """
    # 1) True aggressor side if available
    if "side" in trades.columns:
        s = trades["side"]
        mapper = {1: 1, -1: -1, "B": 1, "S": -1, "BUY": 1, "SELL": -1, "Buy": 1, "Sell": -1}
        side = s.map(mapper).astype(float)
        side = side.ffill().fillna(0.0)

    # 2) Classify via bid/ask if present
    elif {"bid_px", "ask_px"}.issubset(trades.columns):
        px, bid, ask = trades["price"].to_numpy(), trades["bid_px"].to_numpy(), trades["ask_px"].to_numpy()
        side = pd.Series(np.where(px >= ask, 1, np.where(px <= bid, -1, np.nan)), index=trades.index)
        side = side.ffill().fillna(0.0)

    # 3) Tick rule fallback
    else:
        tick = np.sign(trades["price"].diff())
        side = tick.replace({0: np.nan}).ffill().fillna(0.0)

    cvd = (side * trades["size"].astype(float)).cumsum()
    cvd.name = "CVD"
    return cvd

def ind_adx(ohlc: pd.DataFrame, period: int = ADX_PERIOD) -> pd.Series:
    """
    ADX (Wilder, simplified). Needs columns: high, low, close.
    Returns a single Series named e.g., "ADX(14)".
    """
    high, low, close = ohlc["high"].astype(float), ohlc["low"].astype(float), ohlc["close"].astype(float)
    up = high.diff(); down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(high - low).abs(),
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=ohlc.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=ohlc.index).ewm(alpha=1/period, adjust=False).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100.0
    adx = dx.ewm(alpha=1/period, adjust=False).mean().dropna()
    adx.name = f"ADX({period})"
    return adx

def ind_dmi(ohlc, period=ADX_PERIOD):
    high, low, close = ohlc["high"].astype(float), ohlc["low"].astype(float), ohlc["close"].astype(float)
    up, down = high.diff(), -low.diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(high - low),
                    (high - close.shift(1)).abs(),
                    (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)
    plus_di  = 100 * pd.Series(plus_dm, index=ohlc.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=ohlc.index).ewm(alpha=1/period, adjust=False).mean() / atr
    return (plus_di - minus_di).rename(f"DMI({period})")

# ------------------------------------------------------------
# INDICATORS: add your own here (name, function, needs)
#   - needs == "trades": function will receive TRADES
#   - needs == "ohlc":   function will receive OHLC bars
# ------------------------------------------------------------
INDICATORS: list[tuple[str, callable, str]] = [
    ("CVD", ind_cvd, "trades"),
    (f"ADX({ADX_PERIOD})", lambda o: ind_adx(o, ADX_PERIOD), "ohlc"),
    (f"DMI({ADX_PERIOD})", lambda o: ind_dmi(o, ADX_PERIOD), "ohlc")
    # EXAMPLE (uncomment to try a simple momentum indicator):
    # ("ROC(10)", lambda o: o["close"].pct_change(10).rename("ROC(10)"), "ohlc"),
]


# ==========================
# 3) Simple comparison
# ==========================
def forward_returns(close: pd.Series, horizons: list[int]) -> pd.DataFrame:
    return pd.DataFrame({f"fwd_{h}": close.shift(-h) / close - 1.0 for h in horizons}, index=close.index)


def compare_indicator(ind: pd.Series, close: pd.Series, horizons: list[int]) -> pd.DataFrame:
    """Two tiny metrics per horizon: Spearman rho and a sign hit-rate."""
    ind = ind.astype(float)
    fr = forward_returns(close.astype(float), horizons)
    rows = []
    for col in fr.columns:
        y = fr[col].dropna()
        x = ind.reindex(y.index).dropna()
        idx = x.index.intersection(y.index)
        x, y = x.loc[idx], y.loc[idx]
        if len(idx) < 5:
            rows.append({"horizon": col, "spearman": np.nan, "hit_rate": np.nan, "n": len(idx)})
            continue
        rho = x.corr(y, method="spearman")
        slope = x.diff()
        hits = ((slope > 0) & (y > 0)) | ((slope < 0) & (y < 0))
        rows.append({"horizon": col, "spearman": float(rho), "hit_rate": float(hits.mean()), "n": int(len(idx))})
    return pd.DataFrame(rows)


# ==========================
# 4) Run button entry point
# ==========================
def main():
    # Choose DBN
    path = DBN_PATH or _file_picker()
    if not path or not os.path.exists(path):
        print("No .dbn selected. Exiting.")
        return

    # Load + prep
    print(f"\nLoading: {path}")
    raw = load_dbn(path)
    raw = ensure_time_index(raw)
    raw = filter_symbol(raw, SYMBOL)
    if raw.empty:
        print("No data after loading/filtering.")
        return

    # Build datasets (robust to OHLCV-only files)
    cols = set(map(str.lower, raw.columns))

    def _has_any(names):
        return any(n in raw.columns for n in names)

    has_ohlc = {"open", "high", "low", "close"}.issubset(cols)
    has_trade_px = _has_any(("price", "px", "trade_px"))
    has_trade_sz = _has_any(("size", "sz", "qty", "size_lots"))

    trades = None
    ohlc = None

    if has_trade_px and has_trade_sz:
        try:
            trades = standardize_trades(raw)
        except KeyError:
            trades = None  # not a trades file

    if has_ohlc:
        # Use existing OHLCV
        ohlc = raw[[c for c in raw.columns if c.lower() in {"open", "high", "low", "close"}]].astype(float).dropna()
    elif trades is not None:
        # Build OHLCV from trades if possible
        ohlc = resample_ohlcv(trades, BAR)

    if ohlc is None or ohlc.empty:
        print("Could not build OHLC bars from this file (no OHLC and no trades).")
        return

    close = ohlc["close"]
    print(f"Detected datasets -> OHLC: {ohlc is not None}, TRADES: {trades is not None}")

    # ==========================
    # Compute + compare
    # ==========================
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 10)

    all_series: dict[str, pd.Series] = {}

    for name, func, needs in INDICATORS:
        if needs == "trades" and trades is None:
            print(f"Skipping {name}: no trades in this file.")
            continue
        if needs == "ohlc" and ohlc is None:
            print(f"Skipping {name}: no OHLC bars available.")
            continue

        print(f"\nComputing {name} ...")
        series = func(trades) if needs == "trades" else func(ohlc)
        series = series.reindex(close.index).ffill().dropna()
        if series.empty:
            print(f"{name} produced no data after alignment; skipping.")
            continue

        all_series[name] = series

        cmp_df = compare_indicator(series, close, HORIZONS)
        print(f"=== {name} vs forward returns ===")
        print(cmp_df.to_string(index=False))

    if not all_series:
        print("No indicators were computed for this file.")
        return

    # Optional plots
    if PLOTS and plt is not None:
        try:
            close.plot(title="Close", figsize=(10, 3)); plt.tight_layout(); plt.show()
            for name, s in all_series.items():
                s.plot(title=name, figsize=(10, 3)); plt.tight_layout(); plt.show()
        except Exception as e:
            print(f"Plotting error: {e}")

if __name__ == "__main__":
    main()

