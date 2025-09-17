"""
Z-Spread Reversion Algorithm on High-Correlation Equities
Author: Nathan Abell
Date: September 2025

"""

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from databento import DBNStore


# Data structures 
@dataclass
class TradingSignal:
    timestamp: pd.Timestamp
    action: str  # 'long_spread', 'short_spread', 'close_long', 'close_short'
    z_score: float
    spread_value: float
    cvx_price: float
    xom_price: float


# Utils functions 
def sharpe_sortino(
    returns: np.ndarray,
    *,
    MAR: float = 0.0,                 # annualized minimum acceptable return (e.g., risk-free)
    periods_per_year: Optional[int] = None,  # if None, no annualization scaling is applied
) -> Dict[str, float]:
    """Compute annualized Sharpe & Sortino on a series of periodic returns. """
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    out = {"sharpe": 0.0, "sortino": 0.0, "sharpe_ann": 0.0, "sortino_ann": 0.0}
    if arr.size == 0:
        return out

    if periods_per_year is not None and periods_per_year > 0:
        per_period_mar = MAR / periods_per_year
    else:
        per_period_mar = 0.0  # assume MAR already baked in or not desired

    excess = arr - per_period_mar

    mu = float(np.mean(excess))
    sigma = float(np.std(excess, ddof=0))

    downside = np.minimum(excess, 0.0)
    downside_semidev = float(np.sqrt(np.mean(downside ** 2)))

    sharpe = mu / sigma if sigma > 0 else 0.0
    sortino = mu / downside_semidev if downside_semidev > 0 else 0.0

    out["sharpe"] = sharpe
    out["sortino"] = sortino

    if periods_per_year is not None and periods_per_year > 0:
        scale = np.sqrt(periods_per_year)
        out["sharpe_ann"] = sharpe * scale
        out["sortino_ann"] = sortino * scale
    else:
        out["sharpe_ann"] = sharpe
        out["sortino_ann"] = sortino

    return out


def max_drawdown_from_equity(equity: np.ndarray) -> float:
    """Max drawdown from an equity curve (returns a negative number, e.g., -0.23 for -23%)."""
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(eq)
    drawdowns = (eq - peaks) / peaks
    return float(drawdowns.min() if drawdowns.size else 0.0)


# Strategy 
class ZScoreReversalStrategy:
    def __init__(
        self,
        lookback_window: int = 251,
        entry_threshold: float = 1.25,
        exit_threshold: float = 1.00,
        stop_loss_threshold: float = 3.0,
        mar_annual: float = 0.00,          # annualized MAR used in Sortino/Sharpe
        use_trade_returns: bool = True,     # compute metrics on trade-to-trade returns
    ):
        self.lookback_window = lookback_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.mar_annual = mar_annual
        self.use_trade_returns = use_trade_returns

        self.data: Optional[pd.DataFrame] = None
        self.signals: List[TradingSignal] = []
        self.positions = []
        self.performance_metrics: Dict = {}

    # Data
    def fetch_databento_data(self, file_path: str, symbols: List[str] = ["CVX", "XOM"]) -> pd.DataFrame:
        store = DBNStore.from_file(file_path)
        df = store.to_df()

        # Ensure datetime index
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            ts_cols = [c for c in df.columns if "ts" in c.lower() or "time" in c.lower()]
            if not ts_cols:
                raise KeyError("No timestamp column found in DBN data")
            time_col = ts_cols[0]
            df[time_col] = pd.to_datetime(df[time_col])
            df.set_index(time_col, inplace=True)

        # Filter for symbols & pivot close prices
        df = df[df["symbol"].isin(symbols)]
        price_data = (
            df
            .pivot_table(index=df.index, columns="symbol", values="close", aggfunc="first")
            .sort_index()
            .ffill()
            .dropna()
        )
        return price_data

    # Features 
    def calculate_spread_and_zscore(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["log_cvx"] = np.log(df["CVX"])  # log prices
        df["log_xom"] = np.log(df["XOM"])  # log prices

        # Simple log spread (could be replaced with cointegration / hedge ratio)
        df["spread"] = df["log_cvx"] - df["log_xom"]

        # Rolling stats for z-score
        df["spread_mean"] = df["spread"].rolling(window=self.lookback_window).mean()
        df["spread_std"] = df["spread"].rolling(window=self.lookback_window).std()
        df["z_score"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]

        # Useful diagnostics
        df["cvx_returns"] = df["CVX"].pct_change()
        df["xom_returns"] = df["XOM"].pct_change()
        df["rolling_corr"] = df["cvx_returns"].rolling(window=self.lookback_window).corr(df["xom_returns"])

        self.data = df
        return df

    # Signals 
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        signals: List[TradingSignal] = []
        position = 0  # 0: flat, 1: long spread, -1: short spread

        for i, (timestamp, row) in enumerate(data.iterrows()):
            if pd.isna(row["z_score"]) or i < self.lookback_window:
                continue

            z = float(row["z_score"])
            spread_val = float(row["spread"]) if not pd.isna(row["spread"]) else np.nan
            cvx_p = float(row["CVX"]) if not pd.isna(row["CVX"]) else np.nan
            xom_p = float(row["XOM"]) if not pd.isna(row["XOM"]) else np.nan

            if position == 0:
                if z > self.entry_threshold:
                    signals.append(TradingSignal(timestamp, "short_spread", z, spread_val, cvx_p, xom_p))
                    position = -1
                elif z < -self.entry_threshold:
                    signals.append(TradingSignal(timestamp, "long_spread", z, spread_val, cvx_p, xom_p))
                    position = 1

            elif position == 1:  # long spread
                if (z > -self.exit_threshold) or (z < -self.stop_loss_threshold):
                    signals.append(TradingSignal(timestamp, "close_long", z, spread_val, cvx_p, xom_p))
                    position = 0

            elif position == -1:  # short spread
                if (z < self.exit_threshold) or (z > self.stop_loss_threshold):
                    signals.append(TradingSignal(timestamp, "close_short", z, spread_val, cvx_p, xom_p))
                    position = 0

        self.signals = signals
        return signals

    # Backtest 
    def backtest_strategy(self, *, initial_capital: float = 100_000, position_size: float = 0.1) -> Dict:
        if not self.signals:
            raise ValueError("No signals generated. Run generate_signals first.")

        portfolio_value = float(initial_capital)
        trade_pnl: List[float] = []
        trade_returns: List[float] = []
        trade_timestamps: List[pd.Timestamp] = []

        equity_curve: List[float] = [portfolio_value]  # equity points at trade closes
        current_position = None

        for signal in self.signals:
            if signal.action in ("long_spread", "short_spread"):
                trade_cap = portfolio_value * position_size

                if signal.action == "long_spread":
                    cvx_shares = trade_cap / (2 * signal.cvx_price)
                    xom_shares = -trade_cap / (2 * signal.xom_price)
                else:  # short spread
                    cvx_shares = -trade_cap / (2 * signal.cvx_price)
                    xom_shares = trade_cap / (2 * signal.xom_price)

                current_position = {
                    "entry_signal": signal,
                    "cvx_shares": cvx_shares,
                    "xom_shares": xom_shares,
                    "cvx_entry_price": signal.cvx_price,
                    "xom_entry_price": signal.xom_price,
                    "trade_cap": trade_cap,
                }

            elif signal.action in ("close_long", "close_short") and current_position is not None:
                exit_val = (
                    current_position["cvx_shares"] * signal.cvx_price
                    + current_position["xom_shares"] * signal.xom_price
                )
                entry_val = (
                    current_position["cvx_shares"] * current_position["cvx_entry_price"]
                    + current_position["xom_shares"] * current_position["xom_entry_price"]
                )

                pnl = float(exit_val - entry_val)
                trade_pnl.append(pnl)
                portfolio_value += pnl
                equity_curve.append(portfolio_value)

                # per-trade return normalized by allocated capital
                cap = current_position.get("trade_cap", 0.0)
                trade_ret = pnl / cap if cap != 0 else 0.0
                trade_returns.append(trade_ret)
                trade_timestamps.append(signal.timestamp)

                current_position = None

        # Metrics 
        perf: Dict[str, float] = {}
        n_trades = len(trade_pnl)
        total_return = (portfolio_value - initial_capital) / initial_capital if initial_capital != 0 else 0.0
        win_rate = (sum(1 for p in trade_pnl if p > 0) / n_trades) if n_trades > 0 else 0.0
        avg_trade_pnl = float(np.mean(trade_pnl)) if n_trades > 0 else 0.0

        # Estimate trades/year for annualization (based on first/last trade close)
        trades_per_year = None
        if len(trade_timestamps) >= 2:
            days = (trade_timestamps[-1] - trade_timestamps[0]).days
            years = max(days / 365.25, 1e-9)
            trades_per_year = int(round(n_trades / years)) if years > 0 else None

        # Choose return series for ratios
        returns_arr = np.array(trade_returns, dtype=float)
        metrics = sharpe_sortino(returns_arr, MAR=self.mar_annual, periods_per_year=trades_per_year)

        # Proper max drawdown from equity curve
        mdd = max_drawdown_from_equity(np.array(equity_curve, dtype=float))

        self.performance_metrics = {
            "total_return": total_return,
            "final_portfolio_value": portfolio_value,
            "num_trades": n_trades,
            "win_rate": win_rate,
            "avg_trade_pnl": avg_trade_pnl,
            "sharpe_ratio": metrics["sharpe_ann"],   # annualized
            "sortino_ratio": metrics["sortino_ann"], # annualized
            "trades_per_year_est": trades_per_year if trades_per_year is not None else 0,
            "max_drawdown": mdd,  # as a negative fraction (e.g., -0.23)
            "equity_curve": np.array(equity_curve, dtype=float),
            "trade_pnl": np.array(trade_pnl, dtype=float),
            "trade_returns": returns_arr,
        }
        return self.performance_metrics

    # Reporting 
    def plot_results(self):
        if self.data is None:
            raise ValueError("No data available. Run the strategy first.")

        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # Price series
        axes[0].plot(self.data.index, self.data["CVX"], label="CVX", alpha=0.7)
        axes[0].plot(self.data.index, self.data["XOM"], label="XOM", alpha=0.7)
        axes[0].set_title("Price Series")
        axes[0].legend()
        axes[0].grid(True)

        # Spread and signals
        axes[1].plot(self.data.index, self.data["spread"], label="Spread", alpha=0.7)
        axes[1].axhline(y=0, color="black", linestyle="-", alpha=0.3)

        for s in self.signals:
            if s.action == "long_spread":
                axes[1].scatter(s.timestamp, s.spread_value, color="green", marker="^", s=50, alpha=0.8)
            elif s.action == "short_spread":
                axes[1].scatter(s.timestamp, s.spread_value, color="red", marker="v", s=50, alpha=0.8)
        axes[1].set_title("Spread and Trading Signals")
        axes[1].legend()
        axes[1].grid(True)

        # Z-score
        axes[2].plot(self.data.index, self.data["z_score"], label="Z-Score")
        axes[2].axhline(y=self.entry_threshold, color="red", linestyle="--", alpha=0.5)
        axes[2].axhline(y=-self.entry_threshold, color="red", linestyle="--", alpha=0.5)
        axes[2].axhline(y=self.exit_threshold, color="green", linestyle="--", alpha=0.5)
        axes[2].axhline(y=-self.exit_threshold, color="green", linestyle="--", alpha=0.5)
        axes[2].axhline(y=0, color="black", linestyle="-", alpha=0.3)
        axes[2].set_title("Z-Score with Thresholds")
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

    def print_performance_summary(self):
        if not self.performance_metrics:
            print("No performance metrics available. Run backtest first.")
            return

        pm = self.performance_metrics
        print("=== Z-Score Reversion Strategy Performance ===")
        print(f"Total Return: {pm['total_return']:.2%}")
        print(f"Final Portfolio Value: ${pm['final_portfolio_value']:,.2f}")
        print(f"Number of Trades: {pm['num_trades']}")
        print(f"Win Rate: {pm['win_rate']:.2%}")
        print(f"Average Trade P&L: ${pm['avg_trade_pnl']:,.2f}")
        print(f"Sharpe Ratio (annualized): {pm['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio (annualized): {pm['sortino_ratio']:.3f}")
        print(f"Trades/Year (est): {pm['trades_per_year_est']}")
        print(f"Max Drawdown (from equity): {pm['max_drawdown']:.2%}")


# Runner 
def run_zscore_strategy():
    strategy = ZScoreReversalStrategy(
        lookback_window = 50,
        entry_threshold = 0.6546503939723856,
        exit_threshold = 0.6353198417703924,
        stop_loss_threshold = 5.85432814273555,
        mar_annual=0.00,              # set risk-free/MAR here if desired (e.g., 0.02 for 2%)
        use_trade_returns=True,
    )

    #Some Example Hyperparameters but be cautious of overfit
    # Hyperparameters for 1h data
    # lookback_window = 50,
    # entry_threshold = 0.6546503939723856,
    # exit_threshold = 0.6353198417703924,
    # stop_loss_threshold = 5.85432814273555,
    
    # Hyperparameters for 1d data
    # lookback_window: 228
    # entry_threshold: 1.8070954005506952
    # exit_threshold: 0.7022162346847439
    # stop_loss_threshold: 3.1107933779448746
    # position_size: 0.4242684805042215

    file_path_h = r"" #Change this to 1 hour data dbn file path
    file_path_d = r"" #Change this to 1 hour data dbn file path

    data = strategy.fetch_databento_data(file_path=file_path_h)
    processed = strategy.calculate_spread_and_zscore(data)
    _ = strategy.generate_signals(processed)
    performance = strategy.backtest_strategy(initial_capital=100000, position_size=0.2)

    strategy.print_performance_summary()
    strategy.plot_results()
    return strategy


if __name__ == "__main__":
    strategy = run_zscore_strategy()


