import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MomentumContrarianStrategy:

    def __init__(self,
                 data_file: str,
                 start_date: str = '2023-01-01',
                 end_date: str = '2024-12-31',
                 initial_capital: float = 1_000_000,
                 transaction_cost_bps: float = 5.0,
                 max_position_size: float = 10_000,
                 min_price: float = 5.0,
                 min_volume: float = 100_000):
        """
        Initialize the strategy.

        Args:
            data_file: Path to the filtered universe CSV
            start_date: Strategy start date (YYYY-MM-DD)
            end_date: Strategy end date (YYYY-MM-DD)
            initial_capital: Starting capital
            transaction_cost_bps: Transaction costs in basis points (5 bps = 0.05%)
            max_position_size: Maximum position size per stock
            min_price: Minimum stock price filter
            min_volume: Minimum daily volume filter
        """
        self.data_file = data_file
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.max_position_size = max_position_size
        self.min_price = min_price
        self.min_volume = min_volume

        # Strategy parameters
        self.lookback_days = 5
        self.rebalance_frequency = 'weekly'  # Weekly rebalancing

        # Tracking variables
        self.positions = {}  # {symbol: shares}
        self.cash = initial_capital
        self.portfolio_values = []
        self.weekly_returns = []
        self.weekly_turnover = []
        self.weekly_costs = []
        self.weekly_dates = []
        self.trade_log = []

        # Data
        self.data = None
        self.weekly_data = None

    def load_and_prepare_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.data_file}")

        # Load data
        df = pd.read_csv(self.data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Handle timezone issues - make both timezone-naive for comparison
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

        # Ensure comparison dates are timezone-naive
        start_date_naive = self.start_date.tz_localize(None) if self.start_date.tz is not None else self.start_date
        end_date_naive = self.end_date.tz_localize(None) if self.end_date.tz is not None else self.end_date

        # Filter by date range
        df = df[(df['timestamp'] >= start_date_naive) & (df['timestamp'] <= end_date_naive)]

        # Apply filters
        df = df[df['close'] >= self.min_price]
        df = df[df['volume'] >= self.min_volume]

        # Sort by symbol and date
        df = df.sort_values(['symbol', 'timestamp'])

        # Add day of week
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['is_monday'] = df['day_of_week'] == 'Monday'
        df['is_friday'] = df['day_of_week'] == 'Friday'

        # Add week identifier for grouping
        df['year_week'] = df['timestamp'].dt.to_period('W').astype(str)

        logger.info(f"Data loaded: {len(df)} records, {df['symbol'].nunique()} symbols")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        self.data = df
        return df

    def calculate_weekly_returns(self) -> pd.DataFrame:
        logger.info("Calculating weekly returns...")

        weekly_data = []

        for symbol in self.data['symbol'].unique():
            symbol_data = self.data[self.data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('timestamp')

            # Calculate 5-day returns
            symbol_data['return_5d'] = symbol_data['close'].pct_change(periods=self.lookback_days)

            # Get weekly data (Fridays for signal generation, Mondays for execution)
            for year_week in symbol_data['year_week'].unique():
                week_data = symbol_data[symbol_data['year_week'] == year_week]

                # Get Friday's data (for signal calculation)
                friday_data = week_data[week_data['is_friday'] == True]
                if friday_data.empty:
                    # Use last day of week if no Friday
                    friday_data = week_data.iloc[[-1]]

                # Get Monday's data (for execution)
                # Convert to timezone-naive datetime for comparison
                next_week_start = pd.to_datetime(year_week + '-1').tz_localize(None) + timedelta(days=7)
                next_week_end = next_week_start + timedelta(days=7)

                next_week_data = symbol_data[
                    (symbol_data['timestamp'] >= next_week_start) &
                    (symbol_data['timestamp'] < next_week_end)
                    ]
                monday_data = next_week_data[next_week_data['is_monday'] == True]
                if monday_data.empty and not next_week_data.empty:
                    # Use first day of next week if no Monday
                    monday_data = next_week_data.iloc[[0]]

                if not friday_data.empty and not monday_data.empty:
                    weekly_entry = {
                        'symbol': symbol,
                        'year_week': year_week,
                        'signal_date': friday_data.iloc[0]['timestamp'],
                        'execution_date': monday_data.iloc[0]['timestamp'],
                        'signal_close': friday_data.iloc[0]['close'],
                        'execution_open': monday_data.iloc[0]['open'],
                        'return_5d': friday_data.iloc[0]['return_5d'],
                        'volume': friday_data.iloc[0]['volume'],
                    }
                    weekly_data.append(weekly_entry)

        self.weekly_data = pd.DataFrame(weekly_data)
        self.weekly_data = self.weekly_data.sort_values(['year_week', 'symbol'])

        logger.info(f"Calculated weekly returns: {len(self.weekly_data)} entries")
        return self.weekly_data

    def calculate_position_sizes(self, week_data: pd.DataFrame):
        week_data = week_data.copy()

        # Strategy rule: if return is -X%, buy $X; if return is +X%, sell $X
        # Convert percentage to dollar amount
        week_data['return_pct'] = week_data['return_5d'] * 100  # Convert to percentage
        week_data['target_dollar_amount'] = -week_data['return_pct']*10  # Negative of return

        # Cap position sizes
        week_data['target_dollar_amount'] = np.clip(
            week_data['target_dollar_amount'],
            -self.max_position_size,
            self.max_position_size
        )

        # Calculate target shares (positive = buy, negative = sell)
        week_data['target_shares'] = week_data['target_dollar_amount'] / week_data['execution_open']
        week_data['target_shares'] = week_data['target_shares'].round().astype(int)

        # Filter out very small positions
        week_data = week_data[abs(week_data['target_shares']) >= 1]

        return week_data

    def execute_trades(self, week_data: pd.DataFrame, week_date: str):
        total_turnover = 0.0
        total_costs = 0.0

        for _, row in week_data.iterrows():
            symbol = row['symbol']
            target_shares = row['target_shares']
            price = row['execution_open']

            if abs(target_shares) < 1:
                continue

            current_shares = self.positions.get(symbol, 0)
            shares_to_trade = target_shares

            if shares_to_trade == 0:
                continue

            # Calculate trade value
            trade_value = abs(shares_to_trade * price)

            # Check if we have enough cash for buys
            if shares_to_trade > 0 and trade_value > self.cash:
                # Scale down the trade to available cash
                shares_to_trade = int(self.cash / price)
                trade_value = shares_to_trade * price

            if abs(shares_to_trade) < 1:
                continue

            # Calculate transaction costs
            transaction_cost = trade_value * (self.transaction_cost_bps / 10000)

            # Execute trade
            if shares_to_trade > 0:  # Buy
                self.cash -= (trade_value + transaction_cost)
                self.positions[symbol] = current_shares + shares_to_trade
            else:  # Sell
                self.cash += (trade_value - transaction_cost)
                self.positions[symbol] = current_shares + shares_to_trade

            # Remove zero positions
            if self.positions[symbol] == 0:
                del self.positions[symbol]

            # Track metrics
            total_turnover += trade_value
            total_costs += transaction_cost

            # Log trade
            self.trade_log.append({
                'date': row['execution_date'],
                'week': week_date,
                'symbol': symbol,
                'shares': shares_to_trade,
                'price': price,
                'value': trade_value,
                'cost': transaction_cost,
                'return_5d': row['return_5d'],
                'signal': 'BUY' if shares_to_trade > 0 else 'SELL'
            })

        return total_turnover, total_costs

    def calculate_portfolio_value(self, current_date: pd.Timestamp):
        if not self.positions:
            return self.cash

        # Ensure current_date is timezone-naive for comparison
        if current_date.tz is not None:
            current_date = current_date.tz_localize(None)

        # Get current prices
        current_data = self.data[self.data['timestamp'] <= current_date]
        if current_data.empty:
            return self.cash

        latest_prices = current_data.groupby('symbol')['close'].last()

        position_value = 0.0
        for symbol, shares in self.positions.items():
            if symbol in latest_prices:
                position_value += shares * latest_prices[symbol]

        return self.cash + position_value

    def run_backtest(self):
        logger.info("Starting backtest...")

        # Load and prepare data
        self.load_and_prepare_data()
        self.calculate_weekly_returns()

        # Filter out NaN returns
        self.weekly_data = self.weekly_data.dropna(subset=['return_5d'])

        # Get unique weeks for iteration
        weeks = sorted(self.weekly_data['year_week'].unique())

        logger.info(f"Running backtest for {len(weeks)} weeks")

        for i, week in enumerate(weeks):
            week_data = self.weekly_data[self.weekly_data['year_week'] == week].copy()

            if week_data.empty:
                continue

            # Calculate position sizes
            week_data = self.calculate_position_sizes(week_data)

            # Get portfolio value before trades
            execution_date = week_data.iloc[0]['execution_date']
            portfolio_value_before = self.calculate_portfolio_value(execution_date)

            # Execute trades
            turnover, costs = self.execute_trades(week_data, week)

            # Calculate portfolio value after trades
            portfolio_value_after = self.calculate_portfolio_value(execution_date)

            # Calculate weekly return
            if i == 0:
                weekly_return = 0.0
                prev_value = self.initial_capital
            else:
                weekly_return = (portfolio_value_after - prev_value) / prev_value

            prev_value = portfolio_value_after

            # Store metrics
            self.portfolio_values.append(portfolio_value_after)
            self.weekly_returns.append(weekly_return)
            self.weekly_turnover.append(turnover)
            self.weekly_costs.append(costs)
            self.weekly_dates.append(execution_date)

            if i % 10 == 0:
                logger.info(f"Week {i + 1}/{len(weeks)}: Portfolio Value = ${portfolio_value_after:,.2f}")

        logger.info("Backtest completed")

        # Calculate final metrics
        return self.calculate_performance_metrics()

    def calculate_performance_metrics(self):
        if not self.weekly_returns:
            return {}

        returns = np.array(self.weekly_returns[1:])  # Exclude first zero return
        portfolio_values = np.array(self.portfolio_values)

        # Basic metrics
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        num_weeks = len(returns)
        num_years = num_weeks / 52.0

        # CAGR
        cagr = (portfolio_values[-1] / self.initial_capital) ** (1 / num_years) - 1 if num_years > 0 else 0

        # Volatility (annualized)
        weekly_volatility = np.std(returns, ddof=1) if len(returns) > 1 else 0
        annual_volatility = weekly_volatility * np.sqrt(52)

        # Sharpe Ratio (assuming 0% risk-free rate)
        mean_weekly_return = np.mean(returns) if len(returns) > 0 else 0
        sharpe_ratio = (mean_weekly_return * 52) / annual_volatility if annual_volatility > 0 else 0

        # Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_vol = np.std(negative_returns, ddof=1) * np.sqrt(52) if len(negative_returns) > 1 else 0
        sortino_ratio = (mean_weekly_return * 52) / downside_vol if downside_vol > 0 else 0

        # Maximum Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)

        # Calmar Ratio
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # Hit Rate
        positive_returns = np.sum(returns > 0)
        hit_rate = positive_returns / len(returns) if len(returns) > 0 else 0

        # Average turnover and costs
        avg_weekly_turnover = np.mean(self.weekly_turnover) if self.weekly_turnover else 0
        avg_weekly_costs = np.mean(self.weekly_costs) if self.weekly_costs else 0
        total_costs = sum(self.weekly_costs)

        # Cost as percentage of returns
        gross_return = total_return + (total_costs / self.initial_capital)
        cost_drag = (total_costs / self.initial_capital) / gross_return if gross_return != 0 else 0

        metrics = {
            'total_return': total_return,
            'cagr': cagr,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'hit_rate': hit_rate,
            'num_trades': len(self.trade_log),
            'avg_weekly_turnover': avg_weekly_turnover,
            'avg_weekly_costs': avg_weekly_costs,
            'total_costs': total_costs,
            'cost_drag': cost_drag,
            'final_portfolio_value': portfolio_values[-1] if len(portfolio_values) > 0 else self.initial_capital,
            'num_weeks': num_weeks,
            'num_years': num_years,
        }

        return metrics

    def print_results(self, metrics: Dict) :
        print("\n" + "=" * 80)
        print("MOMENTUM-CONTRARIAN STRATEGY BACKTEST RESULTS")
        print("=" * 80)

        print(f"\nSTRATEGY PARAMETERS:")
        print(f"  Lookback Period: {self.lookback_days} days")
        print(f"  Rebalancing: Weekly")
        print(f"  Transaction Costs: {self.transaction_cost_bps} bps")
        print(f"  Max Position Size: ${self.max_position_size:,}")
        print(f"  Date Range: {self.start_date.date()} to {self.end_date.date()}")

        print(f"\nPORTFOLIO PERFORMANCE:")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Value: ${metrics['final_portfolio_value']:,.2f}")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  CAGR: {metrics['cagr']:.2%}")
        print(f"  Annual Volatility: {metrics['annual_volatility']:.2%}")

        print(f"\nRISK METRICS:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"  Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")
        print(f"  Hit Rate: {metrics['hit_rate']:.2%}")

        print(f"\nTRADING ACTIVITY:")
        print(f"  Total Trades: {metrics['num_trades']:,}")
        print(f"  Avg Weekly Turnover: ${metrics['avg_weekly_turnover']:,.2f}")
        print(f"  Total Transaction Costs: ${metrics['total_costs']:,.2f}")
        print(f"  Cost Drag: {metrics['cost_drag']:.2%}")

        print(f"\nTIME PERIOD:")
        print(f"  Number of Weeks: {metrics['num_weeks']}")
        print(f"  Number of Years: {metrics['num_years']:.2f}")

        # Portfolio Value Progression (sample)
        if len(self.portfolio_values) > 0:
            print(f"\nPORTFOLIO VALUE PROGRESSION (Sample):")
            sample_indices = np.linspace(0, len(self.portfolio_values) - 1, min(10, len(self.portfolio_values)),
                                         dtype=int)
            for i in sample_indices:
                date = self.weekly_dates[i].strftime('%Y-%m-%d')
                value = self.portfolio_values[i]
                print(f"  {date}: ${value:,.2f}")

        # Weekly Returns (last 10)
        if len(self.weekly_returns) > 1:
            print(f"\nRECENT WEEKLY RETURNS:")
            recent_returns = self.weekly_returns[-10:]
            recent_dates = self.weekly_dates[-10:]
            for date, ret in zip(recent_dates, recent_returns):
                print(f"  {date.strftime('%Y-%m-%d')}: {ret:.2%}")

    def save_detailed_results(self, output_dir: str = None):
        if output_dir is None:
            output_dir = "backtest_results"

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Portfolio values
        portfolio_df = pd.DataFrame({
            'date': self.weekly_dates,
            'portfolio_value': self.portfolio_values,
            'weekly_return': [0] + self.weekly_returns[1:],
            'weekly_turnover': self.weekly_turnover,
            'weekly_costs': self.weekly_costs
        })
        portfolio_df.to_csv(output_path / 'portfolio_performance.csv', index=False)

        # Trade log
        if self.trade_log:
            trades_df = pd.DataFrame(self.trade_log)
            trades_df.to_csv(output_path / 'trade_log.csv', index=False)

        logger.info(f"Detailed results saved to {output_path}")


def main():
    """
    Main function to run the backtest.
    """
    # Configuration
    DATA_FILE = r"C:\Users\habel\OneDrive\Desktop\Algo Trade\Data\filtered_universe.csv"
    START_DATE = "2023-01-01"
    END_DATE = "2024-12-31"
    INITIAL_CAPITAL = 100000
    TRANSACTION_COST_BPS = 0.0  # Change this based on your own values 5 basis points (0.05%)
    MAX_POSITION_SIZE = 10_000  # Max $10K per position

    # Create strategy instance
    strategy = MomentumContrarianStrategy(
        data_file=DATA_FILE,
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        max_position_size=MAX_POSITION_SIZE,
        min_price=5.0,  # Min $5 stock price
        min_volume=100_000  # Min 100K daily volume
    )

    # Run backtest
    results = strategy.run_backtest()

    # Print results
    strategy.print_results(results)

    # Save detailed results
    strategy.save_detailed_results("backtest_results")

    print(f"\nBacktest completed! Detailed results saved to 'backtest_results' folder.")


if __name__ == "__main__":
    main()