import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingUniverseFilter:
    """
    Filters a consolidated trading universe CSV to the top N companies based on various criteria.
    """

    def __init__(self, input_csv: str, output_csv: str = "filtered_universe_5000.csv", top_n: int = 5000):
        """
        Initialize the filter.

        Args:
            input_csv (str): Path to consolidated trading universe CSV
            output_csv (str): Output CSV filename for filtered universe
            top_n (int): Number of top companies to keep
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.top_n = top_n
        self.df = None
        self.filtered_df = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the consolidated trading universe data.

        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {self.input_csv}")

        try:
            self.df = pd.read_csv(self.input_csv)
            logger.info(f"Loaded {len(self.df):,} records with {self.df['symbol'].nunique()} unique symbols")

            # Convert timestamp to datetime if it's not already
            if 'timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

            return self.df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def analyze_data_quality(self) -> pd.DataFrame:
        """
        Analyze data quality metrics for each symbol.

        Returns:
            DataFrame with quality metrics per symbol
        """
        logger.info("Analyzing data quality metrics...")

        # Group by symbol and calculate metrics
        symbol_metrics = []

        for symbol in self.df['symbol'].unique():
            symbol_data = self.df[self.df['symbol'] == symbol].copy()

            if symbol_data.empty:
                continue

            # Sort by timestamp
            symbol_data = symbol_data.sort_values('timestamp')

            # Calculate metrics
            metrics = {
                'symbol': symbol,
                'total_records': len(symbol_data),
                'date_range_days': (symbol_data['timestamp'].max() - symbol_data['timestamp'].min()).days + 1,
                'avg_daily_volume': symbol_data['volume'].mean(),
                'median_daily_volume': symbol_data['volume'].median(),
                'total_volume': symbol_data['volume'].sum(),
                'avg_price': symbol_data['close'].mean(),
                'median_price': symbol_data['close'].median(),
                'price_volatility': symbol_data['close'].std() / symbol_data['close'].mean() if symbol_data[
                                                                                                    'close'].mean() > 0 else 0,
                'zero_volume_days': (symbol_data['volume'] == 0).sum(),
                'missing_data_pct': symbol_data.isnull().sum().sum() / (
                            len(symbol_data) * len(symbol_data.columns)) * 100,
                'first_date': symbol_data['timestamp'].min(),
                'last_date': symbol_data['timestamp'].max(),
                'price_range': symbol_data['close'].max() - symbol_data['close'].min(),
                'avg_dollar_volume': (symbol_data['close'] * symbol_data['volume']).mean(),
            }

            # Data consistency checks
            metrics['valid_ohlc'] = (
                    (symbol_data['low'] <= symbol_data['high']).all() and
                    (symbol_data['low'] <= symbol_data['open']).all() and
                    (symbol_data['low'] <= symbol_data['close']).all() and
                    (symbol_data['open'] <= symbol_data['high']).all() and
                    (symbol_data['close'] <= symbol_data['high']).all()
            )

            symbol_metrics.append(metrics)

        metrics_df = pd.DataFrame(symbol_metrics)
        logger.info(f"Calculated quality metrics for {len(metrics_df)} symbols")

        return metrics_df

    def calculate_scoring_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite scoring metrics for ranking symbols.

        Args:
            metrics_df: DataFrame with quality metrics

        Returns:
            DataFrame with scoring metrics added
        """
        logger.info("Calculating composite scores...")

        # Create a copy to avoid modifying original
        scored_df = metrics_df.copy()

        # Normalize metrics to 0-1 scale for scoring
        def normalize_metric(series, higher_better=True):
            if series.std() == 0:
                return pd.Series([0.5] * len(series), index=series.index)

            normalized = (series - series.min()) / (series.max() - series.min())
            return normalized if higher_better else (1 - normalized)

        # Volume score (higher is better)
        scored_df['volume_score'] = (
                normalize_metric(scored_df['avg_daily_volume']) * 0.4 +
                normalize_metric(scored_df['median_daily_volume']) * 0.3 +
                normalize_metric(scored_df['avg_dollar_volume']) * 0.3
        )

        # Data quality score (higher is better)
        scored_df['quality_score'] = (
                normalize_metric(scored_df['total_records']) * 0.3 +
                normalize_metric(scored_df['date_range_days']) * 0.2 +
                normalize_metric(scored_df['missing_data_pct'], higher_better=False) * 0.2 +
                normalize_metric(scored_df['zero_volume_days'], higher_better=False) * 0.2 +
                (scored_df['valid_ohlc'].astype(float)) * 0.1
        )

        # Liquidity score (higher is better)
        scored_df['liquidity_score'] = (
                normalize_metric(scored_df['avg_daily_volume']) * 0.5 +
                normalize_metric(scored_df['avg_dollar_volume']) * 0.5
        )

        # Price stability score (moderate volatility is preferred)
        # Too low volatility = possibly inactive, too high = too risky
        volatility_normalized = normalize_metric(scored_df['price_volatility'])
        scored_df['stability_score'] = 1 - np.abs(volatility_normalized - 0.5) * 2

        # Market cap proxy (using average dollar volume as proxy)
        scored_df['market_cap_proxy_score'] = normalize_metric(scored_df['avg_dollar_volume'])

        # Composite score
        scored_df['composite_score'] = (
                scored_df['volume_score'] * 0.30 +
                scored_df['quality_score'] * 0.25 +
                scored_df['liquidity_score'] * 0.20 +
                scored_df['stability_score'] * 0.15 +
                scored_df['market_cap_proxy_score'] * 0.10
        )

        return scored_df

    def apply_filters(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply basic quality filters before ranking.

        Args:
            scored_df: DataFrame with scoring metrics

        Returns:
            Filtered DataFrame
        """
        logger.info("Applying quality filters...")

        initial_count = len(scored_df)

        # Filter criteria
        filters = {
            'minimum_records': scored_df['total_records'] >= 30,  # At least 30 days of data
            'minimum_volume': scored_df['avg_daily_volume'] > 1000,  # Minimum average volume
            'minimum_price': scored_df['avg_price'] >= 1.0,  # Minimum $1 average price
            'maximum_missing_data': scored_df['missing_data_pct'] <= 5.0,  # Max 5% missing data
            'valid_ohlc': scored_df['valid_ohlc'] == True,  # Valid OHLC data
            'minimum_date_range': scored_df['date_range_days'] >= 30,  # At least 30 days range
        }

        # Apply filters and log impact
        filtered_df = scored_df.copy()
        for filter_name, condition in filters.items():
            before_count = len(filtered_df)
            filtered_df = filtered_df[condition]
            after_count = len(filtered_df)
            removed = before_count - after_count
            logger.info(f"{filter_name}: Removed {removed} symbols, {after_count} remaining")

        logger.info(f"Total filtered: {initial_count - len(filtered_df)} symbols, {len(filtered_df)} remaining")

        return filtered_df

    def select_top_symbols(self, filtered_df: pd.DataFrame) -> List[str]:
        """
        Select top N symbols based on composite score.

        Args:
            filtered_df: Filtered DataFrame with scores

        Returns:
            List of top N symbols
        """
        logger.info(f"Selecting top {self.top_n} symbols...")

        # Sort by composite score (descending)
        ranked_df = filtered_df.sort_values('composite_score', ascending=False)

        # Take top N
        top_symbols = ranked_df.head(self.top_n)['symbol'].tolist()

        logger.info(f"Selected {len(top_symbols)} symbols")

        # Log some statistics about selected symbols
        if len(top_symbols) > 0:
            top_metrics = ranked_df.head(self.top_n)
            logger.info(
                f"Score range: {top_metrics['composite_score'].min():.3f} - {top_metrics['composite_score'].max():.3f}")
            logger.info(
                f"Avg volume range: {top_metrics['avg_daily_volume'].min():,.0f} - {top_metrics['avg_daily_volume'].max():,.0f}")
            logger.info(f"Price range: ${top_metrics['avg_price'].min():.2f} - ${top_metrics['avg_price'].max():.2f}")

        return top_symbols

    def create_filtered_dataset(self, top_symbols: List[str]) -> pd.DataFrame:
        """
        Create the filtered dataset with only top symbols.

        Args:
            top_symbols: List of symbols to include

        Returns:
            Filtered DataFrame
        """
        logger.info("Creating filtered dataset...")

        # Filter original data to only include top symbols
        self.filtered_df = self.df[self.df['symbol'].isin(top_symbols)].copy()

        # Sort by timestamp and symbol
        self.filtered_df = self.filtered_df.sort_values(['timestamp', 'symbol'])

        # Reset index
        self.filtered_df = self.filtered_df.reset_index(drop=True)

        logger.info(f"Filtered dataset shape: {self.filtered_df.shape}")

        return self.filtered_df

    def save_filtered_data(self) -> None:
        """
        Save the filtered dataset to CSV.
        """
        if self.filtered_df is None:
            raise ValueError("No filtered data to save. Run the filtering process first.")

        logger.info(f"Saving filtered data to {self.output_csv}")
        self.filtered_df.to_csv(self.output_csv, index=False)
        logger.info(f"Successfully saved {len(self.filtered_df)} records")

    def save_selection_report(self, scored_df: pd.DataFrame, top_symbols: List[str]) -> None:
        """
        Save a detailed report of the selection process.

        Args:
            scored_df: DataFrame with all scores
            top_symbols: List of selected symbols
        """
        report_file = self.output_csv.replace('.csv', '_selection_report.csv')

        # Create report with top symbols and their metrics
        report_df = scored_df[scored_df['symbol'].isin(top_symbols)].copy()
        report_df = report_df.sort_values('composite_score', ascending=False)

        # Round numeric columns for readability
        numeric_cols = report_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'score' in col:
                report_df[col] = report_df[col].round(4)
            elif col in ['avg_daily_volume', 'total_volume', 'avg_dollar_volume']:
                report_df[col] = report_df[col].round(0)
            elif col in ['avg_price', 'median_price', 'price_range']:
                report_df[col] = report_df[col].round(2)

        report_df.to_csv(report_file, index=False)
        logger.info(f"Saved selection report to {report_file}")

    def generate_summary(self, metrics_df: pd.DataFrame, top_symbols: List[str]) -> None:
        """
        Generate and display summary statistics.

        Args:
            metrics_df: DataFrame with all metrics
            top_symbols: List of selected symbols
        """
        print("\n" + "=" * 60)
        print("TRADING UNIVERSE FILTERING SUMMARY")
        print("=" * 60)

        print(f"Original universe: {len(metrics_df):,} symbols")
        print(f"Filtered universe: {len(top_symbols):,} symbols")
        print(f"Reduction: {((len(metrics_df) - len(top_symbols)) / len(metrics_df) * 100):.1f}%")

        if self.filtered_df is not None:
            print(f"Total records in filtered dataset: {len(self.filtered_df):,}")
            print(f"Date range: {self.filtered_df['timestamp'].min()} to {self.filtered_df['timestamp'].max()}")

            # Volume statistics
            volume_stats = self.filtered_df.groupby('symbol')['volume'].mean().describe()
            print(f"\nAverage Daily Volume Statistics:")
            print(f"  Mean: {volume_stats['mean']:,.0f}")
            print(f"  Median: {volume_stats['50%']:,.0f}")
            print(f"  Min: {volume_stats['min']:,.0f}")
            print(f"  Max: {volume_stats['max']:,.0f}")

            # Price statistics
            price_stats = self.filtered_df.groupby('symbol')['close'].mean().describe()
            print(f"\nAverage Price Statistics:")
            print(f"  Mean: ${price_stats['mean']:.2f}")
            print(f"  Median: ${price_stats['50%']:.2f}")
            print(f"  Min: ${price_stats['min']:.2f}")
            print(f"  Max: ${price_stats['max']:.2f}")

            # Show top 20 symbols
            top_20 = top_symbols[:20]
            print(f"\nTop 20 selected symbols: {', '.join(top_20)}")

            if len(top_symbols) > 20:
                print(f"... and {len(top_symbols) - 20} more")

    def run(self) -> None:
        """
        Execute the complete filtering process.
        """
        try:
            logger.info(f"Starting universe filtering process (top {self.top_n})...")

            # Step 1: Load data
            self.load_data()

            # Step 2: Analyze data quality
            metrics_df = self.analyze_data_quality()

            # Step 3: Calculate scores
            scored_df = self.calculate_scoring_metrics(metrics_df)

            # Step 4: Apply filters
            filtered_scored_df = self.apply_filters(scored_df)

            # Step 5: Select top symbols
            top_symbols = self.select_top_symbols(filtered_scored_df)

            # Step 6: Create filtered dataset
            self.create_filtered_dataset(top_symbols)

            # Step 7: Save results
            self.save_filtered_data()
            self.save_selection_report(filtered_scored_df, top_symbols)

            # Step 8: Generate summary
            self.generate_summary(metrics_df, top_symbols)

            logger.info("Universe filtering completed successfully!")

        except Exception as e:
            logger.error(f"Filtering process failed: {str(e)}")
            raise


def main():
    """
    Main function to run the filtering process.
    """
    # Configuration
    INPUT_CSV = r"" #Change file name to your stock universe CSV (From Universe Creation)
    OUTPUT_CSV = r"" #Change file name to your new CSV file 
    TOP_N = 5000  # Number of top companies to select

    # Create filter instance
    filter_instance = TradingUniverseFilter(INPUT_CSV, OUTPUT_CSV, TOP_N)

    # Run filtering
    filter_instance.run()


if __name__ == "__main__":
    main()
