import os
import pandas as pd
from pathlib import Path
import databento as db
from typing import List, Optional
import logging
from datetime import datetime
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DBNConsolidator:
    """
    Consolidates .dbn OHLCV files from a folder into a single CSV file for backtesting.
    """

    def __init__(self, input_folder: str, output_file: str = "trading_universe.csv"):
        """
        Initialize the consolidator.

        Args:
            input_folder (str): Path to folder containing .dbn files
            output_file (str): Output CSV filename
        """
        self.input_folder = Path(input_folder)
        self.output_file = output_file
        self.consolidated_data = []

    def get_dbn_files(self) -> List[Path]:
        """
        Get all .dbn files from the input folder.

        Returns:
            List of Path objects for .dbn files
        """
        dbn_files = list(self.input_folder.glob("*.dbn"))
        logger.info(f"Found {len(dbn_files)} .dbn files")
        return dbn_files

    def extract_symbol_from_filename(self, filepath: Path) -> str:
        """
        Extract symbol from filename with multiple naming convention support.
        For Databento bundle files, this will return a bundle identifier.

        Args:
            filepath: Path to the .dbn file

        Returns:
            Symbol string or bundle identifier
        """
        filename = filepath.stem  # Gets filename without .dbn extension
        logger.debug(f"Extracting identifier from filename: {filename}")

        # Check if this is a Databento bundle file (contains multiple symbols)
        if 'equs-mini-' in filename.lower() or 'mini-' in filename.lower():
            logger.info(f"Detected Databento bundle file: {filename}")
            return "BUNDLE"  # This will be overridden by data extraction

        # Try different extraction patterns for individual symbol files
        patterns = [
            r'^([A-Z]+)_',  # Pattern: AAPL_1d, AAPL_daily, etc.
            r'^([A-Z]+)\.',  # Pattern: AAPL.something
            r'^([A-Z]+)-',  # Pattern: AAPL-1d, AAPL-daily
            r'^([A-Z]{1,5})(?=\d)',  # Pattern: AAPL123, MSFT456 (symbol followed by numbers)
            r'^([A-Z]+)',  # Pattern: Just the symbol at start
        ]

        # First, try the regex patterns
        for pattern in patterns:
            match = re.match(pattern, filename.upper())
            if match:
                symbol = match.group(1)
                logger.debug(f"Extracted symbol '{symbol}' using pattern '{pattern}'")
                return symbol

        # If no pattern matches, try removing common suffixes
        suffixes_to_remove = [
            '_1d', '_daily', '_1day', '_d1',
            '_1h', '_hourly', '_h1',
            '_1m', '_minute', '_min', '_m1',
            '_ohlcv', '_bars', '_candles',
            '_data', '_historical', '_hist'
        ]

        cleaned_filename = filename.upper()
        for suffix in suffixes_to_remove:
            if cleaned_filename.endswith(suffix.upper()):
                cleaned_filename = cleaned_filename.replace(suffix.upper(), '')
                break

        # Remove any remaining non-alphabetic characters and take only letters
        symbol = re.sub(r'[^A-Z]', '', cleaned_filename)

        # If we still don't have a valid symbol, use the first part of the filename
        if not symbol or len(symbol) > 10:  # Reasonable max length for a ticker
            # Split by common delimiters and take the first part
            parts = re.split(r'[_\-\.\s]+', filename.upper())
            if parts:
                symbol = re.sub(r'[^A-Z]', '', parts[0])

        # Final fallback - use entire filename (cleaned)
        if not symbol:
            symbol = re.sub(r'[^A-Z]', '', filename.upper())[:5]  # Max 5 chars

        logger.info(f"Final extracted symbol: '{symbol}' from filename: '{filename}'")
        return symbol if symbol else "UNKNOWN"

    def extract_symbol_from_data(self, df: pd.DataFrame) -> Optional[str]:
        """
        Try to extract symbol from the data itself.
        Enhanced for Databento files which often have symbols in the data.

        Args:
            df: DataFrame from the .dbn file

        Returns:
            Symbol string if found, None otherwise
        """
        # Common column names that might contain the symbol
        symbol_columns = [
            'symbol', 'ticker', 'instrument', 'instrument_id', 'raw_symbol',
            'Symbol', 'SYMBOL', 'Ticker', 'TICKER'
        ]

        for col in symbol_columns:
            if col in df.columns:
                unique_symbols = df[col].dropna().unique()
                if len(unique_symbols) == 1:
                    symbol = str(unique_symbols[0]).upper()
                    logger.info(f"Found symbol '{symbol}' in data column '{col}'")
                    return symbol
                elif len(unique_symbols) > 1:
                    logger.info(f"Multiple symbols found in column '{col}': {unique_symbols}")
                    # For multi-symbol files, we'll handle this differently
                    return "MULTI_SYMBOL"

        return None

    def process_dbn_file(self, filepath: Path) -> Optional[pd.DataFrame]:
        """
        Process a single .dbn file and extract OHLCV data.
        Enhanced to handle multi-symbol Databento bundle files.

        Args:
            filepath: Path to the .dbn file

        Returns:
            DataFrame with OHLCV data or None if processing fails
        """
        try:
            logger.info(f"Processing {filepath.name}")

            # Read the .dbn file using databento
            store = db.DBNStore.from_file(filepath)

            # Convert to DataFrame
            df = store.to_df()

            if df.empty:
                logger.warning(f"No data found in {filepath.name}")
                return None

            # Check if this is a multi-symbol file
            symbol_from_data = self.extract_symbol_from_data(df)

            if symbol_from_data == "MULTI_SYMBOL":
                logger.info(f"Processing multi-symbol bundle file: {filepath.name}")
                # For multi-symbol files, the symbol should already be in the data
                # We don't need to add it - just use what's there
                pass
            else:
                # Single symbol file or no symbol in data
                if not symbol_from_data:
                    symbol_from_data = self.extract_symbol_from_filename(filepath)

                # Add or update symbol column for single-symbol files
                if symbol_from_data and symbol_from_data != "BUNDLE":
                    df['symbol'] = symbol_from_data

            # Ensure we have the required OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.warning(f"Missing columns in {filepath.name}: {missing_cols}")
                return None

            # Ensure we have a symbol column
            if 'symbol' not in df.columns:
                logger.error(f"No symbol column found or created for {filepath.name}")
                return None

            # Reset index to make timestamp a column if it's currently the index
            if df.index.name == 'ts_event' or 'ts_event' in str(df.index.name):
                df = df.reset_index()

            # Rename timestamp column to standard format
            timestamp_cols = ['ts_event', 'timestamp', 'date', 'datetime', 'ts_recv']
            timestamp_col = None
            for col in timestamp_cols:
                if col in df.columns:
                    timestamp_col = col
                    break

            if timestamp_col:
                df = df.rename(columns={timestamp_col: 'timestamp'})
            else:
                logger.warning(f"No timestamp column found in {filepath.name}")
                return None

            # Select and reorder columns
            columns_order = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in columns_order if col in df.columns]
            df = df[available_cols]

            # Sort by timestamp
            df = df.sort_values('timestamp')

            # Log summary by symbol for multi-symbol files
            if symbol_from_data == "MULTI_SYMBOL":
                symbol_counts = df['symbol'].value_counts()
                logger.info(f"Successfully processed bundle with {len(symbol_counts)} symbols:")
                for symbol, count in symbol_counts.head(10).items():  # Show top 10
                    logger.info(f"  {symbol}: {count} records")
                if len(symbol_counts) > 10:
                    logger.info(f"  ... and {len(symbol_counts) - 10} more symbols")
            else:
                symbol = df['symbol'].iloc[0] if not df.empty else "Unknown"
                logger.info(f"Successfully processed {symbol}: {len(df)} records")

            return df

        except Exception as e:
            logger.error(f"Error processing {filepath.name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def debug_file_structure(self) -> None:
        """
        Debug method to inspect the structure of .dbn files and their naming.
        """
        dbn_files = self.get_dbn_files()

        print("\n" + "=" * 60)
        print("FILE STRUCTURE DEBUG")
        print("=" * 60)

        for i, filepath in enumerate(dbn_files[:5]):  # Check first 5 files
            print(f"\nFile {i + 1}: {filepath.name}")
            print(f"  Full path: {filepath}")
            print(f"  Stem (without extension): {filepath.stem}")

            # Try extracting symbol
            extracted_symbol = self.extract_symbol_from_filename(filepath)
            print(f"  Extracted symbol: {extracted_symbol}")

            # Try to peek at the actual data
            try:
                store = db.DBNStore.from_file(filepath)
                df = store.to_df()
                if not df.empty:
                    print(f"  Data shape: {df.shape}")
                    print(f"  Columns: {list(df.columns)}")

                    # Check if symbol exists in data
                    data_symbol = self.extract_symbol_from_data(df)
                    if data_symbol:
                        print(f"  Symbol in data: {data_symbol}")

                    # Show first few rows (limited columns)
                    display_cols = [col for col in ['timestamp', 'ts_event', 'symbol', 'open', 'close'] if
                                    col in df.columns]
                    if display_cols:
                        print(f"  Sample data:\n{df[display_cols].head(2)}")
                else:
                    print("  No data in file")
            except Exception as e:
                print(f"  Error reading file: {e}")

        if len(dbn_files) > 5:
            print(f"\n... and {len(dbn_files) - 5} more files")

    def consolidate_files(self) -> pd.DataFrame:
        """
        Process all .dbn files and consolidate into a single DataFrame.

        Returns:
            Consolidated DataFrame
        """
        dbn_files = self.get_dbn_files()

        if not dbn_files:
            raise ValueError(f"No .dbn files found in {self.input_folder}")

        all_dataframes = []

        for filepath in dbn_files:
            df = self.process_dbn_file(filepath)
            if df is not None:
                all_dataframes.append(df)

        if not all_dataframes:
            raise ValueError("No data could be extracted from any files")

        # Concatenate all DataFrames
        logger.info("Consolidating all data...")
        consolidated_df = pd.concat(all_dataframes, ignore_index=True)

        # Sort by timestamp and symbol
        consolidated_df = consolidated_df.sort_values(['timestamp', 'symbol'])

        # Reset index
        consolidated_df = consolidated_df.reset_index(drop=True)

        logger.info(f"Consolidated data shape: {consolidated_df.shape}")
        logger.info(f"Symbols in dataset: {sorted(consolidated_df['symbol'].unique())}")
        logger.info(f"Date range: {consolidated_df['timestamp'].min()} to {consolidated_df['timestamp'].max()}")

        return consolidated_df

    def save_to_csv(self, df: pd.DataFrame) -> None:
        """
        Save the consolidated DataFrame to CSV.

        Args:
            df: DataFrame to save
        """
        logger.info(f"Saving to {self.output_file}")
        df.to_csv(self.output_file, index=False)
        logger.info(f"Successfully saved {len(df)} records to {self.output_file}")

    def generate_summary_stats(self, df: pd.DataFrame) -> None:
        """
        Generate and display summary statistics.

        Args:
            df: Consolidated DataFrame
        """
        print("\n" + "=" * 50)
        print("TRADING UNIVERSE SUMMARY")
        print("=" * 50)
        print(f"Total records: {len(df):,}")
        print(f"Number of symbols: {df['symbol'].nunique()}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Symbols: {', '.join(sorted(df['symbol'].unique()))}")

        # Records per symbol
        print("\nRecords per symbol:")
        symbol_counts = df['symbol'].value_counts().sort_index()
        for symbol, count in symbol_counts.items():
            print(f"  {symbol}: {count:,} records")

        # Data quality check
        print("\nData quality:")
        print(f"  Missing values: {df.isnull().sum().sum()}")
        print(f"  Duplicate rows: {df.duplicated().sum()}")

    def run(self, debug_mode: bool = False) -> None:
        """
        Execute the full consolidation process.

        Args:
            debug_mode: If True, run debug analysis first
        """
        try:
            logger.info("Starting DBN consolidation process...")

            if debug_mode:
                self.debug_file_structure()
                response = input("\nContinue with consolidation? (y/n): ")
                if response.lower() != 'y':
                    return

            # Consolidate files
            consolidated_df = self.consolidate_files()

            # Save to CSV
            self.save_to_csv(consolidated_df)

            # Generate summary
            self.generate_summary_stats(consolidated_df)

            logger.info("Consolidation process completed successfully!")

        except Exception as e:
            logger.error(f"Consolidation process failed: {str(e)}")
            raise


def main():
    """
    Main function to run the consolidation process.
    """
    # Configuration
    INPUT_FOLDER = r"C:\Users\habel\OneDrive\Desktop\Algo Trade\Data\US Equity DBN"  # Update this path
    OUTPUT_FILE = r"C:\Users\habel\OneDrive\Desktop\Algo Trade\Data\trading_universe.csv"

    # Create consolidator instance
    consolidator = DBNConsolidator(INPUT_FOLDER, OUTPUT_FILE)

    # Run in debug mode first to see file structure
    print("Running in DEBUG mode to analyze file structure...")
    consolidator.run(debug_mode=True)


if __name__ == "__main__":
    main()