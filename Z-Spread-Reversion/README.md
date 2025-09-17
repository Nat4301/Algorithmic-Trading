<section>
  <h1>Z-Spread Reversion on High-Correlation Equities (CVX/XOM)</h1>

  <p><strong>What it does</strong><br>
  Implements a mean-reversion pairs strategy on Chevron (CVX) and Exxon (XOM). It builds a <em>log-price spread</em>, converts it to a <em>rolling z-score</em>, and trades reversion: short the spread when it’s rich (positive z), long when it’s cheap (negative z), and close on mean-reversion or stop-loss.</p>

  <hr>

  <h2>Data &amp; Dependencies</h2>
  <ul>
    <li><strong>Data source:</strong> Databento <code>.dbn</code> file loaded via <code>DBNStore.from_file</code>.</li>
    <li><strong>Symbols:</strong> defaults to <code>["CVX", "XOM"]</code>.</li>
    <li><strong>Libraries:</strong> numpy, pandas, matplotlib, dataclasses, databento.</li>
  </ul>
  <p><code>fetch_databento_data()</code> reads the file, ensures a datetime index, filters to the two symbols, pivots to a tidy close-price frame, forward-fills, and drops missing rows.</p>

  <hr>

  <h2>Feature Engineering</h2>
  <p><code>calculate_spread_and_zscore()</code> produces:</p>
  <ul>
    <li><code>log_cvx</code>, <code>log_xom</code> – log prices.</li>
    <li><strong>Spread:</strong> <code>spread = log_cvx − log_xom</code>.</li>
    <li><strong>Rolling stats:</strong> over <code>lookback_window</code>, compute <code>spread_mean</code>, <code>spread_std</code>.</li>
    <li><strong>Z-score:</strong> <code>(spread − spread_mean) / spread_std</code>.</li>
    <li><strong>Diagnostics:</strong> <code>cvx_returns</code>, <code>xom_returns</code>, and <code>rolling_corr</code> (same window).</li>
  </ul>
  <p><em>Note:</em> The hedge ratio is implicitly 1:−1 (simple log-spread). The code comments invite replacing this with a cointegration/hedge-ratio estimate if desired.</p>

  <hr>

  <h2>Signal Logic</h2>
  <p><code>generate_signals()</code> walks the time series and emits <code>TradingSignal</code> events (dataclass with timestamp, action, z, spread, and prices):</p>
  <ul>
    <li><strong>Flat → Enter</strong>
      <ul>
        <li>If <code>z &gt; entry_threshold</code>: <strong>short_spread</strong> (short CVX / long XOM notionally).</li>
        <li>If <code>z &lt; −entry_threshold</code>: <strong>long_spread</strong> (long CVX / short XOM notionally).</li>
      </ul>
    </li>
    <li><strong>In Position → Exit</strong>
      <ul>
        <li><strong>Long spread:</strong> close if <code>z &gt; −exit_threshold</code> (reverted) or <code>z &lt; −stop_loss_threshold</code>.</li>
        <li><strong>Short spread:</strong> close if <code>z &lt; exit_threshold</code> or <code>z &gt; stop_loss_threshold</code>.</li>
      </ul>
    </li>
  </ul>
  <p>The strategy remains flat between discrete entry/exit signals (no pyramiding).</p>

  <hr>

  <h2>Backtest Mechanics</h2>
  <p><code>backtest_strategy(initial_capital=100_000, position_size=0.1)</code> executes trades only at signal timestamps:</p>
  <ul>
    <li><strong>Capital allocation:</strong> each trade uses <code>trade_cap = current_portfolio * position_size</code>.</li>
    <li><strong>Position sizing:</strong> splits capital evenly across legs using current prices:
      <ul>
        <li>Long spread: <code>+CVX</code>, <code>−XOM</code>; Short spread: <code>−CVX</code>, <code>+XOM</code>.</li>
      </ul>
    </li>
    <li><strong>P&amp;L:</strong> realized on the corresponding close signal:
      <br><code>pnl = (shares_cvx*exit_cvx + shares_xom*exit_xom) − (shares_cvx*entry_cvx + shares_xom*entry_xom)</code>.
    </li>
    <li><strong>Equity curve:</strong> updated on trade closes only (stepwise).</li>
    <li><strong>Returns for ratios:</strong> per-trade return = <code>pnl / trade_cap</code>.</li>
  </ul>

  <h3>Performance Metrics</h3>
  <p>Stored in <code>self.performance_metrics</code>:</p>
  <ul>
    <li><code>total_return</code>, <code>final_portfolio_value</code></li>
    <li><code>num_trades</code>, <code>win_rate</code>, <code>avg_trade_pnl</code></li>
    <li><strong>Risk-adjusted:</strong> annualized <strong>Sharpe</strong> &amp; <strong>Sortino</strong> via <code>sharpe_sortino()</code> (annualization uses estimated <code>trades_per_year</code>).</li>
    <li><strong>Drawdown:</strong> <code>max_drawdown_from_equity()</code> on the equity curve (negative fraction).</li>
    <li>Also returns arrays: <code>equity_curve</code>, <code>trade_pnl</code>, <code>trade_returns</code>.</li>
  </ul>

  <hr>

  <h2>Reporting</h2>
  <ul>
    <li><code>print_performance_summary()</code> – concise console summary of the metrics above.</li>
    <li><code>plot_results()</code> – three vertically stacked charts:
      <ol>
        <li>CVX/XOM price series</li>
        <li>Spread with entry markers (green ▲ for long spread, red ▼ for short spread)</li>
        <li>Z-score with entry/exit bands and zero line</li>
      </ol>
    </li>
  </ul>

  <hr>

  <h2>Configuration &amp; Tuning</h2>
  <p>Constructor parameters in <code>ZScoreReversalStrategy</code>:</p>
  <ul>
    <li><code>lookback_window</code> – rolling window for mean/std &amp; correlation.</li>
    <li><code>entry_threshold</code>, <code>exit_threshold</code> – signal gates.</li>
    <li><code>stop_loss_threshold</code> – z-score hard stop.</li>
    <li><code>mar_annual</code> – annualized minimum acceptable return (for Sortino/Sharpe).</li>
    <li><code>use_trade_returns</code> – whether to compute ratios on per-trade returns.</li>
  </ul>
  <p><code>run_zscore_strategy()</code> shows example hyperparameter sets for:</p>
  <ul>
    <li><strong>1-hour data</strong> (default in code)</li>
    <li><strong>21-period</strong> variant</li>
    <li><strong>1-day data</strong> (with example <code>position_size</code>)</li>
  </ul>
  <p>Set <code>file_path_h</code> (or <code>file_path_d</code>) to your Databento file and run.</p>

  <hr>

  <h2>Results and Returns</h2>
  <p>This section summarizes the performance of the Z-Spread Reversion Strategy on historical CVX/XOM data. 
  The metrics below are taken directly from the backtest output.</p>

  <h3> Returns from 1 Hour Price Data after optimization <h3>
  <ul>
    <li><strong>Total Return:</strong> 39.60%</li>
    <li><strong>Number of Trades:</strong> 1797 </li>
    <li><strong>Win Rate:</strong> 85.42%</li>
    <li><strong>Average Trade P&amp;L:</strong> $22.04 </li>
    <li><strong>Sharpe Ratio (annualized):</strong> 4.883 </li>
    <li><strong>Sortino Ratio (annualized):</strong> 5.985 </li>
    <li><strong>Trades/Year (est):</strong> 361 </li>
    <li><strong>Max Drawdown:</strong> -1.36%</li>
  </ul>

  <h3> Returns from 1 Day Price Data after optimization <h3>
  <ul>
    <li><strong>Total Return:</strong> 8.79%</li>
    <li><strong>Number of Trades:</strong> 17 </li>
    <li><strong>Win Rate:</strong> 70.59%</li>
    <li><strong>Average Trade P&amp;L:</strong> $517.25 </li>
    <li><strong>Sharpe Ratio (annualized):</strong> 1.205 </li>
    <li><strong>Sortino Ratio (annualized):</strong> 2.644 </li>
    <li><strong>Trades/Year (est):</strong> 5 </li>
    <li><strong>Max Drawdown:</strong> -1.44%</li>
  </ul>

  <hr>

  <h2>Assumptions &amp; Notes</h2>
  <ul>
    <li><strong>Hedge ratio = 1:</strong> The spread is simple log-difference; no regression/Kalman/cointegration is applied out of the box.</li>
    <li><strong>Slippage/fees not modeled.</strong> Results are optimistic relative to live trading.</li>
    <li><strong>Execution:</strong> at signal timestamps using contemporaneous prices.</li>
    <li><strong>Discrete trades:</strong> the strategy holds a single spread at a time; no compounding mid-trade.</li>
  </ul>

  <hr>

  <h2>Extending the Strategy</h2>
  <ul>
    <li>Replace the fixed 1:−1 hedge with <strong>OLS/cointegration</strong> or a <strong>Kalman filter</strong> for dynamic hedge ratios.</li>
    <li>Add <strong>transaction costs</strong>, <strong>borrow fees</strong>, and <strong>slippage</strong>.</li>
    <li>Switch to <strong>bar-close/next-bar</strong> execution to reduce look-ahead bias.</li>
    <li>Add <strong>position sizing</strong> by volatility, <strong>Kelly fraction</strong>, or <strong>risk parity</strong>.</li>
    <li>Introduce <strong>regime filters</strong> (e.g., require high rolling correlation, or macro filters).</li>
    <li>Generalize to a <strong>symbol pairs list</strong> and run a portfolio of spreads.</li>
  </ul>

  <hr>

  <h2>Quick Start</h2>
  <ol>
    <li>Install requirements and ensure access to a Databento <code>.dbn</code> containing CVX/XOM with <code>close</code> prices.</li>
    <li>Set <code>file_path_h</code> (or <code>file_path_d</code>) in <code>run_zscore_strategy()</code>.</li>
    <li>Run the script.</li>
  </ol>
</section>

