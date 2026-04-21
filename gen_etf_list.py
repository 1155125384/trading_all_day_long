import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import time
import warnings
import logging
import os
import sys
from datetime import datetime, timedelta
import concurrent.futures
import random
import ssl



def get_complete_us_etf_list():
    ssl._create_default_https_context = ssl._create_unverified_context
    
    nasdaq_url = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt"
    other_url = "ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt"
    
    try:
        # Read Nasdaq file
        df_nasdaq = pd.read_csv(nasdaq_url, sep="|")[:-1] # Remove footer
        etfs_nasdaq = df_nasdaq[df_nasdaq['ETF'] == 'Y']['Symbol'].tolist()
        
        # Read Other (NYSE/ARCA) file
        df_other = pd.read_csv(other_url, sep="|")[:-1]
        # In this file, it's 'ETF' as well
        etfs_other = df_other[df_other['ETF'] == 'Y']['NASDAQ Symbol'].tolist()
        
        all_etfs = sorted(list(set(etfs_nasdaq + etfs_other)))
        return all_etfs
    except Exception as e:
        print(f"Error fetching: {e}")
        return []

print("Phase 1/4: Fetching complete list of US ETFs...")
etf_list = get_complete_us_etf_list()
print(f"Total ETFs found: {len(etf_list)}")



warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

class SuppressOutput:
    def __enter__(self):
        self._original_stdout, self._original_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout, sys.stderr = self._original_stdout, self._original_stderr

def get_sentiment(df_slice):
    if df_slice.empty: return 0
    
    # Extract native numpy arrays for massive speedup
    close_vals = df_slice['Close'].values
    vol_vals = df_slice['Volume'].values
    
    # Calculate diff natively (prepend nan to match length)
    diff = np.diff(close_vals, prepend=np.nan)
    
    # Create masks
    buys_mask = diff > 0
    sells_mask = diff < 0
    
    # Sum volumes based on masks
    buys = np.sum(vol_vals[buys_mask])
    sells = np.sum(vol_vals[sells_mask])
    
    total = buys + sells
    return round((buys / total) * 100, 2) if total > 0 else 0

def analyze_etf_momentum_pipeline(ticker_list, batch_size=20):
    all_results = []
    two_years_ago_ts = (datetime.now() - timedelta(days=730)).timestamp()

    for i in tqdm(range(0, len(ticker_list), batch_size), desc="Scanning Market"):
        batch = ticker_list[i:i+batch_size]
        
        with SuppressOutput():
            # 1m data kept at 5d (yfinance limit for 1m interval)
            d1m = yf.download(batch, period="5d", interval="1m", group_by='ticker', progress=False)
            # Daily data set to 1y for a 1-year lookback
            d1d = yf.download(batch, period="1y", interval="1d", group_by='ticker', progress=False)

        # Define the processing logic for a single ticker to allow multithreading
        def process_ticker(ticker):
            try:
                t_obj = yf.Ticker(ticker)
                
                # Quick establishment check (Network Bound - benefits most from multithreading)
                inception_ts = t_obj.info.get('firstTradeDateEpochUtc') or t_obj.info.get('inceptionDate')
                if inception_ts and inception_ts > two_years_ago_ts:
                    return None

                if len(batch) > 1:
                    df1m = d1m[ticker].dropna()
                    df1d = d1d[ticker].dropna()
                else:
                    df1m = d1m.dropna()
                    df1d = d1d.dropna()
                
                if df1m.empty or df1d.empty: return None

                # 1) 3-Day Average Volume Filter
                avg_vol_3d = df1d['Volume'].iloc[-3:].mean()
                if avg_vol_3d <= 1000:
                    return None

                # Calculate Windows (Now heavily optimized with NumPy)
                m60 = get_sentiment(df1m.iloc[-60:])
                h3 = get_sentiment(df1m.iloc[-180:])
                h6 = get_sentiment(df1m.iloc[-360:])
                d1 = get_sentiment(df1m.iloc[-390:])
                d3 = get_sentiment(df1m.iloc[-1170:])

                # Cumulative Momentum Score Weights
                cum_score = (m60 * 0.20) + (h3 * 0.30) + (h6 * 0.30) + (d1 * 0.10) + (d3 * 0.10)

                # Gate: Immediate 5m momentum must still be strong (65%)
                if cum_score < 50: return None

                # 2) Last Peak Price & Mark Calculation (1 Year lookback, excluding last 7 days)
                current_price = df1d['Close'].iloc[-1]
                last_peak_price = df1d['High'].iloc[:-7].max() 
                
                # --- UPDATED: Extended Multiplier Logic ---
                pct_diff = ((last_peak_price - current_price) / last_peak_price) * 100
                raw_peak_mark = 50 + (pct_diff * 4)
                peak_mark = round(max(0, min(100, raw_peak_mark)), 2)

                # --- NEW: Exclude anything with a Peak Mark of 30 or lower ---
                if peak_mark <= 30:
                    return None

                return {
                    'Ticker': ticker,
                    'Price': round(current_price, 2),
                    'Avg_Vol': int(avg_vol_3d),
                    'Peak_Price': round(last_peak_price, 2),
                    'Peak_Mark': peak_mark,
                    'Cum_Buy_%': round(cum_score, 2),
                    'Buy_1h_%': m60,
                    'Buy_3h_%': h3,
                    'Buy_6h_%': h6,
                    'Buy_1d_%': d1,
                    'Buy_3d_%': d3
                }
            except Exception:
                return None

        # Execute the network/processing loop concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Map returns results in the same order as the batch iterable
            results = executor.map(process_ticker, batch)
            
            # Filter out the None returns (tickers that failed conditions or errored)
            for res in results:
                if res is not None:
                    all_results.append(res)
            
        time.sleep(0.5)

    df = pd.DataFrame(all_results)
    if df.empty: return df
    
    # Sort by Cumulative Score primarily, then by the 5m trigger
    return df.sort_values(by=['Peak_Mark','Cum_Buy_%', 'Buy_1h_%'], ascending=False)

# Example Execution (Ensure etf_list is defined)
print("Phase 2/4: Analyzing ETF momentum across multiple timeframes...")
final_report = analyze_etf_momentum_pipeline(etf_list)
print(final_report.to_string(index=False))



class ETFAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.etf = yf.Ticker(self.ticker)
        self.df = self.etf.history(period="5y")
        self.info = self.etf.info
        
        if self.df.empty:
            raise ValueError(f"No data found for ticker {self.ticker}.")

    def _proportional_score(self, value, worst_val, best_val, max_score):
        # Handle NaNs
        if pd.isna(value): return 0
        
        if best_val > worst_val:
            if value >= best_val: return max_score
            if value <= worst_val: return 0
        else:
            if value <= best_val: return max_score
            if value >= worst_val: return 0
        return ((value - worst_val) / (best_val - worst_val)) * max_score

    def _calculate_rsi(self, window=14):
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self):
        exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    def _calculate_atr(self, window=14):
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=window).mean()

    def get_short_term_score(self):
        score = 0
        latest = self.df.iloc[-1]
        
        # 1. Dollar Volume (Price * Volume) -> True Liquidity Measure
        avg_volume = self.df['Volume'].tail(20).mean()
        dollar_volume = avg_volume * latest['Close']
        score += self._proportional_score(dollar_volume, worst_val=2_000_000, best_val=50_000_000, max_score=25)

        # 2. RSI Momentum (Linear from 40 to 70)
        rsi = self._calculate_rsi().iloc[-1]
        score += self._proportional_score(rsi, worst_val=40, best_val=70, max_score=25)
        
        # 3. MACD Normalized by ATR (Volatility Adjusted)
        macd, signal = self._calculate_macd()
        atr = self._calculate_atr().iloc[-1]
        
        if atr > 0:
            macd_diff_atr = (macd.iloc[-1] - signal.iloc[-1]) / atr
        else:
            macd_diff_atr = 0
            
        # A MACD/ATR divergence of 0.5 is a strong directional push
        score += self._proportional_score(macd_diff_atr, worst_val=-0.5, best_val=0.5, max_score=25)

        # 4. Distance to SMA20
        sma20 = self.df['Close'].tail(20).mean()
        dist_sma20 = (latest['Close'] - sma20) / sma20
        score += self._proportional_score(dist_sma20, worst_val=-0.05, best_val=0.05, max_score=25)

        return round(score, 2)

    def get_medium_term_score(self):
        if len(self.df) < 252: return 50.0 
        score = 0
        
        # 1. Trend Strength
        sma50 = self.df['Close'].tail(50).mean()
        sma200 = self.df['Close'].tail(200).mean()
        trend_strength = (sma50 - sma200) / sma200
        score += self._proportional_score(trend_strength, worst_val=-0.10, best_val=0.10, max_score=30)
        
        # 2. Sharpe Ratio
        df_1y = self.df.tail(252).copy()
        df_1y['Daily_Return'] = df_1y['Close'].pct_change()
        mean_return = df_1y['Daily_Return'].mean() * 252
        volatility = df_1y['Daily_Return'].std() * np.sqrt(252)
        
        sharpe = (mean_return - 0.04) / volatility if volatility > 0 else 0
        score += self._proportional_score(sharpe, worst_val=0.0, best_val=1.5, max_score=40)
        
        # 3. YoY Return
        yoy_return = (self.df['Close'].iloc[-1] / self.df['Close'].iloc[-252]) - 1
        score += self._proportional_score(yoy_return, worst_val=0.0, best_val=0.20, max_score=30)

        return round(score, 2)

    def get_long_term_score(self):
        score = 0
        
        # 1. Expense Ratio (Fees)
        fees = self.info.get('expenseRatio') or self.info.get('annualReportExpenseRatio') or 0.005
        score += self._proportional_score(fees, worst_val=0.006, best_val=0.0005, max_score=30)
        
        # 2. Max Drawdown
        if len(self.df) > 252:
            roll_max = self.df['Close'].cummax()
            drawdown = self.df['Close'] / roll_max - 1.0
            max_dd = drawdown.min()
            score += self._proportional_score(max_dd, worst_val=-0.40, best_val=-0.10, max_score=35)
        else:
            score += 17.5 

        # 3. Total Assets (AUM)
        total_assets = self.info.get('totalAssets') or 100_000_000
        score += self._proportional_score(total_assets, worst_val=100_000_000, best_val=5_000_000_000, max_score=35)

        return round(score, 2)

# === Execution Block ===
df = final_report.copy()
print("Phase 3/4: Grading for ETFs...")

def process_single_ticker(ticker, max_retries=5):
    for attempt in range(max_retries):
        try:
            # Base delay to prevent initial spam
            time.sleep(random.uniform(2, 5))
            
            analyzer = ETFAnalyzer(ticker)
            short = analyzer.get_short_term_score()
            med = analyzer.get_medium_term_score()
            long_t = analyzer.get_long_term_score()
            total = round((short * 0.35) + (med * 0.35) + (long_t * 0.3), 2)
            
            return ticker, total, short, med, long_t
            
        except Exception as e:
            error_msg = str(e)
            # Check if it's a rate limit error
            if "Too Many Requests" in error_msg or "Rate limited" in error_msg or "429" in error_msg:
                wait_time = (attempt + 1) * 20  # Wait 15s, then 30s, then 45s
                print(f"\n[!] Rate limited on {ticker} (Attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                # If it's a different error (e.g., delisted ticker), print and break
                print(f"\n[!] Error processing {ticker}: {error_msg}")
                break 
                
    # If we exhaust all retries or hit a non-recoverable error, return NaNs
    print(f"\n[X] Failed to process {ticker} after {max_retries} attempts.")
    return ticker, np.nan, np.nan, np.nan, np.nan

results_map = {}

# Reduced max_workers from 10 to 4. It is still much faster than sequential, 
# but slow enough to stay under Yahoo's radar.
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    futures = {executor.submit(process_single_ticker, ticker): ticker for ticker in df['Ticker']}
    
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(df), desc="Analyzing ETFs", unit="ticker"):
        ticker, total, short, med, long_t = future.result()
        results_map[ticker] = {
            'Grading': total,
            'Short_Term': short,
            'Medium_Term': med,
            'Long_Term': long_t
        }

df['Grading'] = df['Ticker'].map(lambda x: results_map.get(x, {}).get('Grading', np.nan))
df['Short_Term'] = df['Ticker'].map(lambda x: results_map.get(x, {}).get('Short_Term', np.nan))
df['Medium_Term'] = df['Ticker'].map(lambda x: results_map.get(x, {}).get('Medium_Term', np.nan))
df['Long_Term'] = df['Ticker'].map(lambda x: results_map.get(x, {}).get('Long_Term', np.nan))

column_order = [
    'Ticker', 'Peak_Mark', 'Grading', 'Cum_Buy_%', 'Price', 
    'Buy_5m_%', 'Buy_1h_%', 'Buy_6h_%', 'Buy_1d_%', 'Buy_3d_%', 
    'Short_Term', 'Medium_Term', 'Long_Term',
    'Avg_Vol', 'Peak_Price'
]

available_columns = [col for col in column_order if col in df.columns]
df = df[available_columns]

# Print the pre-filtered dataframe to check if NaNs are still present
print(f"Total rows analyzed: {len(df)}")
print(f"Rows with successful Grading: {df['Grading'].notna().sum()}")

df = df.sort_values(by=['Grading'], ascending=False).reset_index(drop=True)

# Keep the original DataFrame intact for debugging, filter into a new one
df_filtered = df[df['Grading'] >= 45].copy()

if df_filtered.empty:
    print("\n[!] The filtered DataFrame is still empty. Look at the printed errors above to see what failed.")
else:
    print(df_filtered.to_string())




warnings.filterwarnings('ignore', category=FutureWarning)

def calculate_rsi(data, periods=14):
    """
    Calculates RSI using Wilder's Smoothing (Standard RSI),
    which is much more accurate than a simple rolling mean.
    """
    delta = data.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's Smoothing (Exponential Moving Average)
    avg_gain = gain.ewm(com=periods - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=periods - 1, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def process_scoring(tickers, df_filtered):
    """
    Batch downloads data and calculates swing scores to avoid API rate limits
    and drastically improve calculation speed.
    """
    # Batch download is significantly faster than querying ticker by ticker
    hist_data = yf.download(tickers, period="60d", group_by='ticker', threads=True, progress=False)
    
    scores = {}
    
    print("Phase 4/4: Calculating detailed momentum scores for each ETF...")
    for ticker_symbol in tqdm(tickers, desc="Scoring"):
        try:
            # Handle both single-ticker and multi-ticker yf.download output structures
            if len(tickers) == 1:
                df = hist_data.copy()
            else:
                if ticker_symbol not in hist_data.columns.levels[0]:
                    scores[ticker_symbol] = np.nan
                    continue
                df = hist_data[ticker_symbol].copy()

            df.dropna(subset=['Close'], inplace=True)
            
            # Guard clause: Require at least 40 days of data for valid 30-day lookbacks and RSI
            if len(df) < 40:
                scores[ticker_symbol] = np.nan
                continue

            # Calculate indicators
            df['RSI'] = calculate_rsi(df['Close'])
            df['SMA_50'] = df['Close'].rolling(window=50).mean() # Trend filter
            
            recent_10d_low = df['Low'].tail(10).min()
            recent_30d_high = df['High'].tail(30).max()
            current_price = df['Close'].iloc[-1]
            
            adv_20 = df['Volume'].tail(20).mean()
            rebound_avg_vol = df['Volume'].tail(3).mean()
            current_rsi = df['RSI'].iloc[-1]
            sma_50 = df['SMA_50'].iloc[-1]

            # --- TRADING LOGIC SCORING ---

            # 1. Momentum Score (Max 30)
            rebound_pct = ((current_price - recent_10d_low) / recent_10d_low) * 100
            score_momentum = min(30.0, max(0.0, rebound_pct * 10.0))

            # 2. Volume Score with Accumulation/Distribution Check (Max 20)
            vol_ratio = rebound_avg_vol / adv_20 if adv_20 > 0 else 0
            
            # Check if the last 3 days were generally bullish or bearish
            recent_return = (current_price - df['Close'].iloc[-4]) / df['Close'].iloc[-4]
            if recent_return > 0:
                # Accumulation: Reward high volume on up moves
                score_volume = min(20.0, max(0.0, (vol_ratio - 1.0) * 40.0))
            else:
                # Distribution: Penalize high volume on down moves
                score_volume = max(0.0, 10.0 - (vol_ratio * 10.0)) 

            # 3. RSI Score (Max 20) - Buying the dip (30-60 range preferred)
            if pd.isna(current_rsi) or current_rsi < 30 or current_rsi > 60:
                score_rsi = 0.0
            elif 30 <= current_rsi <= 40:
                score_rsi = (current_rsi - 30) * 2.0
            elif 40 < current_rsi <= 60:
                score_rsi = 20.0 - (current_rsi - 40)

            # 4. Dip Severity Score (Max 20)
            dip_pct = ((recent_30d_high - recent_10d_low) / recent_30d_high) * 100
            if dip_pct < 5:
                score_dip = (dip_pct / 5.0) * 20.0
            elif 5 <= dip_pct <= 10:
                score_dip = 20.0
            else:
                score_dip = max(0.0, 20.0 - ((dip_pct - 10.0) * 2.0))

            # 5. Liquidity Score (Max 10)
            score_liquidity = min(10.0, (adv_20 / 2_000_000.0) * 10.0)

            # Calculate total
            total_score = score_momentum + score_volume + score_rsi + score_dip + score_liquidity
            
            # 6. Trend Filter Penalty (Crucial for Swing Trading)
            # If price is below the 50 SMA, penalize the score by 30% to avoid catching falling knives.
            if not pd.isna(sma_50) and current_price < sma_50:
                total_score *= 0.70 

            scores[ticker_symbol] = round(total_score, 2)
            
        except Exception as e:
            # Silently pass errors to avoid cluttering the progress bar, just return NaN
            scores[ticker_symbol] = np.nan

    return scores

df_final = df_filtered.copy()

# Extract list of tickers to batch download
ticker_list = df_final['Ticker'].tolist()

# Get scores dictionary via our optimized batch function
momentum_scores_dict = process_scoring(ticker_list, df_final)

# Map the scores back to the dataframe
df_final['Momentum_Score'] = df_final['Ticker'].map(momentum_scores_dict)

WEIGHT_MOMENTUM = 0.55
WEIGHT_GRADING = 0.35
WEIGHT_PEAK = 0.10

df_final['Total_Score'] = (
    (df_final['Momentum_Score'] * WEIGHT_MOMENTUM) +
    (df_final['Grading'] * WEIGHT_GRADING) +
    (df_final['Peak_Mark'] * WEIGHT_PEAK)
)

df_final['Total_Score'] = df_final['Total_Score'].round(2)

column_order = [
    'Ticker', 'Total_Score', 'Momentum_Score', 'Grading', 'Peak_Mark', 
    'Cum_Buy_%', 'Price', 'Buy_5m_%', 'Buy_1h_%', 'Buy_6h_%', 
    'Buy_1d_%', 'Buy_3d_%', 'Short_Term', 'Medium_Term', 'Long_Term',
    'Avg_Vol', 'Peak_Price'
]

# Reorder columns (ensure all exist in your actual df_filtered)
df_final = df_final[[col for col in column_order if col in df_final.columns]]

# Sort the DataFrame by the new score
df_final = df_final.sort_values(by=['Total_Score', 'Momentum_Score', 'Grading', 'Peak_Mark','Cum_Buy_%'], ascending=False).reset_index(drop=True)

print("\nFinal Ranked Results:")
print(df_final.to_string())



if not df_final.empty:
    print("\nSaving final results to 'etf_data.csv'...")
    df_final.to_csv('etf_data.csv', index=False)
    print("File saved successfully.")
else:
    print("DataFrame is empty; no file was created.")
