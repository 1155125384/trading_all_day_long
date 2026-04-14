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
    diff = df_slice['Close'].diff()
    buys = df_slice[diff > 0]['Volume'].sum()
    total = df_slice[diff > 0]['Volume'].sum() + df_slice[diff < 0]['Volume'].sum()
    return round((buys / total) * 100, 2) if total > 0 else 0

def analyze_etf_momentum_pipeline(ticker_list, batch_size=20):
    all_results = []
    two_years_ago_ts = (datetime.now() - timedelta(days=730)).timestamp()

    for i in tqdm(range(0, len(ticker_list), batch_size), desc="Scanning Market"):
        batch = ticker_list[i:i+batch_size]
        
        with SuppressOutput():
            d1m = yf.download(batch, period="5d", interval="1m", group_by='ticker', progress=False)
            d1d = yf.download(batch, period="5d", interval="1d", group_by='ticker', progress=False)

        for ticker in batch:
            try:
                t_obj = yf.Ticker(ticker)
                # Quick establishment check
                inception_ts = t_obj.info.get('firstTradeDateEpochUtc') or t_obj.info.get('inceptionDate')
                if inception_ts and inception_ts > two_years_ago_ts:
                    continue 

                if len(batch) > 1:
                    df1m = d1m[ticker].dropna()
                    df1d = d1d[ticker].dropna()
                else:
                    df1m = d1m.dropna()
                    df1d = d1d.dropna()
                
                if df1m.empty or df1d.empty: continue

                # Calculate Windows
                m5 = get_sentiment(df1m.iloc[-5:])
                m60 = get_sentiment(df1m.iloc[-60:])
                h6 = get_sentiment(df1m.iloc[-360:])
                d1 = get_sentiment(df1m.iloc[-390:])
                d3 = get_sentiment(df1m.iloc[-1170:])

                # --- NEW: Cumulative Momentum Score ---
                # Logic: (5m * 0.30) + (1h * 0.25) + (6h * 0.20) + (1d * 0.10) + (3d * 0.10)
                cum_score = (m5 * 0.30) + (m60 * 0.25) + (h6 * 0.25) + (d1 * 0.10) + (d3 * 0.10)

                # Gate: Immediate 5m momentum must still be strong (65%)
                if cum_score < 65: continue

                res = {
                    'Ticker': ticker,
                    'Price': round(df1d['Close'].iloc[-1], 2),
                    'Cum_Buy_%': round(cum_score, 2),
                    'Buy_5m_%': m5,
                    'Buy_1h_%': m60,
                    'Buy_6h_%': h6,
                    'Buy_1d_%': d1,
                    'Buy_3d_%': d3
                }
                all_results.append(res)
            except:
                continue
            
        time.sleep(0.5)

    df = pd.DataFrame(all_results)
    if df.empty: return df
    
    # Sort by Cumulative Score primarily, then by the 5m trigger
    return df.sort_values(by=['Cum_Buy_%', 'Buy_5m_%'], ascending=False)

print("Phase 2/4: Analyzing ETF momentum across multiple timeframes...")
final_report = analyze_etf_momentum_pipeline(etf_list)
print(final_report.to_string(index=False))


# 1. The Proportional Scoring Class
class ETFAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.etf = yf.Ticker(self.ticker)
        self.df = self.etf.history(period="5y")
        self.info = self.etf.info
        
        if self.df.empty:
            raise ValueError(f"No data found for ticker {self.ticker}.")

    def _proportional_score(self, value, worst_val, best_val, max_score):
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

    def get_short_term_score(self):
        score = 0
        latest = self.df.iloc[-1]
        
        avg_volume = self.df['Volume'].tail(20).mean()
        score += self._proportional_score(avg_volume, worst_val=100_000, best_val=2_000_000, max_score=25)

        rsi = self._calculate_rsi().iloc[-1]
        if rsi <= 55: score += self._proportional_score(rsi, worst_val=30, best_val=55, max_score=25)
        else: score += self._proportional_score(rsi, worst_val=80, best_val=55, max_score=25)
        
        macd, signal = self._calculate_macd()
        macd_diff = (macd.iloc[-1] - signal.iloc[-1]) / latest['Close']
        score += self._proportional_score(macd_diff, worst_val=-0.01, best_val=0.01, max_score=25)

        sma20 = self.df['Close'].tail(20).mean()
        dist_sma20 = (latest['Close'] - sma20) / sma20
        score += self._proportional_score(dist_sma20, worst_val=-0.05, best_val=0.05, max_score=25)

        return round(score, 2)

    def get_medium_term_score(self):
        if len(self.df) < 252: return 50.0 
        score = 0
        
        sma50 = self.df['Close'].tail(50).mean()
        sma200 = self.df['Close'].tail(200).mean()
        trend_strength = (sma50 - sma200) / sma200
        score += self._proportional_score(trend_strength, worst_val=-0.10, best_val=0.10, max_score=30)
        
        df_1y = self.df.tail(252).copy()
        df_1y['Daily_Return'] = df_1y['Close'].pct_change()
        mean_return = df_1y['Daily_Return'].mean() * 252
        volatility = df_1y['Daily_Return'].std() * np.sqrt(252)
        
        sharpe = (mean_return - 0.04) / volatility if volatility != 0 else 0
        score += self._proportional_score(sharpe, worst_val=0.0, best_val=1.5, max_score=40)
        
        yoy_return = (self.df['Close'].iloc[-1] / self.df['Close'].iloc[-252]) - 1
        score += self._proportional_score(yoy_return, worst_val=0.0, best_val=0.20, max_score=30)

        return round(score, 2)

    def get_long_term_score(self):
        score = 0
        
        fees = self.info.get('expenseRatio', self.info.get('annualReportExpenseRatio', 0.005))
        if fees is None: fees = 0.005
        score += self._proportional_score(fees, worst_val=0.006, best_val=0.0005, max_score=30)
        
        if len(self.df) > 252:
            roll_max = self.df['Close'].cummax()
            drawdown = self.df['Close'] / roll_max - 1.0
            max_dd = drawdown.min()
            score += self._proportional_score(max_dd, worst_val=-0.40, best_val=-0.10, max_score=35)
        else:
            score += 17.5 

        total_assets = self.info.get('totalAssets', 100_000_000)
        if total_assets is None: total_assets = 100_000_000
        score += self._proportional_score(total_assets, worst_val=100_000_000, best_val=5_000_000_000, max_score=35)

        return round(score, 2)

df = final_report.copy()

print("Phase 3/4: Calculating comprehensive momentum scores for each ETF...")
# Initialize lists
short_scores, medium_scores, long_scores, total_scores = [], [], [], []

# Wrap the dataframe iterrows in tqdm for the progress bar
# 'desc' sets the prefix text; 'unit' labels each iteration
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Analyzing ETFs", unit="ticker"):
    ticker = row['Ticker']
    
    try:
        analyzer = ETFAnalyzer(ticker)
        
        short = analyzer.get_short_term_score()
        med = analyzer.get_medium_term_score()
        long_t = analyzer.get_long_term_score()
        
        # Applying your updated weights (25/35/40)
        total = round((short * 0.25) + (med * 0.35) + (long_t * 0.40), 2)
        
        total_scores.append(total)
        short_scores.append(short)
        medium_scores.append(med)
        long_scores.append(long_t)
        
    except Exception as e:
        # We append NaN to keep list lengths consistent with the DataFrame
        total_scores.append(np.nan)
        short_scores.append(np.nan)
        medium_scores.append(np.nan)
        long_scores.append(np.nan)
        
    time.sleep(0.5) 

# Append results to DataFrame
df['Total_Score'] = total_scores
df['Short_Term'] = short_scores
df['Medium_Term'] = medium_scores
df['Long_Term'] = long_scores

# Final Sort
column_order = [
    'Ticker', 'Cum_Buy_%', 'Total_Score', 'Price', 
    'Buy_5m_%', 'Buy_1h_%', 'Buy_6h_%', 'Buy_1d_%', 'Buy_3d_%', 
    'Short_Term', 'Medium_Term', 'Long_Term'
]

df = df[column_order]
df = df.sort_values(by=['Total_Score','Cum_Buy_%'], ascending=False).reset_index(drop=True)
df_filtered = df[df['Total_Score'] >= 45].copy()

print(df_filtered.to_string())



def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_swing_score(ticker_symbol):
    try:
        # Suppress yfinance warnings for cleaner output
        ticker = yf.Ticker(ticker_symbol)
        history_df = ticker.history(period="60d")
        
        # Guard clause in case data is missing
        if history_df.empty or len(history_df) < 30:
            return np.nan

        # Calculate indicators
        history_df['RSI'] = calculate_rsi(history_df['Close'])
        recent_10d_low = history_df['Low'].tail(10).min()
        recent_30d_high = history_df['High'].tail(30).max()
        current_price = history_df['Close'].iloc[-1]
        
        adv_20 = history_df['Volume'].tail(20).mean()
        rebound_avg_vol = history_df['Volume'].tail(3).mean()
        current_rsi = history_df['RSI'].iloc[-1]

        # 1. Momentum Score (Max 30)
        rebound_pct = ((current_price - recent_10d_low) / recent_10d_low) * 100
        score_momentum = min(30.0, max(0.0, rebound_pct * 10.0))

        # 2. Volume Score (Max 20)
        vol_ratio = rebound_avg_vol / adv_20 if adv_20 > 0 else 0
        score_volume = min(20.0, max(0.0, (vol_ratio - 1.0) * 40.0))

        # 3. RSI Score (Max 20)
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

        total_score = score_momentum + score_volume + score_rsi + score_dip + score_liquidity
        return round(total_score, 2)
        
    except Exception as e:
        print(f"Error scoring {ticker_symbol}: {e}")
        return np.nan
    
tqdm.pandas(desc="Calculating Momentum Scores") # 2. Set up the progress bar

print("Phase 4/4: Calculating detailed momentum scores for each ETF and finalizing rankings...")

# 4. Apply the function to the DataFrame to create the new column
df_final = df_filtered.copy()
df_final['Momentum_Score'] = df_final['Ticker'].copy().progress_apply(get_swing_score)

column_order = [
    'Ticker', 'Cum_Buy_%', 'Total_Score', 'Momentum_Score', 'Price', 
    'Buy_5m_%', 'Buy_1h_%', 'Buy_6h_%', 'Buy_1d_%', 'Buy_3d_%', 
    'Short_Term', 'Medium_Term', 'Long_Term'
]

df_final = df_final[column_order]

# 5. Sort the DataFrame by the new score to bubble the best setups to the top
df_final = df_final.sort_values(by=['Momentum_Score','Total_Score','Cum_Buy_%'], ascending=False).reset_index(drop=True)

# Display the updated DataFrame with the new column
print(df_final.to_string())

if not df_final.empty:
    print("\nSaving final results to 'etf_data.csv'...")
    df_final.to_csv('etf_data.csv', index=False)
    print("File saved successfully.")
else:
    print("DataFrame is empty; no file was created.")

