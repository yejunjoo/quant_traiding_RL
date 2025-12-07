# runner
import yfinance as yf
from Environment import Environment as ENV
from algo.ppo import ppo as PPO

## todo: Add yaml file reading for hyperparameters ##

N_tickers = 1
start_date = "2010-01-01"
end_date = "2024-01-01"
tickers_candidate = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
# -------------------------- #

tickers = tickers_candidate[0:N_tickers]
df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
df_stacked = df.stack(level=1, future_stack=True)
df_stacked = df_stacked.sort_index(level=[0, 1])

raw_data = df_stacked.values
n_days = len(df.index)
n_features = raw_data.shape[1]

# [Time step, ticker_index, feature_index]
# features: Close, High, Low, Open, Volume
data_matrix = raw_data.reshape(n_days, N_tickers, n_features)
env = ENV()