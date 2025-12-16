# runner
import yfinance as yf
from Environment import StockTradingEnv
import gymnasium as gym
from algo.ppo import PPO, Actor, Critic
from algo.modules  import MLP, MultivariateGaussianDiagonalCovariance

## todo: Add yaml file reading for hyperparameters ##

# Data
N_tickers = 1
Start_date = "2023-01-01" #"2010-01-01"
End_date = "2024-01-01"
Tickers_candidate = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]

# Learning
MAX_EPOCH = int(1e6) # 백만
Rollout_storage = 2048

# Env
Bankrupt_coef = 0.3
Termination_reward = -1e4
Max_balance = 1e7
# -------------------------- #



def make_env(data_matrix, bankrupt_coef, termination_reward, max_balance):
    env = StockTradingEnv(df_matrix=data_matrix,
                          bankrupt_coef=bankrupt_coef,
                          termination_reward=termination_reward,
                          max_balance=max_balance)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def shape_data_matrix(tickers):
    df = yf.download(tickers, start=Start_date, end=End_date, auto_adjust=True)
    df_stacked = df.stack(level=1, future_stack=True)
    df_stacked = df_stacked.sort_index(level=[0, 1])

    raw_data = df_stacked.values
    n_days = len(df.index)
    n_features = raw_data.shape[1]

    # [Time step, ticker_index, feature_index]
    # features: Close, High, Low, Open, Volume
    data_matrix = raw_data.reshape(n_days, N_tickers, n_features)   # shape: 250*1*5
    print(f"Data matrix shape: {data_matrix.shape}")
    return data_matrix

data_matrix = shape_data_matrix(Tickers_candidate[0:N_tickers])
env = make_env(data_matrix, Bankrupt_coef, Termination_reward, Max_balance)
obs_shape = env.observation_space.shape
action_shape = env.action_space.shape
print(f"Obs shape\t: {obs_shape}")
print(f"Action shape\t: {action_shape}")
assert len(obs_shape) == 1
assert len(action_shape) == 1

actor = Actor(MLP(),
              MultivariateGaussianDiagonalCovariance())
critic = Critic(MLP())

ppo = PPO(actor=actor,
          critic=critic,
          rollout_storage=Rollout_storage,
          obs_shape=obs_shape[0],
          action_shape=action_shape[0])

new_obs, info = env.reset()

for epoch in range(MAX_EPOCH):
    for batch in range(Rollout_storage):
        curr_obs = new_obs
        action = ppo.act(curr_obs)
        new_obs, reward, truncated, terminated, info = env.step(action)
        ppo.step(obs=curr_obs, reward=reward, truncated=truncated, terminated=terminated)
    ppo.update(new_obs)














# tickers = tickers_candidate[0:N_tickers]
# df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
# df_stacked = df.stack(level=1, future_stack=True)
# df_stacked = df_stacked.sort_index(level=[0, 1])
#
# raw_data = df_stacked.values
# n_days = len(df.index)
# n_features = raw_data.shape[1]
#
# # [Time step, ticker_index, feature_index]
# # features: Close, High, Low, Open, Volume
# data_matrix = raw_data.reshape(n_days, N_tickers, n_features)   # shape: 250*1*5
# print(f"Data matrix shape: {data_matrix.shape}")
#
# env = StockTradingEnv(df_matrix=data_matrix)
# # actor = Actor()
# # critic = Critic()
# # ppo = PPO(actor, critic, NUM_BATCH)

# new_obs, info = env.reset()
#
# for epoch in range(MAX_EPOCH):
#     for batch in range(NUM_BATCH):
#         curr_obs = new_obs
#         action = ppo.act(curr_obs)
#         new_obs, reward, truncated, terminated, info = env.step(action)
#         ppo.step(obs=curr_obs, reward=reward, truncated=truncated, terminated=terminated)
#     ppo.update(new_obs)
#
