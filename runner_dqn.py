# runner.py
# for DQN

import os
import subprocess
import webbrowser
import time
import torch
import pickle
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import yfinance as yf
import gymnasium as gym
from Environment import StockTradingEnv
from algo.dqn import DQN



# Data
Start_date = "2019-01-01"
End_date = "2024-01-01"
# Tickers_candidate = ['AAPL', 'AMZN', 'F', 'GOOGL', 'JPM', 'META', 'MSFT', 'NVDA', 'TSLA', 'UBER', 'XOM']
# Tickers_candidate = ['AAPL', 'F', 'JPM', 'META', 'NVDA', 'TSLA', 'UBER', 'XOM']

# 항상 위에 주어진 순서대로 줘야함.
Tickers_candidate = ['AAPL']
N_tickers = len(Tickers_candidate)

# Learning
SAVE_EVERY_STEP = 5e5   # 1e6
MAX_STEP = int(SAVE_EVERY_STEP * 100)

# Env
Bankrupt_coef = 0.0
Termination_reward = -0.5
Max_balance = 1e4*N_tickers
Balance_rand = True    # if False, set to max balance
Max_trade = 50

run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_DQN_{N_tickers}"
log_dir = f"runs/{run_name}"
save_dir = f"saved_models/{run_name}"

writer = SummaryWriter(log_dir)
os.makedirs(save_dir, exist_ok=True)
# ==========================================

def tensorboard_launcher(directory_path, port=6006):
    cmd = ["python", "-m", "tensorboard", "--logdir", directory_path, "--port", str(port)]
    print(f"TensorBoard launching on http://localhost:{port}")
    process = subprocess.Popen(cmd)
    time.sleep(3)
    webbrowser.open(f"http://localhost:{port}/")
    return process

def make_env(data_matrix, balance_rand, bankrupt_coef, termination_reward, max_balance, max_trade):
    env = StockTradingEnv(df_matrix=data_matrix,
                          balance_rand=balance_rand,
                          bankrupt_coef=bankrupt_coef,
                          termination_reward=termination_reward,
                          max_balance=max_balance,
                          max_trade=max_trade)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
    env = gym.wrappers.ClipAction(env)

    return env

def shape_data_matrix(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)
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

data_matrix = shape_data_matrix(Tickers_candidate[0:N_tickers], start=Start_date, end=End_date)
env = make_env(data_matrix, Balance_rand, Bankrupt_coef, Termination_reward, Max_balance, Max_trade)
obs_shape = env.observation_space.shape
action_shape = env.action_space.shape
print(f"Obs shape\t: {obs_shape}")
print(f"Action shape\t: {action_shape}")
assert len(obs_shape) == 1
assert len(action_shape) == 1


dqn_agent = DQN(obs_dim=obs_shape[0],
                action_dim=21, # 21 discretized actions; unit: 5 stocks
                n_tickers=N_tickers,
                buffer_size=50000,
                batch_size=64)

new_obs, info = env.reset()
tensorboard_launcher("runs")

for global_step in range(MAX_STEP):

    curr_obs = new_obs
    action_continuous, action_idx = dqn_agent.act(curr_obs)

    new_obs, reward, truncated, terminated, info = env.step(action_continuous)

    if "episode" in info:
        epi_return = info['episode']['r']
        epi_length = info['episode']['l']
        if isinstance(epi_return, (np.ndarray, list)):
            epi_return = epi_return[0]
            epi_length = epi_length[0]

        print(f"Step={global_step}, Return={epi_return}, Epsilon={dqn_agent.epsilon:.3f}")
        writer.add_scalar("charts/episodic_return", epi_return, global_step)
        writer.add_scalar("charts/epsilon", dqn_agent.epsilon, global_step)

    dqn_agent.step(curr_obs, action_idx, reward, new_obs, truncated or terminated)

    loss = dqn_agent.update()
    writer.add_scalar("losses/q_loss", loss, global_step)

    if truncated or terminated:
        new_obs, _ = env.reset()

    if global_step % SAVE_EVERY_STEP == 0:
        torch.save(dqn_agent.q_net.state_dict(), os.path.join(save_dir, f"dqn_step_{global_step}.pth"))

        obs_rms = env.get_wrapper_attr('obs_rms')
        with open(os.path.join(save_dir, f"obs_rms_step_{global_step}.pkl"), "wb") as f:
            pickle.dump(obs_rms, f)

        print(f"Saved model at step {global_step}")

writer.close()