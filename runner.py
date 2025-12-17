# runner.py

import yfinance as yf
from Environment import StockTradingEnv
import gymnasium as gym
from algo.ppo import PPO, Actor, Critic
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import pickle

import os
import subprocess
import webbrowser
import time

# $ tensorboard --logdir=runs --port=6006 --bind_all

## todo: Add yaml file reading for hyperparameters ##

# Data
N_tickers = 1
Start_date = "2023-01-01"
End_date = "2024-01-01"
Tickers_candidate = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]

# Learning
MAX_EPOCH = int(1e8) # 백만
Rollout_storage = 10 # 10240
SAVE_EVERY_EPOCH = 1 # 500

# Env
Bankrupt_coef = 0.3
Termination_reward = -1.0
Max_balance = 1e7
Balance_rand = False    # if False, set to max balance
Max_trade = 50

run_name = f"StockTrading_PPO_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
log_dir = f"runs/{run_name}"
save_dir = f"saved_models/{run_name}"

writer = SummaryWriter(log_dir)
os.makedirs(save_dir, exist_ok=True)
# ==========================================

def tensorboard_launcher(directory_path, port=6006):
    """
    TensorBoard를 백그라운드에서 실행하고 웹 브라우저를 엽니다.
    """
    # 1. 텐서보드 실행 명령 (백그라운드 프로세스 Popen 사용)
    # 윈도우/리눅스 환경에 따라 tensorboard 실행 파일 경로 문제가 있을 수 있으므로
    # 'python -m tensorboard' 방식으로 실행하는 것이 가장 안전합니다.
    cmd = ["python", "-m", "tensorboard", "--logdir", directory_path, "--port", str(port)]

    print(f"🚀 TensorBoard launching on http://localhost:{port}")
    process = subprocess.Popen(cmd)

    # 2. 서버가 시작될 때까지 잠시 대기 (3초)
    time.sleep(3)

    # 3. 브라우저 자동 실행
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
    env = gym.wrappers.RescaleAction(env, min_action=(-1)*Max_trade, max_action=Max_trade)
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

actor = Actor(obs_dim=obs_shape[0], action_dim=action_shape[0]).to('cuda')
critic = Critic(obs_dim=obs_shape[0]).to('cuda')

ppo = PPO(actor=actor,
          critic=critic,
          rollout_storage=Rollout_storage,
          obs_shape=obs_shape[0],
          action_shape=action_shape[0])

new_obs, info = env.reset()
global_step = 0

tensorboard_launcher("runs")

for epoch in range(MAX_EPOCH):
    print(f"\n\n\t/// ------- EPOCH {epoch}  ------- ///")
    for batch in range(Rollout_storage):
        global_step += 1
        curr_obs = new_obs
        action = ppo.act(curr_obs)
        new_obs, reward, truncated, terminated, info = env.step(action)
        if "episode" in info:
            epi_return = info['episode']['r']
            epi_length = info['episode']['l']
            if isinstance(epi_return, (np.ndarray, list)):
                epi_return = epi_return[0]
                epi_length = epi_length[0]

            print(f"global_step={global_step}, episodic_return={epi_return}")
            writer.add_scalar("charts/episodic_return", epi_return, global_step)
            writer.add_scalar("charts/episodic_length", epi_length, global_step)


        ppo.step(obs=curr_obs, reward=reward, truncated=truncated, terminated=terminated)
    next_obs_tensor = torch.tensor(new_obs, dtype=torch.float32).to('cuda')
    loss = ppo.update(next_obs_tensor)

    writer.add_scalar("losses/total_loss", loss, global_step)

    if epoch % SAVE_EVERY_EPOCH == 0:
        actor_path = os.path.join(save_dir, f"actor_epoch_{epoch}.pth")
        critic_path = os.path.join(save_dir, f"critic_epoch_{epoch}.pth")

        torch.save(actor.state_dict(), actor_path)
        torch.save(critic.state_dict(), critic_path)


        obs_rms = env.get_wrapper_attr('obs_rms')
        with open(os.path.join(save_dir, f"obs_rms_epoch_{epoch}.pkl"), "wb") as f:
            pickle.dump(obs_rms, f)

        print(f"Saved model checkpoint at epoch {epoch}")
writer.close()