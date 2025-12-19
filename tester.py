# tester.py

import yfinance as yf
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D
import numpy as np

from Environment import StockTradingEnv
from algo.ppo import Actor

MODEL_NAME = "20251219-231440_PPO_4"
EPOCH = "1000"

# 다른 종목 테스트하려면, N_tickers로 주지말고, 직접 ticker 리스트를 정해서 주기
# Tickers_candidate = ['AAPL', 'AMZN', 'F', 'GOOGL', 'JPM', 'META', 'MSFT', 'NVDA', 'TSLA', 'UBER', 'XOM']
# Tickers_candidate = ['AAPL', 'F', 'JPM', 'META', 'NVDA', 'TSLA', 'UBER', 'XOM']

# 리스트 줄 때 항상 순서 맞춰서 주기
Tickers_candidate = ['F', 'TSLA', 'UBER', "XOM"]
# Tickers_candidate = ['AAPL', 'AMZN', 'GOOGL']
# Tickers_candidate = ['AAPL', 'META', 'MSFT']
# Tickers_candidate = ['META', 'MSFT', 'NVDA']

START_DATE = "2024-01-01"
END_DATE = "2025-01-01"


N_tickers = len(Tickers_candidate)

MODEL_PATH = f"saved_models/{MODEL_NAME}/actor_epoch_{EPOCH}.pth"
STATS_PATH = f"saved_models/{MODEL_NAME}/obs_rms_epoch_{EPOCH}.pkl"

BANKRUPT_COEF = 0.0
TERMINATION_REWARD = -0.5
MAX_BALANCE = 1e4 *N_tickers
BALANCE_RAND = False
device = 'cuda'
MAX_TRADE = 50
# ==========================================

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

def make_env_for_test(data_matrix, balance_rand, bankrupt_coef, termination_reward, max_balance, max_trade, stats_path):
    env = StockTradingEnv(df_matrix=data_matrix,
                          balance_rand=balance_rand,
                          bankrupt_coef=bankrupt_coef,
                          termination_reward=termination_reward,
                          max_trade=max_trade,
                          max_balance=max_balance)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.NormalizeObservation(env)

    with open(stats_path, "rb") as f:
        loaded_obs_rms = pickle.load(f)
    env.obs_rms = loaded_obs_rms
    print(f"Loaded observation statistics from {stats_path}")
    env.training = False

    env = gym.wrappers.NormalizeReward(env)

    env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
    env = gym.wrappers.ClipAction(env)
    return env


def test():
    target_tickers = Tickers_candidate[0:N_tickers]
    data_matrix = shape_data_matrix(target_tickers, START_DATE, END_DATE)
    env = make_env_for_test(data_matrix=data_matrix,
                            balance_rand=BALANCE_RAND,
                            bankrupt_coef=BANKRUPT_COEF,
                            termination_reward=TERMINATION_REWARD,
                            max_balance=MAX_BALANCE,
                            max_trade=MAX_TRADE,
                            stats_path=STATS_PATH)
    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]

    actor = Actor(obs_dim=obs_shape, action_dim=action_shape).to(device)
    actor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Loaded pre-trained model from \n{MODEL_PATH}")
    actor.eval()

    raw_env = env.unwrapped
    obs, info = env.reset()
    done = False

    current_balances = [raw_env.portfolio_value]
    rewards = []
    policy_actions = []
    stock_prices_obs = []

    stock_prices_gt = data_matrix[:,:,0]


    print("Start Testing ...")

    while not done:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
            action_tensor = actor(obs_tensor)
            action = action_tensor.cpu().numpy()

        current_prices = []
        for i in range(N_tickers):
            p = raw_env.obs_dict['market'][i*5]
            current_prices.append(p)
        stock_prices_obs.append(current_prices)


        policy_actions.append(action[0] *MAX_TRADE)

        next_obs, reward, truncated, terminated, info = env.step(action.flatten())

        current_balances.append(raw_env.portfolio_value)
        rewards.append(raw_env.reward)

        obs = next_obs
        done = terminated or truncated

    current_balances = current_balances[:-1]
    # last value is blance that is reset
    print("Done Testing.")
    print(f"Prices: {len(stock_prices_obs)}, Actions: {len(policy_actions)}")
    print(f"Rewards: {len(rewards)}, Balances: {len(current_balances)}")

    stock_prices_obs = np.array(stock_prices_obs)
    policy_actions = np.array(policy_actions)


    steps = range(len(current_balances))
    stock_prices_gt = stock_prices_gt[:len(steps)]


    for i in range(N_tickers):
        ticker_name = target_tickers[i]
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        fig.canvas.manager.set_window_title(f"Analysis Report - {ticker_name}")

        # 1. Portfolio Balance
        ax1 = axes[0]
        total_return = (current_balances[-1] - current_balances[0]) / current_balances[0] * 100
        ax1.set_title(f"1. Portfolio Balance (Total Return: {total_return:.2f}%)", fontweight='bold')
        ax1.plot(steps, current_balances, color='tab:red', linewidth=2)
        ax1.set_ylabel('Balance (Won)')
        ax1.grid(True, alpha=0.3)

        # 2. Reward
        ax2 = axes[1]
        ax2.set_title("2. Step Reward", fontweight='bold')
        ax2.fill_between(steps, rewards, color='gray', alpha=0.5)
        ax2.plot(steps, rewards, color='black', linewidth=0.5, alpha=0.3)
        ax2.set_ylabel('Reward')
        ax2.grid(True, alpha=0.3)

        # 3. Stock Prices
        ax3 = axes[2]
        ax3.set_title(f"3. Stock Price ({ticker_name})", fontweight='bold')
        ax3.plot(steps, stock_prices_gt[:, i], color='black', linestyle='--', label='Ground Truth')
        ax3.plot(steps, stock_prices_obs[:,i], color='tab:blue', label='Observed')
        ax3.set_ylabel('Price')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)

        # 4. Policy Actions
        ax4 = axes[3]
        ax4.set_title(f"4. Agent Action Volume ({ticker_name})", fontweight='bold')
        actions_i = policy_actions[:, i]
        action_colors = ['green' if x >= 0 else 'red' for x in actions_i]
        ax4.bar(steps, actions_i, color=action_colors, width=1.0)
        ax4.axhline(0, color='black', linewidth=0.8)
        ax4.set_ylabel('Volume')
        ax4.set_xlabel('Steps')
        ax4.grid(True, alpha=0.3)
        legend_elements = [Line2D([0], [0], color='green', lw=4, label='Buy'),
                           Line2D([0], [0], color='red', lw=4, label='Sell')]
        ax4.legend(handles=legend_elements, loc='upper left')
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    test()