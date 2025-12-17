import yfinance as yf
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.lines import Line2D
import os

from Environment import StockTradingEnv
from algo.dqn import DQN  # [변경] PPO Actor 대신 DQN 임포트

# ==========================================
# [설정] 테스트 파라미터 (저장된 모델 경로 확인 필수)
# ==========================================
# 실행 전 runs/ 폴더 내의 실제 모델 폴더명으로 변경하세요.
MODEL_NAME = "StockTrading_PPO_20251217-XXXXXX"
STEP = "10000" # 저장된 스텝 번호 (예: 10000, 20000...)

# DQN은 step 단위로 저장됨
MODEL_PATH = f"saved_models/{MODEL_NAME}/dqn_step_{STEP}.pth"
STATS_PATH = f"saved_models/{MODEL_NAME}/obs_rms_step_{STEP}.pkl"

Tickers_candidate = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]
START_DATE = "2023-01-01"
END_DATE = "2024-01-01"
N_tickers = 1

BANKRUPT_COEF = 0.3
TERMINATION_REWARD = -1.0
MAX_BALANCE = 1e7
BALANCE_RAND = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_TRADE = 50
# ==========================================

def shape_data_matrix(tickers, start, end):
    print(f"📥 {tickers} 데이터 다운로드 중...")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)

    if len(tickers) == 1:
        raw_data = df.values
        n_days = raw_data.shape[0]
        n_features = raw_data.shape[1]
        data_matrix = raw_data.reshape(n_days, 1, n_features)
    else:
        df_stacked = df.stack(level=1, future_stack=True)
        df_stacked = df_stacked.sort_index(level=[0, 1])
        raw_data = df_stacked.values
        n_days = len(df.index)
        n_features = raw_data.shape[1]
        data_matrix = raw_data.reshape(n_days, len(tickers), n_features)

    print(f"✅ Data matrix shape: {data_matrix.shape}")
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

    # [중요] 학습 때 저장한 통계값(obs_rms) 불러오기
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            loaded_obs_rms = pickle.load(f)
        env.obs_rms = loaded_obs_rms
        print(f"✅ Loaded observation statistics from {stats_path}")
    else:
        print(f"⚠️ Warning: Stats file not found at {stats_path}. Running without stats load.")

    # 테스트 모드 설정 (통계 업데이트 중지)
    env.training = False

    env = gym.wrappers.NormalizeReward(env)

    # [중요] Runner와 동일하게 RescaleAction 적용 (-1~1 -> -50~50)
    env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
    env = gym.wrappers.ClipAction(env)
    return env


def test():
    # 1. 데이터 준비
    data_matrix = shape_data_matrix(Tickers_candidate[0:N_tickers], START_DATE, END_DATE)

    # 2. 환경 생성
    env = make_env_for_test(data_matrix=data_matrix,
                            balance_rand=BALANCE_RAND,
                            bankrupt_coef=BANKRUPT_COEF,
                            termination_reward=TERMINATION_REWARD,
                            max_balance=MAX_BALANCE,
                            max_trade=MAX_TRADE,
                            stats_path=STATS_PATH)

    obs_shape = env.observation_space.shape[0]

    # 3. DQN 에이전트 생성 및 로드
    # DQN 클래스 초기화 (Action Dim은 1이지만 내부적으로 Discrete 매핑)
    dqn_agent = DQN(obs_dim=obs_shape, action_dim=1)

    try:
        dqn_agent.q_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"✅ Loaded pre-trained model from \n{MODEL_PATH}")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return

    dqn_agent.q_net.eval() # 평가 모드 (Dropout 등 비활성화)

    # 4. 테스트 루프 준비
    raw_env = env.unwrapped
    obs, info = env.reset()
    done = False

    current_balances = [raw_env.curr_balance]
    rewards = []
    policy_actions = []
    stock_prices_obs = []

    stock_prices_gt = data_matrix[:,0,0] # Ground Truth Prices

    print("🚀 Start Testing ...")

    while not done:
        # [변경] DQN Action 결정 (Eval Mode=True로 Epsilon Greedy 끔)
        # dqn_agent.act returns (continuous_action, action_idx)
        action_continuous, action_idx = dqn_agent.act(obs, eval_mode=True)

        # 기록: 현재 주가
        current_price = raw_env.obs_dict['market'][0] if hasattr(raw_env, 'obs_dict') else 0
        stock_prices_obs.append(current_price)

        # 기록: 행동 (실제 물량으로 변환)
        # RescaleAction이 적용되어 있으므로 action_continuous(-1~1) * MAX_TRADE가 실제 물량과 비례함
        policy_actions.append(action_continuous[0] * MAX_TRADE)

        # 환경 진행 (연속값 전달)
        next_obs, reward, truncated, terminated, info = env.step(action_continuous)

        # 기록: 잔고 및 보상
        current_balances.append(raw_env.curr_balance)
        rewards.append(raw_env.reward)

        obs = next_obs
        done = terminated or truncated

    print("🏁 Done Testing.")

    # 그래프를 위한 데이터 정리
    # current_balances는 초기값이 있어 1개 더 많으므로 마지막 스텝 제외하거나 길이를 맞춤
    if len(current_balances) > len(rewards):
        current_balances = current_balances[:-1] # 길이를 맞춤

    print(f"Prices: {len(stock_prices_obs)}, Actions: {len(policy_actions)}")
    print(f"Rewards: {len(rewards)}, Balances: {len(current_balances)}")

    # =========================================================
    # 5. 시각화 (Visualization)
    # =========================================================
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    steps = range(len(current_balances))

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
    ax3.set_title("3. Stock Prices (Ground Truth vs Observed)", fontweight='bold')
    sliced_gt = stock_prices_gt[:len(steps)]
    ax3.plot(steps, sliced_gt, color='black', linestyle='--', label='Ground Truth')
    ax3.plot(steps, stock_prices_obs, color='tab:blue', label='Observed')
    ax3.set_ylabel('Price')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # 4. Policy Actions
    ax4 = axes[3]
    ax4.set_title("4. Agent Actions (Buy/Sell Volume)", fontweight='bold')

    # 매수(초록)/매도(빨강) 색상 지정
    action_colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in policy_actions]

    ax4.bar(steps, policy_actions, color=action_colors, width=1.0)
    ax4.axhline(0, color='black', linewidth=0.8) # 0 기준선
    ax4.set_ylabel('Volume')
    ax4.set_xlabel('Steps')
    ax4.grid(True, alpha=0.3)

    legend_elements = [Line2D([0], [0], color='green', lw=4, label='Buy'),
                       Line2D([0], [0], color='red', lw=4, label='Sell'),
                       Line2D([0], [0], color='gray', lw=4, label='Hold')]
    ax4.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test()