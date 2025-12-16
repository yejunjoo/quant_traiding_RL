import yfinance as yf
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# 기존에 작성했던 모듈들 import
from Environment import StockTradingEnv
from algo.ppo import Actor # Actor 클래스 정의가 필요합니다

# ==========================================
# [설정] 테스트 파라미터
# ==========================================
MODEL_PATH = "saved_models/StockTrading_PPO_20251217-023317/actor_epoch_590.pth" # <- 실제 저장된 경로로 수정 필수!
TICKER = ["AAPL"] # 학습 때와 동일한 종목
START_DATE = "2024-01-01"
END_DATE = "2025-01-01" # 현재 시점 (Context 기준)

# 환경 파라미터 (학습 때와 동일하게)
BANKRUPT_COEF = 0.3
TERMINATION_REWARD = -1e4
MAX_BALANCE = 1e7
# ==========================================

def shape_data_matrix(tickers, start, end):
    print(f"📥 {tickers} 데이터 다운로드 중 ({start} ~ {end})...")
    # auto_adjust=True는 종가(Close)가 수정주가로 반영됨
    df = yf.download(tickers, start=start, end=end, auto_adjust=True)

    # [중요] 단일 종목일 경우 MultiIndex가 아닐 수 있음 -> 강제 변환 필요 가능성 확인
    # yfinance 버전에 따라 다르지만, 보통 단일 종목은 stack이 필요 없음
    if len(tickers) == 1:
        # 단일 종목: (Days, Features) -> (Days, 1, Features)로 변환
        raw_data = df.values
        # Feature 순서 파악 (나중에 Close 가격 찾기 위해)
        feature_columns = list(df.columns)

        n_days = raw_data.shape[0]
        n_features = raw_data.shape[1]

        # (Days, 1, Features) 형태로 Reshape
        data_matrix = raw_data.reshape(n_days, 1, n_features)

    else:
        # 다중 종목: 기존 로직 유지
        # columns가 (Ticker, Feature) 형태인지 확인 필요
        df_stacked = df.stack(level=1, future_stack=True)
        df_stacked = df_stacked.sort_index(level=[0, 1])
        raw_data = df_stacked.values
        feature_columns = list(df_stacked.columns) # 정확하지 않을 수 있음 (구조에 따라 다름)

        n_days = len(df.index)
        n_features = raw_data.shape[1]
        data_matrix = raw_data.reshape(n_days, len(tickers), n_features)

    print(f"✅ 데이터 준비 완료. Shape: {data_matrix.shape}")

    # 'Close' 컬럼이 몇 번째 인덱스인지 찾기
    close_index = 0
    # 보통 yfinance 컬럼은 알파벳 순: Close, High, Low, Open, Volume
    # auto_adjust=True면 Adj Close는 없음.
    # 대소문자 구분 없이 'Close'가 포함된 컬럼 찾기
    for i, col in enumerate(df.columns):
        if "Close" in str(col):
            close_index = i
            break

    print(f"ℹ️ Close Price Index: {close_index} (Column: {df.columns[close_index]})")

    return data_matrix, df.index, close_index

def make_env_for_test(data_matrix):
    """
    학습 때와 '똑같은' 전처리 과정을 거치는 환경 생성
    """
    env = StockTradingEnv(df_matrix=data_matrix,
                          bankrupt_coef=BANKRUPT_COEF,
                          termination_reward=TERMINATION_REWARD,
                          max_balance=MAX_BALANCE)

    # Wrapper도 학습 때와 동일하게 씌워줘야 신경망이 입력을 이해함
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.NormalizeObservation(env) # 주의: 테스트 시에는 통계치가 초기화된 상태로 시작함
    env = gym.wrappers.NormalizeReward(env)      # 테스트 시 보상 정규화는 결과 확인용으로만 동작
    env = gym.wrappers.ClipAction(env)

    return env
def test():
    # 1. 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")

    # 2. 데이터 준비 (close_index 받아오기 추가)
    data_matrix, dates, close_idx = shape_data_matrix(TICKER, START_DATE, END_DATE)

    # 3. 환경 생성
    env = make_env_for_test(data_matrix)
    obs_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]

    # 4. 모델 로드
    print(f"📂 모델 로딩 중: {MODEL_PATH}")
    actor = Actor(obs_dim=obs_shape, action_dim=action_shape).to(device)

    try:
        actor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ 모델 가중치 로드 성공!")
    except FileNotFoundError:
        print("❌ 모델 파일을 찾을 수 없습니다. MODEL_PATH를 확인해주세요.")
        return

    # 5. 평가 모드
    actor.eval()

    # 6. 테스트 루프
    obs, info = env.reset()
    done = False

    portfolio_values = []
    rewards = []
    stock_prices = []
    actions_history = []

    print("🚀 백테스팅 시작...")

    while not done:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            action_tensor = actor(obs_tensor)
            action = action_tensor.cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- [수정된 부분] 데이터 수집 로직 ---
        raw_env = env.unwrapped

        # (1) 자산 가치
        if hasattr(raw_env, 'portfolio_value'):
            portfolio_values.append(raw_env.portfolio_value[0] if isinstance(raw_env.portfolio_value, list) else raw_env.portfolio_value)
        else:
            # 기본 자산 계산 (Environment 구현에 따라 다를 수 있음)
            # raw_env.state[0]이 잔고(balance)라고 가정하는 경우가 많음
            portfolio_values.append(MAX_BALANCE)

            # (2) 주가 (정확한 인덱싱)
        try:
            current_step = raw_env.timestep
            # raw_env.data shape: [Days, Tickers, Features]
            # 우리가 필요한 것: [Current Day, 0번째 Ticker, Close Feature]
            if current_step < len(raw_env.data):
                # [수정] 피쳐 벡터 전체([0])가 아니라, 그 안의 close_idx를 가져옴
                price = raw_env.data[current_step][0][close_idx]
                stock_prices.append(float(price))
            else:
                stock_prices.append(stock_prices[-1])
        except Exception as e:
            # 디버깅을 위해 에러 출력
            if len(stock_prices) == 0: print(f"Price Error: {e}")
            stock_prices.append(0)

        rewards.append(reward)
        actions_history.append(action[0])

    print("🏁 백테스팅 종료.")

    # (이하 시각화 코드는 동일)
    # ...
    if len(portfolio_values) > 0:
        initial_value = MAX_BALANCE
        final_value = portfolio_values[-1]
        profit_pct = ((final_value - initial_value) / initial_value) * 100

        print(f"💰 초기 자산: {initial_value:,.0f}")
        print(f"💰 최종 자산: {final_value:,.0f}")
        print(f"📈 수익률: {profit_pct:.2f}%")

        plt.figure(figsize=(15, 12))

        plt.subplot(4, 1, 1)
        plt.plot(portfolio_values, label='My Portfolio Value', color='red', linewidth=2)
        plt.axhline(y=initial_value, color='gray', linestyle='--', label='Initial Balance')
        plt.title(f'1. Portfolio Performance (Profit: {profit_pct:.2f}%)')
        plt.ylabel('Value (Won/Dollar)')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(stock_prices, label=f'{TICKER[0]} Price', color='blue')
        plt.title(f'2. Stock Price Movement ({TICKER[0]})')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(rewards, label='Step Reward', color='purple', alpha=0.7)
        plt.title('3. Rewards per Step')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 4)
        plt.bar(range(len(actions_history)), actions_history, color='green', label='Action (Buy/Sell)', width=1.0)
        plt.title('4. Agent Actions')
        plt.ylabel('Strength')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ 데이터가 기록되지 않았습니다.")

if __name__ == "__main__":
    test()