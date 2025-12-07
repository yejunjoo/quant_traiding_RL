import gymnasium as gym
from gymnasium import spaces
import numpy as np


class Environment(gym.Env):
    def __init__(self, start_date, df_matrix, max_balance=1e8):
        super(Environment, self).__init__()
        self.max_timestep, self.num_ticker, self.num_feat = df_matrix.shape
        assert self.num_ticker == 1, "For now, only for single ticker"


        self.start_date = start_date
        self.timestep = 0

        # df_matrix
        # [Time step, ticker_index, feature_index]
        # features: Close, High, Low, Open, Volume
        self.d_close = np.squeeze(df_matrix[:,:,0])
        self.d_high = np.squeeze(df_matrix[:,:,1])
        self.d_low = np.squeeze(df_matrix[:,:,2])
        self.d_open = np.squeeze(df_matrix[:,:,3])
        self.d_volume = np.squeeze(df_matrix[:,:,4])

        self.min_balance = np.min(self.d_close)
        self.max_balance = max_balance
        # todo: multi ticker system


        self.agent_obs_dim = 2
        # curr_balance
        # num_stock
        agent_min = np.array([0.0, 0], np.float32)
        agent_max = np.array([np.inf, np.inf], np.float32)
        assert len(agent_min) == self.agent_obs_dim
        assert len(agent_max) == self.agent_obs_dim

        self.market_obs_dim = 5
        market_min = np.array([0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
        market_max = np.array([np.inf, np.inf, np.inf, np.inf, np.inf], np.float32)
        assert len(market_min) == self.market_obs_dim
        assert len(market_max) == self.market_obs_dim

        self.obs_dim = self.agent_obs_dim + self.market_obs_dim

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=agent_min,
                    high=agent_max,
                    shape=(self.agent_obs_dim,),
                    dtype=np.float32
                ),
                "market": spaces.Box(
                    low=market_min,
                    high=market_max,
                    shape=(self.market_obs_dim,),
                    dtype=np.float32
                )
            })

        self.action_dim = self.num_ticker
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.action_dim,),
            dtype=np.float32
        )


        self.curr_balance = -1.0
        self.num_stock = -1
        self.obs_dict = {}
        self.portfolio_value = -1.0
        self.reset()

    def step(self, action):
        terminated = False

        last_price = self.d_close[self.timestep]

        assert len(action) == 1, "For now, only for single ticker"
        n_trade = int(round(action[0]))

        curr_balance = self.curr_balance
        max_buy = int(curr_balance // last_price)
        max_sell = self.num_stock
        # normalize된 값 조심

        if n_trade >0:
            # buy
            actual_buy = min(max_buy, n_trade)
            self.curr_balance -= last_price*actual_buy
            self.num_stock += actual_buy
        elif n_trade<0:
            # sell
            n_sell = -n_trade
            actual_sell = min(max_sell, n_sell)
            self.curr_balance += last_price*actual_sell
            self.num_stock -= actual_sell
        else:
            # Do nothing
            pass

        self.timestep += 1
        terminated = terminated or (self.timestep >= self.max_timestep - 1)
        # 이거 끝내는 step 맞나? 맞는듯..?
        truncated = False

        if terminated:
            next_price = last_price
        else:
            next_price = self.d_close[self.timestep]

        new_portfolio_value = self.curr_balance + self.num_stock*next_price
        reward = new_portfolio_value - self.portfolio_value
        self.portfolio_value = new_portfolio_value

        self._get_obs()
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.curr_balance,
            'num_stock': self.num_stock
        }
        return self.obs_dict, reward, terminated, truncated, info

    def reset(self, seed=42, options=None):
        super().reset(seed=seed)

        self.timestep = 0
        self.curr_balance = self.np_random.uniform(low=self.min_balance,
                                                   high=self.max_balance)
        self.num_stock = 0

        self._get_obs()
        self.portfolio_value = self.curr_balance

        # todo: add curriculum or RSI #

        return self.obs_dict, {'portfolio_value': self.portfolio_value}

    def _get_obs(self):
        market_obs = np.array([
            self.d_close[self.timestep],
            self.d_high[self.timestep],
            self.d_low[self.timestep],
            self.d_open[self.timestep],
            self.d_volume[self.timestep]
        ], dtype=np.float32)

        agent_obs = np.array([
            self.curr_balance,
            self.num_stock
        ], dtype=np.float32)

        self.obs_dict = {"agent": agent_obs,
                         "market": market_obs}