# Environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class StockTradingEnv(gym.Env):
    def __init__(self,
                 df_matrix,
                 balance_rand,
                 max_trade,
                 balance_unit=500*1e4,
                 bankrupt_coef=0.3,
                 termination_reward=-1e4,
                 max_balance=1e7):
        super().__init__()


        self.max_timestep, self.num_ticker, self.num_feat = df_matrix.shape
        # assert self.num_ticker == 1, "For now, only for single ticker"
        self.bankrupt_coef = bankrupt_coef
        self.termination_reward = termination_reward
        self.balance_rand = balance_rand
        self.balance_unit = balance_unit
        self.max_trade = max_trade

        self.timestep = 0
        self.reward = 0

        # df_matrix
        # [Time step, ticker_index, feature_index]
        # features: Close, High, Low, Open, Volume
        self.d_close = df_matrix[:,:,0]
        self.d_high = df_matrix[:,:,1]
        self.d_low = df_matrix[:,:,2]
        self.d_open = df_matrix[:,:,3]
        self.d_volume = df_matrix[:,:,4]

        self.min_balance = np.ceil(np.min(self.d_close)/self.balance_unit) *self.balance_unit
        self.max_balance = max_balance
        # todo: multi ticker system


        self.agent_obs_dim = 1 + self.num_ticker
        # curr_balance + num_stock(per ticker)
        agent_min = np.zeros(self.agent_obs_dim, np.float32)
        agent_max = np.full_like(agent_min, np.inf)
        assert len(agent_min) == self.agent_obs_dim
        assert len(agent_max) == self.agent_obs_dim

        self.market_obs_dim = 5 * self.num_ticker
        market_min = np.zeros(self.market_obs_dim, np.float32)
        market_max = np.full_like(market_min, np.inf)
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

        action_min = np.full((self.action_dim,), (-1)*self.max_trade, dtype=np.float32)
        action_max = np.full((self.action_dim,), self.max_trade, dtype=np.float32)

        self.action_space = spaces.Box(
            low=action_min,
            high=action_max,
            shape=(self.action_dim,),
            dtype=np.float32
        )


        self.curr_balance = -1.0
        self.init_balance = -1.0
        self.num_stocks = np.zeros(self.num_ticker, dtype=int)
        self.obs_dict = {}
        self.portfolio_value = -1.0
        self.reset()



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.timestep = 0
        self.reward = 0
        if self.balance_rand:
            # self.curr_balance = self.np_random.uniform(low=self.min_balance,
            #                                            high=self.max_balance)

            num_steps = int((self.max_balance - self.min_balance) / self.balance_unit)
            random_step = self.np_random.integers(0, num_steps + 1)
            self.curr_balance = self.min_balance + (random_step * self.balance_unit)
        else:
            self.curr_balance = self.max_balance

        self.init_balance = self.curr_balance
        self.num_stocks = np.zeros(self.num_ticker, dtype=int)

        self._get_obs()
        self.portfolio_value = self.curr_balance

        # todo: add curriculum or RSI #
        # random for curr balance and even random num stock

        return self.obs_dict, { 'portfolio_value': self.portfolio_value,
                                'balance': self.curr_balance,
                                'num_stock': self.num_stocks }




    def step(self, action):
        terminated = False

        # [Time step, Ticker index]
        last_prices = self.d_close[self.timestep]

        assert len(action) == self.num_ticker, "Ticker num difference!"
        n_trades = np.round(action).astype(int)

        curr_balance = self.curr_balance
        max_buys = (curr_balance // last_prices).astype(int)
        max_sells = self.num_stocks
        # normalize된 값 조심

        print(f"\nTimestep: {self.timestep}")
        print(f"Balance prev\t: {self.curr_balance//10000}")
        print(f"Last prices\t: {last_prices}")

        for stock_idx in range(self.num_ticker):
            print(f"For Stock idx: {stock_idx}")
            n_trade_per_stock = n_trades[stock_idx]
            max_buy_per_stock = max_buys[stock_idx]
            max_sell_per_stock = max_sells[stock_idx]
            last_price_per_stock = last_prices[stock_idx]

            if n_trade_per_stock >0:
                # buy
                actual_buy = min(max_buy_per_stock, n_trade_per_stock)
                self.curr_balance -= last_price_per_stock *actual_buy
                self.num_stocks[stock_idx] += actual_buy
                # print(f"Buy\t\t: {actual_buy}")

            elif n_trade_per_stock<0:
                # sell
                n_sell = -n_trade_per_stock
                actual_sell = min(max_sell_per_stock, n_sell)
                self.curr_balance += last_price_per_stock*actual_sell
                self.num_stocks[stock_idx] -= actual_sell
                # print(f"Sell\t\t: {actual_sell}")

            else:
                # Do nothing
                pass
        # print(f"Balance after\t: {self.curr_balance}")


        last_day_idx = self.max_timestep -1
        truncated = (self.timestep >= last_day_idx -1)  # tmw is last day
        terminated = (self.curr_balance < self.init_balance * self.bankrupt_coef)

        self.reward = self.compute_reward()
        print(f"Reward\t: {self.reward}")

        if terminated:
            self.reward = self.termination_reward
            self.reset()
        elif truncated:
            self.reset()
        else:
            self.timestep += 1
            self._get_obs()

        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.curr_balance,
            'num_stocks': self.num_stocks
        }

        return self.obs_dict, self.reward, truncated, terminated, info



    def _get_obs(self):
        agent_obs = np.concatenate([
            np.array([self.curr_balance]),
            self.num_stocks.flatten()
        ]).astype(np.float32)

        market_obs = np.array([
            self.d_close[self.timestep, :],
            self.d_high[self.timestep, :],
            self.d_low[self.timestep, :],
            self.d_open[self.timestep, :],
            self.d_volume[self.timestep, :]
        ], dtype=np.float32)

        self.obs_dict = {"agent": agent_obs,
                         "market": market_obs}

        return self.obs_dict

    def compute_reward(self):
        next_prices = self.d_close[self.timestep+1]
        new_portfolio_value = self.curr_balance + np.sum(self.num_stocks *next_prices)
        reward = (new_portfolio_value - self.portfolio_value)/self.init_balance
        self.portfolio_value = new_portfolio_value
        return reward