# Environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class StockTradingEnv(gym.Env):
    def __init__(self,
                 df_matrix,
                 ticker_info_list,
                 balance_rand,
                 max_trade,
                 window_size,
                 fee_rate,
                 balance_unit=2500,
                 bankrupt_coef=0.3,
                 termination_reward=-1e4,
                 max_balance=1e4
                 ):
        super().__init__()



        self.df_matrix = df_matrix
        self.max_timestep, self.num_ticker, self.num_feat = df_matrix.shape
        self.bankrupt_coef = bankrupt_coef
        self.termination_reward = termination_reward
        self.balance_rand = balance_rand
        self.balance_unit = balance_unit
        self.max_trade = max_trade
        self.window_size = window_size
        self.fee_rate = fee_rate

        self.ticker_extra_features = self._process_extra_features(ticker_info_list)


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

        self.rsi_values = self._calculate_rsi(window=14)

        self.min_balance = np.ceil(np.min(self.d_close)/self.balance_unit) *self.balance_unit
        self.max_balance = max_balance
        # todo: multi ticker system

        # 잔고, 보유수, 전고점대비, 애널리스트 목표가대비
        self.agent_obs_dim = 1 + (3 *self.num_ticker)

        # curr_balance + num_stock(per ticker)
        # agent_min = np.zeros(self.agent_obs_dim, np.float32)
        # agent_max = np.full_like(agent_min, np.inf)
        # assert len(agent_min) == self.agent_obs_dim
        # assert len(agent_max) == self.agent_obs_dim

        # Market Obs: (기본 5개 + RSI 1개 + 추세 1개 + 고정지표 2개) * window_size * ticker
        self.per_ticker_feat = 7 + 2
        self.market_obs_dim = self.per_ticker_feat * self.num_ticker *self.window_size
        # market_min = np.zeros(self.market_obs_dim, np.float32)
        # market_max = np.full_like(market_min, np.inf)
        # assert len(market_min) == self.market_obs_dim
        # assert len(market_max) == self.market_obs_dim

        self.obs_dim = self.agent_obs_dim + self.market_obs_dim

        # [Time step, ticker_index, feature_index]
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.agent_obs_dim,),
                    dtype=np.float32
                ),
                "market": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
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

        self.prev_action = np.zeros(self.num_ticker)
        self.actual_action = np.zeros(self.num_ticker)
        self.action_penalty_coef = 1e-4
        # self.log_return_coef = 10.0

        self.max_prices_seen = np.zeros(self.num_ticker)

        self.overflow = False

        self.curr_balance = -1.0
        self.init_balance = -1.0
        self.num_stocks = np.zeros(self.num_ticker, dtype=int)
        self.obs_dict = {}
        self.portfolio_value = -1.0
        self.reset()


    def _process_extra_features(self, info_list):
        features = []
        for info in info_list:
            target = info.get('targetMeanPrice', 0)
            pe = info.get('forwardPE', 0) if info.get('forwardPE') else 0
            div = info.get('dividendYield', 0) if info.get('dividendYield') else 0
            features.append({
                'target': target,
                'pe': pe,
                'div': div
            })
        return features

    def _calculate_rsi(self, window=14):
        diff = np.diff(self.d_close, axis=0)
        gain = np.where(diff > 0, diff, 0)
        loss = np.where(diff < 0, -diff, 0)
        rsi_matrix = np.zeros_like(self.d_close)
        for i in range(self.num_ticker):
            avg_gain = np.convolve(gain[:, i], np.ones(window)/window, mode='valid')
            avg_loss = np.convolve(loss[:, i], np.ones(window)/window, mode='valid')
            rs = avg_gain / (avg_loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            rsi_matrix[window:, i] = rsi
            rsi_matrix[:window, i] = 50.0
        return rsi_matrix

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.overflow = False
        self.prev_action = np.zeros(self.num_ticker)

        self.timestep = self.window_size - 1
        self.max_prices_seen = self.d_close[self.timestep].copy()
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
        self.overflow = False
        self.max_prices_seen = np.maximum(self.max_prices_seen, self.d_close[self.timestep])

        prev_portfolio_value = self.curr_balance + np.sum(self.num_stocks * self.d_close[self.timestep])
        total_fee_paid = 0

        # [Time step, Ticker index]
        last_prices = self.d_close[self.timestep]

        assert len(action) == self.num_ticker, "Ticker num difference!"
        n_trades = np.round(action).astype(int)

        print(f"\nTimestep: {self.timestep}")
        print(f"Balance prev\t: {self.curr_balance//10000}")
        print(f"Last prices\t: {last_prices}")

        for stock_idx in range(self.num_ticker):
            n_trade_per_stock = n_trades[stock_idx]
            last_price_per_stock = last_prices[stock_idx]
            price_with_fee = last_price_per_stock  *(1 + self.fee_rate)
            max_buy = (self.curr_balance // price_with_fee).astype(int)
            max_sell = self.num_stocks[stock_idx]

            print(f"For Stock idx: {stock_idx}")
            if n_trade_per_stock >0:
                # buy
                if n_trade_per_stock > max_buy:
                    self.overflow = True
                    actual_buy = max_buy
                else:
                    actual_buy = n_trade_per_stock
                self.curr_balance -= price_with_fee *actual_buy
                self.num_stocks[stock_idx] += actual_buy
                # print(f"Buy\t\t: {actual_buy}")
                self.actual_action[stock_idx] = actual_buy

                total_fee_paid += actual_buy * (last_price_per_stock * self.fee_rate)

            elif n_trade_per_stock<0:
                # sell
                n_sell = -n_trade_per_stock
                if n_sell > max_sell:
                    self.overflow = True
                    actual_sell = max_sell
                else:
                    actual_sell = n_sell
                self.curr_balance += last_price_per_stock *(1 - self.fee_rate) *actual_sell
                self.num_stocks[stock_idx] -= actual_sell
                # print(f"Sell\t\t: {actual_sell}")
                self.actual_action[stock_idx] = -actual_sell

                total_fee_paid += actual_sell * (last_price_per_stock * self.fee_rate)

            else:
                # Do nothing
                self.actual_action[stock_idx] = 0
                pass
        # print(f"Balance after\t: {self.curr_balance}")


        last_day_idx = self.max_timestep -1
        truncated = (self.timestep >= last_day_idx -1)  # tmw is last day


        self.reward = self.compute_reward(self.actual_action)
        print(f"Reward\t: {self.reward}")

        # new termination condition
        terminated = (self.portfolio_value < self.init_balance * self.bankrupt_coef)

        if terminated:
            self.reward = self.termination_reward

        self.timestep += 1
        self._get_obs()

        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.curr_balance,
            'num_stocks': self.num_stocks
        }

        return self.obs_dict, self.reward, truncated, terminated, info



    def _get_obs(self):
        curr_prices = self.d_close[self.timestep]
        upside_ratios = []
        for i in range(self.num_ticker):
            target = self.ticker_extra_features[i]['target']
            ratio = target / (curr_prices[i] + 1e-9) if target > 0 else 1.0
            upside_ratios.append(ratio)

        agent_obs = np.concatenate([
            [self.curr_balance / self.init_balance],
            self.num_stocks.flatten(),
            (curr_prices / (self.max_prices_seen + 1e-9)).flatten(),
            np.array(upside_ratios)
        ]).astype(np.float32)


        market_data_list = []
        for t in range(self.timestep - self.window_size + 1, self.timestep + 1):
            base_feat = self.df_matrix[t] # [Ticker, 5]
            rsi_feat = self.rsi_values[t].reshape(-1, 1)

            extra_vals = []
            for i in range(self.num_ticker):
                extra_vals.append([
                    self.ticker_extra_features[i]['pe'],
                    self.ticker_extra_features[i]['div']
                ])

            combined = np.hstack([base_feat, rsi_feat, np.zeros((self.num_ticker, 1)), np.array(extra_vals)])
            market_data_list.append(combined.flatten())

        market_obs = np.concatenate(market_data_list).astype(np.float32)
        self.obs_dict = {"agent": agent_obs, "market": market_obs}
        return self.obs_dict

    def compute_reward(self, action):
        action_diff = np.abs(action - self.prev_action)
        action_penalty = np.sum(action_diff) / self.max_trade * self.action_penalty_coef
        self.prev_action = action.copy()

        next_prices = self.d_close[self.timestep+1]
        new_portfolio_value = self.curr_balance + np.sum(self.num_stocks *next_prices)
        log_return = np.log(new_portfolio_value + 1e-9) - np.log(self.portfolio_value + 1e-9)
        # reward = self.log_return_coef *log_return - action_penalty
        reward = log_return - action_penalty
        # reward = (new_portfolio_value - self.portfolio_value)/self.init_balance - action_penalty
        self.portfolio_value = new_portfolio_value
        return reward