import gymnasium as gym
from gymnasium import spaces
import yfinance as yF

class Environment(gym.Env):
    def __init__(self, balance:float,
                 start_date:str = "2020-10-19",
                 end_date:str = "2024-10-25",
                 code:str = "AAPL"):
    # Define your state space and action space.
    # The folloiwng is an example:
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=np.array([0, 0, 0], dtype=np.float32),
                    high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
                    shape=(3,),
                    dtype=float
                ),
                "market": spaces.Box(
                    low=data.min().to_numpy(dtype=np.float32),
                    high=data.max().to_numpy(dtype=np.float32),
                    shape=(len(self.column_names),),
                    dtype=float
                )
            })

        # We have 3 actions, corresponding to "sell", "buy", "hold".
        self.action_space = spaces.Discrete(3)

        start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
        end_date = dt.datetime.strptime(end_date, "%Y-%m-%d")

        ticker = yF.Ticker(code)
        df = ticker.history(interval='1d',
                            start=start_date.strftime("%Y-%m-%d"),
                            end=end_date.strftime("%Y-%m-%d"), auto_adjust=False)
    def step(self, action):
        # implemnt your logic
    def reset(self):
        # implement your logic