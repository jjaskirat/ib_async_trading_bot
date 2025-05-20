"""
---------------------------------------------------------------------------------------------------
Check out: https://github.com/opendilab/DI-engine/blob/main/dizoo/gym_anytrading/envs/README.md
---------------------------------------------------------------------------------------------------
This is a slightly modified version on DI-engine AnyTrading

DI-engine AnyTrading

AnyTrading is a collection of OpenAI Gym environments for reinforcement learning-based trading algorithms.

Based on original gym-anytrading environment (you can see that at https://github.com/AminHP/gym-anytrading), there are lots of modifications done to improve the original environment.

In our environment, TradingEnv is an abstract environment which is defined to support all kinds of trading environments. StocksEnv, inheriting and extending TradingEnv, backtests the trading data of Google stock from 2009 to 2018.
Environment Properties

The original design of gym-anytrading is quite simple, which aims at making the agent learn in a faster and more efficient way. However, we find that many defects of the original environment make it difficult to train agents, and the incomplete original environment is difficult to describe the real trading environment. Therefore, lots of modifications have been done. In the several following subsections, I will explain why these modifications are meaningful.
State Machine

We use a state machine to describe how the TradingEnv interact with agent as well as how an agent make profits.

As shown below, the state machine use three kinds of trading positions and five (action "Hold" does not shown) kinds of trading actions to describe how the transaction goes over time.

state machine
Trading Positions

Short: If the current env is in Short state, it means that the agent borrowed stocks from the securities companies.

Flat: If the current env is in Flat state, it means that the agent does not hold shares.

Long: If the current env is in Long state, it means that the agent has changed all the funds into stocks.
Trading Actions

Double_Sell: means agent want sell all the stocks it holds as well as the stocks it borrows from securities companies.

Sell: means sell the stocks agent holds.

Hold: maintain current status.

Buy: means buy the stocks at current close price.

Double_Buy: means return shares to securities companies and exchange all the funds on hand for stocks at current close price.
How did the profit and loss happen

If profit or loss occurs, it means that one of the following two cycles in state machine has occurred.

    buying long
        Flat -> Long -> Flat
    short selling
        Flat -> Short -> Flat

Current Profit Calculation

According to the above definition, we can easily know that the formula of accumulative profit is:

∏ b u y i n g   l o n g ( r c u r r / r p r e   ∗   c o s t ) ∗ ∏ s h o r t   s e l l i n g ( ( 2 − r c u r r / r p r e )   ∗   c o s t )
Reward Function

Comparing the objective function ( E τ ∑   r ) in reinforcement learning and the formula of profit, we can get that the reward function is:

    buying long:
        l o g ( c l o s e c u r r / c l o s e p r e ) + l o g ( c o s t )
    short selling:
        l o g ( 2 − c l o s e c u r r / c l o s e p r e ) + l o g ( c o s t )
    otherwise:
        0

so that maximize $\mathbb{E}{\tau} \sum r$ is equivalent to maximize $\mathbb{E}{\tau}[\prod_{buying\ long}(r_{curr}/r_{pre}\ *\ cost) + \prod_{short\ selling}((2-r_{curr}/r_{pre})\ *\ cost)]$

The experimental results show that such a definition is better than the original gym-anytrading accumulated reward function :$\sum(r_{curr} - r_{pre})$.
Render Function

As you see, you can use render method to plot the position and profit at one episode.

    The position figure:
        The x-axis of the position figure is trading days. In this case, it is 252 trading days.
        The y-axis of the position figure is the closing price of each day.
        Besides, the red inverted triangle, the green positive triangle and the blue circle represent the position of the agent every trading day respectively.

position

    The profit figure:
        Similarly, The x-axis of the profit figure is trading days. In this case, it is 252 trading days. (a pair of pictures keep the same time interval)
        The y-axis of the profit figure is the profit of each day. 1.5 means the rate of return is 150%.

profit
"""


from abc import abstractmethod
from cmath import inf
from enum import Enum
from typing import Any, List

import numpy as np
import numbers
import pandas as pd
import torch
# from ding.torch_utils import to_ndarray
# from ding.utils import ENV_REGISTRY
# from easydict import EasyDict
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class Actions(int, Enum):
    DOUBLE_SELL = 0
    SELL = 1
    HOLD = 2
    BUY = 3
    DOUBLE_BUY = 4


class Positions(int, Enum):
    SHORT = -1.
    FLAT = 0.
    LONG = 1.
    
def to_ndarray(item: Any, dtype: np.dtype = None) -> Any:
    """
    Overview:
        Convert ``torch.Tensor`` to ``numpy.ndarray``.
    Arguments:
        - item (:obj:`Any`): The ``torch.Tensor`` objects to be converted. It can be exactly a ``torch.Tensor`` \
            object or a container (list, tuple or dict) that contains several ``torch.Tensor`` objects.
        - dtype (:obj:`np.dtype`): The type of wanted array. If set to ``None``, its dtype will be unchanged.
    Returns:
        - item (:obj:`object`): The changed arrays.

    Examples (ndarray):
        >>> t = torch.randn(3, 5)
        >>> tarray1 = to_ndarray(t)
        >>> assert tarray1.shape == (3, 5)
        >>> assert isinstance(tarray1, np.ndarray)

    Examples (list):
        >>> t = [torch.randn(5, ) for i in range(3)]
        >>> tarray1 = to_ndarray(t, np.float32)
        >>> assert isinstance(tarray1, list)
        >>> assert tarray1[0].shape == (5, )
        >>> assert isinstance(tarray1[0], np.ndarray)

    .. note:

        Now supports item type: :obj:`torch.Tensor`,  :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`.
    """

    def transform(d):
        if dtype is None:
            return np.array(d)
        else:
            return np.array(d, dtype=dtype)

    if isinstance(item, dict):
        new_data = {}
        for k, v in item.items():
            new_data[k] = to_ndarray(v, dtype)
        return new_data
    elif isinstance(item, list) or isinstance(item, tuple):
        if len(item) == 0:
            return None
        elif isinstance(item[0], numbers.Integral) or isinstance(item[0], numbers.Real):
            return transform(item)
        elif hasattr(item, '_fields'):  # namedtuple
            return type(item)(*[to_ndarray(t, dtype) for t in item])
        else:
            new_data = []
            for t in item:
                new_data.append(to_ndarray(t, dtype))
            return new_data
    elif isinstance(item, torch.Tensor):
        if dtype is None:
            return item.numpy()
        else:
            return item.numpy().astype(dtype)
    elif isinstance(item, np.ndarray):
        if dtype is None:
            return item
        else:
            return item.astype(dtype)
    elif isinstance(item, bool) or isinstance(item, str):
        return item
    elif np.isscalar(item):
        if dtype is None:
            return np.array(item)
        else:
            return np.array(item, dtype=dtype)
    elif item is None:
        return None
    else:
        raise TypeError("not support item type: {}".format(type(item)))


def transform(position: Positions, action: int) -> Any:
    '''
    Overview:
        used by env.tep().
        This func is used to transform the env's position from
        the input (position, action) pair according to the status machine.
    Arguments:
        - position(Positions) : Long, Short or Flat
        - action(int) : Doulbe_Sell, Sell, Hold, Buy, Double_Buy
    Returns:
        - next_position(Positions) : the position after transformation.
    '''
    if action == Actions.SELL:

        if position == Positions.LONG:
            return Positions.FLAT, False

        if position == Positions.FLAT:
            return Positions.SHORT, True

    if action == Actions.BUY:

        if position == Positions.SHORT:
            return Positions.FLAT, False

        if position == Positions.FLAT:
            return Positions.LONG, True

    if action == Actions.DOUBLE_SELL and (position == Positions.LONG or position == Positions.FLAT):
        return Positions.SHORT, True

    if action == Actions.DOUBLE_BUY and (position == Positions.SHORT or position == Positions.FLAT):
        return Positions.LONG, True

    return position, False


# @ENV_REGISTRY.register('base_trading_1')
class TradingEnv(gym.Env):

    def __init__(
        self,
        df: pd.core.frame.DataFrame,
        target_name: str = 'wavelet_transformed_close',
        all_feature_name: List = ['wavelet_transformed_close', 'high', 'low', 'close', 'volume'],
        window_size: int = 64,
        plot_freq: int = 10,
        save_path: str = './',
        **kwargs
        ) -> None:

        # self._cfg = cfg
        self._env_id = '0' # TODO: Improve
        #======== param to plot =========
        self.cnt = 0

        self.plot_freq = plot_freq
        self.save_path = save_path
        #================================

        # self.train_range = cfg.train_range
        # self.test_range = cfg.test_range
        self.window_size = window_size
        self.df = df
        self.target_name = target_name
        self.eps_length = None
        self.prices = None
        self.target_prices = None
        self.signal_features = None
        self.all_feature_name = all_feature_name
        self.feature_dim_len = len(all_feature_name)
        # self.shape = (window_size * self.feature_dim_len + 2, )

        #======== param about episode =========
        self._start_tick = 0
        self._end_tick = 0
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        #======================================

        self._init_flag = True
        # init the following variables variable at first reset.
        self._action_space = None
        self._observation_space = None
        self._reward_space = None
        
        self._action_space = spaces.Discrete(len(Actions))
        self._observation_space = spaces.Dict(
            {
                'features': gym.spaces.Box(low=0, high=np.inf, shape=(window_size, self.feature_dim_len)),
                'position': gym.spaces.Box(low=-1, high=1, shape=(1,1)),
                # 'tick': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,1)),
            }
        )
        self._reward_space = gym.spaces.Box(-inf, inf, shape=(1, ), dtype=np.float32)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
        self.np_random, seed = seeding.np_random(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.cnt += 1
        self.eps_length = len(self.df) - self.window_size
        self.prices, self.signal_features = self._process_data(None)
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.FLAT
        self._position_history = []
        self._profit_history = [1.]
        self._total_reward = 0.
        info = dict(
            total_reward=self._total_reward,
            position=self._position.value,
        )
        
        return self._get_observation(), info

    def random_action(self) -> Any:
        return np.array([self.action_space.sample()])

    def step(self, action: np.ndarray):

        self._done = False
        self._current_tick += 1

        if self._current_tick >= self._end_tick:
            self._done = True
            if self._position == Positions.LONG:
                action = Actions.SELL.value
            elif self._position == Positions.SHORT:
                action = Actions.BUY.value
            else:
                action = Actions.HOLD.value

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward
        step_profit = self._calculate_profit(action)

        self._position, trade = transform(self._position, action)

        if trade:
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        # self._profit_history.append(float(np.exp(self._total_reward)))
        self._profit_history.append(step_profit)
        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            position=self._position.value,
        )

        if self._done:
            if self._env_id[-1] == 'e' and self.cnt % self.plot_freq == 0:
                self.render()
            info['max_possible_profit'] = np.log(self.max_possible_profit())
            info['eval_episode_return'] = self._total_reward
            info['profit_history'] = self._profit_history

        return observation, step_reward, False, self._done, info

    def _get_observation(self) -> np.ndarray:
        obs = to_ndarray(self.signal_features[(self._current_tick - self.window_size + 1):self._current_tick + 1]
                         ).astype(np.float32)
        obs = {
            'features': obs,
            'position': np.array([[self._position.value]]).astype(float),
        }
        return obs

    def render(self) -> None:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.xlabel('trading days')
        plt.ylabel('profit')
        plt.plot(self._profit_history)

        plt.clf()
        plt.xlabel('trading days')
        plt.ylabel('close price')
        window_ticks = np.arange(self._end_tick - self._start_tick)
        target_price = self.target_prices[self._start_tick - self.window_size + 1:self._end_tick + 1]
        eps_price = self.raw_prices[self._start_tick - self.window_size + 1:self._end_tick + 1]
        plt.plot(eps_price)
        plt.plot(target_price)

        short_ticks = []
        long_ticks = []
        flat_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.SHORT:
                short_ticks.append(tick + self.window_size)
            elif self._position_history[i] == Positions.LONG:
                long_ticks.append(tick + self.window_size)
            else:
                flat_ticks.append(tick + self.window_size)

        plt.plot(long_ticks, eps_price[long_ticks], 'g^', markersize=3, label="Long")
        plt.plot(flat_ticks, eps_price[flat_ticks], 'bo', markersize=3, label="Flat")
        plt.plot(short_ticks, eps_price[short_ticks], 'rv', markersize=3, label="Short")
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
        plt.savefig(self.save_path + str(self._env_id) + '-price.png')

    def close(self):
        import matplotlib.pyplot as plt
        plt.close()

    @abstractmethod
    def _process_data(self):
        raise NotImplementedError

    @abstractmethod
    def _calculate_reward(self, action):
        raise NotImplementedError

    @abstractmethod
    def _calculate_profit(self, action):
        raise NotImplementedError

    @abstractmethod
    def max_possible_profit(self):
        raise NotImplementedError

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "DI-engine Trading Env"


# @ENV_REGISTRY.register('stocks_1-v0')
class StocksEnvDizoo(TradingEnv):

    def __init__(
        self,
        trade_fee_bid_percent,
        trade_fee_ask_percent,
        **kwargs
        ):

        super().__init__(**kwargs)

        # ====== load Google stocks data =======
        # raw_data = load_dataset(self._cfg.stocks_data_filename, 'Date')
        # raw_data = deepcopy(self.df)
        # ======================================

        # set cost
        self.trade_fee_bid_percent = trade_fee_bid_percent  # unit
        self.trade_fee_ask_percent = trade_fee_ask_percent  # unit

    # override
    def _process_data(self, start_idx: int = None) -> Any:
        '''
        Overview:
            used by env.reset(), process the raw data.
        Arguments:
            - start_idx (int): the start tick; if None, then randomly select.
        Returns:
            - prices: the close.
            - signal_features: feature map
            - feature_dim_len: the dimension length of selected feature
        '''

        # ====== build feature map ========
        # 'wavelet_transformed_close', 'High', 'Low', 'Close', 'Volume'
        all_feature = {k: self.df.loc[:, k].to_numpy() for k in self.all_feature_name}
        # add feature "Diff"
        prices = self.df.loc[:, self.target_name].to_numpy()
        if 'Diff' in self.all_feature_name:
            diff = np.insert(np.diff(prices), 0, 0)
            self.all_feature_name.append('Diff')
            all_feature['Diff'] = diff
        # =================================

        # you can select features you want
        all_feature_stack = np.column_stack([all_feature[k] for k in all_feature])

        # validate index
        if start_idx is None:
            self.start_idx = self.window_size
        else:
            self.start_idx = start_idx
        self._start_tick = self.start_idx
        self._end_tick = self._start_tick + self.eps_length - 1

        return prices, all_feature_stack

    # override
    def _calculate_reward(self, action: int) -> np.float32:
        step_reward = 0.
        current_price = (self.raw_prices[self._current_tick])
        last_trade_price = (self.raw_prices[self._last_trade_tick])
        ratio = current_price / last_trade_price
        # Cost is always negative
        cost = np.log((1 - self.trade_fee_ask_percent)\
             * (1 - self.trade_fee_bid_percent))
        # cost = 0

        if action == Actions.BUY and self._position == Positions.SHORT:
            # step_reward = np.log(1 - ratio) + cost
            step_reward = -1*np.log(ratio) + cost

        if action == Actions.SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_BUY and self._position == Positions.SHORT:
            # step_reward = np.log(1 - ratio) + cost
            step_reward = -1*np.log(ratio) + cost

        step_reward = float(step_reward)

        return step_reward
    
    # override
    def _calculate_profit(self, action: int) -> np.float32:
        step_profit = 0.
        # current_price = (self.raw_prices[self._current_tick])
        # last_trade_price = (self.raw_prices[self._last_trade_tick])
        current_price = (self.target_prices[self._current_tick])
        last_trade_price = (self.target_prices[self._last_trade_tick])
        # Cost is always positive
        cost = (current_price * self.trade_fee_ask_percent\
             + current_price * self.trade_fee_bid_percent)\
             / last_trade_price
        pct_change = (current_price - last_trade_price)\
                   / last_trade_price

        if action == Actions.BUY and self._position == Positions.SHORT:
            step_profit = -1 * pct_change - cost

        if action == Actions.SELL and self._position == Positions.LONG:
            step_profit = pct_change - cost

        if action == Actions.DOUBLE_SELL and self._position == Positions.LONG:
            step_profit = pct_change - cost

        if action == Actions.DOUBLE_BUY and self._position == Positions.SHORT:
            step_profit = -1 * pct_change - cost

        step_profit = float(step_profit)

        return step_profit

    # override
    def max_possible_profit(self) -> float:
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:

            if self.raw_prices[current_tick] < self.raw_prices[current_tick - 1]:
                while (current_tick <= self._end_tick
                       and self.raw_prices[current_tick] < self.raw_prices[current_tick - 1]):
                    current_tick += 1

                current_price = self.raw_prices[current_tick - 1]
                last_trade_price = self.raw_prices[last_trade_tick]
                tmp_profit = profit * (2 - (current_price / last_trade_price)) * (1 - self.trade_fee_ask_percent
                                                                                  ) * (1 - self.trade_fee_bid_percent)
                profit = max(profit, tmp_profit)
            else:
                while (current_tick <= self._end_tick
                       and self.raw_prices[current_tick] >= self.raw_prices[current_tick - 1]):
                    current_tick += 1

                current_price = self.raw_prices[current_tick - 1]
                last_trade_price = self.raw_prices[last_trade_tick]
                tmp_profit = profit * (current_price / last_trade_price) * (1 - self.trade_fee_ask_percent
                                                                            ) * (1 - self.trade_fee_bid_percent)
                profit = max(profit, tmp_profit)
            last_trade_tick = current_tick - 1

        return profit

    def __repr__(self) -> str:
        return "DI-engine Stocks Trading Env"    
    
    
class StocksEnvDizooMultiDataset(StocksEnvDizoo):
    def __init__(self,
                df_list,
                window_size,
                target_name: str = 'wavelet_transformed_close',
                normalize: bool = True,
                episodes_between_dataset_switch = 1,
                total_days = 32,
                filter_pct: int = 0.1,
                **kwargs):
        self._episodes_on_this_dataset = 0
        self.episodes_between_dataset_switch = episodes_between_dataset_switch
        self.total_days = total_days
        self.filter_pct = filter_pct
        self.window_size = window_size
        self.target_name = target_name
        self.normalize = normalize
        self.total_datasets = len(df_list)
        self.df_list = df_list
        self.next_data_generator = self._get_next_dataset_generator()
        
        super().__init__(
            # Note over here we instantiate the generator
            # So that we start from 0 when _get_next_dataset_generator
            # is called
            df=next(self._get_next_dataset_generator()),
            window_size=window_size,
            target_name=target_name,
            **kwargs
        )
    
    def _get_next_dataset_generator(self):
        self._episodes_on_this_dataset = 0
        total_day_num = None
        while True:
            while True:
                rand_df_idx = np.random.randint(self.total_datasets)
                df = self.df_list[rand_df_idx]
                if self.total_days is None:
                    df_to_yield = df
                    break
                if isinstance(self.total_days, list):
                    total_day_num = np.random.choice(self.total_days)
                else:
                    total_day_num = self.total_days
                rand_month_idx = np.random.randint(len(df)-self.window_size - total_day_num)
                df_to_yield = df.iloc[rand_month_idx: rand_month_idx + self.window_size + total_day_num]
                # df_to_yield.reset_index(inplace=True, drop=True)
                last_week_close = df_to_yield['close'][:self.window_size]
                pct_diff = lambda y: (y.max() - y.min()) / (y.max())
                if (pct_diff(last_week_close) > self.filter_pct):
                    break
            yield df_to_yield.copy()
    
    def reset(self, seed=None):
        self._episodes_on_this_dataset += 1
        if self._episodes_on_this_dataset % self.episodes_between_dataset_switch == 0:
            self.df = next(self.next_data_generator)
            self.raw_prices = self.df.loc[:, self.target_name].to_numpy()
            self.target_prices = self.df.loc[:, 'close'].to_numpy()
            EPS = 1e-10
            # if self.train_range is None or self.test_range is None:
            self.df = self.df.set_index('date')
            if self.normalize:
                self.df = self.df.apply(lambda x: (x - x.mean()) / (x.std() + EPS), axis=0)
        return super().reset(seed)
    

class StocksEnvDizooMultiDatasetVal(StocksEnvDizooMultiDataset):
    def __init__(self,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)
    
    def _get_next_dataset_generator(self):
        self.episodes_between_dataset_switch = 1
        self._episodes_on_this_dataset = 0
        for df in self.df_list:
            yield df.copy()
            
class StocksEnvDizooMultiDatasetTest(StocksEnvDizooMultiDatasetVal):
    def __init__(self,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)
            
    def reset(self, seed=None, options=None, position=Positions.FLAT):
        super(TradingEnv, self).reset(seed=seed)
        
        self._episodes_on_this_dataset += 1
        if self._episodes_on_this_dataset % self.episodes_between_dataset_switch == 0:
            self.df = next(self.next_data_generator)
            self.raw_prices = self.df.loc[:, self.target_name].to_numpy()
            self.target_prices = self.df.loc[:, 'close'].to_numpy()
            EPS = 1e-10
            # if self.train_range is None or self.test_range is None:
            self.df = self.df.set_index('date') if 'date' in self.df else self.df
            if self.normalize:
                self.df = self.df.apply(lambda x: (x - x.mean()) / (x.std() + EPS), axis=0)
        self._position = position
        self.cnt += 1
        self.eps_length = len(self.df) - self.window_size
        self.prices, self.signal_features = self._process_data(None)
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position_history = []
        self._profit_history = [1.]
        self._total_reward = 0.
        info = dict(
            total_reward=self._total_reward,
            position=self._position.value,
        )
        
        return self._get_observation(), info            