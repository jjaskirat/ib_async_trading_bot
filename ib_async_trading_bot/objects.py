from dataclasses import dataclass, field, fields, asdict
from enum import Enum
from typing import Any, Callable, List, Union

import torch

CONTEXT_LENGTH = 16
PREDICTION_LENGTH = 16

NUM_ENVS = 16

@dataclass
class Base:
    def get_config(self):
        # return asdict(self)
        config = {}
        for field in fields(self):
            if isinstance(getattr(self, field.name), Base):
                config[field.name] = field.get_config()
            else:
                config[field.name] = getattr(self, field.name)
        return config

@dataclass
class DataSplitNativeConfig(Base):
    sizes: List = field(default_factory=lambda: [0.998, 0.002])
    context_length: int = CONTEXT_LENGTH
    prediction_length: int = 1
    num_pred_days: int = PREDICTION_LENGTH
    
@dataclass
class PolicyMlpPPOConfig(Base):
    net_arch: List = field(default_factory=lambda:[64, 64])
    optimizer_class: torch.optim.Optimizer = torch.optim.Adamax
    optimizer_kwargs: dict = field(default_factory=lambda:{
        'weight_decay': 1e-8,
    })
    
@dataclass
class PolicyMlpRecurrentPPOConfig(Base):
    net_arch: List = field(default_factory=lambda:[64, 64])
    n_lstm_layers: int = 3
    optimizer_class: torch.optim.Optimizer = torch.optim.Adamax
    optimizer_kwargs: dict = field(default_factory=lambda:{
        'weight_decay': 1e-8,
    })
    
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        min_lr = 1e-7
        new_lr = progress_remaining * initial_value
        return max(min_lr, new_lr)

    return func
    
@dataclass
class HyperparamsPPOConfig(Base):
    policy: str = 'MultiInputPolicy'
    # policy: str = 'MultiInputLstmPolicy'
    tensorboard_log: str = None
    # use_sde: bool = True
    # num_envs * n_steps
    batch_size: int = 128
    learning_rate: float = 5e-7
    # learning_rate: Callable = linear_schedule(1e-6)
    n_steps: int = 2048
    n_epochs: int = 10
    gamma: float = 0.95
    gae_lambda: float = 0.95
    ent_coef: float = 0.15
    policy_kwargs: dict = field(default_factory=lambda:\
        PolicyMlpPPOConfig().get_config())
    # 'lr_schedule': linear_schedule(LR),
    device: str ='cuda'
    verbose: int = 0
    
@dataclass
class EnvironmentDizooConfig(Base):
    window_size: int = CONTEXT_LENGTH
    normalize: bool = True
    target_name: str = 'close'
    all_feature_name: List = field(default_factory=lambda:[
        'open',
        'high',
        'low',
        'close',
        'volume',
    ])
    trade_fee_bid_percent: float = 0.005
    trade_fee_ask_percent: float = 0.005
    plot_freq: int = 10
    save_path: str = './'
    
@dataclass
class EnvironmentMultidatasetConfig(Base):
    episodes_between_dataset_switch: int = 1
    #TODO: list of random prediction length
    total_days: Union[int, List] = field(default_factory=lambda:[
        4,
        8,
        16
    ])
    filter_pct: float = 0.0
    
@dataclass
class DataTimeSeriesPandasDfObject(Base):
    type: str = 'data_time_series_pandasdf'
    target: str = 'close'
    columns: List = field(default_factory=lambda:[
        'open',
        'high',
        'low',
        'close',
        'volume',
    ])
    data_split: dict = field(default_factory=lambda:\
        DataSplitNativeConfig().get_config())
    dataloader: dict = field(default_factory=lambda: {})
    
@dataclass
class AIModelRLObject(Base):
    type: str = 'reinforcement_learning'
    model_name: str = 'stable_baselines3.PPO'
    model_path: str = None
    pretrained_path: str = None
    env_name: str = 'ib_async_trading_bot.environment.StocksEnvDizooMultiDataset'
    model_hyperparams: dict = field(default_factory=lambda:\
        HyperparamsPPOConfig().get_config())
    env_hyperparams: dict = field(default_factory=lambda:{
        **EnvironmentDizooConfig().get_config(),
        **EnvironmentMultidatasetConfig().get_config(),
    })
    device: str = 'cuda'
    
@dataclass
class ModelEngineSB3Object(Base):
    type: str = 'model_engine_sb3'
    # n_envs: int = 1024*32
    n_envs: int = NUM_ENVS
    training: dict = field(default_factory=lambda:{
        'total_timesteps': 10_000_000,
        'progress_bar': True,
        'callback': {
            'name': 'CheckpointCallback',
            'config': {
                'save_freq': 500_000 // NUM_ENVS,
                'save_path': "/models/rabadium/",
                'name_prefix': "rl_model",
                'save_replay_buffer': True,
                'save_vecnormalize': True,
            }
        }
    })

@dataclass
class TradingBotSB3Object(Base):
    type: str = 'application_tb_sb3'
    # backend_obj: BackendObject = field(default_factory=lambda:\
    #         BackendMongoDBObject())
    stock_symbol_list: List = field(default_factory=lambda:[])
    data_obj: DataTimeSeriesPandasDfObject = field(default_factory=lambda:\
            DataTimeSeriesPandasDfObject())
    ai_model_obj: AIModelRLObject = field(default_factory=lambda:\
            AIModelRLObject())
    model_engine_obj: ModelEngineSB3Object = field(default_factory=\
        lambda: ModelEngineSB3Object())
    
@dataclass
class DataSeq(Base):
    seq: Union[List, dict] = field(default_factory=lambda:{})
    
@dataclass
class StockSymbol(Base):
    stock_symbol: str = "AAPL"

@dataclass
class InterfaceTriggerObject(Base):
    type: str = 'interface_trigger'
    trigger: str = ''
    config: dict = field(default_factory=lambda: {})

@dataclass
class InterfaceObject(Base):
    type: str = 'interface'
    host: str = '127.0.0.1'
    port: int = 7496
    client_id: int = 1
    application_obj: TradingBotSB3Object = None
    positions: dict = field(default_factory=lambda:{})
    market_data: DataSeq = field(default_factory=lambda:DataSeq({}))
    top_gainers: List = field(default_factory=lambda:[])
    stocks_to_consider: List = field(default_factory=lambda: [])
    contracts: dict = field(default_factory=lambda: {})
    # model_loaded: Any = None
    # env_loaded: Any = None