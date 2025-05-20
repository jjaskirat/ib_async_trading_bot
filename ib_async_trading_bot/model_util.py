from functools import partial
from importlib import import_module
from typing import List

from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan
from stable_baselines3.common.env_util import make_vec_env

from ib_async_trading_bot.objects import ModelEngineSB3Object, AIModelRLObject

def get_model(ai_model: AIModelRLObject, mode: str = 'train'):
    ai_model_config = ai_model.get_config()
    model_import_str = ai_model_config['model_name']
    model_cls = get_class_from_str(model_import_str)
    if ai_model_config['model_path'] is not None:
        model = model_cls.load(ai_model_config['model_path'], print_system_info=True)
        model.tensorboard_log = ai_model_config['model_hyperparams']['tensorboard_log']
    else:
        model = partial(model_cls, **ai_model_config['model_hyperparams'])
    
    if mode == 'val':
        env_import_str = ai_model_config['env_name'] + 'Val'
    elif mode == 'test':
        env_import_str = ai_model_config['env_name'] + 'Test'
    else:
        env_import_str = ai_model_config['env_name']
    env_cls = get_class_from_str(env_import_str)
    env = partial(env_cls, **ai_model_config['env_hyperparams'])
    return model, env

def get_integrated_model(
    model_engine_obj: ModelEngineSB3Object,
    ai_model_obj: AIModelRLObject,
    dataset: List = None,
    mode: str = 'train',
    **kwargs
    ):
    model, env = get_model(ai_model_obj, mode=mode, **kwargs)
    if model_engine_obj.n_envs > 1 and mode == 'train':
        env = make_vec_env(
            env,
            n_envs = model_engine_obj.n_envs,
            env_kwargs={'df_list': dataset},
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={'start_method': 'spawn'}
        )
        env = VecCheckNan(env, raise_exception=True)
    else:
        # NOQA
        env = env(dataset)
    try:
        model = model(env=env)
    except Exception as e:
        print(e)
        model.env = env
        for name, parameter in model.policy.named_parameters():
            parameter.requires_grad = True
        print("Unfreeze")
    # Load Pretrained
    if ai_model_obj.pretrained_path is not None and ai_model_obj.model_path is None:
        ai_model_obj.model_path = ai_model_obj.pretrained_path
        model_pretrained, _ = get_model(ai_model_obj, mode=mode, **kwargs)
        # Load state dict with Type Mismatch
        current_model_dict = model.policy.state_dict()
        loaded_state_dict = model_pretrained.policy.state_dict()
        new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
        model.policy.load_state_dict(new_state_dict, strict=False)
        ai_model_obj.model_path = None
    
    return model, env

def get_class_from_str(class_str: str):
    """gets class name from the import string
    EX: 'rabadium.closing_value_predictor.rnn.GRU.GRUNet'
        will import GRUNet from rabadium/closing_value_predictor/rnn/GRU.py

    Args:
        class_str (str): import string

    Raises:
        ImportError: if an error is raised

    Returns:
        class: class of object
    """
    cl = None
    try:
        module_path, class_name = class_str.rsplit('.', 1)
        module = import_module(module_path)
        cl = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(class_str)
    return cl