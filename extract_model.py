from dataclasses import MISSING, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import tensorflow as tf
import gym
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
import ray
from geese.env import SoloEnv
from singleppo import Parameter

MODEL_NAME = "model_name"
ENV_NAME = "env_name"


def load_agent(
    path: str,
    config: Dict[str, Any],
) -> PPOTrainer:
    # ray.init(num_gpus=config["num_gpus"])
    if not ray.is_initialized():
        ray.init()
    if "env" not in config:
        raise KeyError("Config must contains key 'env'")
    if "model" not in config:
        raise KeyError("Config must contains key 'model'")
    agent = PPOTrainer(config=config)
    agent.restore(path)
    return agent


def get_env_factory(parameter: Parameter) -> Callable[[Dict[str, Any]], gym.Env]:
    def env_factory(_: Dict[str, Any]) -> gym.Env:
        return SoloEnv(**parameter.env_parameter)

    return env_factory


param = Parameter()
env_factory = get_env_factory(param)
register_env(ENV_NAME, env_factory)
path = "/root/ray_results/PPO_2021-07-25_21-55-25/PPO_env_name_943b2_00001_1_clip_param=0.17341,lambda=0.9278,lr=0.00013053,train_batch_size=430_2021-07-25_21-55-25/checkpoint_000003/checkpoint-3"
agent = load_agent(path, param.config)
model: tf.keras.models.Model = agent.get_policy().model.base_model
save_path = "./temp_model"
print(model.summary())
model.save(save_path)
