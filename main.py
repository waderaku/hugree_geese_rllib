import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import gym
import ray
import tensorflow as tf
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from geese.constants import RewardFunc
from geese.env import SoloEnv
from geese.model import BaseModel
from geese.utils.modelv2wrapper import create_modelv2

MODEL_NAME = "model_name"
ENV_NAME = "env_name"


@dataclass
class Parameter:
    num_layers: int = 3
    num_filters: int = 16
    kernel_size: List[int] = field(default_factory=lambda: [3, 3])
    bn: bool = True
    use_gpu: bool = True
    reward_func: str = "RAW"
    reward_list: Optional[List[float]] = field(
        default_factory=lambda: [-1, -0.5, 0.5, 1]
    )
    scale_flg: bool = False
    press_flg: bool = True
    max_reward_value: float = 20099
    learning_rate: float = 1e-5

    @property
    def model_parameter(self) -> Dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "bn": self.bn,
            "use_gpu": self.use_gpu,
        }

    @property
    def env_parameter(self) -> Dict[str, Any]:
        if self.reward_func == "RAW":
            reward_func = RewardFunc.RAW
        elif self.reward_func == "RANK":
            reward_func = RewardFunc.RANK
        else:
            raise ValueError("Unexpected Reward Function")

        return {
            "reward_func": reward_func,
            "reward_list": self.reward_list,
            "scale_flg": self.scale_flg,
            "press_flg": self.press_flg,
            "max_reward_value": self.max_reward_value,
        }

    @property
    def tune_config(self):
        return {
            "framework": "tf",
            "model": {"custom_model": MODEL_NAME, "custom_model_config": {}},
            "env": ENV_NAME,
            "lr": self.learning_rate,
        }


def get_model_factory(parameter: Parameter) -> Callable[[], tf.keras.Model]:
    def model_factory() -> Callable[[], gym.Env]:
        return BaseModel(**parameter.model_parameter)

    return model_factory


def get_env_factory(parameter: Parameter) -> Callable[[], gym.Env]:
    def env_factory(_: Dict[str, Any]) -> gym.Env:
        return SoloEnv(**parameter.env_parameter)

    return env_factory


if __name__ == "__main__":
    with open("./conf/parameter.json", "r") as f:
        param_json = json.load(f)
    parameter = Parameter(**param_json)
    model_factory = get_model_factory(parameter)
    model_class = create_modelv2(model_factory)
    env_factory = get_env_factory(parameter)
    ModelCatalog.register_custom_model(MODEL_NAME, model_class)
    register_env(ENV_NAME, env_factory)

    ray.init()
    tune.run(PPOTrainer, config=parameter.tune_config)
