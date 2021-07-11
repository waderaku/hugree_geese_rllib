from dataclasses import dataclass
from typing import Any, Callable, Dict

import gym
import ray
import tensorflow as tf
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from geese.env import SoloEnv
from geese.model import TorusModel
from geese.utils.modelv2wrapper import create_modelv2


@dataclass
class Parameter:
    @property
    def model_parameter(self) -> Dict[str, Any]:
        pass

    @property
    def env_parameter(self) -> Dict[str, Any]:
        pass


def get_model_factory(parameter: Parameter) -> Callable[[], tf.keras.Model]:
    def model_factory() -> Callable[[]]:
        return TorusModel(**parameter.model_parameter)

    return model_factory


def get_env_factory(parameter: Parameter) -> Callable[[], gym.Env]:
    def env_factory(_: Dict[str, Any]) -> gym.Env:
        return SoloEnv(**parameter.env_parameter)

    return env_factory


parameter = Parameter()

# Run

model_factory = get_model_factory(parameter)
model_class = create_modelv2(model_factory)
env_factory = get_env_factory(parameter)
ModelCatalog.register_custom_model("mymodel", model_class)
register_env("myenv", env_factory)
config = {
    "framework": "tf",
    "model": {"custom_model": "mymodel", "custom_model_config": {}},
    "env": "myenv",
}

ray.init()
tune.run(PPOTrainer, config=config)
