import json
import random
from dataclasses import MISSING, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import gym
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.tune.schedulers.pb2 import PB2

from geese.constants import RewardFunc
from geese.env import SoloEnv

MODEL_NAME = "model_name"
ENV_NAME = "env_name"


@dataclass
class Parameter:
    conv_filters: List[List[Union[int, List[int]]]] = field(
        default_factory=lambda: [[16, [3, 3], 1], [32, [3, 3], 1], [64, [7, 11], 1]]
    )
    reward_func: str = "RAW"
    reward_list: Optional[List[float]] = field(
        default_factory=lambda: [-1, -0.5, 0.5, 1]
    )
    scale_flg: bool = False
    press_flg: bool = True
    max_reward_value: float = 20099
    # PPOParameter
    learning_rate: List[float] = field(
        default_factory=lambda: [1e-3, 1e-6], default=MISSING
    )
    batch_size: List[int] = field(default_factory=lambda: [128, 512], default=MISSING)
    param_lambda: List[float] = field(
        default_factory=lambda: [0.9, 1.0], default=MISSING
    )
    clip_param: List[float] = field(default_factory=lambda: [0.1, 0.5], default=MISSING)
    # PB2Parameter
    perturbation_interval: int = 1000
    # RLlib Parameter
    num_samples: int = 8
    num_workers: int = 1

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
    def tune_arguments(self) -> Dict[str, Any]:
        run_or_experiment = PPOTrainer
        num_samples = self.num_samples
        config = {
            "framework": "tf",
            "model": {"conv_filters": self.conv_filters},
            "env": ENV_NAME,
            "lambda": tune.sample_from(lambda _: random.uniform(*self.param_lambda)),
            "clip_param": tune.sample_from(lambda _: random.uniform(*self.clip_param)),
            "lr": tune.sample_from(lambda _: random.uniform(*self.learning_rate)),
            "train_batch_size": tune.sample_from(
                lambda _: random.randint(*self.batch_size)
            ),
            "num_workers": self.num_workers,
        }
        pb2_scheduler = PB2(
            time_attr="timesteps_total",
            metric="episode_reward_mean",
            mode="max",
            perturbation_interval=self.perturbation_interval,
            quantile_fraction=0.25,
            hyperparam_bounds={
                "lambda": self.param_lambda,
                "clip_param": self.clip_param,
                "lr": self.learning_rate,
                "train_batch_size": self.batch_size,
            },
            synch=True,
        )
        arguments = {
            "run_or_experiment": run_or_experiment,
            "scheduler": pb2_scheduler,
            "config": config,
            "num_samples": num_samples,
        }
        return arguments


def get_env_factory(parameter: Parameter) -> Callable[[], gym.Env]:
    def env_factory(_: Dict[str, Any]) -> gym.Env:
        return SoloEnv(**parameter.env_parameter)

    return env_factory


if __name__ == "__main__":
    # with open("./conf/parameter.json", "r") as f:
    #     param_json = json.load(f)
    # parameter = Parameter(**param_json)
    parameter = Parameter()
    env_factory = get_env_factory(parameter)
    register_env(ENV_NAME, env_factory)
    tune.run(**parameter.tune_arguments)
