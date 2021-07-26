from geese.utils.modelv2wrapper import create_modelv2
from model.model import Model
from dataclasses import MISSING, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from ray.rllib.models import ModelCatalog
import gym
import tensorflow as tf
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

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
    learning_rate: float = 5e-6
    batch_size: int = 512
    param_lambda: float = 0.9
    clip_param: float = 0.1
    # RLlib Parameter
    num_samples: int = 1
    num_workers: int = 7
    stop: Dict[str, int] = field(
        default_factory=lambda: {"timesteps_total": 10_000_000}, default=MISSING
    )

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
    def config(self) -> Dict[str, Any]:
        config = {
            "framework": "tf2",
            # "model": {"conv_filters": self.conv_filters},
            "model": {
                "custom_model": MODEL_NAME,
                "custom_model_config": {},
            },
            "env": ENV_NAME,
            "lambda": self.param_lambda,
            "clip_param": self.clip_param,
            "lr": self.learning_rate,
            "train_batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "num_gpus": 0,
        }
        return config

    @property
    def tune_arguments(self) -> Dict[str, Any]:
        run_or_experiment = PPOTrainer
        num_samples = self.num_samples
        config = self.config

        arguments = {
            "stop": self.stop,
            "run_or_experiment": run_or_experiment,
            "config": config,
            "num_samples": num_samples,
            "checkpoint_at_end": True,
        }
        return arguments


def get_env_factory(parameter: Parameter) -> Callable[[Dict[str, Any]], gym.Env]:
    def env_factory(_: Dict[str, Any]) -> gym.Env:
        return SoloEnv(**parameter.env_parameter)

    return env_factory


def get_model_factory(
    env_factory: Callable[[Dict[str, Any]], gym.Env]
) -> Callable[[], tf.keras.models.Model]:
    env = env_factory(None)
    obs = env.observation_space.sample()
    obs = obs[None, :, :, :]

    def model_factory() -> tf.keras.models.Model:
        model = Model()
        model.build(obs.shape)
        return model

    return model_factory


if __name__ == "__main__":
    parameter = Parameter()
    env_factory = get_env_factory(parameter)
    register_env(ENV_NAME, env_factory)
    model_factory = get_model_factory(env_factory)
    model_class = create_modelv2(model_factory)
    ModelCatalog.register_custom_model(MODEL_NAME, model_class)
    analysis = tune.run(**parameter.tune_arguments)
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean", mode="max"),
        metric="episode_reward_mean",
    )
    checkpoint_path = checkpoints[0][0]
    print(f"Best trial's checkpoint path: {checkpoint_path}")
