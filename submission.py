import random
from dataclasses import MISSING, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import gym
import numpy as np
import ray
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
        default_factory=lambda: [1e-8, 1e-3], default=MISSING
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
    num_workers: int = 0
    stop: Dict[str, int] = field(
        default_factory=lambda: {"time_steps_total": 100_000}, default=MISSING
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
        return config

    @property
    def tune_arguments(self) -> Dict[str, Any]:
        run_or_experiment = PPOTrainer
        num_samples = self.num_samples
        config = self.config
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
            synch=False,
        )
        arguments = {
            "stop": self.stop,
            "run_or_experiment": run_or_experiment,
            "scheduler": pb2_scheduler,
            "config": config,
            "num_samples": num_samples,
            "checkpoint_at_end": True,
        }
        return arguments


def get_env_factory(parameter: Parameter) -> Callable[[Dict[str, Any]], gym.Env]:
    def env_factory(_: Dict[str, Any]) -> gym.Env:
        return SoloEnv(**parameter.env_parameter)

    return env_factory


# Input for Neural Network
def centerize(b: np.ndarray):
    dy, dx = np.where(b[0])
    centerize_y = (np.arange(0, 7) - 3 + dy[0]) % 7
    centerize_x = (np.arange(0, 11) - 5 + dx[0]) % 11

    b = b[:, centerize_y, :]
    b = b[:, :, centerize_x]

    return b


def make_input(obses: List[Dict]):
    b: np.ndarray = np.zeros((17, 7 * 11), dtype=np.float32)
    obs = obses[-1]

    for p, pos_list in enumerate(obs["geese"]):
        # head position
        for pos in pos_list[:1]:
            b[0 + (p - obs["index"]) % 4, pos] = 1
        # tip position
        for pos in pos_list[-1:]:
            b[4 + (p - obs["index"]) % 4, pos] = 1
        # whole position
        for pos in pos_list:
            b[8 + (p - obs["index"]) % 4, pos] = 1

    # previous head position
    if len(obses) > 1:
        obs_prev = obses[-2]
        for p, pos_list in enumerate(obs_prev["geese"]):
            for pos in pos_list[:1]:
                b[12 + (p - obs["index"]) % 4, pos] = 1

    # food
    for pos in obs["food"]:
        b[16, pos] = 1

    b = b.reshape(-1, 7, 11)
    b = centerize(b)

    return b


def load_agent(
    path: str,
    config: Dict[str, Any],
) -> PPOTrainer:
    ray.init(num_gpus=config["num_gpus"])
    # ray.init()
    if "env" not in config:
        raise KeyError("Config must contains key 'env'")
    if "model" not in config:
        raise KeyError("Config must contains key 'model'")
    agent = PPOTrainer(config=config)
    agent.restore(path)
    return agent


# p = Path("/kaggle_simulations/agent/")
# if p.exists():
#     sys.path.append(str(p))
# else:
#     p = Path("__file__").resolve().parent
print("aaa")

path = "/root/ray_results/PPO_2021-07-18_14-33-59/PPO_env_name_c0dc1_00007_7_clip_param=0.35753,lambda=0.9926,lr=0.00047974,train_batch_size=409_2021-07-18_14-34-00/checkpoint_000003/checkpoint-3"
parameter = Parameter()
env_factory = get_env_factory(parameter)
register_env(ENV_NAME, env_factory)
# model: tf.keras.models.Model = tf.keras.models.load_model(str(p / "my_model.h5"))
obses = []
_agent = load_agent(
    path,
    {
        "num_gpus": 0,
        "env": ENV_NAME,
        "model": {"conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1], [64, [7, 11], 1]]},
    },
)


def agent(obs_dict, config_dict):
    obses.append(obs_dict)
    X_test = make_input(obses)
    X_test = np.transpose(X_test, (1, 2, 0))
    # X_test = X_test.reshape(-1, 7, 11, 17)  # channel last.

    # avoid suicide
    # obstacles = X_test[:, :, [8, 9, 10, 11, 12]].max(axis=2) - X_test[
    #     :, :, [4, 5, 6, 7]
    # ].max(
    #     axis=2
    # )  # body + opposite_side - my tail
    # obstacles = np.array(
    #     [obstacles[0, 2, 5], obstacles[0, 4, 5], obstacles[0, 3, 4], obstacles[0, 3, 6]]
    # )

    y_pred = _agent.compute_action(X_test)
    # - obstacles

    actions = ["NORTH", "SOUTH", "WEST", "EAST"]
    return actions[np.argmax(y_pred)]
