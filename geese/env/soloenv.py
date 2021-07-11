from geese.env.env import Env
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
from geese.constants import (
    ACTIONLIST,
    NUM_GEESE,
    RewardFunc,
    FIELD_HEIGHT,
    FIELD_WIDTH,
    NUM_CHANNELS,
)

DUMMY_ACTION = ACTIONLIST[0]


class SoloEnv(gym.Env):
    def __init__(
        self,
        reward_func: RewardFunc,
        reward_list: Optional[List[float]],
        max_reward_value: float,
        press_flg: bool,
        scale_flg: bool,
    ):
        self._env = Env(
            reward_func=reward_func,
            reward_list=reward_list,
            max_reward_value=max_reward_value,
            press_flg=press_flg,
            scale_flg=scale_flg,
        )

    def reset(self) -> np.ndarray:
        raise self._env.reset()[0]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        pre_done = [
            self._env.dena_env.env.state[p]["status"] != "ACTIVE"
            for p in range(NUM_GEESE)
        ]
        other_actions = [
            ACTIONLIST[self._env.dena_env.rule_based_action(player)]
            if not pre_done[player]
            else DUMMY_ACTION
            for player in range(1, NUM_GEESE)
        ]
        actions = [action] + other_actions
        obs, reward, done = self._env.step(actions)
        if done[0]:
            obs = self._env.reset()
        return obs[0], reward[0], done[0], dict()

    @property
    def observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
            low=np.zeros((NUM_CHANNELS, FIELD_HEIGHT, FIELD_WIDTH)),
            high=np.ones((NUM_CHANNELS, FIELD_HEIGHT, FIELD_WIDTH)),
        )

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(ACTIONLIST))
