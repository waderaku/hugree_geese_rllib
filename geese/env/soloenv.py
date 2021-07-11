from typing import Any, Dict, Tuple

import gym
import numpy as np


class SoloEnv(gym.Env):
    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        raise NotImplementedError

    @property
    def observation_space(self) -> gym.spaces.Space:
        raise NotImplementedError

    @property
    def action_space(self) -> gym.spaces.Space:
        raise NotImplementedError
