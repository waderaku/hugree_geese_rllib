from typing import List, Optional, Tuple
import math
import numpy as np
from geese.constants import NUM_GEESE, RewardFunc
from geese.env.dena_env import Environment as DenaEnv
from geese.utils.converter import action2int
from kaggle_environments.envs.hungry_geese.hungry_geese import Action


class Env:
    def __init__(
        self,
        reward_func: RewardFunc,
        reward_list: Optional[List[float]],
        max_reward_value: float,
        press_flg: bool,
        scale_flg: bool,
    ):
        self._dena_env = DenaEnv()
        self._env = self._dena_env.env
        self._reward_func = reward_func
        self._reward_list = reward_list
        self._max_reward_value = max_reward_value
        self._press_flg = press_flg
        self._scale_flg = scale_flg
        if self._scale_flg:
            self._num = 0
            self._ave = 0
            self._squared_ave = 0

    def reset(self) -> List[np.ndarray]:
        self._dena_env.reset()
        return [self._dena_env.observation(p) for p in range(NUM_GEESE)]

    def step(
        self, actions: List[Action]
    ) -> Tuple[List[np.ndarray], List[float], List[bool]]:
        # 今回死ぬGooseを判定するために、１個前のStateですでに死んでいるかどうかを保持
        pre_done = np.array(
            [
                self._dena_env.env.state[p]["status"] != "ACTIVE"
                for p in range(NUM_GEESE)
            ]
        )
        actions = {p: action2int(actions[p]) for p in range(NUM_GEESE)}
        # Envを次の状態へ遷移させる
        self._dena_env.step(actions)

        # Gooseごとの終了判定
        done: np.ndarray = np.array(
            [
                self._dena_env.env.state[p]["status"] != "ACTIVE"
                for p in range(NUM_GEESE)
            ],
            dtype=np.float,
        )

        # Envの報酬
        env_reward = [
            self._dena_env.env.state[p]["reward"] for p in range(len(actions))
        ]

        # 報酬関数の適用
        raw_reward = self._compute_reward(env_reward)

        if self._press_flg:
            raw_reward = list(
                map(lambda x: (x / self._max_reward_value - 0.5) * 2, raw_reward)
            )

        if self._scale_flg:
            raw_reward = self._update_reward(raw_reward, done, pre_done)

        # 前回生きていて(1 - pre_done)今回死んだ(done)GooseにのみRewardをリターン
        reward: list = ((1 - pre_done) * done * raw_reward).tolist()

        # 全Geeseが終了したらリセット
        if sum(map(int, done)) == NUM_GEESE:
            self._dena_env.reset()

        # Gooseごとの観測
        observation = [self._dena_env.observation(p) for p in range(NUM_GEESE)]
        done = done.astype(np.bool).tolist()
        return observation, reward, done

    def __str__(self) -> str:
        return str(self._dena_env)

    def _compute_reward(self, env_rewards: List[float]) -> np.ndarray:
        if self._reward_func == RewardFunc.RANK:
            # env_rewardが小さい順に(index, env_reward)を格納したリスト
            target = [(i, v) for i, v in zip(range(len(env_rewards)), env_rewards)]
            target.sort(key=lambda x: x[1])

            # 順位配列（スコアが等しいときは繰り下げ順位を適用）
            charge = 1
            rank = NUM_GEESE
            state = -1
            rank_array = [None for _ in range(len(target))]
            for i, v in target:
                if state != v:
                    rank -= charge
                    charge = 1
                    state = v
                else:
                    charge += 1
                assert 0 <= rank < NUM_GEESE
                rank_array[i] = rank

            return np.array(
                [self._reward_list[rank] for rank in rank_array], dtype=np.float
            )
        else:
            return np.array(env_rewards, dtype=np.float)

    def _update_reward(
        self, reward: List[float], done: np.ndarray, pre_done: np.ndarray
    ) -> None:
        sum_reward = 0
        sum_squared_reward = 0
        num = 0
        for r, d, p_d in zip(reward, done, pre_done):
            if d != p_d:
                sum_reward += r
                sum_squared_reward += r * r
                num += 1
        new_num = self._num + num

        if new_num == 0:
            return reward

        old_per = self._num / new_num

        self._ave = self._ave * old_per + sum_reward / new_num
        self._squared_ave = self._squared_ave * old_per + sum_squared_reward / new_num

        self._num = new_num

        siguma = math.sqrt(self._squared_ave - (self._ave * self._ave))

        if siguma == 0:
            return reward

        return list(map(lambda x: x / siguma, reward))

    @property
    def dena_env(self) -> DenaEnv:
        return self._dena_env
