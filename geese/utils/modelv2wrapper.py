from typing import Any, Callable, Dict, List, Type

import gym
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2


def create_modelv2(model_factory: Callable[[], tf.keras.Model]) -> Type[TFModelV2]:
    class ModelV2(TFModelV2):
        def __init__(
            self,
            obs_space: gym.Space,
            action_space: gym.Space,
            num_outputs: int,
            model_config: Dict[str, Any],
            name: str,
        ):
            super().__init__(obs_space, action_space, num_outputs, model_config, name)
            self.base_model = model_factory()

        def forward(
            self, input_dict: Dict[str, Any], _state: List[Any], _max_len: Any
        ) -> tf.Tensor:
            obs = input_dict["obs"]
            if isinstance(obs, dict):
                policy, value = self.base_model(**obs)
            elif isinstance(obs, tuple):
                policy, value = self.base_model(*obs)
            else:
                policy, value = self.base_model(obs)
            self._value = value
            return policy, list()

        def value_function(self) -> tf.Tensor:
            return self._value

    return ModelV2
