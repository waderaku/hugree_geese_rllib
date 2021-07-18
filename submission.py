from typing import Any, Dict

import ray
from ray.rllib.agents.ppo import PPOTrainer


def load_agent(
    path: str,
    config: Dict[str, Any],
) -> PPOTrainer:
    ray.init(num_gpus=config["num_gpus"])
    if "env" not in config:
        raise KeyError("Config must contains key 'env'")
    if "model" not in config:
        raise KeyError("Config must contains key 'model'")
    agent = PPOTrainer(config=config)
    agent.restore(path)
    return agent
