# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# game environment

import importlib

import numpy as np

# You need to install kaggle_environments, requests
from kaggle_environments import make

ENVS = {
    "TicTacToe": "handyrl.envs.tictactoe",
    "Geister": "handyrl.envs.geister",
    "ParallelTicTacToe": "handyrl.envs.parallel_tictactoe",
    "HungryGeese": "handyrl.envs.kaggle.hungry_geese",
}


def prepare_env(env_args):
    env_name = env_args["env"]
    env_source = ENVS.get(env_name, env_name)
    env_module = importlib.import_module(env_source)

    if env_module is None:
        print("No environment %s" % env_name)
    elif hasattr(env_module, "prepare"):
        env_module.prepare()


def make_env(env_args):
    env_name = env_args["env"]
    env_source = ENVS.get(env_name, env_name)
    env_module = importlib.import_module(env_source)

    if env_module is None:
        print("No environment %s" % env_name)
    else:
        return env_module.Environment(env_args)


# base class of Environment


class BaseEnvironment:
    def __init__(self, args={}):
        pass

    def __str__(self):
        return ""

    #
    # Should be defined in all games
    #
    def reset(self, args={}):
        raise NotImplementedError()

    #
    # Should be defined in all games except you implement original step() function
    #
    def play(self, action, player):
        raise NotImplementedError()

    #
    # Should be defined in games which has simultaneous trainsition
    #
    def step(self, actions):
        for p, action in actions.items():
            if action is not None:
                self.play(action, p)

    #
    # Should be defined if you use multiplayer sequential action game
    #
    def turn(self):
        return 0

    #
    # Should be defined if you use multiplayer simultaneous action game
    #
    def turns(self):
        return [self.turn()]

    #
    # Should be defined in all games
    #
    def terminal(self):
        raise NotImplementedError()

    #
    # Should be defined if you use immediate reward
    #
    def reward(self):
        return {}

    #
    # Should be defined in all games
    #
    def outcome(self):
        raise NotImplementedError()

    #
    # Should be defined in all games
    #
    def legal_actions(self, player):
        raise NotImplementedError()

    #
    # Should be defined in all games
    #
    def action_length(self):
        raise NotImplementedError()

    #
    # Should be defined if you use multiplayer game or add name to each player
    #
    def players(self):
        return [0]

    #
    # Should be defined in all games
    #
    def observation(self, player=None):
        raise NotImplementedError()

    #
    # Should be defined if you encode action as special string
    #
    def action2str(self, a, player=None):
        return str(a)

    #
    # Should be defined if you encode action as special string
    #
    def str2action(self, s, player=None):
        return int(s)

    #
    # Should be defined if you use network battle mode
    #
    def diff_info(self, player=None):
        return ""

    #
    # Should be defined if you use network battle mode
    #
    def update(self, info, reset):
        raise NotImplementedError()


class Environment(BaseEnvironment):
    ACTION = ["NORTH", "SOUTH", "WEST", "EAST"]
    DIRECTION = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    NUM_AGENTS = 4

    def __init__(self, args={}):
        super().__init__()
        self.env = make("hungry_geese")
        self.reset()

    def reset(self, args={}):
        obs = self.env.reset(num_agents=self.NUM_AGENTS)
        self.update((obs, {}), True)

    def update(self, info, reset):
        obs, last_actions = info
        if reset:
            self.obs_list = []
        self.obs_list.append(obs)
        self.last_actions = last_actions

    def action2str(self, a, player=None):
        return self.ACTION[a]

    def str2action(self, s, player=None):
        return self.ACTION.index(s)

    def direction(self, pos_from, pos_to):
        if pos_from is None or pos_to is None:
            return None
        x, y = pos_from // 11, pos_from % 11
        for i, d in enumerate(self.DIRECTION):
            nx, ny = (x + d[0]) % 7, (y + d[1]) % 11
            if nx * 11 + ny == pos_to:
                return i
        return None

    def __str__(self):
        # output state
        obs = self.obs_list[-1][0]["observation"]
        colors = ["\033[33m", "\033[34m", "\033[32m", "\033[31m"]
        color_end = "\033[0m"

        def check_cell(pos):
            for i, geese in enumerate(obs["geese"]):
                if pos in geese:
                    if pos == geese[0]:
                        return i, "h"
                    if pos == geese[-1]:
                        return i, "t"
                    index = geese.index(pos)
                    pos_prev = geese[index - 1] if index > 0 else None
                    pos_next = geese[index + 1] if index < len(geese) - 1 else None
                    directions = [
                        self.direction(pos, pos_prev),
                        self.direction(pos, pos_next),
                    ]
                    return i, directions
            if pos in obs["food"]:
                return "f"
            return None

        def cell_string(cell):
            if cell is None:
                return "."
            elif cell == "f":
                return "f"
            else:
                index, directions = cell
                if directions == "h":
                    return colors[index] + "@" + color_end
                elif directions == "t":
                    return colors[index] + "*" + color_end
                elif max(directions) < 2:
                    return colors[index] + "|" + color_end
                elif min(directions) >= 2:
                    return colors[index] + "-" + color_end
                else:
                    return colors[index] + "+" + color_end

        cell_status = [check_cell(pos) for pos in range(7 * 11)]

        s = "turn %d\n" % len(self.obs_list)
        for x in range(7):
            for y in range(11):
                pos = x * 11 + y
                s += cell_string(cell_status[pos])
            s += "\n"
        for i, geese in enumerate(obs["geese"]):
            s += colors[i] + str(len(geese) or "-") + color_end + " "
        return s

    def step(self, actions):
        # state transition
        obs = self.env.step(
            [self.action2str(actions.get(p, None) or 0) for p in self.players()]
        )
        self.update((obs, actions), False)

    def diff_info(self, _):
        return self.obs_list[-1], self.last_actions

    def turns(self):
        # players to move
        return [p for p in self.players() if self.obs_list[-1][p]["status"] == "ACTIVE"]

    def terminal(self):
        # check whether terminal state or not
        for obs in self.obs_list[-1]:
            if obs["status"] == "ACTIVE":
                return False
        return True

    def outcome(self):
        # return terminal outcomes
        # 1st: 1.0 2nd: 0.33 3rd: -0.33 4th: -1.00
        rewards = {o["observation"]["index"]: o["reward"] for o in self.obs_list[-1]}
        outcomes = {p: 0 for p in self.players()}
        for p, r in rewards.items():
            for pp, rr in rewards.items():
                if p != pp:
                    if r > rr:
                        outcomes[p] += 1 / (self.NUM_AGENTS - 1)
                    elif r < rr:
                        outcomes[p] -= 1 / (self.NUM_AGENTS - 1)
        return outcomes

    def legal_actions(self, player):
        # return legal action list
        return list(range(len(self.ACTION)))

    def action_length(self):
        # maximum action label (it determines output size of policy function)
        return len(self.ACTION)

    def players(self):
        return list(range(self.NUM_AGENTS))

    def rule_based_action(self, player):
        from kaggle_environments.envs.hungry_geese.hungry_geese import (
            Observation,
            Configuration,
            Action,
            GreedyAgent,
        )

        action_map = {
            "N": Action.NORTH,
            "S": Action.SOUTH,
            "W": Action.WEST,
            "E": Action.EAST,
        }

        agent = GreedyAgent(Configuration({"rows": 7, "columns": 11}))
        agent.last_action = (
            action_map[self.ACTION[self.last_actions[player]][0]]
            if player in self.last_actions
            else None
        )
        obs = {
            **self.obs_list[-1][0]["observation"],
            **self.obs_list[-1][player]["observation"],
        }
        action = agent(Observation(obs))
        return self.ACTION.index(action)

    def observation(self, player=None):
        if player is None:
            player = 0

        b = np.zeros((self.NUM_AGENTS * 4 + 1, 7 * 11), dtype=np.float32)
        obs = self.obs_list[-1][0]["observation"]

        for p, geese in enumerate(obs["geese"]):
            # head position
            for pos in geese[:1]:
                b[0 + (p - player) % self.NUM_AGENTS, pos] = 1
            # tip position
            for pos in geese[-1:]:
                b[4 + (p - player) % self.NUM_AGENTS, pos] = 1
            # whole position
            for pos in geese:
                b[8 + (p - player) % self.NUM_AGENTS, pos] = 1

        # previous head position
        if len(self.obs_list) > 1:
            obs_prev = self.obs_list[-2][0]["observation"]
            for p, geese in enumerate(obs_prev["geese"]):
                for pos in geese[:1]:
                    b[12 + (p - player) % self.NUM_AGENTS, pos] = 1

        # food
        for pos in obs["food"]:
            b[16, pos] = 1

        return b.reshape(-1, 7, 11)
