from pathlib import Path
import sys
from typing import Dict, List
import numpy as np
import tensorflow as tf

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


p = Path("/kaggle_simulations/agent/")
if p.exists():
    sys.path.append(str(p))
else:
    p = Path("__file__").resolve().parent

model: tf.keras.models.Model = tf.keras.models.load_model(str(p / "my_model.h5"))
obses = []


def agent(obs_dict, config_dict):
    obses.append(obs_dict)

    X_test = make_input(obses)
    X_test = np.transpose(X_test, (1, 2, 0))
    X_test = X_test.reshape(-1, 7, 11, 17)  # channel last.

    # avoid suicide
    obstacles = X_test[:, :, :, [8, 9, 10, 11, 12]].max(axis=3) - X_test[
        :, :, :, [4, 5, 6, 7]
    ].max(
        axis=3
    )  # body + opposite_side - my tail
    obstacles = np.array(
        [obstacles[0, 2, 5], obstacles[0, 4, 5], obstacles[0, 3, 4], obstacles[0, 3, 6]]
    )

    y_pred = model.predict(X_test) - obstacles

    actions = ["NORTH", "SOUTH", "WEST", "EAST"]
    return actions[np.argmax(y_pred)]
