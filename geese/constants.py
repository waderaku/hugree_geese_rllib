import enum
from kaggle_environments.envs.hungry_geese.hungry_geese import Action

# Field Size
FIELD_HEIGHT = 7
FIELD_WIDTH = 11

NUM_CHANNELS = 17

# ActionList
ACTIONLIST = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]

NUM_GEESE = 4

TIME_LIMIT = 1.0

NO_GPU_MSG = "GPU is not available."

# save info
SAVE_DIR = "./trained_models"

# log info
LOG_BASE_DIR = "logs"


class RewardFunc(enum.Enum):
    RAW = enum.auto()
    RANK = enum.auto()
