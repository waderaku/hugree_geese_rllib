from kaggle_environments.envs.hungry_geese.hungry_geese import Action


def action2int(action: Action) -> int:
    if action == Action.NORTH:
        return 0
    elif action == Action.SOUTH:
        return 1
    elif action == Action.WEST:
        return 2
    elif action == Action.EAST:
        return 3
    else:
        raise ValueError("Unexpected Action Input")
