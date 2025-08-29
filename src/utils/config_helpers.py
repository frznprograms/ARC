import yaml
from itertools import product
import copy


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def expand_config(base_config: dict):
    """
    Expand config values that are lists into multiple configs.
    """
    flat = {}
    for section, params in base_config["pipeline"].items():
        for key, value in params.items():
            flat[(section, key)] = value if isinstance(value, list) else [value]

    # cartesian product of all parameter options
    keys, values = zip(*flat.items())
    for combo in product(*values):
        new_config = copy.deepcopy(base_config)
        for (section, key), val in zip(keys, combo):
            new_config["pipeline"][section][key] = val
        yield new_config
