import argparse
import itertools
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from models.MLP import MLP
from trainer.train import train
from trainer.train_semi import train_semi

model_dict = {
    "random_forest": RandomForestRegressor,
    "linear_regression": LinearRegression,
    "neural_network": MLP,
}

method_dict = {"supervised": train, "semi_supervised": train_semi}


def parse_json_and_create_tests(json_data):
    # Parse the models and test_size directly
    models = json_data["models"]
    test_size = json_data["test_size"]
    methods = json_data["methods"]
    target_features = json_data["target_features"]
    # Handle random_state: if it's a list, use it directly, otherwise generate random integers
    random_states = json_data["random_states"]
    if isinstance(random_states, int):
        random_states = [
            random.sample(range(1000), random_states)
        ]  # Generate random numbers

    # Generate all combinations of models, random_state, and test_size
    test_instances = []
    for model, states, size, method, target_feature in itertools.product(
        models, random_states, test_size, methods, target_features
    ):
        instance = {
            "model": model["name"],
            "model_args": {k: v for k, v in model.items() if k != "name"},
            "random_state": states,
            "test_size": size,
            "method": method["name"],
            "method_args": {k: v for k, v in method.items() if k != "name"},
            "target_features": target_feature,
            "drop_y_nans": not "semi" in method["name"],
        }
        test_instances.append(instance)

    return test_instances


def parse_instances(instances):
    parsed = []
    for instance in instances:
        parsed.append(instance.copy())
        parsed[-1]["model"] = model_dict[instance["model"]]
        parsed[-1]["method"] = method_dict[instance["method"]]
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Parse config file path.")
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to the config file"
    )

    args = parser.parse_args()
    return args
