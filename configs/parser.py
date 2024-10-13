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
    methods = json_data["methods"]
    # Handle random_state: if it's a list, use it directly, otherwise generate random integers
    if isinstance(json_data["data_args"]["random_state"], int):
        json_data["data_args"]["random_state"] = \
            random.sample(range(1000), json_data["data_args"]["random_state"]) # Generate random numbers
    
    # Transform all data_args into lists
    for data_arg, data_arg_value in json_data["data_args"].items():
        if not isinstance(data_arg_value, list):
            json_data["data_args"][data_arg] = [data_arg_value]

    # Generate all combinations of models, data_args, train_args
    test_instances = []
    for model, method, *args in itertools.product(
        models, methods, *json_data["data_args"].values()
    ):
        instance = {
            "model": model["name"],
            "model_args": {k: v for k, v in model.items() if k != "name"},
            "method": method["name"],
            "method_args": {k: v for k, v in method.items() if k != "name"},
            "data_args": {**{
                key: value
                for key, value in zip(json_data["data_args"].keys(), args)
            }, "drop_y_nans": not "semi" in method["name"]},
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
    
    parser.add_argument(
        "--output", type=str, help="Name of the output file"
    )

    args = parser.parse_args()
    return args
