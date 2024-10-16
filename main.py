import argparse
import json
import os

from tqdm import tqdm

from configs.parser import *
from trainer.train import calculate_scores, print_scores
from utils.data import get_data

if __name__ == "__main__":
    args = parse_args()
    config_path = args.cfg
    output_name = args.output if args.output else "output.json"
    print(config_path)
    with open(config_path, "r") as f:
        json_obj = json.load(f)
    # create run dir
    run_dir = f"runs/{config_path.split('/')[-1][:-5]}"
    os.makedirs(run_dir, exist_ok=True)

    instances = parse_json_and_create_tests(json_obj)
    parsed = parse_instances(instances)
    
    for i in tqdm(range(len(parsed))):
        instance = parsed[i]
        final_results = {}
        
        X_train, X_test, y_train, y_test = get_data(**instance["data_args"])
        
        model = instance["method"](
            instance["model"],
            X_train,
            y_train,
            **instance["method_args"],
            **instance["model_args"],
        )

        # drop NaNs in y for evaluation
        # Note : This does nothing if there's already no missing value
        valid_indices = ~y_test.isna().any(axis=1)
        X_test = X_test[valid_indices]
        y_test = y_test[valid_indices]

        valid_indices = ~y_train.isna().any(axis=1)
        X_train = X_train[valid_indices]
        y_train = y_train[valid_indices]

        state_score = calculate_scores(model, X_train, X_test, y_train, y_test)
        # print_scores(model, X_train, X_test, y_train, y_test)
        for key, value in state_score.items():
            final_results[key] = (
                value
                if final_results.get(key) is None
                else final_results.get(key) + value
            )
        # at the end of all the states calculate mean of the scores
        for key, value in final_results.items():
            final_results[key] = value
        instances[i].update(final_results)
    
    with open(f"{run_dir}/{output_name}", "w") as f:
        json.dump(instances, f, indent=4)

    exit()
