from model_helper import *
import argparse
import json
import time
import numpy as np
import joblib
from xgboost import XGBClassifier, XGBRegressor
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


DATASET = "airline_classification"
MODEL = "xgboost"
TREES = None
DEPTH = None


def parse_arguments(config):
    global DATASET, MODEL
    parser = argparse.ArgumentParser(description='Arguments for train_model.')
    parser.add_argument("-d", "--dataset", type=str, choices=['higgs', 'airline_regression', 'airline_classification', 'fraud', 'year', 'epsilon', 'bosch', 'covtype', 'tpcxai_fraud'],
        help="Dataset to be trained. Choose from ['higgs', 'airline_regression', 'airline_classification', 'fraud', 'year', 'epsilon', 'bosch', 'covtype', 'tpcxai_fraud']")
    parser.add_argument("-m", "--model", type=str, choices=['randomforest', 'xgboost', 'lightgbm'],
                        help="Model name. Choose from ['randomforest', 'xgboost', 'lightgbm']")
    parser.add_argument(
        "-D", "--depth", type=int,
        choices=[8],
        help="Depth of trees[Optional default is 8]. Choose from [8].")
    parser.add_argument("-t", "--num_trees", type=int, choices=[10, 500, 1600],
                        help="Number of trees for the model. Choose from ['10', '500', '1600']")
    args = parser.parse_args()
    if args.dataset:
        DATASET = args.dataset
    if args.model:
        MODEL = args.model
    if args.num_trees:
        TREES = args.num_trees
        config["num_trees"] = args.num_trees
    if args.depth:
        DEPTH = args.depth
        config["depth"] = args.depth
    check_argument_conflicts(args)
    print(f"DATASET: {DATASET}")
    print(f"MODEL: {MODEL}")
    return args


def train(config, df_train):
    print("TRAINING START...")

    # Prepare data
    y_col = config[DATASET]["y_col"]
    x_col = list(df_train.columns)
    x_col.remove(y_col)
    x = np.array(df_train[x_col])
    y = np.array(df_train[y_col])
    # Load model
    # The settings of the models are consistent with Hummingbird: https://github.com/microsoft/hummingbird/blob/main/benchmarks/trees/train.py
    if MODEL == 'randomforest':
        if config[DATASET]["type"] == "classification":
            ModelClass = RandomForestClassifier
        else:
            ModelClass = RandomForestRegressor
        model = ModelClass(
            n_estimators=config["num_trees"],
            max_depth=config["depth"],
            verbose=0,
            n_jobs=-1
        )
    elif MODEL == "xgboost":
        task_spec_args = {}
        if config[DATASET]["type"] == "classification":
            ModelClass = XGBClassifier
            task_spec_args["scale_pos_weight"] = len(y) / np.count_nonzero(y)
            task_spec_args["objective"] = "binary:logistic"
        elif config[DATASET]["type"] == "regression":
            ModelClass = XGBRegressor
            task_spec_args["objective"] = "reg:squarederror"
        else:
            raise ValueError(
                "Task type in config.json must be one of ['classification', 'regression']")
        model = ModelClass(
            max_depth=config["depth"],
            n_estimators=config["num_trees"],
            max_leaves=256,
            learning_rate=0.1,
            tree_method="hist",
            reg_lambda=1,
            verbosity=0,
            n_jobs=-1,
            **task_spec_args
        )
    elif MODEL == "lightgbm":
        task_spec_args = {}
        from lightgbm import LGBMClassifier, LGBMRegressor
        if config[DATASET]["type"] == "classification":
            ModelClass = LGBMClassifier
            task_spec_args["scale_pos_weight"] = len(y) / np.count_nonzero(y)
            task_spec_args["objective"] = "binary"
        elif config[DATASET]["type"] == "regression":
            ModelClass = LGBMRegressor
            task_spec_args["objective"] = "regression"
        else:
            raise ValueError(
                "Task type in config.json must be one of ['classification', 'regression']")
        model = ModelClass(
            max_depth=config["depth"],
            n_estimators=config["num_trees"],
            num_leaves=256,
            reg_lambda=1,
            n_jobs=-1,
            **task_spec_args
        )

    # Train model
    train_start_time = time.time()
    model.fit(x, y)
    train_end_time = time.time()
    print(
        f"Time taken to train the model: {calculate_time(train_start_time, train_end_time)}")

    # Compute metrics
    if config[DATASET]["type"] == "classification":
        metrics_method = metrics.classification_report
    else:
        metrics_method = metrics.mean_squared_error
    print(metrics_method(df_train[y_col], model.predict(df_train[x_col])))

    # Save the model using joblib
    joblib_time_start = time.time()
    # TODO: For LightGBM, use model.save_model equivalent to save to .txt file along with this.
    joblib.dump(model, relative2abspath(
        "models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.pkl"))
    # TODO: If LightGBM, one option to convert to LLVM is to store the txt file. Else, read the pkl, then store it as txt file in the converter itself.
    joblib_time_end = time.time()
    print(
        f"Time taken to save model using joblib: {calculate_time(joblib_time_start, joblib_time_end)}")

    if MODEL == 'xgboost':
        save_model_time_start = time.time()
        model.save_model(relative2abspath(
            "models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.model"))
        save_model_time_end = time.time()
        print(
            f"Time taken to save model using joblib: {calculate_time(save_model_time_start, save_model_time_end)}")


if __name__ == "__main__":
    config = json.load(open(relative2abspath("config.json")))
    parse_arguments(config)
    print(f"DEPTH: {config['depth']}")
    print(f"TREES: {config['num_trees']}")
    df_train = fetch_data(DATASET, config, "train")

print(f"Number of training examples: {len(df_train)}")
train(config, df_train)
