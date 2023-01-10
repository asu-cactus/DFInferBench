import warnings
warnings.filterwarnings('ignore')

import xgboost
import joblib
import time
import json
import os
import argparse
from model_helper import *

DATASET = None
MODEL = None
FRAMEWORKS = None

def parse_arguments(config):
    # check_argument_conflicts(args)  # TODO: Move this function from the bottom to here, after checking with Prof.
    global DATASET, MODEL, FRAMEWORKS
    parser = argparse.ArgumentParser(description='Arguments for train_model.')
    parser.add_argument("-d", "--dataset", required=True, type=str, choices=['higgs', 'airline_classification', 'airline_regression', 'fraud', 'year', 'epsilon', 'bosch', 'covtype', 'tpcxai_fraud', 'criteo'],
        help="Dataset to be trained. Choose from ['higgs', 'airline_classification', 'airline_regression', 'fraud', 'year', 'epsilon', 'bosch', 'covtype', 'tpcxai_fraud', 'criteo]")

    parser.add_argument("-m", "--model", required=True, type=str, choices=['randomforest', 'xgboost', 'lightgbm'],
        help="Model name. Choose from ['randomforest', 'xgboost', 'lightgbm']")
    parser.add_argument("-f", "--frameworks", required=True, type=str,
        help="Zero to multiple values from ['pytorch', 'torch', 'tf-df', 'onnx', 'treelite', 'lleaves', 'netsdb', 'xgboost', 'lightgbm'], seperated by ','")
    parser.add_argument(
        "-t", "--num_trees", type=int, default=10,
        choices=[10, 500, 1600],
        help="Number of trees in the model. Choose from ['10', '500', '1600']")
    parser.add_argument(
        "-D", "--depth", type=int, default=8,
        choices=[6, 8],
        help="Choose from [6, 8].")
    args = parser.parse_args()
    config['depth'] = args.depth
    config['num_trees'] = args.num_trees
    if args.dataset:
        DATASET = args.dataset.lower()
    if args.model:
        MODEL = args.model.lower()
    if args.frameworks:
        framework_options = ['pytorch', 'torch', 'tf-df', 'onnx', 'treelite', 'lleaves', 'netsdb', 'xgboost', 'lightgbm']
        for framework in args.frameworks.lower().split(","):
            if framework not in framework_options:
                raise ValueError(f"Framework {framework} is not supported. Choose from ['pytorch', 'torch', 'tf-df', 'onnx', 'treelite', 'lleaves', 'netsdb', 'xgboost', 'lightgbm']")
        FRAMEWORKS = args.frameworks  # TODO: Better to store these as a List? Instead of as a string.
    check_argument_conflicts(args)  # TODO: Maybe, this is good to do it at the beginning of function itself?
    print(f"DATASET: {DATASET}")
    print(f"MODEL: {MODEL}")
    print(f"FRAMEWORKS: {FRAMEWORKS}")

    return args

def convert_to_pytorch_model(model, config):
    import hummingbird.ml as hml
    pytorch_time_start = time.time()
    model = hml.convert(model, 'pytorch')
    pytorch_time_start = time.time()
    convert_time = calculate_time(pytorch_time_start, pytorch_time_start)
    print(f"Time Taken to convert HummingbirdPyTorch: {convert_time}")
    
    pytorch_time_start = time.time()
    model.save(relative2abspath("models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}_pytorch.pkl"))
    pytorch_time_start = time.time()
    save_time = calculate_time(pytorch_time_start, pytorch_time_start)
    print(f"Time Taken to save HummingbirdPyTorch: {save_time}")
    print(f"{DATASET} {MODEL} pytorch {config['num_trees']} total time: {convert_time + save_time}")

def convert_to_torch_model(model, config):
    import hummingbird.ml as hml
    import torch
    torch_time_start = time.time()
    model = hml.convert(model, 'torch')
    torch_time_end = time.time()
    convert_time = calculate_time(torch_time_start, torch_time_end)
    print(f"Time taken to convert torch model {convert_time}")
    
    torch_time_start = time.time()
    torch.save(model, relative2abspath("models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}_torch.pkl"))
    torch_time_end = time.time()
    save_time = calculate_time(torch_time_start, torch_time_end)
    print(f"Time taken to save torch model {save_time}")
    print(f"{DATASET} {MODEL} torch {config['num_trees']} total time: {convert_time + save_time}")

def convert_to_tf_df_model(model, config):
    # Converting to TF-DF model
    import tensorflow as tf
    import external.scikit_learn_model_converter as sk2tfdf_converter  # TODO: Can we rename this file or move it, so that it is clear this is only meant for TFDF, and not used anywhere else.
    import xgboost_model_converter as xgb2tfdf_converter # TODO: Can we rename this file or move it, so that it is clear this is only meant for TFDF, and not used anywhere else.
    import shutil

    intermediate_write_path="intermediate_path"
    if MODEL == "randomforest":
        tfdf_time_start = time.time()
        tensorflow_model = sk2tfdf_converter.convert(model,  intermediate_write_path=intermediate_write_path)
        tfdf_time_end = time.time()
        convert_time = calculate_time(tfdf_time_start, tfdf_time_end)
        print(f"Time taken to convert tfdf randomforest model {convert_time}")

        libpath = relative2abspath("models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}_tfdf")
        tfdf_time_start = time.time()    
        tf.saved_model.save(obj=tensorflow_model, export_dir=libpath)
        tfdf_time_end = time.time()
        save_time = calculate_time(tfdf_time_start, tfdf_time_end)
        print(f"Time taken to save tfdf randomforest model {save_time}")

    elif MODEL == "xgboost":
        tfdf_time_start = time.time()
        tensorflow_model = xgb2tfdf_converter.convert(model, intermediate_write_path=intermediate_write_path)
        tfdf_time_end = time.time()
        convert_time = calculate_time(tfdf_time_start, tfdf_time_end)
        print(f"Time taken to convert tfdf xgboost model {convert_time}")

        libpath = relative2abspath("models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}_tfdf")
        tfdf_time_start = time.time()
        tf.saved_model.save(obj=tensorflow_model, export_dir=libpath)
        tfdf_time_end = time.time()
        save_time = calculate_time(tfdf_time_start, tfdf_time_end)
        print(f"Time taken to save tfdf xgboost model {save_time}")
    
    else:
        raise ValueError(f"lightgbm is currently not supported for tf-df.")
    print(f"{DATASET} {MODEL} tf-df {config['num_trees']} total time: {convert_time + save_time}")
    # Clean up
    shutil.rmtree(intermediate_write_path)

    

def convert_to_onnx_model(model, config):
    from skl2onnx.common.data_types import FloatTensorType
    from skl2onnx import convert_sklearn
    import onnxmltools
   
    initial_types = [('float_input', FloatTensorType([None, config[DATASET]['num_features']]))]
    if MODEL == "randomforest":
        onnx_time_start = time.time()
        model_onnx = convert_sklearn(model,'pipeline_randomforest', initial_types=initial_types) 
        onnx_time_end = time.time()
        convert_time = calculate_time(onnx_time_start, onnx_time_end)
        print(f"Time taken to convert onnx: {convert_time}") 
        onnx_write_time_start = time.time()
        with open(relative2abspath("models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.onnx"), "wb") as f:
            f.write(model_onnx.SerializeToString())
        onnx_write_time_end = time.time()
        save_time = calculate_time(onnx_write_time_start, onnx_write_time_end)
        print(f"Time taken to write onnx model {save_time}")

    elif MODEL == "xgboost":
        onnx_time_start = time.time()
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_types)
        onnx_time_end = time.time()
        convert_time = calculate_time(onnx_time_start, onnx_time_end)
        print(f"Time taken to convert onnx: {convert_time}")
        
        model_save_path = relative2abspath("models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.onnx")
        onnx_write_time_start = time.time()
        onnxmltools.utils.save_model(onnx_model, model_save_path)
        onnx_write_time_end = time.time()
        save_time = calculate_time(onnx_write_time_start, onnx_write_time_end)
        print(f"Time taken to write onnx model: {save_time}")
    elif MODEL == "lightgbm":
        onnx_time_start = time.time()
        onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_types)
        onnx_time_end = time.time()
        convert_time = calculate_time(onnx_time_start, onnx_time_end)
        print(f"Time taken to convert onnx: {convert_time}")
        
        model_save_path = relative2abspath("models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.onnx")
        onnx_write_time_start = time.time()
        onnxmltools.utils.save_model(onnx_model, model_save_path)
        onnx_write_time_end = time.time()
        save_time = calculate_time(onnx_write_time_start, onnx_write_time_end)
        print(f"Time taken to write onnx model: {save_time}")

    print(f"{DATASET} {MODEL} onnx {config['num_trees']} total time: {convert_time + save_time}")

def convert_to_treelite_model(model, config):
    import treelite
    treelite_time_start = time.time()
    if MODEL == "randomforest":
        treelite_model = treelite.sklearn.import_model(model)
    elif MODEL == "xgboost":
        treelite_model = treelite.Model.from_xgboost(model.get_booster())
    elif MODEL == "lightgbm":
        treelite_model = treelite.Model.from_lightgbm(model.booster_)
    treelite_time_end = time.time()
    convert_time = calculate_time(treelite_time_start, treelite_time_end)
    print(f"Time taken to convert treelite model {convert_time}")

    treelite_time_start = time.time()
    libpath = relative2abspath("models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.so")
    treelite_model.export_lib(toolchain='gcc', libpath=libpath, verbose=True, params={"parallel_comp":os.cpu_count()})
    treelite_time_end = time.time()
    save_time = calculate_time(treelite_time_start, treelite_time_end)
    print(f"Time taken to write treelite model: {save_time}")
    print(f"{DATASET} {MODEL} treelite {config['num_trees']} total time: {convert_time + save_time}")


def convert_to_lleaves_model(model, config):
    # TODO: Option 1: Read pkl, write to txt, read txt model file from lleaves package.
    # TODO: Option 2: Write txt during model saving itself in train_model.py, and read it.
    # Implementing Option1 as it can be a drop-in to existing implmentation
    import lleaves
    if MODEL == 'lightgbm':
        lleaves_start_time = time.time()
        model_path = relative2abspath("models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.txt")
        model.booster_.save_model(model_path)
        lleaves_model = lleaves.Model(model_file=model_path)
        model_cache_path = relative2abspath("models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.elf")
        lleaves_model.compile(cache=model_cache_path)  # NOTE: Same logic to be used for testing. This time, the elf file is loaded instead of compiled.
        lleaves_end_time = time.time()
        total_time = calculate_time(lleaves_start_time, lleaves_end_time)
        print(f"{DATASET} {MODEL} lleaves {config['num_trees']} total time: {total_time}")
    else:
        print(f"LLeaves is only supported for LightGBM at the moment. Does not support {MODEL}.")


def convert_to_netsdb_model(model, config):
    from sklearn.tree import export_graphviz
    import os
    
    netsdb_model_path = os.path.join("models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}_netsdb")
    if not os.path.exists(netsdb_model_path):
        os.mkdir(netsdb_model_path)
    
    start_time = time.time()
    if MODEL == "randomforest":
        for index, model in enumerate(model.estimators_):
            output_file_path = os.path.join(netsdb_model_path, f'{index}.txt')
            data = export_graphviz(model, class_names=True)
            with open(output_file_path, 'w') as f:
                f.write(data) 

    elif MODEL == "xgboost":
        num_trees = config['num_trees']
        for index in range(num_trees):
            output_file_path = os.path.join(netsdb_model_path, f'{index}.txt') 
            data = xgboost.to_graphviz(model, num_trees=index)
            with open(output_file_path, 'w') as f:
                f.write(str(data)) 
   
    elif MODEL == "lightgbm":
        df = model.booster_.trees_to_dataframe()
        for index, tree in df.groupby("tree_index"):
            output_file_path = os.path.join(netsdb_model_path, f"{index}.csv")
            tree.to_csv(output_file_path, index=False)

        # for index in range(config['num_trees']):
        #     output_file_path = os.path.join(netsdb_model_path, str(index)+'.txt')
        #     data = lightgbm.create_tree_digraph(model, tree_index=index)
        #     with open(output_file_path, 'w') as f:
        #         f.write(str(data))
    end_time = time.time()
    total_time = calculate_time(start_time, end_time)
    print(f"{DATASET} {MODEL} netsdb {config['num_trees']} total time: {total_time}")
        

def convert_to_xgboost_model(model,config):
    # clf = joblib.load(relative2abspath('models',model_file))
    # clf.save_model(relative2abspath('models',model_file.split('.')[0]+'.model'))
    new_model = f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.model"
    if (MODEL == "xgboost") and (new_model not in os.listdir("models")):
        model_path = relative2abspath("models", new_model)
        model.save_model(model_path)
    else:
        raise("Model not xgboost or model already exists")


def convert_to_lightgbm_model(model,config):
    new_model = f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.model"
    if (MODEL == "lightgbm") and (new_model not in os.listdir("models")):
        model_path = relative2abspath("models", new_model)
        model.booster_.save_model(model_path)
    else:
        raise("Model not lightgbm or model already exists")

def convert(model, config):
    def print_logs(function,model,config,framework_name):
        border = '-'*30
        print(border)
        print(f'Converting model to {DATASET} {MODEL} {framework_name} {config["num_trees"]}...')
        print(border)
        function(model,config)
        print(border)
        print(f'Converted model to {DATASET} {MODEL} {framework_name} {config["num_trees"]}')
        print(border + '\n\n')

    frameworks = FRAMEWORKS.lower().split(",")
    if "pytorch" in frameworks:
        print_logs(convert_to_pytorch_model,model,config,"PyTorch")
    if "torch" in frameworks:
        print_logs(convert_to_torch_model,model,config,"Torch")
    if "tf-df" in frameworks or 'tfdf' in frameworks:
        print_logs(convert_to_tf_df_model,model,config,"TF-DF")
    if "onnx" in frameworks:
        print_logs(convert_to_onnx_model,model,config,"ONNX")
    if "treelite" in frameworks:
        print_logs(convert_to_treelite_model,model,config,"TreeLite")
    if "lleaves" in frameworks:
        print_logs(convert_to_lleaves_model,model,config,"Lleaves")
    if "netsdb" in frameworks:
        print_logs(convert_to_netsdb_model, model, config, "netsdb")
    if "xgboost" in frameworks:
        print_logs(convert_to_xgboost_model, model, config, "xgboost")
    if "lightgbm" in frameworks:
        print_logs(convert_to_lightgbm_model, model, config, "lightgbm")

def load_model(config):
    model = joblib.load(relative2abspath("models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.pkl"))
    return model

if __name__ ==  "__main__":
    config = json.load(open(relative2abspath("config.json")))
    parse_arguments(config)
    model = load_model(config)
    convert(model, config)
