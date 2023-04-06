from model_helper import *
import argparse
import csv
import os
import json
import time
import numpy as np
import joblib
import treelite_runtime
import warnings
warnings.filterwarnings('ignore')


# Default arguments
DATASET = None
MODEL = None
FRAMEWORK = None


def parse_arguments(config):
    global DATASET, MODEL, FRAMEWORK
    parser = argparse.ArgumentParser(description="""
        Parse arguments for test_model.py
        Usage: python3 ./test_model.py DATASET MODEL FRAMEWORK QUERY_SIZE BATCH_SIZE [gpu/cpu]
        For Sklearn, TreeLite, ONNX, only QUERY_SIZE is used to split the inference into multiple queries, and BATCH_SIZE will not be used.
        For other platforms, both QUERY_SIZE and BATCH_SIZE will be used.
    """)
    parser.add_argument(
        "-d", "--dataset", required=True, type=str, 
        choices=[
            'higgs', 
            'airline_regression', 
            'airline_classification', 
            'fraud', 
            'year', 
            'epsilon', 
            'bosch', 
            'covtype', 
            'criteo',
            'tpcxai_fraud'],
        help="Dataset to be tested.")
    parser.add_argument(
        "-m", "--model", required=True, type=str,
        choices=['randomforest', 'xgboost', 'lightgbm'],
        help="Model name. Choose from ['randomforest', 'xgboost', 'lightgbm']")
    parser.add_argument(
        "-f", "--framework", required=True, type=str,
        choices=[
            'Sklearn',
            'TreeLite',
            'HummingbirdPytorchCPU',
            'HummingbirdTorchScriptCPU',
            'HummingbirdTVMCPU',
            'TFDF',
            'ONNXCPU',
            'LightGBM',
            'Lleaves',
            'HummingbirdPytorchGPU',
            'HummingbirdTorchScriptGPU',
            'ONNXGPU',
            'HummingbirdTVMGPU',
            'NvidiaFILGPU',
            'XGBoostGPU',
            'Spark'
        ],
        help="Framework to run the decision forest model.")
    parser.add_argument(
        "-t", "--num_trees", type=int, default=10,
        help="Number of trees in the model. [Default value is 10]")
    parser.add_argument(
        "-D", "--depth", type=int, default=8,
        help="Depth of trees [Default value is 8].")
    parser.add_argument("--batch_size", type=int,
                        help="Batch size for testing. For Sklearn, TreeLite, ONNX, batch_size will not be used.")
    parser.add_argument("--query_size", type=int,
                        help="Query size for testing.")
    parser.add_argument("--threads", type=int,
                        help="Number of threads for testing.")
    args = parser.parse_args()
    config['threads'] = args.threads if args.threads else -1
    config['depth'] = args.depth
    config['num_trees'] = args.num_trees
    check_argument_conflicts(args)

    DATASET = args.dataset
    MODEL = args.model
    FRAMEWORK = args.framework
    config['batch_size'] = config[DATASET]["batch_size"] if args.batch_size is None else args.batch_size
    config['query_size'] = config[DATASET]["query_size"] if args.query_size is None else args.query_size
    if FRAMEWORK != 'TFDF':
        assert config['batch_size'] == config['query_size'], "If framework is not TFDF, batch_size must equal to query_size"

    if DATASET == "year" or DATASET == "airline_regression":
        config['task_type'] = "regression"
    else:
        config['task_type'] = "classification"
    # Print arguments
    print(f"DATASET: {DATASET}")
    print(f"MODEL: {MODEL}")
    print(f"FRAMEWORK: {FRAMEWORK}")
    print(f"Query Size: {config['query_size']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Trees: {config['num_trees']}")
    print(f"Depth: {config['depth']}")
    print(f"Threads: {config['threads']}")
    return args


def load_data(config, time_consume):
    test_data = fetch_data(DATASET, config, "test", time_consume=time_consume)
    if isinstance(test_data, tuple):
        features, label = test_data
        return (features, label)
        
    y_col = config[DATASET]["y_col"]
    x_col = list(test_data.columns)
    x_col.remove(y_col)
    features = test_data[x_col].to_numpy(dtype=np.float32)
    label = test_data[y_col]
    return (features, label)


def load_sklearn_model(config, time_consume):
    print("LOADING: ", relative2abspath(
        "models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.pkl"))
    start_time = time.time()
    relative_path = relative2abspath(
        "models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.pkl")
    sklearnmodel = joblib.load(relative_path)
    sklearnmodel.set_params(verbose=0)
    sklearnmodel.set_params(n_jobs=config['threads'])
    if MODEL == 'xgboost': #TODO: current code can not make lightgbm run on custom nthreads
        nthread = config['threads'] if config['threads'] != -1 else os.cpu_count()
        sklearnmodel.set_params(nthread=nthread)
    load_time = time.time()
    sklearnmodel_loading_time = calculate_time(start_time, load_time)
    time_consume["sklearn loading time"] = sklearnmodel_loading_time
    print(f"Time Taken to load sklearn model: {sklearnmodel_loading_time}")
    return sklearnmodel


def load_spark_model(config,time_consume):
    from pyspark.ml.classification import RandomForestClassifier,RandomForestClassificationModel
    print("LOADING: ", relative2abspath(
        "models", f"{DATASET}_{MODEL}_{FRAMEWORK}_{config['num_trees']}_{config['depth']}"))
    relative_path = relative2abspath(
        "models", f"{DATASET}_{MODEL}_{FRAMEWORK}_{config['num_trees']}_{config['depth']}")
    start_time = time.time()
    model = RandomForestClassificationModel.load(relative_path)
    time_consume["spark model loading time"] = calculate_time(start_time, time.time())
    model.explainParams()
    return model



def test(*argv):
    if FRAMEWORK.endswith("GPU"):
        test_postprocess(*test_gpu(*argv))
    else:
        test_postprocess(*test_cpu(*argv))


def test_spark(*argv):
    if FRAMEWORK.endswith("GPU"):
        #TODO : Add gpu support
        print("GPU support not yet added")
    else:
        test_postprocess(*test_cpu_spark(*argv))


def test_cpu_spark(test_data, model, config, time_consume):
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    start_time = time.time()

    featureCols = test_data.schema.names
    featureCols = featureCols[1:]
    assembler = VectorAssembler(inputCols=featureCols, outputCol="features")
    test_data = assembler.transform(test_data)

    test_start_time = time.time()
    predictions = model.transform(test_data)
    # This is just for consistency and is not the actual inference time
    time_consume["inference time"] = calculate_time(test_start_time,time.time()) 

    start = time.time()
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions)
    print("Accuracy = %s" % (accuracy))
    print("Test Error = %s" % (1.0 - accuracy))
    print(f"Evaluation time : {calculate_time(start, time.time())}")
    
    total_framework_time = calculate_time(start_time, time.time())
    return (time_consume, 0, total_framework_time, config)


def test_cpu(features, label, sklearnmodel, config, time_consume):
    input_size = len(label)
    is_classification = config[DATASET]["type"] == "classification"
    batch_size = config['batch_size']
    query_size = config['query_size']
    if FRAMEWORK == "Sklearn":
        start_time = time.time()
        # scikit-learn will use all data in a query as one batch
        conversion_time = 0.0
        results = run_inference(FRAMEWORK, features, input_size, query_size,sklearnmodel.predict, time_consume, is_classification)
        write_data(FRAMEWORK, results, time_consume) 
        total_framework_time = calculate_time(start_time, time.time())

    elif FRAMEWORK == "TreeLite":
        start_time = time.time()
        libpath = relative2abspath(
            "models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.so")
        if config['threads'] == -1:
            predictor = treelite_runtime.Predictor(libpath, verbose=True)
        else:
            predictor = treelite_runtime.Predictor(libpath, verbose=True, nthread=config['threads'])
        conversion_time = calculate_time(start_time, time.time())
        results = run_inference(FRAMEWORK, features, input_size, query_size, predictor.predict, time_consume, is_classification)
        write_data(FRAMEWORK, results, time_consume)
        total_framework_time = calculate_time(start_time, time.time())

    # https://github.com/microsoft/hummingbird/blob/main/hummingbird/ml/convert.py#L447
    elif FRAMEWORK == "HummingbirdPytorchCPU":
        start_time = time.time()
        model = convert_to_hummingbird_model(
            sklearnmodel, "torch", features, batch_size, "cpu", config['threads'])
        conversion_time = calculate_time(start_time, time.time())
        results = run_inference(FRAMEWORK, features, input_size, query_size, model.predict, time_consume, is_classification)
        write_data(FRAMEWORK, results, time_consume)
        total_framework_time = calculate_time(start_time, time.time())

    elif FRAMEWORK == "HummingbirdTorchScriptCPU":
        start_time = time.time()
        model = convert_to_hummingbird_model(
            sklearnmodel, "torch.jit", features, batch_size, "cpu", config['threads'])
        conversion_time = calculate_time(start_time, time.time())

        def predict(batch):
            return model.predict(batch)

        results = run_inference(FRAMEWORK, features, input_size, query_size, predict, time_consume, is_classification)
        write_data(FRAMEWORK, results, time_consume)
        total_framework_time = calculate_time(start_time, time.time())

    elif FRAMEWORK == "HummingbirdTVMCPU":
        start_time = time.time()
        model = convert_to_hummingbird_model(
            sklearnmodel, "tvm", features, batch_size, "cpu", config['threads'])
        conversion_time = calculate_time(start_time, time.time())
        remainder_size = input_size % batch_size
        if remainder_size > 0:
            remainder_model = convert_to_hummingbird_model(
                sklearnmodel, "tvm", features, remainder_size, "cpu", config['threads'])

        def predict(batch, use_remainder_model):
            if use_remainder_model:
                return remainder_model.predict(batch)
            return model.predict(batch)

        results = run_inference(FRAMEWORK, features, input_size, query_size, predict, time_consume, is_classification)
        write_data(FRAMEWORK, results, time_consume)
        total_framework_time = calculate_time(start_time, time.time())

    elif FRAMEWORK == "TFDF":
        import tensorflow as tf
        import tensorflow_decision_forests as tfdf
        import external.scikit_learn_model_converter as scikit_learn_model_converter
        import xgboost_model_converter
        import shutil
        intermediate_write_path="intermediate_path"
        start_time = time.time()
        if MODEL == "randomforest":
            model = scikit_learn_model_converter.convert(
                sklearnmodel, intermediate_write_path=intermediate_write_path)
        else:
            model = xgboost_model_converter.convert(
                sklearnmodel, intermediate_write_path=intermediate_write_path)
        conversion_time = calculate_time(start_time, time.time())

        def predict(batch):
            batch = tf.constant(batch)
            return model.predict(batch, batch_size=batch_size)

        results = run_inference(FRAMEWORK, features, input_size, query_size, predict, time_consume, is_classification)
        write_data(FRAMEWORK, results, time_consume)
        total_framework_time = calculate_time(start_time, time.time())
        # Clean up
        shutil.rmtree(intermediate_write_path)

    elif FRAMEWORK == "ONNXCPU":
        import onnxruntime as rt
        # https://github.com/microsoft/onnxruntime-openenclave/blob/openenclave-public/docs/ONNX_Runtime_Perf_Tuning.md
        sess_opt = rt.SessionOptions()
        sess_opt.intra_op_num_threads = os.cpu_count() if config['threads'] == -1 else config['threads']
        sess_opt.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        start_time = time.time()
        relative_path = relative2abspath(
            "models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.onnx")
        sess = rt.InferenceSession(relative_path, providers=[
                                   'CPUExecutionProvider'], sess_options=sess_opt)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        conversion_time = calculate_time(start_time, time.time())

        def predict(batch):
            output = sess.run([label_name], {input_name: batch})[0]
            return output

        results = run_inference(FRAMEWORK, features, input_size, query_size, predict, time_consume, is_classification)
        write_data(FRAMEWORK, results, time_consume)
        total_framework_time = calculate_time(start_time, time.time())
    elif FRAMEWORK == "LightGBM":
        import lightgbm
        # LightGBM Model Conversion & Inference
        start_time = time.time()
        model_path = relative2abspath(
            "models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.txt")
        model = lightgbm.Booster(model_file=model_path)
        conversion_time = calculate_time(start_time, time.time())

        def predict(batch):
            return model.predict(batch)

        results = run_inference(FRAMEWORK, features, input_size, query_size, predict, time_consume, is_classification)
        write_data(FRAMEWORK, results, time_consume)
        total_framework_time = calculate_time(start_time, time.time())
    elif FRAMEWORK == "Lleaves":
        import lleaves
        # Lleaves Model Conversion & Inference
        start_time = time.time()
        model_path = relative2abspath(
            "models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.txt")
        model = lleaves.Model(model_file=model_path)
        model_cache_path = relative2abspath(
            "models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.elf")
        # NOTE: Using Cache because of the extremely long compilation times for 500, 1600 Trees Models.
        model.compile(cache=model_cache_path)
        conversion_time = calculate_time(start_time, time.time())

        def predict(batch):
            return model.predict(batch)

        results = run_inference(FRAMEWORK, features, input_size, query_size, predict, time_consume, is_classification)
        write_data(FRAMEWORK, results, time_consume)
        total_framework_time = calculate_time(start_time, time.time())
    else:
        raise ValueError(f"{FRAMEWORK} is not supported.")
    if config['task_type'] == "classification":
        find_accuracy(FRAMEWORK, label, results)
    else:
        find_MSE(FRAMEWORK, label, results)
    return (time_consume, conversion_time, total_framework_time, config)


def test_gpu(features, label, sklearnmodel, config, time_consume):
    print("Running GPU Test", FRAMEWORK)
    import torch
    import gc
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    input_size = len(label)
    is_classification = config[DATASET]["type"] == "classification"
    batch_size = config['batch_size']
    query_size = config['query_size']
    if FRAMEWORK == "HummingbirdPytorchGPU":
        import torch
        device = torch.device('cuda')
        start_time = time.time()
        relative_path = relative2abspath(
            "models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}_torch.pkl")
        model = convert_to_hummingbird_model(
            sklearnmodel, "torch", features, batch_size, "cuda", config['threads'])
        conversion_time = calculate_time(start_time, time.time())
        results = run_inference(FRAMEWORK, features, input_size, query_size, model.predict, time_consume, is_classification)
        write_data(FRAMEWORK, results, time_consume)
        total_framework_time = calculate_time(start_time, time.time())

    elif FRAMEWORK == "HummingbirdTorchScriptGPU":
        import hummingbird.ml as hml
        start_time = time.time()
        torch_data = features[0:query_size]
        model = hml.convert(sklearnmodel, "torch.jit", torch_data, "cuda")
        conversion_time = calculate_time(start_time, time.time())

        def predict(batch):
            return model.predict(batch)

        results = run_inference(FRAMEWORK, features, input_size, query_size, predict, time_consume, is_classification)
        write_data(FRAMEWORK, results, time_consume)
        total_framework_time = calculate_time(start_time, time.time())

    elif FRAMEWORK == "ONNXGPU":
        import onnxruntime as rt
        start_time = time.time()
        relative_path = relative2abspath(
            "models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.onnx")
        sess = rt.InferenceSession(relative_path, providers=[
                                   'CUDAExecutionProvider'])
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        conversion_time = calculate_time(start_time, time.time())

        def predict(batch):
            output = sess.run([label_name], {input_name: batch})[0]
            return output

        results = run_inference(FRAMEWORK, features, input_size, query_size, predict, time_consume, is_classification)
        write_data(FRAMEWORK, results, time_consume)
        total_framework_time = calculate_time(start_time, time.time())

    elif FRAMEWORK == "HummingbirdTVMGPU":
        start_time = time.time()
        model = convert_to_hummingbird_model(
            sklearnmodel, "tvm", features, batch_size, "cuda", config['threads'])
        conversion_time = calculate_time(start_time, time.time())
        remainder_size = input_size % batch_size
        if remainder_size > 0:
            remainder_model = convert_to_hummingbird_model(
                sklearnmodel, "tvm", features, remainder_size, "cuda", config['threads'])
        
        def predict(batch, use_remainder_model):
            if use_remainder_model:
                return remainder_model.predict(batch)
            return model.predict(batch)

        results = run_inference(FRAMEWORK, features, input_size, query_size, predict, time_consume, is_classification)
        write_data(FRAMEWORK, results, time_consume)
        total_framework_time = calculate_time(start_time, time.time())

    elif FRAMEWORK == "XGBoostGPU":
        if MODEL != 'xgboost':
            exit()
        start_time = time.time()
        # scikit-learn will use all data in a query as one batch
        conversion_time = 0.0
        sklearnmodel.set_params(predictor="gpu_predictor")  # NOT safe!
        sklearnmodel.set_params(n_jobs=-1)
        results = run_inference(FRAMEWORK, features, input_size, query_size, sklearnmodel.predict, time_consume, is_classification)
        write_data(FRAMEWORK, results, time_consume)
        total_framework_time = calculate_time(start_time, time.time())

    elif FRAMEWORK == "NvidiaFILGPU":
        from cuml import ForestInference
        start_time = time.time()
        model = None
        if MODEL == 'randomforest':
            model = ForestInference.load_from_sklearn(
                sklearnmodel, output_class=True, storage_type='auto')
        elif MODEL == "lightgbm":
            relative_path = relative2abspath(
                "models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.model")
            model = ForestInference.load(
                relative_path, output_class=True, storage_type='auto', model_type="lightgbm")
        elif MODEL == 'xgboost':
            relative_path = relative2abspath(
                "models", f"{DATASET}_{MODEL}_{config['num_trees']}_{config['depth']}.model")
            model = ForestInference.load(
                relative_path, output_class=True, storage_type='auto', model_type="xgboost")
        else:
            print(MODEL + " support will be added to " + FRAMEWORK)
            exit()

        # model = ForestInference.load_from_sklearn(sklearnmodel,output_class=True, storage_type='auto')
        conversion_time = calculate_time(start_time, time.time())
        results = run_inference(FRAMEWORK, features, input_size, query_size, model.predict, time_consume, is_classification)
        write_data(FRAMEWORK, results, time_consume)
        total_framework_time = calculate_time(start_time, time.time())

    else:
        raise ValueError(f"{FRAMEWORK} is not supported.")
    if config['task_type'] == "classification":
        find_accuracy(FRAMEWORK, label, results)
    else:
        find_MSE(FRAMEWORK, label, results)
    return (time_consume, conversion_time, total_framework_time, config)


def test_postprocess(time_consume, conversion_time, total_framework_time, config):
    # Print conversion time and total time used on framework
    print(f"Time Taken to convert {FRAMEWORK} model: {conversion_time}")
    print(f"TOTAL Time Taken for {FRAMEWORK}: {total_framework_time}")

    # Update output dictionary (time consumption at each step)
    time_consume["conversion time"] = conversion_time
    time_consume["total framework time"] = total_framework_time

    # Save output dictionary to csv
    filename_suffix = "GPU" if FRAMEWORK.endswith("GPU") else "CPU"
    num_trees, depth = config['num_trees'], config['depth']
    output_file_path = relative2abspath(
        "results", f"{DATASET}_{num_trees}_{depth}_{filename_suffix}.csv")
    file_exists = os.path.isfile(output_file_path)
    with open(output_file_path, 'a') as csvfile:
        headers = list(time_consume.keys())
        writer = csv.DictWriter(csvfile, delimiter=',',
                                lineterminator='\n', fieldnames=headers)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow(time_consume)


if __name__ == "__main__":
    print("\n\n\n==============EXPERIMENT STARTING=========================")
    config = json.load(open(relative2abspath("config.json")))
    parse_arguments(config)

    time_consume = {
        "query size": config['query_size'],
        "batch_size": config['batch_size'],
        "model": MODEL,
        "framework": FRAMEWORK}
    if FRAMEWORK == "Spark":
        if not validate_spark_params(DATASET, MODEL):
            exit()
        spark = get_spark_session(config["spark"])
        from sparkmeasure import StageMetrics
        stagemetrics = StageMetrics(spark)
        stagemetrics.begin()
        test_data = fetch_data_spark(spark, DATASET, config, "test")
        print((test_data.count(), len(test_data.columns))) 
        model = load_spark_model(config, time_consume)
        test_spark(test_data, model, config, time_consume)
        stagemetrics.end()
        stagemetrics.print_report()
        spark.stop()
    else:
        features, label = load_data(config, time_consume)
        sklearnmodel = load_sklearn_model(config, time_consume)
        test(features, label, sklearnmodel, config, time_consume)
    print("==============EXPERIMENT ENDING=========================\n")
