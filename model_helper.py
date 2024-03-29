import pickle
import time
import os
import numpy as np
import math
from sklearn.metrics import classification_report, mean_squared_error
from hummingbird.ml import constants
from hummingbird.ml import convert, convert_batch

dataset_folder = "dataset/"
SPARK_DATASETS = ["higgs","fraud","epsilon","airline_classification","year","criteo"]

def calculate_time(start_time, end_time):
    diff = (end_time-start_time)*1000
    return diff


def load_data_from_pickle(dataset, config, suffix, time_consume):
    start_time = time.time()
    pkl_path = relative2abspath(
        dataset_folder, f"{config[dataset]['filename']}_{suffix}.pkl")
    dataframe = pickle.load(open(pkl_path, "rb"))
    end_time = time.time()
    data_loading_time = calculate_time(start_time, end_time)
    if time_consume is not None:
        time_consume["data loading time"] = data_loading_time
    print(
        f"Time Taken to load {dataset} as a dataframe is: {data_loading_time}")
    return dataframe

def fetch_criteo(suffix, time_consume):
    from sklearn import datasets
    start_time = time.time()
    path = relative2abspath(dataset_folder, "criteo.kaggle2014.svm", f"{suffix}.txt.svm")
    x, y = datasets.load_svmlight_file(path, dtype=np.float32)
    data_loading_time = calculate_time(start_time,time.time())
    if time_consume is not None:
        time_consume["data loading time"] = data_loading_time
    y = y.astype(np.int8, copy=False)
    return (x, y)

def fetch_data(dataset, config, suffix, time_consume=None):
    if dataset == "criteo":        
        return fetch_criteo(suffix, time_consume)
    print("LOADING " + dataset + " " + suffix)
    
    try:
        import connectorx as cx
        import psycopg2
        pgsqlconfig = config["pgsqlconfig"]
        datasetconfig = config[dataset]
        query = datasetconfig["query"]+"_"+suffix
        dbURL = "postgresql://"+pgsqlconfig["username"]+":"+pgsqlconfig["password"] + \
            "@"+pgsqlconfig["host"]+":" + \
                pgsqlconfig["port"]+"/"+pgsqlconfig["dbname"]
        # print(dbURL)
        # print(query)
        start_time = time.time()
        dataframe = cx.read_sql(dbURL, query)
        if dataset == 'epsilon':
            unpacked = zip(*list(dataframe['row'].values))
            for i in range(1, 2001):
                dataframe[i] = next(unpacked)

            dataframe.drop('row', axis=1, inplace=True)
            # dataframe['row'] = dataframe['row'].apply(lambda row:np.array(row))
        end_time = time.time()
        data_loading_time = calculate_time(start_time, end_time)
        if time_consume is not None:
            time_consume["data loading time"] = data_loading_time
        print(
            f"Time Taken to load {dataset} as a dataframe is: {data_loading_time}")

        if datasetconfig["type"] == "classification":
            dataframe = dataframe.astype({datasetconfig["y_col"]: int})
        return dataframe
    except psycopg2.Error as e:
        print("Postgres Database error: " + e + "/n")


def validate_spark_params(dataset, model):
    if model != "randomforest" or dataset not in SPARK_DATASETS:
        print(f"Invalid params for spark. Models supported : randomforest, dataset supported : {SPARK_DATASETS}")
        return False
    return True


def fetch_criteo_spark(spark,  config, suffix, time_consumed):
    path = relative2abspath(dataset_folder, "criteo.kaggle2014.svm", f"{suffix}.txt.svm")
    df = spark.read.format("libsvm").option("numFeatures",config["criteo"]["num_features"]).load(path)
    df = df.repartition(spark.sparkContext.defaultParallelism)
    df.cache().count()
    return df

def fetch_data_spark(spark, dataset, config, suffix, time_consumed=None): 
   
    if dataset == "criteo":
        return fetch_criteo_spark(spark,config,suffix, time_consumed)

    pgsqlconfig = config["pgsqlconfig"]
    datasetconfig = config[dataset]
    query = datasetconfig["query"]+"_"+suffix
    dbURL =  "jdbc:postgresql://" + pgsqlconfig["host"]+":" + pgsqlconfig["port"]+"/"+pgsqlconfig["dbname"]
    start_time = time.time()
    try:
        df = spark.read \
            .format("jdbc") \
            .option("url",  dbURL) \
            .option("query", query) \
            .option("user", pgsqlconfig["username"]) \
            .option("driver", "org.postgresql.Driver") \
            .option("password", pgsqlconfig["password"]) \
            .load()

        if dataset == "epsilon":
            length = len(df.head()["row"])
            df = df.select(['label'] + [df.row[x] for x in range(length)])
        
        df = df.repartition(spark.sparkContext.defaultParallelism)
        df.cache().count()
        
        end_time = time.time()
        data_loading_time = calculate_time(start_time, end_time)
        if time_consumed is not None:
            time_consumed["data loading time"] = data_loading_time
        print(
            f"Time Taken to load {dataset} as a dataframe is: {data_loading_time}")    
        return df
    except Exception as e:
        print(e)
        

def get_spark_session(conf):
    from pyspark.sql.session import SparkSession
    from pyspark import SparkContext, SparkConf
    import psutil

    memory_gb = int(psutil.virtual_memory()[1]/1000000000) 
    spark_conf = SparkConf().setAll(list(conf.items()))
    spark_conf.set("spark.driver.memory", str(memory_gb) + "g")
    sc = SparkContext(conf = spark_conf).getOrCreate("DFInferBench")
    return SparkSession(sc)

def convert_to_hummingbird_model(model, backend, test_data, batch_size, device, nthreads):
    remainder_size = test_data.shape[0] % batch_size
    extra_config = {constants.N_THREADS: os.cpu_count() if nthreads == -1 else nthreads}
    batch_data = test_data[0:batch_size]
    if backend == "tvm":
        model = convert(model, backend, batch_data,
                        device=device, extra_config=extra_config)
    else:
        model = convert_batch(model, backend, batch_data, remainder_size=remainder_size,
                              device=device, extra_config=extra_config)
    return model

def run_inference(framework, features, input_size, query_size, predict, time_consume, is_classification):
    start_time = time.time()
    results = []
    iterations = math.ceil(input_size/query_size)
    if framework == "TreeLite":
        import treelite_runtime
        def aggregate_function():
            def append(output):
                results.append(output)

            def extend(output):
                results.extend(output)
            return append if query_size == 1 else extend

        aggregate_func = aggregate_function()
        for i in range(iterations):
            query_data = treelite_runtime.DMatrix(
                features[i*query_size:(i+1)*query_size])
            output = predict(query_data)
            if is_classification:
                output = np.where(output > 0.5, 1, 0)
            aggregate_func(output)
    elif framework == "TFDF":
        for i in range(iterations):
            query_data = features[i*query_size:(i+1)*query_size]
            output = predict(query_data).flatten()
            if is_classification:
                output = np.where(output > 0.5, 1, 0)
            results.extend(output)
    elif framework == "HummingbirdTVMCPU" or framework == "HummingbirdTVMGPU":
        for i in range(iterations):
            query_data = features[i*query_size:(i+1)*query_size]
            output = predict(query_data, len(query_data) != query_size)
            results.extend(output)
    elif framework in {"Lleaves", "LightGBM"}:
        for i in range(iterations):
            query_data = features[i*query_size:(i+1)*query_size]
            output = predict(query_data)
            if is_classification:
                output = np.where(output > 0.5, 1, 0)
            results.extend(output)
    else:
        for i in range(iterations):
            query_data = features[i*query_size:(i+1)*query_size]
            # converting sparse to dense
            # query_data = query_data.todense()
            output = predict(query_data)
            results.extend(output)

    inference_time = calculate_time(start_time, time.time())
    time_consume["inference time"] = inference_time
    print(f"Time Taken to predict on {framework} is {inference_time}")
    return results


def write_data(framework, results, time_consume):
    start_time = time.time()
    # arr = np.array(results)
    # df = pd.DataFrame(arr)
    # df.to_csv(os.path.join('results','results.txt'), index=False)
    # print(results[0:10])
    with open(os.path.join('results', 'results.txt'), 'w') as f:
        for item in results:
            f.write("%s\n" % int(item))

    writing_time = calculate_time(start_time, time.time())
    time_consume["result writing time"] = writing_time
    print(
        f"Time Taken to write results to a text file for {framework} is {writing_time}")


def find_accuracy(framework, y_actual, y_pred):
    print("Classification Report", framework)
    print(classification_report(y_actual, y_pred))
    print("################")


def find_MSE(framework, y_actual, y_pred):
    print("Regression Report", framework)
    print(f"MSE: {mean_squared_error(y_actual, y_pred)}")
    print("################")


def relative2abspath(path, *paths):
    return os.path.join(
        os.path.dirname(__file__),
        path,
        *paths
    )

def check_argument_conflicts(args):
    model = args.model.lower()
    if hasattr(args, "frameworks"):
        frameworks = args.frameworks.lower().split(",")
        if "lleaves" in frameworks and not model == "lightgbm":
            raise ValueError(
                "LLeaves Framework supports compilation of LightGBM Models.")

    dataset = args.dataset.lower()
    if dataset == "bosch" and model == "randomforest":
        raise ValueError(
            "Sklearn implementation of randomforest algorithm does not support datasets with missing values.")
