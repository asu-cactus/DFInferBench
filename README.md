# Decision Forest Inference Benchmark Suite

We provide a comprehensive benchmark framework for decision forest model inferences. 

The framework covers three algorithms: RandomForest, XGBoost, and LightGBM. 

The framework supports most of the popular decision forest inference platforms, including Scikit-Learn, XGBoost, LightGBM, ONNX, HummingBird, TreeLite, lleaves, TensorFlow TFDF, cu-ml, and FIL.

The framework also supports multiple well-known workloads, including Higgs, Airline, TPCx-AI fraud detection, Fraud, Year, Bosch, Epsilon, and Crieto.

<!-- toc -->
- [System requirements](#system-requirements)
- [Installation](#installation)
  - [PostgreSQL](#postgresql)
  - [Platforms and other tools](#platforms-and-other-tools)
- [Run benchmark](#run-benchmark)
  - [Platforms with a Python interface](#platforms-with-a-python-interface)
  - [Single thread experiments](#single-thread-experiments)
  - [Yggdrasil](#yggdrasil)
  - [netsDB](#netsdb)
- [Datasets](#datasets)
  - [TPCxAI](#generating-synthetic-data-using-tpcxai)

<!-- tocstop -->

## System requirements

## Installation
### PostgreSQL
We used PostgreSQL to manage data for non-netsDB platforms. Please refer to [here](https://www.postgresql.org/download/) to install it. We used the default username and password in our code. Please either leave it as default, or modify the username and password in the `config.json` file.

### Platforms and other tools
It is recommended to use [conda](https://docs.conda.io/en/latest/miniconda.html) to manage your environment because one of the required package, TVM, is much easier to be installed using `conda`. TVM also recommends a Python version of 3.7.X+ or 3.8.X+, so we also recommend to create a conda virtual environment with Python 3.7 or 3.8. 
```bash
conda create -n [env-name] python=3.8
```
Then activate the virtual environment.
```bash
conda activate [env-name]
```

Install some useful tools.
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```

It is important to install TVM first, because it might uninstall some other packages. We provide an easy way to install TVM in the following code block, which is tested in our environment. For other installation methods and other details, please refer to [here](https://tvm.apache.org/docs/install/from_source.html). 

(Note: If you choose to build TVM with CMake, you may meet error "collect2: fatal error: cannot find ‘ld’", try to change the linker, e.g., you may change 'fuse-ld=lld' to 'fuse-ld=gold' in the ./CMakeFiles/tvm_runtime.dir/link.txt, ./CMakeFiles/tvm.dir/link.txt, and ./CMakeFiles/tvm_allvisible.dir/link.txt.
Remember to run 'make install' from the build directory after successfully compiling tvm to shared libraries.)
```bash
git clone --recursive https://github.com/apache/tvm tvm
# Update the current conda environment with the dependencies specified by the yaml
conda env update --file conda/build-environment.yaml
# Build TVM
conda build --output-folder=conda/pkg  conda/recipe
# Run conda/build_cuda.sh to build with cuda enabled
conda install tvm -c ./conda/pkg
```

Install Nvidia cuML ([here for instructions](https://github.com/rapidsai/cuml/blob/branch-23.02/BUILD.md)) to support Nvidia FIL.

Install Python packages. The command below install all packages for our benchmarck, but feel free to select some from these packages if you only want to run a subset of these frameworks.
```bash
pip install scikit-learn xgboost lightgbm pandas onnxruntime onnxruntime-gpu skl2onnx onnxmltools torch tensorflow tensorflow_decision_forests hummingbird-ml[extra] treelite treelite_runtime connectorx lleaves catboost py-xgboost-gpu pyyaml psycopg2-binary plotly
```

## Run benchmark

### Platforms with a Python interface


To run a certain experiment

Datasets: higgs

Classifiers: xgboost, randomforest, lightgbm

Frameworks: Sklearn, ONNXCPU, TreeLite, HummingbirdPytorchCPU, HummingbirdTorchScriptCPU, HummingbirdTVMCPU, LightGBM, Lleaves

```
python data_processing.py -d [dataset]

python train_model.py -d [dataset] -m [model]

python convert_trained_model_to_framework.py -d [dataset] -m [model] -f [frameworks]

python test_model.py -d [dataset] -m [model] -f [framework] --batch_size [batch_size] --query_size [query_size]
```

**Examples**
```
python data_processing.py -d higgs

python train_model.py -d higgs -m randomforest --num_trees 10
python train_model.py -d higgs -m xgboost --num_trees 10
python train_model.py -d higgs -m lightgbm --num_trees 10

python convert_trained_model_to_framework.py -d higgs -m randomforest -f pytorch,torch,tf-df,onnx,netsdb --num_trees 10
python convert_trained_model_to_framework.py -d higgs -m xgboost -f pytorch,torch,onnx,treelite,tf-df,netsdb --num_trees 10
python convert_trained_model_to_framework.py -d higgs -m lightgbm -f pytorch,torch,onnx,treelite,lightgbm,lleaves,netsdb --num_trees 10

python test_model.py -d higgs -m xgboost -f TreeLite --batch_size 1000 --query_size 1000 --num_trees 10
```
or modify and run `run_test.sh`
```
nohup ./run_test.sh &> ./results/test_output.txt &
```


### Single thread experiments

Add `threads` argument to `python test_model.py`, for example, 
```
python test_model.py -d higgs -m xgboost -f TreeLite --batch_size 100000 --query_size 100000 --num_trees 10 --threads 1
```
### Yggdrasil
To run Yggdrasil, which implements QuickScorer algorithm, first download the binaries from https://github.com/google/yggdrasil-decision-forests/releases to a separate directory and unzip it. Next, put the dataset and model to the right place. Yggdrasil requires the dataset has a header to generate meta data, which you don't need to care much about, but you should manually add a header to the first line of the dataset. For example, add a header to the fraud dataset, run 
```
sed -i '1i feature_0,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9,feature_10,feature_11,feature_12,feature_13,feature_14,feature_15,feature_16,feature_17,feature_18,feature_19,feature_20,feature_21,feature_22,feature_23,feature_24,feature_25,feature_26,feature_27,label' datasets/creditcard_test.csv
```
Then run the benchmark:
```
./benchmark_inference --dataset=csv:datasets/creditcard_test.csv --model=models/fraud_xgboost_500_6_tfdf/assets/ --generic=false --num_runs=1 --batch_size=56962
```

### Get CPU Usage

```
echo "CPU Usage: "$[100-$(vmstat 1 2|tail -1|awk '{print $15}')]"%"
```

## Datasets

### Generating Synthetic Data using TPCxAI
* Tool Download Link: [TPCxAI Tool](https://www.tpc.org/tpc_documents_current_versions/download_programs/tools-download-request5.asp?bm_type=TPCX-AI&bm_vers=1.0.2&mode=CURRENT-ONLY)
* Documentation Link: [TPCxAI Documentation](https://www.tpc.org/tpc_documents_current_versions/pdf/tpcx-ai_v1.0.2.pdf)
#### Setup & Instructions
1. Once Downloaded, in the root folder open file *setenv.sh* and find environment variable `TPCxAI_SCALE_FACTOR`.
2. Based on the required size, change the value of the Scale Factor. This value represents the size of the generated datasets across all the 10 Use Cases that TPCxAI supports (For more details on the use-cases, check the Documentation). 
    | Scale Factor  | Size  |
    | ------------- | ----- |
    | 1             | 1GB   |
    | 3             | 3GB   |
    | 10            | 10GB  |
    | 30            | 30GB  |
    | 100           | 100GB |
    | ...           | ...   |
    | 10,000        | 10TB  |
 > TPCxAI Supports Scale Factors in multiples of form `(1|3)*10^x` upto `10,000`. *(i.e.: 1, 3, 10, 30, 100, 300, ..., 10,000)*
3. Once the value is set, save and close the file.
4. Run the file `TPCx-AI_Benchmarkrun.sh`. It takes a while depending on the Scale Factor.
5. Once done, the generated datasets should be available at `[tool_root_dir]/output/data/`