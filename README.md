# Decision Forest Inference Benchmark Suite

We provide a comprehensive benchmark framework for decision forest model inferences. 

The framework covers three algorithms: RandomForest, XGBoost, and LightGBM. 

The framework supports most of the popular decision forest inference platforms, including Scikit-Learn, XGBoost, LightGBM, ONNX, HummingBird, TreeLite, lleaves, TensorFlow TFDF, cu-ml, and so on.

The framework also supports multiple well-known workloads, including Higgs, Airline, Fraud, TPCx-AI fraud detection, year, Bosch, Epsilon, Crieto.

# Benchmark Environment Setup

## Python version
TVM requires Python version >= 3.6 and < 3.9. 

## Benchmark Platforms

### Scikit-Learn

https://scikit-learn.org/stable/install.html

### ONNX

https://onnxruntime.ai/docs/install/

### TreeLite

https://treelite.readthedocs.io/en/latest/install.html

### HummingBird

https://github.com/microsoft/hummingbird#installation

#### To Use TVM as backend

https://tvm.apache.org/docs/install/from_source.html


**There are two ways to build TVM -- build with CMake and buld with conda, see the documents in the above link for details**

If you build TVM with CMake, you may meet error "collect2: fatal error: cannot find ‘ld’", try to change the linker, e.g., you may change 'fuse-ld=lld' to 'fuse-ld=gold' in the ./CMakeFiles/tvm_runtime.dir/link.txt, ./CMakeFiles/tvm.dir/link.txt, and ./CMakeFiles/tvm_allvisible.dir/link.txt.

Remember to run 'make install' from the build directory after successfully compiling tvm to shared libraries.

#### To Use PyTorch/TorchScript as backend

https://pytorch.org/

## Tensorflow

https://www.tensorflow.org/install

## PostgreSQL

https://www.postgresql.org/download/

## ConnectX

https://github.com/sfu-db/connector-x#installation

## XGBoost

pip3 install xgboost

## LightGBM

pip3 install lightgbm

### LLeaves (Model Compiler for LightGBM Model)

pip3 install lleaves (or) conda install -c conda-forge lleaves

## Catboost (This is not a framework. Only used to load the "epsilon" dataset)

pip3 install catboost


# Decision Tree Experiments

## Project Setup

### PostgreSQL

```
#Install tools to download and unzip dataset.
sudo apt update
sudo apt install wget gzip bzip2 unzip xz-utils
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
gzip -d HIGGS.csv.gz

#Install postgres DBMS
sudo apt install postgresql postgresql-contrib

##The following steps are optional. They show you how to login and create a table/relation.
##The python script split_data.py will create the
#Login in
sudo -i -u postgres

#Run postgres DBMS
psql

#Default username and password are both "postgres", which are used in our config.json file.
#In case you want to change the password, you can do the following:
postgres=# \password postgres
Enter new password: <new-password>

#Create Table and load
postgres=# CREATE TABLE higgs(label REAL NOT NULL, leptonpT REAL NOT NULL, leptoneta REAL NOT NULL, leptonphi REAL NOT NULL, missingenergymagnitude REAL NOT NULL, missingenergyphi REAL NOT NULL, jet1pt REAL NOT NULL, jet1eta REAL NOT NULL, jet1phi REAL NOT NULL, jet1btag REAL NOT NULL, jet2pt REAL NOT NULL, jet2eta REAL NOT NULL, jet2phi REAL NOT NULL, jet2btag REAL NOT NULL, jet3pt REAL NOT NULL, jet3eta REAL NOT NULL, jet3phi REAL NOT NULL, jet3btag REAL NOT NULL, jet4pt REAL NOT NULL, jet4eta REAL NOT NULL, jet4phi REAL NOT NULL, jet4btag REAL NOT NULL, mjj REAL NOT NULL, mjjj REAL NOT NULL, mlv REAL NOT NULL, mjlv REAL NOT NULL, mbb REAL NOT NULL, mwbb REAL NOT NULL, mwwbb REAL NOT NULL);

postgres=# copy higgs from 'HIGGS.csv' with CSV;

#Quite postgres DBMS
postgres=# \q
```

You must first run split_data.py to split a dataset into training part and testing part, and load both parts to PostgreSQL, which is a prerequisite for running train_model.py and test_model.py.

TO MOUNT THE DRIVE ON EC2

```
sudo file -s /dev/nvme1n1
sudo mkfs -t xfs /dev/nvme1n1
sudo mkdir /mnt/data
sudo mount /dev/nvme1n1 /mnt/data
cd /mnt
sudo chmod 777 data
```

DATASETS

```
#higgs
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz
gzip -d HIGGS.csv.gz

#airline
wget "http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2"
bzip2 -dk airline_14col.data.bz2

#fraud
kaggle datasets download mlg-ulb/creditcardfraud -f creditcard.csv.zip
unzip creditcard.csv.zip

#year
wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"
unzip YearPredictionMSD.txt.zip

#epsilon
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.xz
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.xz
unxz epsilon_normalized.xz
unxz epsilon_normalized.t.xz
```
TPCxAI Dataset Instructions available here: [Link](https://github.com/asu-cactus/netsdb/blob/tpcxai-tool-instructions/model-inference/decisionTree/README.md#generating-synthetic-data-using-tpcxai)

To run a certain experiment

Datasets: higgs

Classifiers: xgboost, randomforest, lightgbm

Frameworks: Sklearn, ONNXCPU, TreeLite, HummingbirdPytorchCPU, HummingbirdTorchScriptCPU, HummingbirdTVMCPU, LightGBM, Lleaves

```
python data_processing.py -d [dataset]

python train_model.py -d [dataset] -m [model]

python convert_trained_model_to_framework.py -d [dataset] -m [model] -f [framework]

python test_model.py -d [dataset] -m [model] -f [framework] --batch_size [batch_size] --query_size [query_size]


**Examples**
python data_processing.py -d higgs

python train_model.py -d higgs -m randomforest --num_trees 10
python train_model.py -d higgs -m xgboost --num_trees 10
python train_model.py -d higgs -m lightgbm --num_trees 10

python convert_trained_model_to_framework.py -d higgs -m randomforest -f pytorch,torch,tf-df,onnx,netsdb --num_trees 10
python convert_trained_model_to_framework.py -d higgs -m xgboost -f pytorch,torch,onnx,treelite,tf-df,netsdb --num_trees 10
python convert_trained_model_to_framework.py -d higgs -m lightgbm -f pytorch,torch,onnx,treelite,lightgbm,lleaves,netsdb --num_trees 10

python test_model.py -d higgs -m xgboost -f TreeLite --batch_size 1000 --query_size 1000 --num_trees 10
or modify and run run_test.sh
nohup ./run_test.sh &> ./results/test_output.txt &
```

Get CPU Usage

```
echo "CPU Usage: "$[100-$(vmstat 1 2|tail -1|awk '{print $15}')]"%"
```

GPU Environment Setup

g2dn.2xlarge, with Deep Learning AMI GPU PyTorch 1.12.0 (Ubuntu 20.04) 20220817 AMI ( ami-0f5b2957914692f92)

Install the following libraries

```
sudo conda install -c conda-forge py-xgboost-gpu
sudo conda install pyyaml
sudo conda install connectorx
sudo conda install psycopg2
conda install -c conda-forge treelite
sudo conda install -c conda-forge pandas
pip install hummingbird-ml[extra]
pip install onnxruntime
pip install skl2onnx
pip install onnxmltools
pip install tensorflow
pip install tensorflow_decision_forests –upgrade
pip install plotly
```

tvm gpu must be built from source, code will be added later.

rapids installation instructions will be updated later

pip install apache-tvm, will install prebuilt library, might not run on multithreaded CPUs and does not support GPU.


# Generating Synthetic Data using TPCxAI
* Tool Download Link: [TPCxAI Tool](https://www.tpc.org/tpc_documents_current_versions/download_programs/tools-download-request5.asp?bm_type=TPCX-AI&bm_vers=1.0.2&mode=CURRENT-ONLY)
* Documentation Link: [TPCxAI Documentation](https://www.tpc.org/tpc_documents_current_versions/pdf/tpcx-ai_v1.0.2.pdf)
## Setup & Instructions
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

