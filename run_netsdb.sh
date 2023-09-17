#!/bin/bash
MODELDATAPATH="model-inference/decisionTree/experiments" 
RUNTEST="testDecisionForestWithCrossProduct"
DATASET="fraud"
MODEL="XGBoost"
TREENUM="10"

case $RUNTEST in 
"testDecisionForest")
    FLAG="F A "
    TREENUMFLAG=""
    ;;
"testDecisionForestWithCrossProduct")
    TREENUMFLAG="$TREENUM "
    ;;
esac

#Relation-Centric
# Higgs: Shuffle Page Size = 512 MB, Each tree result has a size of numTrees*4bytes, the block size should be significantly smaller than 512MB/numTrees/4
# bin/testDecisionForestWithCrossProduct N 2200000 28 50000 0 32 1 ~/data/HIGGS.csv_test.csv ~/model/higgs_xgboost_1600_8_netsdb XGBoost 1600 withoutMissing cla^Cification
case $DATASET in
"higgs")
    ROWNUM="2200000"
    COLNUM="28"
    BLOCKSIZE="50000" 
    LABELCOL="0"
    PAGESIZE="32"
    PARTITIONNUM="1"
    DATASETNAME="HIGGS.csv_test.csv"
    MISSING="withoutMissing"
    TYPE="classification"
    ;;

"airline_classification")
    ROWNUM="23013803"
    COLNUM="13"
    BLOCKSIZE="3000000"
    LABELCOL="13"
    PAGESIZE="160"
    PARTITIONNUM="5"
    DATASETNAME="airline_classification.csv_test.csv"
    MISSING="withoutMissing"
    TYPE="classification"
    ;;
"tpcxai_fraud")
    ROWNUM="131564161"
    COLNUM="7"
    BLOCKSIZE="500000"
    LABELCOL="7"
    PAGESIZE="32"
    PARTITIONNUM="5"
    DATASETNAME="tpcxai_fraud_test.csv"
    MISSING="withoutMissing"
    TYPE="classification"
    ;;
"fraud")
    ROWNUM="56962"
    COLNUM="30"
    BLOCKSIZE="7121"
    LABELCOL="30"
    PAGESIZE="1"
    PARTITIONNUM="1"
    DATASETNAME="creditcard.csv_test.csv"
    MISSING="withoutMissing"
    TYPE="classification"
    ;;
"year")
    ROWNUM="103069"
    COLNUM="90"
    BLOCKSIZE="13000"
    LABELCOL="0"
    PAGESIZE="6"
    PARTITIONNUM="1"
    DATASETNAME="YearPredictionMSD_test.csv"
    MISSING="withoutMissing"
    TYPE="regression"
    ;;
"epsilon")
    ROWNUM="100000"
    COLNUM="2000"
    BLOCKSIZE="5000"
    LABELCOL="0"
    PAGESIZE="42"
    PARTITIONNUM="1"
    DATASETNAME="epsilon_test.csv"
    MISSING="withoutMissing"
    TYPE="classification"
    ;;
"bosch")
    ROWNUM="236750"
    COLNUM="968"
    BLOCKSIZE="30000"
    LABELCOL="968"
    PAGESIZE="120"
    PARTITIONNUM="1"
    DATASETNAME="bosch.csv_test.csv"
    MISSING="withMissing"
    TYPE="classification"
    ;;
"criteo")
    ROWNUM="6042135"
    COLNUM="1000000"
    PAGESIZE="512"
    PARTITIONNUM="1"
    DATASETNAME="criteo.kaggle2014.svm/test.txt.svm"
    MISSING="withMissing"
    TYPE="classification"
    ;;
esac

case $MODEL in
"LightGBM")
    MODELLABEL="lightgbm"
    ;;
"XGBoost")
    MODELLABEL="xgboost"
    ;;
"RandomForest")
    MODELLABEL="randomforest"
    ;;
esac

MODELFILE="${DATASET}_${MODELLABEL}_${TREENUM}_8_netsdb"

./scripts/cleanupNode.sh
./scripts/startPseudoCluster.py 8 30000

if [ $DATASET == "criteo" ]; then
    bin/testDecisionForestSparse Y $ROWNUM $COLNUM $BLOCKSIZE $PARTITIONNUM $MODELDATAPATH/dataset/$DATASETNAME $MODELDATAPATH/models/$MODELFILE $MODEL $TYPE
    bin/testDecisionForestSparse N $ROWNUM $COLNUM $BLOCKSIZE $PARTITIONNUM $MODELDATAPATH/dataset/$DATASETNAME $MODELDATAPATH/models/$MODELFILE $MODEL $TYPE
else
    echo "bin/$RUNTEST Y $ROWNUM $COLNUM $BLOCKSIZE $LABELCOL $FLAG$PAGESIZE $PARTITIONNUM $MODELDATAPATH/dataset/$DATASETNAME $MODELDATAPATH/models/$MODELFILE $MODEL $TREENUMFLAG$MISSING $TYPE"
    bin/$RUNTEST Y $ROWNUM $COLNUM $BLOCKSIZE $LABELCOL $FLAG$PAGESIZE $PARTITIONNUM $MODELDATAPATH/dataset/$DATASETNAME $MODELDATAPATH/models/$MODELFILE $MODEL $TREENUMFLAG$MISSING $TYPE
    bin/$RUNTEST N $ROWNUM $COLNUM $BLOCKSIZE $LABELCOL $FLAG$PAGESIZE $PARTITIONNUM $MODELDATAPATH/dataset/$DATASETNAME $MODELDATAPATH/models/$MODELFILE $MODEL $TREENUMFLAG$MISSING $TYPE
fi

# bin/testDecisionForestSparse Y 6042135 1000000 450 1 $MODELDATAPATH/dataset/criteo.kaggle2014.svm/test.txt.svm $MODELDATAPATH/models/criteo_xgboost_1600_8_netsdb XGBoost classification
# bin/testDecisionForestSparse N 6042135 1000000 450 1 $MODELDATAPATH/dataset/criteo.kaggle2014.svm/test.txt.svm $MODELDATAPATH/models/criteo_xgboost_1600_8_netsdb XGBoost classification
