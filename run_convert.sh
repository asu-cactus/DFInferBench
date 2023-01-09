#! /bin/bash
DATASETS="higgs airline_classification fraud year epsilon tpcxai_fraud"
TREESS="10 500 1600"
MODELS="randomforest xgboost lightgbm"

for DATASET in $DATASETS; do
    for MODEL in $MODELS; do
        for TREES in $TREESS; do 
            echo "python convert_trained_model_to_framework.py -d $DATASET -m $MODEL -f pytorch,torch,onnx,treelite,netsdb  --num_trees $TREES"
            python convert_trained_model_to_framework.py -d $DATASET -m $MODEL -f pytorch,torch,onnx,treelite,netsdb --num_trees $TREES
        done
    done
done

for DATASET in $DATASETS; do
    for TREES in $TREESS; do 
        echo "python convert_trained_model_to_framework.py -d $DATASET -m lightgbm -f lleaves  --num_trees $TREES"
        python convert_trained_model_to_framework.py -d $DATASET -m lightgbm -f lleaves --num_trees $TREES
    done
done


sudo shutdown now -h