-- Fraud

-- Training Relation

CREATE TABLE fraud_train (v1 REAL NOT NULL, v2 REAL NOT NULL, v3 REAL NOT NULL, v4 REAL NOT NULL, v5 REAL NOT NULL, v6 REAL NOT NULL, v7 REAL NOT NULL, v8 REAL NOT NULL, v9 REAL NOT NULL, v10 REAL NOT NULL, v11 REAL NOT NULL, v12 REAL NOT NULL, v13 REAL NOT NULL, v14 REAL NOT NULL, v15 REAL NOT NULL, v16 REAL NOT NULL, v17 REAL NOT NULL, v18 REAL NOT NULL, v19 REAL NOT NULL, v20 REAL NOT NULL, v21 REAL NOT NULL, v22 REAL NOT NULL, v23 REAL NOT NULL, v24 REAL NOT NULL, v25 REAL NOT NULL, v26 REAL NOT NULL, v27 REAL NOT NULL, v28 REAL NOT NULL, class INTEGER NOT NULL);

-- Testing Relation

CREATE TABLE fraud_test (v1 REAL NOT NULL, v2 REAL NOT NULL, v3 REAL NOT NULL, v4 REAL NOT NULL, v5 REAL NOT NULL, v6 REAL NOT NULL, v7 REAL NOT NULL, v8 REAL NOT NULL, v9 REAL NOT NULL, v10 REAL NOT NULL, v11 REAL NOT NULL, v12 REAL NOT NULL, v13 REAL NOT NULL, v14 REAL NOT NULL, v15 REAL NOT NULL, v16 REAL NOT NULL, v17 REAL NOT NULL, v18 REAL NOT NULL, v19 REAL NOT NULL, v20 REAL NOT NULL, v21 REAL NOT NULL, v22 REAL NOT NULL, v23 REAL NOT NULL, v24 REAL NOT NULL, v25 REAL NOT NULL, v26 REAL NOT NULL, v27 REAL NOT NULL, v28 REAL NOT NULL, class INTEGER NOT NULL);

-- Loading Training Data (You need to replace the path to the data in your environment)

COPY fraud_train FROM '/home/ubuntu/data/creditcard_train.csv' DELIMITER ',';

-- Loading Testing Data (You need to replace the path to the data in your environment)

COPY fraud_test FROM '/home/ubuntu/data/creditcard_test.csv' DELIMITER ',';

-- multi-threading

SET max_parallel_workers_per_gather = 8;

-- Run the training process

SELECT * FROM pgml.train(
   				project_name => 'fraud_training',
    				algorithm => 'xgboost',
   				task => 'classification',
				hyperparams => '{ "n_estimators": 10, "max_depth": 8}',
   				relation_name => 'fraud_train', y_column_name => 'class' 
		);


-- Run the inference process


SELECT pgml.predict('fraud_training', ARRAY[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, 
	v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28]) AS prediction 
FROM fraud_test;


SELECT pgml.predict_batch('fraud_training', array_agg(ARRAY[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28])) AS prediction FROM fraud_test;

CREATE TABLE fraud_prediction(result REAL NOT NULL);

pgml.predict_batch('fraud_training', array_agg(ARRAY[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28])) AS prediction FROM fraud_test;
