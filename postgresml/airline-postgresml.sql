-- Airline

-- Training Relation

CREATE TABLE airline_train (Year INTEGER NOT NULL, Month INTEGER NOT NULL, DayOfMonth INTEGER NOT NULL, DayOfWeek INTEGER NOT NULL, CRSDepTime INTEGER NOT NULL, CRSArrTime INTEGER NOT NULL, UniqueCarrier INTEGER NOT NULL, FlightNum INTEGER NOT NULL, ActualElapsedTime INTEGER NOT NULL, Origin INTEGER NOT NULL, Dest INTEGER NOT NULL, Distance INTEGER NOT NULL, Diverted INTEGER NOT NULL, ArrDelay INTEGER NOT NULL)

-- Testing Relation

CREATE TABLE airline_test (Year INTEGER NOT NULL, Month INTEGER NOT NULL, DayOfMonth INTEGER NOT NULL, DayOfWeek INTEGER NOT NULL, CRSDepTime INTEGER NOT NULL, CRSArrTime INTEGER NOT NULL, UniqueCarrier INTEGER NOT NULL, FlightNum INTEGER NOT NULL, ActualElapsedTime INTEGER NOT NULL, Origin INTEGER NOT NULL, Dest INTEGER NOT NULL, Distance INTEGER NOT NULL, Diverted INTEGER NOT NULL, ArrDelay INTEGER NOT NULL)

-- Loading Training Data (You need to replace the path to the data in your environment)

COPY airline_train FROM '/home/ubuntu/data/airline_classification.csv_train.csv' DELIMITER ',';

-- Loading Testing Data (You need to replace the path to the data in your environment)

COPY airline_test FROM '/home/ubuntu/data/airline_classification.csv_test.csv' DELIMITER ',';

-- multi-threading

SET max_parallel_workers_per_gather = 8;

-- Run the training process

SELECT * FROM pgml.train(
   				project_name => 'airline_training_10',
    				algorithm => 'xgboost',
   				task => 'classification',
				hyperparams => '{ "n_estimators": 10, "max_depth": 8}',
   				relation_name => 'airline_train', y_column_name => 'arrdelay' 
		);


-- Run the inference process


SELECT pgml.predict('airline_training_10', ARRAY[Year, Month, DayOfMonth, DayOfWeek, CRSDepTime, CRSArrTime, UniqueCarrier, FlightNum, ActualElapsedTime, Origin, Dest, Distance, Diverted]) AS prediction 
FROM airline_test;

SELECT pgml.predict_batch('airline_training_10', ARRAY[Year, Month, DayOfMonth, DayOfWeek, CRSDepTime, CRSArrTime, UniqueCarrier, FlightNum, ActualElapsedTime, Origin, Dest, Distance, Diverted]) AS prediction                               
FROM airline_test;

CREATE TABLE fraud_prediction(result REAL NOT NULL);

INSERT INTO fraud_prediction(result)
SELECT pgml.predict_batch('airline_training_10', ARRAY[Year, Month, DayOfMonth, DayOfWeek, CRSDepTime, CRSArrTime, UniqueCarrier, FlightNum, ActualElapsedTime, Origin, Dest, Distance, Diverted]) FROM airline_test;
