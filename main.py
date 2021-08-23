# import findspark
# findspark.init()
import pyspark as ps
import warnings
from pyspark.sql import SQLContext
from utils import printfo, clean_dataset
from models import train_lr, train_naive

printfo("Initializing...")
# Initialization Section
try:
    # create SparkContext on all CPUs available: in my case I have 4 CPUs on my laptop
    # sc = ps.SparkContext('spark://digimates.id:7077')
    sc = ps.SparkContext('local[4]')
    sqlContext = SQLContext(sc)
    printfo("SparkContext created")
except ValueError:
    warnings.warn("SparkContext already exists in this scope")

printfo("Reading dataset from ./data/*")
train_df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('./data/train_tweets.csv')
test_df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('./data/test_tweets.csv')
print(train_df.show(5))

train_df = clean_dataset(train_df)
test_df = clean_dataset(test_df)

printfo("Splitting train_df into Training and Validation with 80:20 ratio")
train_set, val_set = train_df.randomSplit([0.80, 0.20], seed=1337)
test_set = test_df

printfo("Dataset Specification")
printfo("Train Data: ", train_set.count())
printfo("Validation Data: ", val_set.count())
printfo("Test Data: ", test_set.count())

dataset = (train_set, val_set, test_set)

# Train Section
printfo("Model 1 Training: TF-ID + Logistic Regression")
train_lr(dataset)

printfo("Model 2 Training: TF-ID + Naive Bayes")
train_naive(dataset)