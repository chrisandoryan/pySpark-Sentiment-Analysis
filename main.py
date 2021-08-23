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
    sc.setLogLevel("OFF")
    printfo("SparkContext created")
except ValueError:
    warnings.warn("SparkContext already exists in this scope")

printfo("Reading dataset from ./data/*")
train_df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('./data/clean_tweet.csv')
print(train_df.show(5))

train_df = clean_dataset(train_df)

printfo("Splitting train_df into Training and Validation with 70:30 ratio")
train_set, test_set = train_df.randomSplit([0.70, 0.30], seed=31337)
train_set = train_df

printfo("Dataset Specification")
printfo("Train Data: ", train_set.count())
printfo("Validation Data: ", test_set.count())

dataset = (train_set, test_set)

# Train Section
printfo("Model 1 Training: TF-ID + Logistic Regression")
train_lr(dataset)

printfo("Model 2 Training: TF-ID + Naive Bayes")
train_naive(dataset)