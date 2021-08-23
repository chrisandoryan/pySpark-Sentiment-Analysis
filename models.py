from utils import printfo

# Model 1: TF-IDF + Logistic Regression
# Reference: https://towardsdatascience.com/sentiment-analysis-with-pyspark-bc8e83f80c35
def train_lr(dataset):
    train_set, val_set, test_set = dataset

    from pyspark.ml.feature import HashingTF, IDF, Tokenizer
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml import Pipeline

    printfo("Executing tokenizer and TF-IDF...")
    tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
    hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
    idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
    label_stringIdx = StringIndexer(inputCol="label", outputCol="label_index")

    printfo("Creating pipeline...")
    pipeline = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx])

    printfo("Performing transformation...")
    pipelineFit = pipeline.fit(train_set)
    train_df = pipelineFit.transform(train_set)
    val_df = pipelineFit.transform(val_set)
    print(train_df.show(5))

    printfo("Start training with Logistic Regression...")
    from pyspark.ml.classification import LogisticRegression
    lr = LogisticRegression(maxIter=100)
    lrModel = lr.fit(train_df)
    prediction_lr = lrModel.transform(val_df)

    printfo("Displaying prediction sample...")
    prediction_lr.select("label", "prediction").show(10,False)
    printfo("Training has been completed.")

    printfo("Starting evaluation...")
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

    printfo("[Logistic Regression] Accuracy: ", evaluator.evaluate(prediction_lr))
    return


# Model 2: TF-IDF + Naive Bayes
# Reference: https://github.com/shikha720/Sentiment-Analysis-and-Text-Classification-Using-PySpark
def train_naive(dataset):
    train_set, val_set, test_set = dataset

    from pyspark.ml.feature import HashingTF, IDF, Tokenizer
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml import Pipeline

    printfo("Executing tokenizer and TF-IDF...")
    tokenizer = Tokenizer(inputCol="tweet", outputCol="words")
    hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
    idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms

    printfo("Start training with NaiveBayes...")
    from pyspark.ml.classification import NaiveBayes
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    pipeline_nb = Pipeline(stages=[tokenizer, hashtf, idf, nb])
    nbModel = pipeline_nb.fit(train_set)
    prediction_nb = nbModel.transform(val_set)

    printfo("Displaying prediction sample...")
    prediction_nb.select("label", "prediction").show(10,False)
    printfo("Training has been completed.")

    printfo("Starting evaluation...")
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

    printfo("Starting evaluation...")
    printfo("[Naive Bayes] Accuracy: ", evaluator.evaluate(prediction_nb))
