from utils import printfo

# Model 1: TF-IDF + Logistic Regression
# Reference: https://towardsdatascience.com/sentiment-analysis-with-pyspark-bc8e83f80c35
def train_lr(dataset):
    train_set, val_set = dataset

    from pyspark.ml.feature import HashingTF, IDF, Tokenizer
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml import Pipeline

    printfo("Executing tokenizer and TF-IDF...")
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
    idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
    label_stringIdx = StringIndexer(inputCol="target", outputCol="label")

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
    lrModel = lr.fit(train_df, )
    prediction_lr = lrModel.transform(val_df)

    printfo("Displaying prediction sample...")
    prediction_lr.select("label", "prediction").show(10,False)
    printfo("Training has been completed.")

    printfo("Starting evaluation...")
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")

    n_correct = prediction_lr.filter(prediction_lr.label == prediction_lr.prediction).count()
    printfo("[Logistic Regression] Correct prediction: %s / %s" % (n_correct, val_set.count()))
    accuracy = n_correct / float(val_set.count())
    printfo("[Logistic Regression] Accuracy: ", accuracy)

    from sklearn.metrics import classification_report, confusion_matrix
    y_true = prediction_lr.select(['label']).collect()
    y_pred = prediction_lr.select(['prediction']).collect()
    print(classification_report(y_true, y_pred))

    return


# Model 2: TF-IDF + Naive Bayes
# Reference: https://github.com/shikha720/Sentiment-Analysis-and-Text-Classification-Using-PySpark
def train_naive(dataset):
    train_set, val_set = dataset

    from pyspark.ml.feature import HashingTF, IDF, Tokenizer
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml import Pipeline

    printfo("Executing tokenizer and TF-IDF...")
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashtf = HashingTF(numFeatures=2**16, inputCol="words", outputCol='tf')
    idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
    label_stringIdx = StringIndexer(inputCol="target", outputCol="label")

    printfo("Start training with NaiveBayes...")
    from pyspark.ml.classification import NaiveBayes
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    pipeline_nb = Pipeline(stages=[tokenizer, hashtf, idf, label_stringIdx, nb])
    nbModel = pipeline_nb.fit(train_set)
    prediction_nb = nbModel.transform(val_set)

    printfo("Displaying prediction sample...")
    prediction_nb.select("label", "prediction").show(10,False)
    printfo("Training has been completed.")

    printfo("Starting evaluation...")
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label")

    printfo("Starting evaluation...")
    n_correct = prediction_nb.filter(prediction_nb.label == prediction_nb.prediction).count()
    printfo("[Naive Bayes] Correct prediction: %s / %s" % (n_correct, val_set.count()))
    accuracy = n_correct / float(val_set.count())
    printfo("[Naive Bayes] Accuracy: ", accuracy)

    from sklearn.metrics import classification_report, confusion_matrix
    y_true = prediction_nb.select(['label']).collect()
    y_pred = prediction_nb.select(['prediction']).collect()
    print(classification_report(y_true, y_pred))

