from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FakeNewsClassification") \
    .getOrCreate()

# Load the CSV file with inferred schema
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)

# Create temporary view
df.createOrReplaceTempView("news_data")

# Basic exploration
print("First 5 rows:")
df.show(5)

print("\nTotal number of articles:")
print(df.count())

print("\nDistinct labels:")
spark.sql("SELECT DISTINCT label FROM news_data").show()

# Save to CSV
df.limit(5).write.csv("task1_output.csv", header=True, mode="overwrite")

from pyspark.sql.functions import col, lower, array_join
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# Convert text to lowercase
df = df.withColumn("text", lower(col("text")))

# Tokenize the text
tokenizer = Tokenizer(inputCol="text", outputCol="words")
words_df = tokenizer.transform(df)

# Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
cleaned_df = remover.transform(words_df)

# Convert array to string for CSV output
cleaned_df = cleaned_df.withColumn("filtered_words_str", array_join(col("filtered_words"), ", "))

# Save tokenized output
cleaned_df.select("id", "title", "filtered_words_str", "label") \
          .limit(5) \
          .write.csv("task2_output.csv", header=True, mode="overwrite")

from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.sql.functions import array_join, udf
from pyspark.sql.types import StringType
from pyspark.ml.linalg import DenseVector
import json

# Function to convert vector to string
def vector_to_string(v):
    if isinstance(v, DenseVector):
        return json.dumps({"size": v.size, "indices": v.indices.tolist(), "values": v.values.tolist()})
    return str(v)

# Register the UDF
vector_to_string_udf = udf(vector_to_string, StringType())

# TF-IDF Vectorization
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
featurized_df = hashingTF.transform(cleaned_df)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(featurized_df)
tfidf_df = idf_model.transform(featurized_df)

# Label indexing
indexer = StringIndexer(inputCol="label", outputCol="label_index")
indexed_df = indexer.fit(tfidf_df).transform(tfidf_df)

# Convert complex types to strings
output_df = indexed_df.withColumn("filtered_words_str", array_join(col("filtered_words"), ", ")) \
                     .withColumn("features_str", vector_to_string_udf(col("features")))

# Save features and labels
output_df.select("id", "filtered_words_str", "features_str", "label_index") \
         .limit(5) \
         .write.csv("task3_output.csv", header=True, mode="overwrite")

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

# Split the data into training and test sets (80% train, 20% test)
(train_df, test_df) = indexed_df.randomSplit([0.8, 0.2], seed=42)

# Initialize Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label_index", maxIter=10)

# Train the model
lr_model = lr.fit(train_df)

# Make predictions on test set
predictions = lr_model.transform(test_df)

# Convert vector columns to string representation for CSV output
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.ml.linalg import DenseVector
import json

def vector_to_string(v):
    if isinstance(v, DenseVector):
        return json.dumps(v.toArray().tolist())
    return str(v)

vector_to_string_udf = udf(vector_to_string, StringType())

# Prepare output with readable formats
output_df = predictions.withColumn("probability_str", vector_to_string_udf(col("probability"))) \
                     .withColumn("rawPrediction_str", vector_to_string_udf(col("rawPrediction")))

# Select relevant columns for output
final_output = output_df.select(
    "id",
    "title",
    "label_index",
    "prediction",
    "probability_str",
    "rawPrediction_str"
)

# Save predictions to CSV
final_output.limit(5).write.csv("task4_output.csv", header=True, mode="overwrite")

# Show sample predictions
print("Sample predictions:")
final_output.limit(5).show(truncate=False)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, count, col

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

## Main Evaluation Metrics
evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index",
    predictionCol="prediction"
)

# Calculate all metrics
metrics = [
    ("Accuracy", evaluator.setMetricName("accuracy").evaluate(predictions)),
    ("F1 Score", evaluator.setMetricName("f1").evaluate(predictions)),
    ("Precision", evaluator.setMetricName("weightedPrecision").evaluate(predictions)),
    ("Recall", evaluator.setMetricName("weightedRecall").evaluate(predictions))
]

# Create and save metrics DataFrame
metrics_df = spark.createDataFrame(metrics, ["Metric", "Value"])
metrics_df.show(truncate=False)
metrics_df.write.csv("task5_output.csv", header=True, mode="overwrite")

## Confusion Matrix (simplified and robust)
confusion_df = predictions.groupBy("label_index", "prediction").count()
confusion_matrix = confusion_df.toPandas().pivot(
    index="label_index",
    columns="prediction",
    values="count"
)

print("\nConfusion Matrix:")
print(confusion_matrix)

# Save confusion matrix
confusion_matrix.to_csv("task5_confusion_matrix.csv")

## Class-wise Metrics (fixed syntax)
class_metrics = predictions.groupBy("label_index").agg(
    count(when(col("prediction") == 0, True)).alias("predicted_fake"),
    count(when(col("prediction") == 1, True)).alias("predicted_real"),
    count(when((col("label_index") == 0) & (col("prediction") == 0), True)).alias("true_fake"),
    count(when((col("label_index") == 1) & (col("prediction") == 1), True)).alias("true_real")
)

print("\nClass-wise Metrics:")
class_metrics.show()

# Save class metrics
class_metrics.write.csv("task5_class_metrics.csv", header=True, mode="overwrite")

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize evaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index",
    predictionCol="prediction"
)

# Calculate metrics
accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
f1_score = evaluator.setMetricName("f1").evaluate(predictions)

# Create results DataFrame
results = spark.createDataFrame([
    ("Accuracy", accuracy),
    ("F1 Score", f1_score)
], ["Metric", "Value"])

# Show results
print("Model Evaluation Results:")
results.show(truncate=False)

# Save to CSV
results.write.csv("task5_output.csv", header=True, mode="overwrite")

