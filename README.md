# Fake News Classification with Spark MLlib - README

## Overview
This project implements a machine learning pipeline using Spark MLlib to classify news articles as FAKE or REAL based on their content. The pipeline includes text preprocessing, feature extraction, model training, and evaluation.

## Dataset
The dataset used is `fake_news_sample.csv` containing:
- `id`: Unique article identifier
- `title`: News article headline
- `text`: Article content
- `label`: Classification label (FAKE or REAL)

## Project Structure
```
fake_news_classification/
├── fake_news_sample.csv        # Input dataset
├── task1_output.csv            # Basic exploration results
├── task2_output.csv            # Preprocessed text
├── task3_output.csv            # Extracted features
├── task4_output.csv            # Model predictions
├── task5_output.csv            # Evaluation metrics
└── fake_news_classification.py # Main Spark application
```

## How to Run

### Prerequisites
- Apache Spark (version 3.0+)
- Python (version 3.6+)
- PySpark package

### Installation
1. Install PySpark:
```bash
pip install pyspark
```

2. Download the dataset and place it in your project directory

### Running the Application
Execute the Spark application using:
```bash
spark-submit fake_news_classification.py
```

### Expected Output Files
1. `task1_output.csv` - First 5 rows of raw data
2. `task2_output.csv` - Tokenized and cleaned text
3. `task3_output.csv` - Extracted TF-IDF features
4. `task4_output.csv` - Model predictions on test set
5. `task5_output.csv` - Evaluation metrics (Accuracy and F1 Score)

## Task Output Samples

### Task 5 Output Example
```
Metric,Value
Accuracy,0.89
F1 Score,0.88
```

## Notes
- The script assumes the input file is named `fake_news_sample.csv` and is in the same directory
- Output files will be overwritten if they exist
- For large datasets, consider increasing Spark's memory allocation using `--driver-memory` and `--executor-memory` options

## Dependencies
- pyspark
- Python standard libraries

This implementation provides a complete pipeline for fake news classification using Spark MLlib, from data loading to model evaluation.
