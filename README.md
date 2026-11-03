# Dry Bean Classification — Machine Learning Assignment 2

## 1. Problem Statement

The objective of this project is to implement and compare five machine learning classification algorithms on the Dry Bean dataset.

The project evaluates the performance of each model using six classification metrics and provides an interactive Streamlit web application for model evaluation.

The five classifiers implemented are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The application allows users to select a trained model, upload labelled test data in CSV format, and view evaluation metrics, a confusion matrix, and a classification report.

The application uses saved models and does not retrain them on uploaded test data.

## 2. Dataset Description

**Dataset:** Dry Bean Dataset

**Source:** UCI Machine Learning Repository

**Dataset link:** https://archive.ics.uci.edu/dataset/602/dry+bean

The dataset contains measurements extracted from images of dry bean grains. These measurements are used to classify beans into seven varieties.

| Property | Description |
|---|---|
| Total instances | 13,611 |
| Input features | 16 |
| Total columns | 17 |
| Target column | `Class` |
| Number of classes | 7 |
| Missing values | 0 |
| Problem type | Multiclass classification |

### Input Features

The 16 numerical input features describe the size, dimensions, and shape of the beans:

- Area
- Perimeter
- MajorAxisLength
- MinorAxisLength
- AspectRation
- Eccentricity
- ConvexArea
- EquivDiameter
- Extent
- Solidity
- roundness
- Compactness
- ShapeFactor1
- ShapeFactor2
- ShapeFactor3
- ShapeFactor4

### Target Classes

The target column, `Class`, contains seven bean varieties:

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

### Data Preparation

The dataset was loaded from CSV using pandas. The target column was separated from the input features.

The data was split into training and testing sets using stratified sampling.

| Dataset split | Number of instances | Percentage |
|---|---:|---:|
| Training set | 10,888 | 80% |
| Testing set | 2,723 | 20% |
| Total | 13,611 | 100% |

The split used `random_state=42` and `stratify=y` to make the experiment reproducible and preserve class proportions.

StandardScaler was used in the Logistic Regression, KNN, and SVM pipelines. The Decision Tree and Gaussian Naive Bayes models were trained without feature scaling.

## 3. GitHub Repository Link

**Repository URL:** https://github.com/mahima2071985-alt/dry-bean-classification-ml

The repository contains:

- Complete source code
- Saved trained models for all five classifiers
- Training and testing datasets
- Model comparison results
- Requirements file
- README documentation
- Streamlit application

## 4. Models Used and Evaluation Results

All five models were trained using the same training set and evaluated on the same held-out test set of 2,723 instances.

### 4.1 Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9199 | 0.9945 | 0.9342 | 0.9291 | 0.9313 | 0.9033 |
| Decision Tree | 0.8920 | 0.9450 | 0.9080 | 0.9090 | 0.9083 | 0.8695 |
| KNN | 0.9166 | 0.9833 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9672 | 0.7744 | 0.7694 | 0.7677 | 0.7154 |
| Support Vector Machine | 0.9221 | 0.9949 | 0.9359 | 0.9330 | 0.9343 | 0.9059 |

**Metric averaging:**
- Precision, Recall, and F1 use macro averaging.
- AUC uses macro-averaged one-vs-rest ROC AUC.
- MCC is the multiclass Matthews correlation coefficient.

### 4.2 Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved 91.99% accuracy and 0.9945 AUC. It performed well across the seven classes, with a macro F1 score of 0.9313. |
| Decision Tree | Achieved 89.20% accuracy. Its performance was lower than Logistic Regression, KNN, and SVM on this test set. |
| KNN | Achieved 91.66% accuracy and a macro F1 score of 0.9293. It performed competitively with Logistic Regression and SVM. |
| Naive Bayes | Achieved 76.39% accuracy, the lowest accuracy among the five models. Its macro F1 score was 0.7677. |
| Support Vector Machine | Achieved the highest accuracy of 92.21%, the highest macro F1 score of 0.9343, and the highest MCC of 0.9059 in this experiment. Its AUC was also the highest at 0.9949. |

### 4.3 Overall Winner

**Support Vector Machine (SVM)** was the best-performing model in this experiment based on the observed test results.

It achieved:

- Accuracy: 0.9221
- AUC: 0.9949
- Macro Precision: 0.9359
- Macro Recall: 0.9330
- Macro F1 Score: 0.9343
- MCC: 0.9059

SVM performed slightly better than Logistic Regression and KNN on the overall metrics. This conclusion applies to the selected dataset, train-test split, and model settings used in this experiment.

## 5. Project Structure

```text
dry-bean-classification-ml/
├── data/
│   └── dry_bean.csv
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
├── app.py
├── train_models.py
├── test_data.csv
├── model_comparison.csv
├── requirements.txt
└── README.md
```

The `.venv` virtual environment is used for local development and should not be committed to the repository.

## 6. Installation and Execution

### Prerequisites

- Python
- pip
- Git

### Install dependencies

Create and activate a virtual environment if one does not already exist:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

### Train and evaluate the models

From the project root directory, run:

```bash
python train_models.py
```

The script trains all five classifiers, calculates the evaluation metrics, saves the trained models under `models/`, and generates `model_comparison.csv` and `test_data.csv`.

### Launch the Streamlit application

```bash
streamlit run app.py
```

The local Streamlit application allows users to select a classifier, upload labelled test data, and inspect evaluation metrics, the confusion matrix, and the classification report.

## 7. Streamlit Application Link

**Live application URL:**

https://dry-bean-classification-ml-xbfqbzlfw4ebdzwhvtak2q.streamlit.app/

The application is deployed using Streamlit Community Cloud.

It allows users to:

- Select one of the five trained classification models.
- Upload labelled test data in CSV format.
- View the evaluation metrics.
- Inspect the confusion matrix.
- View the classification report.

The deployed application loads the saved models and evaluates the uploaded labelled test data without retraining the models.

## 8. Conclusion

This project compares five classification algorithms on the Dry Bean dataset using a consistent train-test split and six evaluation metrics.

The SVM achieved the strongest overall results in the current experiment, followed closely by Logistic Regression and KNN. Decision Tree performed moderately well, while Gaussian Naive Bayes had the lowest accuracy.

The project demonstrates a complete classification workflow, including data preparation, model training, evaluation, model persistence, and interactive application development.



2. UCI Machine Learning Repository. Dry Bean Dataset. https://archive.ics.uci.edu/dataset/602/dry+bean
