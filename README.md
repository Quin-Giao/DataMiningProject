# Heart Disease Prediction – Data Mining Project

## Course Information

* **Course:** Data Mining (IT013IU)
* **University:** Vietnam National University – HCM City
* **School:** International University – School of Computer Science and Engineering

## Project Topic

**Heart Disease Prediction – A Comparative Analysis of Baseline and Advanced Machine Learning Frameworks**

---

## Team Members

1. Võ Lê Quỳnh Giao – ITDSIU23036
2. Nguyễn Huỳnh Ngân Anh – ITDSIU23003
3. Nguyễn Hải Thanh – ITCSIU22146
4. Ngô Vũ Cao Long – ITCSIU21085
5. Võ Gia Ân – ITCSIU22241

---

## Project Overview

This project applies data mining and machine learning techniques to predict heart disease using a medical dataset from Kaggle.
The study compares baseline classifiers with advanced and tuned models to evaluate performance under class imbalance conditions.

The workflow includes:

* Data preprocessing and balancing
* Feature selection and feature engineering
* Classification using multiple algorithms
* Model evaluation and comparison

---

## Dataset

* **Source:** Kaggle – Heart Disease Dataset
https://www.kaggle.com/datasets/oktayrdeki/heart-disease
* **Format:** CSV (converted to ARFF for WEKA)
* **Class label:** Presence of heart disease (Yes / No)
* **Issue:** Severe class imbalance (handled during preprocessing)

All processed datasets are stored in the `data/` folder.

---

## Project Structure

```
DATAMININGPROJECT/
│
├── data/
│   ├── heart_disease.csv
│   ├── heart_disease_cleaned.arff
│   ├── heart_disease_balanced.arff
│   ├── InfoGain_data.arff
│   └── ReliefF_data.arff
│
├── lib/
│   ├── weka.jar
│   └── SMOTE.jar
│
├── src/
│   ├── preprocessing/
│   │   ├── Csv2Arff.java
│   │   ├── LoadData.java
│   │   ├── ResampleSmoteBalance.java
│   │   ├── InfoGainSelection.java
│   │   └── ReliefFSelection.java
│   │
│   ├── basemodel/
│   │   ├── ZeroRClassifier.java
│   │   ├── OneRClassifier.java
│   │   ├── NaiveBayesClassifier.java
│   │   ├── J48Classifier.java
│   │   └── IBkClassifier.java
│   │
│   └── ensemblemodel/
│       ├── RandomForestClassifier.java
│       ├── SVMClassifier.java
│       ├── LogisticClassifier.java
│       └── AdaBoostM1Classifier.java
│
├── README.md
└── report.pdf
```

---

## Preprocessing Summary

Data preprocessing is implemented using Java and the WEKA API.

Main steps:

* Convert CSV to ARFF format
* Clean and validate dataset
* Handle class imbalance using supervised resampling (SMOTE-style)
* Perform feature selection using InfoGain and ReliefF

The output is a balanced and feature-optimized dataset used for all classification models.

---

## Models Implemented

### Baseline Models

* ZeroR
* OneR
* Naive Bayes
* J48 (Decision Tree)

### Advanced & Improved Models

* IBk (K-Nearest Neighbors)
* Random Forest
* Support Vector Machine (SVM)
* Logistic Regression
* AdaBoostM1

All models are evaluated using **10-fold cross-validation**.

---

## Evaluation Metrics

* Accuracy
* Precision, Recall, F1-score (especially for minority class “Yes”)
* ROC Area
* Kappa Statistic
* Runtime (computational cost)

---

## Key Findings

* **Random Forest (tuned on balanced raw data)** achieved the best overall performance.
* **IBk** showed the highest recall for the minority class.
* Baseline models mainly serve as performance references.
* Feature engineering and hyperparameter tuning significantly improved results.

Detailed analysis is presented in `GroupS_DataMining_Project.pdf`.

---

## Requirements

* Java 8 or higher
* WEKA library (`weka.jar`)
* SMOTE library (`SMOTE.jar`)

---

## Notes

* The program supports both **relative and absolute file paths**.
* All datasets and scripts are organized according to the project requirements.

---

## References

* Kaggle Heart Disease Dataset
* WEKA Documentation
* WEKA Java API Tutorials

---

