# Task 37: Logistic Regression (Titanic Dataset)

**Track:** AI & ML Track 

## Overview
This project builds a basic binary classification model using **Logistic Regression** in Scikit-learn to predict passenger survival (`survived`) on the Titanic dataset. The goal is to understand data preparation for classification, model training, probability-based predictions, and performance evaluation.

## Tools & Libraries Used
* **Python**
* **Pandas & NumPy** (Data manipulation and preprocessing)
* **Scikit-learn** (Train-test split, Logistic Regression model, evaluation metrics)
* **Matplotlib & Seaborn** (Confusion matrix visualization)
* **Jupyter Notebook**

## Dataset & Preprocessing
The Titanic dataset (`titanic.csv`, 891 rows) was prepared using the following steps:
1. **Feature Selection:** Selected `pclass`, `sex`, `age`, `sibsp`, `parch`, and `fare` as predictor features, and `survived` as the binary target variable (`0` = Did Not Survive, `1` = Survived).
2. **Handling Missing Values:** Filled missing values in the `age` column using the median age.
3. **Categorical Encoding:** Mapped `sex` from text to binary numerical values (`male`: 0, `female`: 1).
4. **Train-Test Split:** Split the dataset into 80% training data (712 samples) and 20% testing data (179 samples) with `random_state=42`.

## Model Explanation
**Logistic Regression** is a supervised machine learning algorithm used for binary classification. Unlike Linear Regression, which outputs continuous numbers, Logistic Regression passes the linear combination of inputs through the **Sigmoid function** to output a probability score between 0 and 1. Using a standard threshold of 0.5:
* Probabilities greater than or equal to 0.5 are classified as **1** (Survived).
* Probabilities less than 0.5 are classified as **0** (Did Not Survive).

## Model Results & Evaluation
The model achieved an overall accuracy of **81.01%** on the test set (179 passengers).

### Classification Report
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Did Not Survive)** | 0.81 | 0.88 | 0.84 | 105 |
| **1 (Survived)** | 0.80 | 0.72 | 0.76 | 74 |
| **Accuracy** | | | **0.81** | **179** |
| **Macro Avg** | 0.81 | 0.80 | 0.80 | 179 |
| **Weighted Avg** | 0.81 | 0.81 | 0.81 | 179 |

### Confusion Matrix Breakdown
* **True Negatives (92):** Correctly predicted as Did Not Survive.
* **False Positives (13):** Incorrectly predicted as Survived.
* **False Negatives (21):** Incorrectly predicted as Did Not Survive.
* **True Positives (53):** Correctly predicted as Survived.

### Result Screenshot
![Model Results and Confusion Matrix](Screenshot%202026-11-15%20171741.png)

---

## Interview Questions & Answers

### Q1: What is Logistic Regression?
**Answer:** Logistic Regression is a supervised machine learning and statistical classification algorithm used to predict the probability of a categorical dependent variable (most commonly binary, such as 0 or 1, Yes or No) based on one or more independent input features.

### Q2: Why is Logistic Regression used for classification?
**Answer:** Linear Regression outputs unbounded continuous values from negative infinity to positive infinity, which does not work well for discrete categories. Logistic Regression applies the sigmoid (logistic) function to squash the output into a strict range between 0 and 1. This output represents a probability that can easily be converted into discrete class labels using a decision threshold (typically 0.5).

### Q3: What is the difference between Linear and Logistic Regression?
**Answer:**
* **Linear Regression** is used for regression tasks to predict continuous numerical values (such as house prices or temperature) by fitting a straight line (`y = mx + c`).
* **Logistic Regression** is used for classification tasks to predict discrete categories (such as spam detection or survival) by fitting an S-shaped sigmoid curve (`sigma(z) = 1 / (1 + e^(-z))`) that outputs class probabilities.
