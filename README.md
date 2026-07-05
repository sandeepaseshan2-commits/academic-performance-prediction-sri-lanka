# Academic Performance Prediction - Private Campus Students

## IT41043 Intelligent Systems Research Assignment

---

## Group Members

- **W. Seshan Sandeepa** – ITBIN-2312-0024 – Data Preprocessing / Model Evaluation / Documentation
- **Wathsala Kithulgala** – ITBIN-2312-0025 – Dataset Collection / Literature Review / Methodology Writing

---

## Project Overview

This project is developed for the IT41043 Intelligent Systems research assignment.

The research topic is:

**A Comparative Study of Academic Performance Prediction Among Sri Lankan Private Campus Students Using Survey-Based Factors**

The main goal of this research is to predict the academic performance level of Sri Lankan private campus undergraduate students using survey-based academic, behavioural, lifestyle, and learning-access factors.

The academic performance levels used in this project are:

- Low
- Average
- High

This project does not use official university records or exact GPA values. Instead, it uses anonymous survey data collected through Google Forms.

---

## Dataset

The dataset was collected using an anonymous Google Form survey shared among Sri Lankan private campus students.

Current dataset size:

- **111 survey responses**

The survey includes factors such as:

- Attendance level
- Weekly self-study hours
- LMS usage frequency
- Assignment submission habit
- Class participation
- Sleep hours
- Motivation level
- Internet access quality
- Part-time work status
- Academic stress level
- Time management ability
- Study resources

The survey did not collect names, student IDs, phone numbers, email addresses, NIC numbers, or exact GPA values from respondents.

---

## Machine Learning Models

This project compares three machine learning models:

- Logistic Regression
- Decision Tree
- Random Forest

Logistic Regression is used as the baseline model. Decision Tree and Random Forest are used as comparison models.

---

## Evaluation Plan

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Stratified 5-fold cross-validation

Because the dataset has class imbalance, macro-average evaluation metrics will be used to compare the models more fairly.

---

## Tools and Technologies

- Google Forms
- Google Sheets
- Python
- pandas
- NumPy
- GitHub
- Microsoft Word
- draw.io

---

## Folder Structure

```text
data/raw/          Original survey data or data access note
data/processed/    Cleaned dataset
src/               Python code files
diagrams/          System architecture diagram
results/           Future result files
docs/              Survey and project notes
