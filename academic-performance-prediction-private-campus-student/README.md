# Academic Performance Prediction Among Private Campus Students

This repository contains the implementation of the research project titled:

**“A Comparative Study of Academic Performance Prediction Among Sri Lankan Private Campus Students Using Survey-Based Factors.”**

This project was prepared for the **IT41043 Milestone 2 research assignment**.

---

## 1. Project Overview

Academic performance can be affected by several academic and behavioural factors, including attendance, study habits, assignment-submission behaviour, motivation, sleep, stress and access to learning resources.

This study uses survey-based data collected from undergraduate students to compare selected machine-learning models for predicting students’ academic performance levels.

The three predicted performance classes are:

- Low
- Average
- High

This is a quantitative, non-experimental study. The project examines predictive relationships and does not claim that the selected factors directly cause academic performance.

---

## 2. Research Aim

The aim of this study is to compare selected machine-learning models for predicting the academic performance level of Sri Lankan private campus undergraduate students using survey-based academic and behavioural factors.

---

## 3. Research Question

To what extent can survey-based academic and behavioural factors support the prediction of Low, Average and High academic performance among Sri Lankan private campus undergraduate students?

---

## 4. Research Objectives

1. To collect survey-based academic and behavioural data from private campus undergraduate students in Sri Lanka.
2. To preprocess the collected data and prepare it for machine-learning model development.
3. To develop Logistic Regression, Decision Tree and Random Forest classification models.
4. To compare the models using suitable classification metrics and statistical testing.

---

## 5. Dataset

Primary data were collected through an anonymous Google Form questionnaire.

The original dataset contained:

- 126 survey responses
- 23 questionnaire items
- 1 timestamp column
- 24 columns in total
- 1 missing value
- 0 exact duplicate responses

After applying consent and eligibility criteria, **97 records** remained for preliminary model development.

### Current eligible class distribution

| Academic Performance Level | Number of Records | Percentage |
|---|---:|---:|
| Low | 7 | 7.2% |
| Average | 65 | 67.0% |
| High | 25 | 25.8% |
| **Total** | **97** | **100.0%** |

The class distribution is imbalanced because the Average class contains more records than the Low and High classes.

The raw and processed datasets are not publicly included in this repository because they contain participant-derived information.

---

## 6. Eligibility Criteria

A response is included in the model-development dataset when the participant:

- Provided informed consent
- Confirmed that they are currently studying
- Confirmed that they study at a private higher-education institution
- Provided a valid academic-performance target value

Responses that did not meet these conditions were removed from the modelling dataset.

---

## 7. Selected Predictor Variables

The study uses the following 16 academic and behavioural predictor variables:

1. Year of study
2. Degree area
3. Study mode
4. Attendance
5. Daily study hours
6. LMS usage
7. Assignment-submission habits
8. Class participation
9. Sleep hours
10. Motivation level
11. Time-management ability
12. Internet quality
13. Part-time work
14. Academic stress
15. Travel time
16. Availability of study resources

The following fields are excluded from model training:

- Timestamp
- Consent information
- Eligibility-screening questions
- Gender
- Current CA marks range
- Academic-support requirement
- Other administrative fields

The CA marks range is excluded because it is closely connected to the target variable and could cause target leakage.

---

## 8. Target Variable

The target variable is:

```text
academic_performance_level
```

The target classes are encoded as:

| Encoded Value | Academic Performance Level |
|---:|---|
| 0 | Low |
| 1 | Average |
| 2 | High |

---

## 9. Machine-Learning Models

The following supervised classification models are implemented and compared:

### 9.1 Logistic Regression

Logistic Regression is used as the baseline model. It provides a simple and interpretable comparison point for the other models.

### 9.2 Decision Tree

Decision Tree can learn non-linear decision rules and provides an understandable tree-based model.

### 9.3 Random Forest

Random Forest combines multiple decision trees and can provide more stable predictions than a single Decision Tree.

---

## 10. Data-Preprocessing Workflow

The preprocessing process includes:

1. Loading the original Google Form CSV file
2. Cleaning and shortening column names
3. Removing exact duplicate responses
4. Checking missing values
5. Applying consent and eligibility criteria
6. Removing excluded variables
7. Separating predictors and the target variable
8. Encoding nominal variables
9. Encoding ordinal variables
10. Imputing missing predictor values
11. Scaling features only for Logistic Regression
12. Preparing the data for model training

Encoding, imputation and scaling are performed inside machine-learning pipelines to reduce data leakage.

---

## 11. Validation Strategy

The models are evaluated using:

- Stratified 5-Fold Cross-Validation
- Shuffling with a fixed random state
- Small 3-Fold GridSearchCV for hyperparameter tuning
- Balanced class weights

Stratified cross-validation is used to maintain approximately similar class proportions in each fold.

The random state is set to:

```text
42
```

---

## 12. Evaluation Metrics

The models are evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Class-wise Recall
- Confusion Matrix
- Mean and standard deviation across five folds

### Primary evaluation metric

**Macro F1-score** is used as the main model-comparison metric because the target classes are imbalanced.

Macro F1 gives equal importance to the Low, Average and High classes.

---

## 13. Statistical Comparison

The Macro F1-scores obtained from the cross-validation folds are statistically compared using:

1. Friedman test
2. Wilcoxon signed-rank tests, only when the Friedman test is significant
3. Holm correction for multiple pairwise comparisons

The statistical significance level is:

```text
α = 0.05
```

---

## 14. Preliminary Results

The following results are based on the current **97 eligible records**. These are preliminary results because survey-data collection may continue.

### Cross-validation results

| Model | Accuracy Mean ± SD | Macro F1 Mean ± SD |
|---|---:|---:|
| Logistic Regression | 0.6605 ± 0.0743 | 0.6292 ± 0.1535 |
| Decision Tree | 0.7632 ± 0.1137 | 0.7191 ± 0.1867 |
| Random Forest | **0.7842 ± 0.0739** | **0.7429 ± 0.1695** |

Random Forest achieved the highest mean Accuracy and mean Macro F1-score in the preliminary cross-validation results.

### Out-of-fold evaluation results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6598 | 0.6471 | 0.6637 | 0.6500 |
| Decision Tree | 0.7629 | 0.7671 | **0.7725** | 0.7534 |
| Random Forest | **0.7835** | **0.8201** | 0.7581 | **0.7773** |

### Class-wise recall

| Model | Low | Average | High |
|---|---:|---:|---:|
| Logistic Regression | 0.7143 | 0.6769 | 0.6000 |
| Decision Tree | 0.7143 | 0.7231 | **0.8800** |
| Random Forest | 0.7143 | **0.8000** | 0.7600 |

The Low class contains only seven records. Therefore, the class-wise results should be interpreted carefully.

### Statistical result

The Friedman test produced:

```text
Friedman statistic = 5.20
p-value = 0.0743
α = 0.05
```

Because the p-value is greater than 0.05, the test did not identify a statistically significant difference among the three models.

Therefore, pairwise Wilcoxon signed-rank tests were not performed.

---

## 15. System Architecture

The proposed system architecture diagram is available at:

```text
diagrams/system_architecture_milestone2.png
```

The main system workflow is:

```text
Google Form
    ↓
Raw Survey Dataset
    ↓
Data-Quality Checks
    ↓
Consent and Eligibility Filtering
    ↓
Data Preprocessing
    ↓
Stratified 5-Fold Cross-Validation
    ↓
Logistic Regression, Decision Tree and Random Forest
    ↓
Model Evaluation
    ↓
Statistical Comparison
    ↓
Best Model and Interpretation
```

---

## 16. Repository Structure

```text
academic-performance-prediction-private-campus-student/
│
├── data/
│   ├── raw/
│   │   ├── README.md
│   │   └── data_access_note.md
│   │
│   └── processed/
│       └── README.md
│
├── diagrams/
│   ├── README.md
│   └── system_architecture_milestone2.png
│
├── docs/
│
├── results/
│   └── README.md
│
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   └── evaluate_models.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 17. Main Python Files

### `src/preprocessing.py`

This script:

- Loads the raw survey data
- Cleans column names
- Checks missing values and duplicate responses
- Applies consent and eligibility criteria
- Selects the approved predictors
- Encodes the target variable
- Creates the cleaned modelling dataset
- Generates a preprocessing report

### `src/train_models.py`

This script:

- Loads the cleaned dataset
- Creates preprocessing pipelines
- Handles missing predictor values
- Encodes nominal and ordinal variables
- Scales Logistic Regression features
- Trains the three selected models
- Performs nested cross-validation
- Performs small-grid hyperparameter tuning
- Saves fold-level and summary results

### `src/evaluate_models.py`

This script:

- Generates out-of-fold predictions
- Calculates overall evaluation metrics
- Calculates class-wise recall
- Generates confusion matrices
- Runs the Friedman statistical test
- Performs pairwise tests only when required
- Saves evaluation and statistical results

---

## 18. Required Technologies

The project uses:

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- SciPy
- statsmodels
- Git
- GitHub

---

## 19. Installation Instructions

### Step 1: Clone the repository

```bash
git clone https://github.com/sandeepaseshan2-commits/academic-performance-prediction-sri-lanka.git
```

### Step 2: Open the project folder

```bash
cd academic-performance-prediction-sri-lanka/academic-performance-prediction-private-campus-student
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the environment on Windows

```bash
.venv\Scripts\activate
```

### Step 5: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 6: Install the required libraries

```bash
pip install -r requirements.txt
```

---

## 20. Running the Project

### Step 1: Add the private survey dataset

Place the Google Form CSV file inside:

```text
data/raw/
```

Rename the file as:

```text
raw_survey_responses.csv
```

### Step 2: Run preprocessing

```bash
python src/preprocessing.py
```

### Step 3: Train and compare the models

```bash
python src/train_models.py
```

### Step 4: Run the complete evaluation

```bash
python src/evaluate_models.py
```

---

## 21. Generated Output Files

The scripts generate files such as:

```text
data/processed/cleaned_survey_data.csv
data/processed/preprocessing_report.txt

results/cross_validation_scores.csv
results/model_summary.csv
results/best_parameters.json
results/evaluation_metrics.csv
results/class_wise_recall.csv
results/statistical_comparison.csv
results/out_of_fold_predictions.csv
results/confusion_matrix_logistic_regression.png
results/confusion_matrix_decision_tree.png
results/confusion_matrix_random_forest.png
```

These generated participant-derived files are protected through `.gitignore` and are not publicly uploaded.

---

## 22. Data Privacy and Ethical Handling

Participation in the survey was voluntary and based on informed consent.

The questionnaire did not require unnecessary personally identifying information.

The public GitHub repository does not contain:

- Raw participant responses
- Cleaned participant-level datasets
- Individual out-of-fold predictions
- Private generated result files

The following file patterns are excluded through `.gitignore`:

```text
data/raw/*.csv
data/processed/*.csv
data/processed/*.txt
results/*.csv
results/*.json
results/*.png
__pycache__/
*.pyc
.venv/
```

---

## 23. Current Project Status

The following components have been completed:

- GitHub repository structure
- Dataset-quality checking
- Consent and eligibility filtering
- Predictor selection
- Target encoding
- Data-preprocessing pipeline
- Logistic Regression implementation
- Decision Tree implementation
- Random Forest implementation
- Stratified 5-Fold Cross-Validation
- Hyperparameter tuning
- Model evaluation
- Class-wise recall calculation
- Confusion-matrix generation
- Statistical comparison
- System architecture diagram
- Data-privacy protection

The current model outputs are preliminary because survey-data collection may continue. Final conclusions should be based on the final approved dataset.

---

## 24. Group Members and Contributions

### W. Seshan Sandeepa

- **Index Number:** ITBIN-2312-0024
- **Roles:** Repository and Version-Control Coordinator, Data Preprocessing Developer, Model-Evaluation Developer
- **Main Contributions:**
  - Created and maintained the GitHub repository structure
  - Managed Git commits, version control and repository updates
  - Configured the Python virtual environment and project dependencies
  - Developed and tested the data-preprocessing workflow
  - Implemented eligibility filtering and variable selection
  - Implemented model-evaluation metrics
  - Generated confusion matrices and statistical comparisons
  - Integrated and tested the complete project workflow
  - Added the system architecture diagram and project documentation

### Wathsala Kithulgala

- **Index Number:** ITBIN-2312-0025
- **Roles:** Machine-Learning Model Developer, Research and Documentation Coordinator
- **Main Contributions:**
  - Supported the selection of research variables and model requirements
  - Contributed to the questionnaire and data-collection process
  - Supported the implementation of Logistic Regression, Decision Tree and Random Forest
  - Contributed to model-training and hyperparameter-tuning decisions
  - Supported the preparation of research methodology content
  - Contributed to dataset documentation and ethical-handling information
  - Reviewed model outputs and preliminary findings
  - Reviewed the final report and repository documentation

---

## 25. Academic Notice

This repository was created for an undergraduate academic research assignment.

The results should not be used to make high-stakes academic decisions about individual students. The models are developed for educational and research purposes using a limited survey-based dataset.