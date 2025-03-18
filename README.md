# Diabetes Prediction using Machine Learning

## About Project
This project focuses on predicting the onset of diabetes using the **Pima Indians Diabetes Dataset**. The goal is to build a machine learning model that can accurately classify whether a patient has diabetes based on diagnostic measurements. The project is part of a healthcare analytics initiative to improve early diagnosis and patient outcomes.

---

## Project Overview
- **Objective:** Develop a state-of-the-art system to predict diabetes using machine learning.
- **Dataset:** Pima Indians Diabetes Database from Kaggle.
- **Tools Used:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and MLxtend.
- **Models Implemented:** Logistic Regression, Decision Trees, K-Nearest Neighbors (KNN), Random Forest, and Stacking Classifier.
- **Evaluation Metrics:** Accuracy, Precision, Recall, and F1 Score.

---

## Key Insights
1. **Dataset Overview:**
   - The dataset contains 8 features and 1 target variable (`Outcome`).
   - The target variable is imbalanced, with 500 instances of "No Diabetes" and 268 instances of "Diabetes."

2. **Missing Values:**
   - Missing values (represented as 0s) were replaced with the mean of the respective columns.

3. **Feature Distributions:**
   - Features like `Insulin` and `SkinThickness` have significant outliers.
   - `Glucose`, `BMI`, and `Age` are the most important features for prediction.

4. **Model Performance:**
   - **Logistic Regression:** Moderate performance with low recall.
   - **Decision Tree:** Improved performance after hyperparameter tuning.
   - **KNN:** Similar performance to Logistic Regression but computationally expensive.
   - **Random Forest:** Best-performing model with high accuracy and F1 score.
   - **Stacking Classifier:** Achieved the highest accuracy (81%) by combining KNN and Random Forest.

5. **Recommendations:**
   - Address class imbalance using techniques like SMOTE or class weighting.
   - Experiment with advanced models like XGBoost or Neural Networks.
   - Collect more data for the minority class to improve recall.

---

## Analysis Techniques
1. **Data Preprocessing:**
   - Handling missing values by replacing 0s with the mean.
   - Scaling features using `StandardScaler`.

2. **Exploratory Data Analysis (EDA):**
   - Count plots, boxplots, pairplots, histograms, and correlation heatmaps.

3. **Model Building:**
   - Implemented Logistic Regression, Decision Trees, KNN, Random Forest, and Stacking Classifier.
   - Hyperparameter tuning using `RandomizedSearchCV`.

4. **Evaluation:**
   - Metrics: Accuracy, Precision, Recall, and F1 Score.
   - Compared performance across all models.

---

## Dataset
- **Source:** [Pima Indians Diabetes Database | Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features:**
  - `Pregnancies`: Number of times pregnant.
  - `Glucose`: Plasma glucose concentration.
  - `BloodPressure`: Diastolic blood pressure.
  - `SkinThickness`: Triceps skinfold thickness.
  - `Insulin`: 2-Hour serum insulin.
  - `BMI`: Body mass index.
  - `DiabetesPedigreeFunction`: Diabetes pedigree function.
  - `Age`: Age in years.
- **Target Variable:**
  - `Outcome`: 1 (Diabetes), 0 (No Diabetes).

---

## Files
1. **`diabetes.csv`:** The dataset file.
2. **`diabetes_prediction.ipynb`:** Jupyter Notebook containing the complete analysis and code.
3. **`README.md`:** Project overview and documentation.
4. **`requirements.txt`:** List of Python libraries required to run the project.

---

## Results Summary
| Model                  | Accuracy | Precision | Recall | F1 Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression    | 76%      | 69%       | 55%    | 61%      |
| Decision Tree          | 78%      | 72%       | 60%    | 65%      |
| KNN                    | 75%      | 68%       | 58%    | 63%      |
| Random Forest          | 80%      | 75%       | 65%    | 70%      |
| Stacking Classifier    | 81%      | 76%       | 66%    | 71%      |

- **Best Model:** Stacking Classifier (KNN + Random Forest) with 81% accuracy.
- **Key Takeaway:** Ensemble methods like Random Forest and Stacking Classifier outperform simpler models, highlighting their effectiveness in handling imbalanced datasets.

