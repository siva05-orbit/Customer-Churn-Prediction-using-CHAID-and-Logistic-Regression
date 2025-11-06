# Customer-Churn-Prediction-using-CHAID-and-Logistic-Regression
Predicting customer churn using a hybrid approach that combines CHAID (Chi-square Automatic Interaction Detection) for rule-based insights and Logistic Regression for probabilistic prediction. Includes full EDA, model evaluation (ROC, Lift, and Gains charts), and interpretable business insights for retention strategy.


# Customer Churn Prediction using CHAID and Logistic Regression

This project predicts customer churn using both statistical and machine learning methods.  
It combines **CHAID rule induction** for interpretable decision rules and **Logistic Regression** for accurate churn probability estimation.  
The analysis is performed on a telecom churn dataset and includes complete visualization, evaluation, and deployment steps.

---

##  Project Overview

- **Objective:** Identify key factors influencing customer churn and build a predictive model for retention strategy.
- **Techniques Used:**  
  - Exploratory Data Analysis (EDA)  
  - CHAID (Chi-square Automatic Interaction Detection)  
  - Logistic Regression (with scaling and evaluation metrics)
- **Evaluation Metrics:**  
  Accuracy, ROC Curve (AUC), Confusion Matrix, Lift Chart, and Gains Chart.
- **Deployment:**  
  Model exported using `joblib` for future predictions and updating with new customer data.

---

##  Features and Workflow

1. **Data Cleaning and Preprocessing**
   - Removed redundant “charge” columns (derived from minutes)
   - Encoded categorical variables (Yes/No → 1/0)
   - Normalized numeric features

2. **Exploratory Data Analysis (EDA)**
   - Visualized churn distribution and feature correlations  
   - Identified top influencing factors:
     - Customer service calls  
     - International plan  
     - Voicemail plan  
     - Daytime usage minutes  

3. **CHAID Rule Induction**
   - Extracted human-readable decision rules explaining churn patterns  
   - Interpreted rules in a business context for actionable insights

4. **Logistic Regression Model**
   - Built and trained a predictive model on scaled features  
   - Evaluated using AUC, precision, recall, and accuracy metrics

5. **Model Evaluation**
   - ROC Curve (AUC ≈ 0.82)  
   - Lift and Gains Charts to assess business effectiveness  

6. **Model Deployment**
   - Model and scaler exported via `joblib`  
   - Includes a prediction function for new customer data  
   - Describes model updating workflow using fresh data

---

##  Key Results

| Metric | Value |
|--------|--------|
| Accuracy | ~82% |
| ROC AUC | 0.82 |
| Lift (Top Decile) | 1.9× |
| Gains | Captures ~60% churners in top 30% customers |

**Business Insights:**
- Customers with **International Plan** and **>3 customer service calls** are at the highest churn risk.  
- Customers with **Voicemail Plan** show the lowest churn probability.  
- Heavy **daytime usage** and **high billing** increase churn likelihood.

---

## Tech Stack

- **Language:** Python  
- **Libraries:** pandas, seaborn, matplotlib, scikit-learn, CHAID, joblib  
- **Tools:** Jupyter Notebook / VS Code  
- **Dataset:** Telecom churn dataset (`churn.csv`)

---

##  Deployment and Updating

- Model saved using `joblib.dump()`  
- Reload anytime for predictions:
  ```python
  model = joblib.load("churn_model.joblib")
  scaler = joblib.load("scaler.joblib")
