## End to End Machine Learning project
# 💸 Loan Default Prediction & Cost Optimization

![Project Banner](https://images.unsplash.com/photo-1554224155-8d04cb21cd6c?auto=format&fit=crop&q=80&w=1000)

## 📖 Overview

This project focuses on predicting loan defaults using **Logistic Regression** and **XGBoost**. Unlike standard classification tasks that aim for high accuracy, this workflow is optimized for a specific **Business Cost Function**.

In the lending industry, the cost of errors is asymmetric. Approving a borrower who defaults is significantly more expensive (principal loss) than rejecting a good borrower (opportunity cost). This project builds predictive models and fine-tunes the decision threshold to minimize the total financial impact based on these real-world constraints.

----

## 📌 Problem

A bank needs to decide whether to approve or reject loan applications.  
Each prediction has a financial consequence:

- **False Negative (FN)**  
  Borrower **defaults**, but the model predicted **safe**  
  ➝ Loss = **₹100,000**

- **False Positive (FP)**  
  Borrower would have repaid, but the model predicted **risky**  
  ➝ Opportunity loss = **₹1,000**

Since FN is 100× more expensive than FP, simply maximizing accuracy is **not** enough.  
We must choose the threshold that **minimizes business cost**, not classification score.
 ----


## 🛠️ Tech Stack & Libraries

* **Python**
* **Pandas & NumPy:** For data manipulation and numerical operations.
* **Scikit-learn:** For preprocessing (StandardScaler), modeling, and evaluation.
* **XGBoost:** For advanced gradient boosting classification.
* **Matplotlib & Seaborn:** For visualizing cost curves and feature importance.

----

## 💰 Business Constraints & Methodology

The models are evaluated based on a custom financial cost matrix rather than just accuracy or F1-score.
| **False Negative (FN)** | Approving a defaulter | **$100,000** |
| **False Positive (FP)** | Rejecting a repayer | **$1,000** |

* **Threshold Optimization:** The project iterates through probability thresholds (0.01 to 0.99) to find the "sweet spot" where the total business cost is minimized.
* **Handling Imbalance:** `scale_pos_weight` and `class_weight='balanced'` were used to address class imbalance.

----

## 📈 Model Performance & Results

We compared a baseline Logistic Regression model against an XGBoost classifier. While XGBoost showed superior standard performance metrics, the Logistic Regression model was found to be the strategically superior choice, as its optimized decision threshold yielded a more reliable, risk-averse profile essential for minimizing catastrophic financial loss.

* **Metric:** Total Business Cost (Lower is better)
* **Optimization:** The decision threshold was shifted from the standard 0.5 to a cost-optimal value.

### Confusion Matrix (XGBoost at Optimal Threshold)

The table below visualizes the performance of the best model at the cost-minimized threshold.

*(Replace values below with your specific script output)*

| | **Predicted: Repay** | **Predicted: Default** |
| :--- | :--- | :--- |
| **Actual: Repay** | 104 (TN) | 22,302 (FP) |
| **Actual: Default** | 1 (FN) | 7,327 (TP) |

* **True Positives (TP):** Defaulters correctly identified and rejected (Money Saved).
* **False Negatives (FN):** Defaulters incorrectly approved (Money Lost - **Critical**).
* **False Positives (FP):** Good borrowers rejected (Opportunity Lost).

### Top Features Driving Risk

Feature importance analysis was conducted using XGBoost to identify the key drivers of default risk. The top features influencing the predictions include:

1.  **Interest_rate_Spread**
2.  **Upfront_Charges**
3.  **rate_of_interest**
4.  **Credit_type_EQUI**
5.  **lump_sum_payment_not_lpsm**
6.  **LTV**
7.  **Gender_Sex_Not_Available**
8.  **submission_of_application_to_inst**
9.  **dtir1**
10. **income**

*(Note: Features are automatically sanitized and ranked by importance in the script).*

----

## 🏁 Conclusion

The analysis confirms that optimizing for accuracy alone is dangerous in financial modeling. By shifting the decision threshold based on actual business costs ($100k vs $1k), we achieved a strategy that minimizes potential losses.

**Key Findings:**
* The XGBoost model, with its near-perfect metrics (ROC AUC $\approx 1.0$) and extremely low cost ($\$1,000$), indicates an anomaly such as data leakage or a feature directly encoding the target variable. In a real-world financial audit, this result would be rejected as unrealistic.We, therefore, focus on the more conservative and strategically stable performance of the Logistic Regression model.

* The **Optimal Threshold** is significantly different from the default 0.5, reflecting the high cost of False Negatives.

