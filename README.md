# 📊 Telco Customer Churn Analysis & Predictive Modeling

[[https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white]](https://www.python.org/)
[[https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white]](https://scikit-learn.org/)
[[https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=white]](https://pandas.pydata.org/)
[[https://img.shields.io/badge/License-MIT-yellow.svg]](https://opensource.org/licenses/MIT)

> An end-to-end data analytics and machine learning project uncovering customer attrition drivers, segmenting high-risk cohorts via K-Means clustering, and deploying an NLP-inspired Support Vector Classifier achieving **78% churn recall** for proactive customer retention.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Business Insights](#-key-business-insights)
- [Customer Segmentation (K-Means)](#-customer-segmentation-k-means)
- [Subgroup & Behavioral Analysis](#-subgroup--behavioral-analysis)
- [Machine Learning & NLP Formulation](#-machine-learning--nlp-formulation)
- [Strategic Retention Recommendations](#-strategic-retention-recommendations)
- [Project Structure](#-project-structure)
- [Getting Started & Installation](#-getting-started--installation)
- [Usage Guide](#-usage-guide)
- [Technologies Used](#-technologies-used)
- [License](#-license)

---

## 🎯 Project Overview

Customer churn poses a significant revenue leak for telecommunication providers. Retaining an existing customer is estimated to cost **5x less** than acquiring a new one.

This project delivers a multi-stage analytical investigation into a 7,043-customer telecommunications dataset:
1. **Data Cleaning & Preprocessing:** Resolving missing values, numeric type coercions, and target encoding.
2. **Exploratory Data Analysis (EDA):** Identifying multi-dimensional correlations between demographics, contract structures, charges, and churn.
3. **Unsupervised Clustering:** Segmenting customer base into 3 distinct behavioral cohorts using **K-Means** on normalized behavioral features (`tenure`, `MonthlyCharges`, `TotalCharges`).
4. **Predictive Modeling via Text Representation:** Innovatively serializing tabular customer profiles into natural language risk narratives, vectorizing with **TF-IDF (1-3 n-grams)**, and training a cost-sensitive **Linear Support Vector Classifier (LinearSVC)** optimized for **Recall**.
5. **Actionable Business Strategy:** Translating empirical findings into customer retention playbooks.

---

## 🔍 Key Business Insights

| Metric / Cohort | Finding | Business Implication |
|---|---|---|
| **Overall Baseline Churn** | **26.5%** of customer base churned | High urgency for targeted retention interventions. |
| **Contract Structure** | Month-to-month: **42.7%** churn<br>1-Year: **11.3%** churn<br>2-Year: **2.8%** churn | Month-to-month contracts are 15x more likely to churn than 2-year commitments. |
| **Internet Technology** | Fiber Optic: **41.9%** churn<br>DSL: **19.0%** churn<br>No Internet: **7.4%** churn | Fiber optic subscribers suffer from pricing/expectation mismatches. |
| **Tech Support Gap** | Customers without Tech Support: **41.6%** churn<br>With Tech Support: **15.2%** churn | Lack of technical support is a leading driver of customer attrition. |
| **Payment Channel** | Electronic Check users show the highest churn rates across all payment types. | Friction or lack of automation correlates with higher drop-off. |
| **Senior Citizens** | Churn rate of **41.7%** vs **23.6%** for non-seniors | Requires dedicated, high-touch support channels. |

---

## 👥 Customer Segmentation (K-Means)

Features utilized: `tenure`, `MonthlyCharges`, and `TotalCharges` (standardized via `StandardScaler`). Using the **Elbow Method** ($k=1$ to $10$), the optimal cluster count was identified as $k=3$:

| Segment | Label | Avg Tenure | Avg Monthly Charges | Churn Rate | Action Priority |
|:---:|:---|:---:|:---:|:---:|:---|
| **0** | **At-Risk Newcomers** | 13.3 months | $75.07 | **47.09%** | 🚨 **Urgent:** 90-day onboarding discounts & proactive check-ins |
| **1** | **High-Value Loyalists** | 58.6 months | $89.67 | **15.36%** | 💎 **Retention:** VIP loyalty perks & contract renewals |
| **2** | **Budget Steady Users** | 29.4 months | $26.65 | **12.44%** | 🛡️ **Stable:** Upsell opportunities & cross-selling services |

---

## 🧪 Subgroup & Behavioral Analysis

Analyzing the interaction between **Monthly Charges Tiers** and **TechSupport**:
Monthly Charges Tier      TechSupport     Churn Rate
(18.15, 51.75] (Low)       No              36.68%
Yes             14.98%
(51.75, 85.25] (Medium)    No              41.80%
Yes             11.07%  <-- 3.8x churn reduction
(85.25, 118.75] (High)     No              44.23%
Yes             19.26%  <-- 2.3x churn reduction
> **Key Finding:** Having TechSupport reduces churn by up to **73%** in mid-to-high billing tiers.

---

## 🤖 Machine Learning & NLP Formulation

Rather than standard tabular encoding alone, customer profiles are serialized into contextual risk narratives:

```text
"month to month contract extremely short tenure high risk high charges expensive no tech support problem fiber optic expensive fast electronic check risky payment
Test Set Accuracy: 73.1%

              Precision    Recall  F1-Score   Support
   No Churn        0.90      0.71      0.80      1035
      Churn        0.50      0.78      0.61       374

   accuracy                            0.73      1409
  macro avg        0.70      0.75      0.70      1409
weighted avg       0.79      0.73      0.75      1409
