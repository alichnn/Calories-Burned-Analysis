# Data Analysis on Calories Burned In the Gym

This project analyzes gym member workout data to explore differences in calorie expenditure between the genders and to predict calories burned.

---

## Featuring

- **Statistical Inference**  
  Conducted hypothesis testing using equal-tailed and one-sided t-tests to evaluate whether male and female gym members burn the same number of calories on average

- **Predictive Modeling**  
  Built linear regression model to predict calories burned from features such as age, BMI, workout type, and session duration

- **Feature Engineering & Preprocessing**  
  Applied ColumnTransformer from scikit-learn for handling categorical and numerical features

- **Visualization & EDA**  
  Used pandas, seaborn, and matplotlib for exploratory data analysis and visualizing relationships between workout attributes

---
```mermaid
flowchart TD
    A(Data Collection) --> B(Data Cleaning &<br/>Preprocessing)

    B --> C("Inference (Statistics)")
    B --> D("Prediction (ML/Regression)")

    C --> E("Hypothesis Testing<br/>(t-test, p-values)")
    E --> F("Interpret Feature Significance<br/>(α = 0.05, Bonferroni)")

    D --> G(Train/Test Split &<br/>Cross-validation)
    G --> H("Evaluate Predictive<br/>Performance (R², RMSE)")
    H --> F
 style F font-size:13px
```  

---

## About the Data
"This dataset provides a detailed overview of gym members' exercise routines, physical attributes, and fitness metrics. It contains 973 
samples of gym data, including key performance indicators such as heart rate, calories burned, and workout duration. Each entry also 
includes demographic data and experience levels, allowing for comprehensive analysis of fitness patterns, athlete progression, and 
health trends."

From [Kaggle](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset)

---

## Overview

**Libraries used:**
pandas, scikit-learn, scipy, matplotlib, seaborn, statsmodels

