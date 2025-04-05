# Kaggle Models Collection

This repository contains a collection of 12 machine learning models built for various Kaggle competitions. Each model demonstrates different techniques, including feature engineering, various machine learning algorithms, and performance optimization. Below is a brief overview of the projects and the techniques used for each.

## Models Overview

### 1. **Regression of Used Car Prices**
   - **Model:** XGBoost
   - **Description:** This project aimed to predict the prices of used cars based on several features such as car age, mileage, and brand. The task involved cleaning the dataset, handling missing values, and performing feature engineering to extract meaningful   
                      information. XGBoost was selected due to its efficiency in handling regression tasks with tabular data. I experimented with hyperparameter tuning and feature transformations to achieve the best performance for this regression task.

### 2. **Regression with an Insurance Dataset**
   - **Model:** Neural Network (5 layers with skip connections inspired by ResNet), LGBM
   - **Description:** The goal of this project was to predict insurance charges for individuals based on their age, sex, BMI, and other personal factors. I used a neural network model consisting of 5 layers, incorporating skip connections (inspired by ResNet) to allow for better learning and generalization. To compare performance, I also used LGBM, a gradient boosting model known for its fast training and high accuracy, to identify the best model for the task.

### 3. **Exploring Mental Health Data**
   - **Model:** LGBM
   - **Description:** This project focused on analyzing mental health patterns and understanding which demographic groups suffer the most from conditions like anxiety and depression. I performed extensive exploratory data analysis (EDA) to uncover patterns and correlations, such as how occupation, socioeconomic status, and lifestyle factors contribute to mental health outcomes. After identifying important features, I built an LGBM model to predict mental health outcomes and assess the key drivers behind them.
     
### 4. **Loan Approval Prediction**
   - **Model:** XGBClassifier
   - **Description:** The goal of this project was to predict whether a loan application would be approved based on the applicant's personal and financial details. I performed detailed feature engineering to create meaningful inputs for the model, including handling categorical variables and scaling continuous features. I chose XGBoost for classification due to its high performance and ability to handle imbalanced data. Additionally, I conducted analysis on different groups of applicants to understand patterns in loan approval.

### 5. **Student Performance Prediction**
   - **Model:** Various models (XGBoost, RandomForest, etc.)
   - **Description:**  This project involved analyzing student performance based on factors such as study time, parental education, and socioeconomic status. Through extensive EDA, I explored which student groups perform better and how factors like financial constraints or mental health influence academic outcomes. I then built several machine learning models, including XGBoost and RandomForest, to predict student scores and understand the underlying reasons for disparities in performance.

### 6. **Digit Recognizer**
   - **Model:** Neural Network (2 layers)
   - **Description:** The goal of this project was to classify handwritten digits from the MNIST dataset. I built a simple neural network with two layers: 64 units in the first layer with batch normalization and dropout for regularization, and 32 units in the second layer. The model achieved an impressive 97% accuracy, demonstrating the effectiveness of even relatively simple neural networks for image classification tasks.

### 7. **Titanic Survival Prediction**
   - **Model:** XGBClassifier
   - **Description:** This classic Kaggle problem required predicting which passengers survived the Titanic disaster. I performed significant feature engineering, handling missing data and converting categorical features into numerical ones. The model used a 3-layer neural network architecture (64, 32, 16 units) with XGBoost for classification. The model performed well and ranked in the top 300, demonstrating the power of XGBoost combined with thoughtful feature engineering.

### 8. **Spaceship Titanic**
   - **Model:** Logistic Regression
   - **Description:** In this competition, the goal was to predict passenger survival in a spaceship disaster, similar to the Titanic competition but with a futuristic twist. I performed feature engineering, transforming categorical variables and handling missing data. A logistic regression model was used for classification, and I achieved a rank in the top 150 by carefully tuning the model and engineering useful features from the dataset.
     
### 9. **March Machine Learning Mania 2025**
   - **Model:** RandomForestRegressor
   - **Description:** The objective of this competition was to predict the outcome of March Madness basketball games. I performed extensive feature engineering, creating inputs such as team statistics and previous tournament performance. I used a RandomForestRegressor for the task, which allowed for handling both numerical and categorical features effectively. The model ranked in the top 700 out of 1745 participants, showcasing the ability to model sports predictions with machine learning.

### 10. **House Prices - Advanced Regression Techniques**
   - **Model:** XGBRegressor
   - **Description:** This competition involved predicting house prices based on a wide range of features, including the number of rooms, location, and the condition of the property. I used advanced regression techniques to create new features and handle missing data. XGBoost was chosen for regression due to its ability to deal with complex datasets and overfitting, and after thorough feature engineering, the model delivered excellent performance.

### 11. **Binary Prediction with a Rainfall Dataset**
   - **Model:** XGBClassifier
   - **Description:** This binary classification problem aimed to predict whether it would rain on a given day. I performed feature engineering to extract meaningful insights from the dataset, such as the month, temperature, and humidity levels. The model was trained using XGBoost, and I ensured proper data cleaning and handling of missing values to improve prediction accuracy.

### 12. **Binary Prediction of Poisonous Mushrooms**
   - **Model:** XGBClassifier
   - **Description:** The goal was to predict whether a mushroom was poisonous or not based on various features such as cap shape, color, and odor. I applied feature engineering to convert raw data into usable inputs and used XGBoost for classification. The model successfully predicted poisonous mushrooms with high accuracy after performing rigorous feature transformation and data cleaning.

## Folder Structure

```
/models
  ├── Binary Prediction of Poisonous Mushrooms Model
  ├── Binary Prediction with a Rainfall Dataset
  ├── House Prices - Advanced Regression Techniques
  ├── March Mania Model, Feature Engineering & Prediction.ipynb
  ├── README.md
  ├── Spaceship Titanic Model
  ├── Titanic Model
  ├── digit-recognizer.ipynb
  ├── eda-on-student-performance.ipynb
  ├── loan-approval-prediction-Model.ipynb
  ├── mental-health-eda-model.ipynb
  ├── neural-lgbm-insurance-predictor.ipynb
  └── regression-of-used-car-prices-Model.ipynb

```

Each model is organized by competition and includes the following:
- **Jupyter Notebooks**: Step-by-step implementation of the project.
- **Data**: The dataset used provided by Kaggle.
- **Feature Engineering**: A section dedicated to how features were engineered for each model.
- **Model Evaluation**: Metrics and results of model performance.

## Installation

To run the notebooks and models, clone this repository and install the necessary dependencies using the following commands:

```bash
git clone https://github.com/NoobML/kaggle-models.git
cd kaggle-models
pip install -r requirements.txt
```

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- TensorFlow/Keras (for deep learning models)
- Matplotlib/Seaborn (for visualization)

