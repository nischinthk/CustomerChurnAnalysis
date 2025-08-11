
📌 Project Overview
This project demonstrates a binary classification pipeline using Python and multiple machine learning models.
The workflow includes:

Data preprocessing

Exploratory Data Analysis (EDA)

Handling class imbalance with SMOTE

Model training & evaluation using:

Logistic Regression

Random Forest Classifier

Support Vector Machine (SVM)

Dimensionality reduction with PCA

Performance visualization

📂 Dataset
File: binary_classification_data2.csv

The dataset contains multiple features and a target column Churn.

The goal is to predict whether a customer will churn (Yes/No).

⚙️ Technologies Used
Python (Pandas, NumPy, Matplotlib, Seaborn)

Scikit-learn (LogisticRegression, RandomForestClassifier, SVC, PCA, SMOTE)

Imbalanced-learn (SMOTE for class balancing)

📊 Project Workflow
Data Import & Cleaning

Load CSV file

Drop unnecessary columns

Encode categorical features using LabelEncoder

Exploratory Data Analysis (EDA)

Boxplots to detect outliers

Histograms for feature distribution

Correlation heatmap

Data Preprocessing

Scaling numerical features

Balancing dataset using SMOTE

Model Training

Logistic Regression

Random Forest Classifier

Support Vector Machine (SVM) with GridSearch for hyperparameter tuning

Dimensionality Reduction

Principal Component Analysis (PCA) to reduce features and visualize results

Model Evaluation

Accuracy score

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

ROC-AUC Curve# CustomerChurnAnalysis
