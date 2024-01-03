## Loan Approval Prediction Solution

# Overview

This solution is designed to predict loan approval status based on various input features such as gender, age, income, region, loan amount, loan purpose, loan type, use type, credit type, loan limit, credit worthiness, credit score, property value, and more. The predictive model employed in this solution is a Random Forest Classifier, trained on historical data to learn patterns and make predictions regarding loan approval.

# Files Included
# EDA and Model Training File (loan_prediction.ipynb):

The notebook contains exploratory data analysis (EDA) to understand the dataset and the preprocessing steps.
It also includes the training of machine learning models, such as K-Nearest Neighbors, Gaussian Naive Bayes, Random Forest Classifier, and Gradient Boosting Classifier.
Fairness metrics, including Demographic Parity and Equalized Odds, are computed and evaluated.

# Dash App File (app.py):

This is a Dash web application designed to interactively predict loan approval status.
The app loads a pre-trained Random Forest Classifier model and uses it to make predictions based on user input.
Users can input information such as gender, age, income, region, loan amount, and more to receive a loan approval prediction.
Training Data Pickle File (training_data.pkl):

A serialized file containing the training and testing datasets, as well as the corresponding target variables. This file is used by the Dash app to load the necessary data for making predictions.

# Solution Details
Exploratory Data Analysis (EDA)
Data Cleaning and Feature Selection:

Several columns are dropped based on their lack of relevance to the loan approval prediction task.
A new column, 'risk_category,' is created by categorizing the 'Status' column into 'Low Risk' and 'High Risk.'
Handling Missing Values:

Missing values are visualized using a matrix, and then appropriate imputation strategies are applied.
Numerical features are imputed with the mean, while categorical features are imputed with the mode.
Outlier Detection and Removal:

Outliers are detected using the Z-score method, and the corresponding rows are removed to enhance model performance.
Data Encoding:

Label encoding is applied to convert categorical features into numerical format for model training.
Correlation Analysis:

A correlation matrix heatmap is created to visualize the relationships between numerical features.

# Model Training
Machine Learning Models:

K-Nearest Neighbors, Gaussian Naive Bayes, Random Forest Classifier, and Gradient Boosting Classifier are trained on the preprocessed dataset.
Model Evaluation:

Accuracy, confusion matrix, and classification reports are provided for each trained model.
Fairness Evaluation
Demographic Parity Difference:

Demographic parity difference is calculated for gender, region, and age groups to evaluate fairness.
Equalized Odds Difference:

Equalized odds difference is calculated, assuming 'age' as a continuous variable converted to binary.

# Dash Web Application
Interactive Loan Approval Prediction:
The Dash app allows users to input various details, and the pre-trained Random Forest Classifier makes loan approval predictions.
Predictions are displayed on the app interface, indicating whether the loan is approved or not.

# How to Use the Dash App
Run the Dash App:

Execute the Dash app file (app.py) to start the web application.
Access the application through a web browser.
Input User Details:

Enter relevant details such as gender, age, income, region, loan amount, loan purpose, and more.
Click 'Predict':

Click the 'Predict' button to trigger the loan approval prediction based on the provided information.
View Prediction:

The app will display the prediction result, indicating whether the loan is approved or not.

# Important Note
Ensure that you have the necessary Python libraries installed. 
Replace the placeholder model (RandomForestClassifier()) in the Dash app with your actual pre-trained model before deploying it in a production environment.
This comprehensive solution provides a detailed exploration of the loan approval prediction task, including data preprocessing, model training, fairness evaluation, and an interactive web application for real-time predictions.
