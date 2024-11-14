# SONAR Rock vs Mine Prediction with Python

This project is an end-to-end machine learning system that predicts whether an object detected by sonar is a rock or a mine. Using a logistic regression model, it performs binary classification on sonar data to differentiate rocks from mines. The project was implemented using Python in Google Colab, enabling streamlined data processing and model training.

## Overview
The objective of this project is to classify sonar signals reflected from metal cylinders (representing mines) and rocks. A logistic regression model is trained on a labeled dataset to predict whether the object is a rock or a mine, enabling submarines and other sonar-based systems to avoid mines.

## Dataset
The dataset used in this project is sourced from Kaggle, containing 208 samples with 60 features each, representing the response of sonar signals.

- **Download Link:** [Sonar Data on Kaggle](https://www.kaggle.com/datasets/rupakroy/sonarcsv)

## Installation
1. **Environment**: This project runs on Google Colab.
2. **Dependencies**: Install the necessary libraries (these are pre-installed on Google Colab):
    ```python
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    ```

## Project Workflow
1. **Data Loading**: Load the sonar data into a Pandas DataFrame.
2. **Data Preprocessing**: 
    - No column headers are present in the dataset, so it is loaded with `header=None`.
    - Data is split into features (first 60 columns) and labels (last column).
3. **Data Exploration**:
    - Analyze the distribution of rocks and mines.
    - Group by label to view mean feature values per class.
4. **Feature Selection**:
    - Separate data (features) and labels.
    - Split data into training and testing sets.

## Model Training and Evaluation
- **Model Selection**: A logistic regression model is chosen for this binary classification problem.
- **Model Training**: 
    - The model is trained on 90% of the data, leaving 10% for testing.
- **Evaluation**:
    - Accuracy on training and test sets is calculated.
    - Training accuracy is around 83%, while test accuracy is around 76%.

## Prediction System
A prediction system was implemented to classify new samples:
- Input sonar data (for either a rock or mine) is fed to the model, which then outputs a prediction.

## Resources
- **Video Reference**: [SONAR Rock vs Mine Prediction Project Video](https://youtu.be/fiz1ORTBGpY?si=fs5PkvCG36jGEiBi)

## License
This project is for educational purposes. Please refer to the data license provided by Kaggle if using the sonar dataset.
