# Finding Donors for CharityML

## Overview

This project aims to use supervised learning techniques to predict which individuals in the Census dataset earn more than $50,000 per year. This is useful for identifying potential donors for the non-profit organization, CharityML. The project involves data exploration, preprocessing, model evaluation, and feature importance analysis.

## Project Structure

1. **Data Exploration**
   - Calculate the total number of records.
   - Determine the number of individuals with income >$50,000 and <=$50,000.
   - Calculate the percentage of individuals with income >$50,000.

2. **Data Preprocessing**
   - Apply one-hot encoding to categorical features.
   - Process income data into binary labels.
   - Split the data into training and testing sets.

3. **Naive Predictor Benchmark**
   - Establish a benchmark using a naive predictor (assuming all individuals earn >$50,000).
   - Calculate accuracy and F1 score for comparison with actual models.

4. **Model Selection and Evaluation**
   - Implement the following supervised learning models:
     - Decision Trees
     - Random Forest Classifier
     - AdaBoost Classifier
   - Evaluate the pros and cons of each model based on performance metrics such as accuracy, F1 score, and training time.

5. **Model Optimization**
   - Select the best-performing model based on accuracy and F1 score.
   - Tune the model using grid search to optimize hyperparameters.

6. **Feature Importance**
   - Extract feature importances from the model.
   - Rank the top five most relevant features for predicting income.
   - Compare model performance using the top five features against using the entire feature set.

7. **Final Model Evaluation**
   - Report accuracy and F1 score for both the unoptimized and optimized models.
   - Provide a comparison of final model performance with naive predictors and earlier models.

## Installation

You will need the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `jupyter notebook`

Install the dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Running the Project
1. Data Preparation
   - Load the census dataset.
   - Perform one-hot encoding and data preprocessing.
   - Split the dataset into training and testing sets.

2. Model Training and Evaluation
   - Train the supervised learning models and evaluate their performance.
   - Compare results using accuracy, precision, recall, and F1 score.

3. Model Optimization
   - Perform grid search optimization on the chosen model.
   - Evaluate the performance of the optimized model.

4. Feature Importance
   - Analyze and compare the importance of features used by the model.
   - Experiment with reducing the feature set and assess the impact on model performance.

## Results
   - The final optimized model demonstrates significantly improved performance compared to the naive predictor.
   - Feature importance analysis shows which variables have the greatest impact on income prediction.

## Conclusion

This project demonstrates the application of supervised learning techniques to a real-world classification problem. By selecting and optimizing models, the project achieves improved predictive accuracy, making it a valuable tool for identifying potential donors.