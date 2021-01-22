# HouseSalesPrediction

_*The Problem*_

Create a Linear Regression model to predict house prices.

## _*Data*_

The data was extracted from Kaggle at the following link:
https://www.kaggle.com/harlfoxem/housesalesprediction

This data set contains the attributes of the houses as well as the price that was sold and the date of sale.

## _*Data Preparation*_

This data set has no missing values. 
I separated the training and test data, considering 80% for training and 20% for testing.

## _*Model Training*_

Being a regression problem, I used the Linear Regression algorithm.

## _*Performance Metrics*_

In regression problems, the metric is estimated through error calculation. In this case, I used MAE (Mean Absolute Error) and MAPE (Mean Absolute Percentage Error) which returned approximately 25% error.