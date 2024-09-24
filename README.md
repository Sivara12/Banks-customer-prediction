# Bank Customer Churn Prediction

## Problem Statement
Customer attrition, also known as churn, is a significant issue for service-oriented businesses like banks. Customers switching to competing institutions can lead to substantial revenue loss. As a result, banks must proactively identify at-risk customers to implement retention strategies. Understanding the key factors influencing a customer's decision to leave is crucial for minimizing churn and enhancing customer loyalty. Predicting customer churn effectively allows businesses to take preventive action, reducing the risk of losing valuable customers.

## Objective
The goal of this project is to build a neural network-based classifier that can determine whether a customer will leave the bank.

## Data
The dataset contains various features related to bank customers, which are used to predict whether they are likely to leave the bank.

### Data Dictionary:
| Feature        | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `CustomerId`     | A unique identifier assigned to each customer                              |
| `Surname`        | The customer's family name                                                  |
| `CreditScore`    | Represents the customer’s creditworthiness based on their financial history |
| `Geography`      | Indicates the customer’s country or region                                  |
| `Gender`         | Specifies the customer's gender                                             |
| `Age`            | The customer’s age in years                                                 |
| `Tenure`         | The length of time (in years) the customer has been with the bank           |
| `NumOfProducts`  | Total number of banking products the customer holds                         |
| `Balance`        | The customer’s current account balance                                      |
| `HasCrCard`      | Binary indicator for whether the customer holds a credit card (Yes/No)      |
| `EstimatedSalary`| Estimated annual income of the customer                                     |
| `isActiveMember` | Binary indicator for active usage of bank products and services             |
| `Exited`         | Whether the customer left the bank (0 = No, 1 = Yes)                        |


### Project Overview

This project focuses on predicting bank customer churn using neural networks. We explored multiple models and optimizers to identify the best-performing model for this classification task. Churn prediction allows banks to take preemptive actions to retain at-risk customers, reducing potential revenue loss. 


![image](https://github.com/user-attachments/assets/126b1f98-93f8-43c6-8114-327562102986)



