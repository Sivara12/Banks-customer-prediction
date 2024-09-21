# Bank Customer Attrition Prediction

## Problem Statement
Service-oriented businesses, such as banks, must address the challenge of customer attrition, where clients switch to competitors. Identifying the key factors that drive a customer's decision to leave is crucial for improving retention.

## Objective
The goal of this project is to build a neural network-based classifier that can determine whether a customer will leave the bank or not.

## Data
The dataset contains various features related to bank customers, which are used to predict whether they are likely to leave the bank or not.

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

## Approach
We aim to build a neural network model that uses customer data to predict whether a customer will leave the bank within six months.

### Steps:
1. Data Preprocessing
2. Model Building (Neural Network Classifier)
3. Evaluation of the Model's Performance

## Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Sivara12/Banks-customer-prediction.git
