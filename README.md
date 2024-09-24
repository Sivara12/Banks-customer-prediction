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


The **CreditScore** histogram shows a right-skewed distribution, where most customers have a credit score between **600 and 750**, with the highest concentration around **650-700**. This indicates that the majority of customers have moderate to high creditworthiness. 

The left tail of the distribution shows a smaller number of customers with low credit scores, below **500**. The right side has fewer customers with very high credit scores above **800**.

The **boxplot** on the right visualizes the spread and the quartiles of the credit score distribution. Most of the data is concentrated in the middle range, with a few outliers visible on the lower end, indicating some customers with particularly low credit scores below **400**. The absence of outliers on the upper end confirms that extreme high scores are less frequent.


![image](https://github.com/user-attachments/assets/e5418dd9-b943-4ffd-868a-62f7c3b18286)
The **Age** histogram displays a right-skewed distribution where the majority of customers are concentrated between **30 and 40 years old**. The highest concentration is around **35 years**, indicating that the bank has a younger customer base.

The frequency drops significantly as age increases, with very few customers older than **60 years**. The curve suggests a gradual decline in customer count as age increases, with a long tail of older customers up to around **90 years**.

The **boxplot** on the right supports this, showing that most customers fall between the ages of **30 and 50 years**. The presence of several outliers beyond **60 years** indicates that there are some older customers, but they are relatively rare. These outliers are marked on the far right, representing customers older than **60**.


![image](https://github.com/user-attachments/assets/92316a75-4f7d-4175-b38d-34fc17bf64ea)

The **Balance** histogram shows a highly skewed distribution with a significant number of customers having a balance of **zero**. This indicates that many customers either do not maintain a balance or have very low account activity. 

For customers with a positive balance, the distribution is bimodal, with one peak around **100,000** units. The tail of the distribution stretches towards **200,000** units, suggesting that while some customers have high balances, they are relatively few.

The **boxplot** further highlights this skewness. The interquartile range (IQR) indicates that the majority of customers hold a balance between **0** and around **120,000** units. There are no extreme outliers, but the right tail shows that a few customers maintain high balances, approaching **250,000** units.

![image](https://github.com/user-attachments/assets/7632a430-9da2-447d-972f-d278c2256a2f)

![image](https://github.com/user-attachments/assets/7cf8d83b-4c50-491a-ac67-4d27bdc13f06)
