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

1. **Estimated Salary (Top Left)**:
   - The histogram shows a fairly uniform distribution across all salary ranges, with no significant peaks or dips. This suggests that customers' salaries are evenly spread across different income levels.
   - The boxplot indicates a wide spread of salaries, with most values concentrated between approximately 50,000 and 150,000 units. There are no significant outliers in the data.

2. **Geography (Bottom Left)**:
   - The histogram for geography reveals that the majority of customers are from France (encoded as 0), followed by Spain (2), with Germany (1) having the least representation.
   - The boxplot confirms this distribution, with a higher concentration of customers in France and fewer in Germany and Spain.

3. **Tenure (Bottom Right)**:
   - The histogram shows an almost uniform distribution for the tenure feature, meaning customers have varying years of tenure with the bank. There are no extreme peaks, suggesting a balanced distribution across different tenures.
   - The boxplot shows that most customers have been with the bank for 4 to 6 years, with no significant outliers.

These plots provide a clear view of how customers' salary, geography, and tenure vary within the dataset, helping to understand the data's distribution and balance.

![image](https://github.com/user-attachments/assets/23a63955-f880-4fc0-ae67-cc6b3ea89ad1)


This **Correlation Heatmap** shows the relationship between different features in the dataset. Here's an explanation:

1. **CreditScore** has a weak negative correlation with the target variable **Exited** (-0.027), suggesting that customers with lower credit scores may be slightly more likely to churn, but the correlation is very weak.

2. **Geography** shows a weak positive correlation with **Exited** (0.036), indicating that location may have a minimal influence on churn behavior.

3. **Gender** has a negative correlation with **Exited** (-0.11), indicating that gender might have some influence on churn, with women being slightly more likely to leave.

4. **Age** is more strongly correlated with **Exited** (0.29) compared to other features, meaning that older customers are more likely to churn.

5. **Balance** has a weak positive correlation (0.12) with **Exited**, showing that customers with higher balances may have a slightly higher chance of leaving.

6. **IsActiveMember** has a negative correlation with **Exited** (-0.16), indicating that active members are less likely to churn.

7. **NumOfProducts** is weakly negatively correlated with **Exited** (-0.048), implying that customers with more products are slightly less likely to churn.

The correlations in the heatmap are mostly weak, which suggests that churn is influenced by multiple factors with none having an overwhelming impact on its own. **Age** and **IsActiveMember** show the strongest correlations with **Exited**, providing important indicators for predicting customer churn.


![image](https://github.com/user-attachments/assets/9430f96d-bbad-408c-8031-b7a21d7ed47a)

This is a **pair plot** (also known as a scatterplot matrix), showing the relationships between multiple features in the dataset with respect to the **Exited** variable (0 for customers who did not churn and 1 for those who did). Here's a detailed explanation:

### Key Features:

1. **Diagonal Plots (Distribution for each variable)**:
   - The diagonal shows the distribution (density) plots for individual features.
   - For features like **CreditScore**, **Age**, **Balance**, and **EstimatedSalary**, we can observe separate distributions for customers who exited (in orange) versus those who did not (in blue).
   - For example, in the **Age** distribution, it is evident that customers who churn (orange) tend to be older, while customers who stay (blue) are more evenly distributed across age groups.
   - For **Balance**, customers with zero balance (cluster at zero) are more likely to stay, as the orange density (exited customers) is much smaller in this group.

2. **Scatter Plots (Feature-Feature Relationship)**:
   - Off the diagonal, the scatter plots show relationships between pairs of features, with points colored by the **Exited** status (blue = stayed, orange = exited).
   - **CreditScore vs. Age**: There's no clear pattern differentiating exited and non-exited customers, as both groups are evenly spread.
   - **Balance vs. Age**: There are some customers with zero balance (cluster at zero balance), who largely did not churn (blue points), but the distribution is otherwise dispersed.
   - **Geography vs. Exited**: The pair plot shows the encoded geography feature (0 for France, 1 for Germany, 2 for Spain). There appears to be a higher concentration of churned customers from one region (likely Germany).

3. **Insights from Pairwise Comparisons**:
   - **Age vs. Exited**: Older customers (around 50+) are more likely to churn, as there are more orange points concentrated in this region.
   - **Balance vs. Exited**: Customers with higher balances appear to be more prone to churn. The orange points are more distributed among higher balances.
   - **CreditScore** doesn't show a strong correlation with churn, as churned and non-churned customers appear to have similar credit scores.

4. **Notable Trends**:
   - **Tenure**: The tenure variable shows no clear pattern or correlation with churn. Both churned and non-churned customers have a wide spread across the tenure values.
   - **EstimatedSalary**: Similarly, estimated salary does not show a strong differentiation between churned and non-churned customers, as both groups are evenly distributed across salary ranges.

### Summary:
This pair plot allows us to visually identify patterns and relationships between the features and customer churn. Some key takeaways are:
- **Age** and **Balance** show some clear differences between customers who churned and those who stayed.
- **Geography** may play a role in churn, as some regions seem to have more churned customers.
- Other features like **CreditScore**, **EstimatedSalary**, and **Tenure** don’t exhibit strong correlations with churn on their own.




## Model Performance Comparison and Final Selection

Three neural network models were trained and evaluated:

1. **NN with Stochastic Gradient Descent (SGD) Optimizer**
2. **NN with Adam Optimizer**
3. **NN with Adam Optimizer + Dropout Layers**

The performance of these models was compared using **recall** as the primary metric because the goal is to minimize false negatives (i.e., not missing customers who are likely to churn).

### Performance Comparison - Recall (Training Set):

- **NN with SGD**: 0.5139
- **NN with Adam**: 0.6915
- **NN with Adam & Dropout**: 0.7209

### Performance Comparison - Recall (Validation Set):

- **NN with SGD**: 0.5098
- **NN with Adam**: 0.6176
- **NN with Adam & Dropout**: 0.6176

### Final Test Set Performance:

- **NN with SGD**: 0.0041
- **NN with Adam**: 0.0738
- **NN with Adam & Dropout**: 0.1033

### Confusion Matrix & Classification Report for the Best Model:

| Precision | Recall | F1-Score | Support |
|-----------|--------|----------|---------|
| 0 (Not Churned) | 0.91 | 0.87 | 0.89 | 1593 |
| 1 (Churned) | 0.57 | 0.67 | 0.62 | 407 |
| **Accuracy** | | 0.83 | | 2000 |
| **Macro Avg** | 0.74 | 0.77 | 0.76 | 2000 |
| **Weighted Avg** | 0.84 | 0.83 | 0.84 | 2000 |

### Conclusion on Model Selection:
The **NN with Adam Optimizer + Dropout** model was selected as the final model for deployment. It had the highest recall on both the training and validation sets, showing better generalization and resilience to overfitting thanks to the dropout layers.

- **Training Recall**: 72.1%
- **Validation Recall**: 61.8%
- **Test Recall**: 67% for predicting customer churn

This model was chosen for its ability to identify more at-risk customers while balancing training and validation recall scores, ensuring better performance on unseen data.

---

## Model Insights and Predictions

After selecting the best model, it was run on the dataset to predict customer churn, identifying which existing customers are likely to leave the bank.

### Key Results:

- **Precision (Churn Class)**: 57% 
- **Recall (Churn Class)**: 67% 
- **Overall Accuracy**: 83%

### Prediction of At-Risk Customers:
- The final model predicted that **901 existing customers** are likely to leave the bank based on their characteristics.

This insight is valuable for the bank, as targeted retention strategies can be applied to these customers, potentially preventing them from leaving.

---

## Final Summary:

We successfully built and evaluated a neural network classifier that can predict whether a bank customer is likely to churn. The **Adam Optimizer with Dropout** model demonstrated the best performance by achieving high recall rates, ensuring that the majority of at-risk customers were identified. 

### Key Highlights:
1. The **dropout layer** helped prevent overfitting and improved the model’s generalization ability.
2. The model provides actionable insights by identifying **901 existing customers** who are at risk of churning.
3. The bank can use these insights to deploy personalized retention strategies, such as offering loyalty programs, enhanced customer service, or special offers to these customers.

---

## Recommendations for Future Work:

1. **Model Refinement**:
   - Further tuning of hyperparameters or exploring different architectures (like convolutional or recurrent neural networks) could improve performance.
   
2. **Feature Expansion**:
   - Including additional data such as transaction history, customer service interactions, or customer satisfaction scores may enhance the predictive power of the model.
   
3. **Regular Model Retraining**:
   - Customer behaviour is dynamic, and retraining the model periodically with new data will ensure that predictions remain accurate over time.

## Conclusion:

This project demonstrates how machine learning can be used to predict customer churn in the banking sector. With the **NN with Adam + Dropout** model, the bank can take proactive measures to retain customers and minimize churn, potentially saving over 900 customers from leaving.

