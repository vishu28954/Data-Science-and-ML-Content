# Part 1 — Why Decision Trees Exist

In Machine Learning, we often want to make predictions from data.

Examples:

```text
Will a user click an ad?
Will a customer churn?
Is this email spam?
Is this URL junk?
What will the revenue?
What will the house price?
```

A simple model like **Linear Regression** or **Logistic Regression** tries to learn a smooth relationship between features and the target.

For example:

```text
prediction = w1*x1 + w2*x2 + w3*x3 + b
```

This is powerful, but it has one limitation:

> It assumes that the prediction can be represented using a weighted combination of features.

However, many real-world decisions are naturally **rule-based**.

For example:

```text
If user is on mobile
and time is evening
and publisher category is sports
then click probability may be high.
```

Another example:

```text
If page has very low content
and many redirects
and high spam signal
then URL may be junk.
```

This kind of reasoning cannot always be represented naturally using a single straight line or linear decision boundary.

Decision Trees exist because they can learn **if-else rules directly from data**.

---

# Core Intuition

A Decision Tree is similar to a **flowchart**.

For example:

```text
Is previous CTR > 0.10?
        |
      Yes/No
        |
Is device mobile?
        |
      Yes/No
        |
Final prediction
```

Instead of manually writing these rules, the Decision Tree learns them automatically from the training data.

So, a Decision Tree is basically:

> A machine-learned if-else rule system.

### Memory Hook

```text
Decision Tree = learned if-else rules.
```

---

# Why Is This Useful?

Decision Trees are useful because they can naturally model:

- Non-linear relationships
- Feature interactions
- Threshold effects
- Rule-like behavior

## 1. Threshold Effects

Consider age as a feature.

```text
If age < 25:
    behavior may be different.

If age >= 25:
    behavior may be different.
```

The effect of age does not necessarily have to increase or decrease smoothly.

There may instead be specific thresholds where behavior changes.

Decision Trees are naturally designed to discover such thresholds.

---

## 2. Feature Interactions

Consider the following behavior:

```text
Mobile users may click more during the evening.

Desktop users may click more during working hours.
```

Here, the prediction depends on an interaction between:

```text
Device Type
    +
Time of Day
```

A Linear Model does not automatically capture such interactions unless we manually create interaction features.

For example:

```text
mobile_evening = is_mobile * is_evening
```

Decision Trees, however, can naturally discover these interactions through sequential splits.

For example:

```text
Is device mobile?
        |
       Yes
        |
Is time evening?
        |
       Yes
        |
Higher click probability
```

---

# Simple Example

Suppose we want to predict whether someone will play badminton.

Our dataset looks like this:

| Weather | Windy | Play? |
| ------- | ----- | ----- |
| Sunny   | No    | Yes   |
| Sunny   | Yes   | No    |
| Rainy   | No    | Yes   |
| Rainy   | Yes   | No    |

A Decision Tree might learn:

```text
Is Windy?
    |
    ├── Yes → No
    |
    └── No  → Yes
```

The model discovered that `Windy` is the important feature for making the prediction.

Notice that the `Weather` feature was not even required.

This is one reason Decision Trees are intuitive: we can often directly understand **why the model made a prediction**.

---

# Why Not Always Use Decision Trees?

Single Decision Trees have one major problem:

> **They overfit easily.**

A Decision Tree can continue creating increasingly specific rules in order to perfectly classify the training data.

For example:

```text
If user_id = 129381
and timestamp = Tuesday 8:03 PM
and device = iPhone
then clicked
```

Such a rule may perfectly explain one observation in the training dataset.

However, it is unlikely to generalize to new users.

The tree has essentially **memorized the training data** instead of learning a general pattern.

Therefore, Decision Trees are powerful but can also be unstable.

---

# From Decision Trees to Ensemble Models

The tendency of Decision Trees to overfit is one of the main reasons ensemble methods were developed.

A simplified evolution looks like this:

```text
Decision Tree
    ↓
Random Forest
    ↓
Gradient Boosting
    ↓
XGBoost / LightGBM / CatBoost
```

Many of the most powerful Machine Learning algorithms for **tabular data** are built using Decision Trees as their fundamental building blocks.

---

# What Are Decision Trees Good At?

Decision Trees are especially useful for **tabular or structured data**.

Examples include:

```text
Customer data
Transaction data
Ad impression logs
User behavior features
Medical records
Financial risk features
Product metrics
```

The features may look like:

```text
age
country
device_type
CTR
income
number_of_visits
previous_purchases
error_ratio
dwell_time
```

This is why tree-based models are extremely common in:

- Applied Machine Learning
- Business Machine Learning
- Fraud Detection
- Recommendation Systems
- Ad Prediction
- Ranking Systems
- Risk Modeling
- Customer Churn Prediction

---

# First-Level Summary

Decision Trees exist because many prediction problems can naturally be represented as a **sequence of decisions**.

They learn rule-based splits directly from data.

Decision Trees can naturally capture:

- Non-linear relationships
- Feature interactions
- Threshold effects
- Rule-based behavior

They are also highly interpretable compared with many other Machine Learning models.

However, a single Decision Tree can easily overfit the training data.

This limitation motivates more powerful ensemble methods such as:

- **Random Forest**
- **Gradient Boosting**
- **XGBoost**
- **LightGBM**
- **CatBoost**

---

# Memory Hooks

```text
Decision Tree = learned if-else rules.

Random Forest = many trees to reduce overfitting.

Boosting = many trees correcting previous mistakes.
```
