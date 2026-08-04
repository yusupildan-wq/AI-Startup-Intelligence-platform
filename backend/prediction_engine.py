import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def generate_training_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    days_since_login = rng.integers(0, 90, n)
    support_tickets = rng.integers(0, 5, n)
    price = rng.choice([30, 50, 80], n)

    # Known pattern baked in on purpose, so we can check afterward whether the model actually learned it.
    linear_score = 0.05 * days_since_login + 0.3 * support_tickets + 0.01 * price - 5
    churn_probability = 1 / (1 + np.exp(-linear_score))
    churned = rng.binomial(1, churn_probability)

    features = np.column_stack([days_since_login, support_tickets, price])
    return features, churned


def train_churn_model():
    features, labels = generate_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
    }
    return model, metrics


def predict_churn_probability(model, days_since_login, support_tickets, price):
    features = np.array([[days_since_login, support_tickets, price]])
    return model.predict_proba(features)[0][1]


if __name__ == "__main__":
    model, metrics = train_churn_model()
    print("Test set metrics:", metrics)
    print("Learned weights (days_since_login, support_tickets, price):", model.coef_)

    low_risk = predict_churn_probability(model, days_since_login=3, support_tickets=0, price=30)
    high_risk = predict_churn_probability(model, days_since_login=80, support_tickets=4, price=80)
    print("Low-risk customer churn probability:", low_risk)
    print("High-risk customer churn probability:", high_risk)
