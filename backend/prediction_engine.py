import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, r2_score, recall_score
from xgboost import XGBClassifier


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


def benchmark_churn_models():
    features, labels = generate_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    candidates = {
        "logistic_regression": LogisticRegression(),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(eval_metric="logloss", random_state=42),
    }

    feature_names = ["days_since_login", "support_tickets", "price"]
    results = {}

    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        predictions = candidate.predict(X_test)

        if hasattr(candidate, "coef_"):
            importance = dict(zip(feature_names, candidate.coef_[0].tolist()))
        else:
            importance = dict(zip(feature_names, candidate.feature_importances_.tolist()))

        results[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "feature_importance": importance,
        }

    return results


def generate_growth_training_data(n=200, seed=7):
    rng = np.random.default_rng(seed)
    marketing_spend = rng.uniform(0, 5000, n)
    price = rng.choice([20, 30, 40, 50, 80], n)
    existing_customers = rng.integers(0, 2000, n)

    # Diminishing returns on marketing spend, organic word-of-mouth from existing base, price sensitivity.
    organic_growth = existing_customers * 0.01
    marketing_growth = 0.8 * np.sqrt(marketing_spend)
    price_penalty = price * 0.05
    noise = rng.normal(0, 3, n)

    new_customers = np.maximum(0, organic_growth + marketing_growth - price_penalty + noise)

    features = np.column_stack([marketing_spend, price, existing_customers])
    return features, new_customers


def train_growth_model():
    features, labels = generate_growth_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=7
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }
    return model, metrics


def predict_new_customers(model, marketing_spend, price, existing_customers):
    features = np.array([[marketing_spend, price, existing_customers]])
    prediction = model.predict(features)[0]
    return max(0, round(prediction))


def generate_fundraising_training_data(n=200, seed=99):
    rng = np.random.default_rng(seed)
    growth_rate = rng.uniform(-0.2, 0.5, n)
    runway_months = rng.uniform(0, 24, n)
    revenue = rng.uniform(0, 50000, n)

    # Investors reward growth most, then runway (survivability), then absolute revenue.
    linear_score = 8 * growth_rate + 0.15 * runway_months + 0.00005 * revenue - 2.5
    success_probability = 1 / (1 + np.exp(-linear_score))
    raised = rng.binomial(1, success_probability)

    features = np.column_stack([growth_rate, runway_months, revenue])
    return features, raised


def train_fundraising_model():
    features, labels = generate_fundraising_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=99
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


def predict_fundraising_success(model, growth_rate, runway_months, revenue):
    features = np.array([[growth_rate, runway_months, revenue]])
    return model.predict_proba(features)[0][1]


if __name__ == "__main__":
    model, metrics = train_churn_model()
    print("Test set metrics:", metrics)
    print("Learned weights (days_since_login, support_tickets, price):", model.coef_)

    low_risk = predict_churn_probability(model, days_since_login=3, support_tickets=0, price=30)
    high_risk = predict_churn_probability(model, days_since_login=80, support_tickets=4, price=80)
    print("Low-risk customer churn probability:", low_risk)
    print("High-risk customer churn probability:", high_risk)
