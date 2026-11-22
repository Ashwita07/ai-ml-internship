
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from data_preprocessing import load_and_preprocess_data


# Project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# MLflow configuration
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_DIR / 'mlflow.db'}")
mlflow.set_experiment("Breast_Cancer_Classification")


def train_model():

    # Load preprocessed data
    X_train, X_test, y_train, y_test, scaler = (
        load_and_preprocess_data()
    )

    # Initialize model
    model = LogisticRegression(max_iter=1000)

    with mlflow.start_run():

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log parameters
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        # Log trained model
        mlflow.sklearn.log_model(
            model,
            name="logistic_regression_model"
        )

        # Log the scaler
        mlflow.sklearn.log_model(
            scaler,
            name="standard_scaler"
        )

        # Print results
        print("\nModel Evaluation Results")
        print("------------------------")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_model()
